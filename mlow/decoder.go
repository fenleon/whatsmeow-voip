// MLow top-level decoder: RED strip -> TOC routing -> active-frame decode
// (3 chained 20 ms internal frames: LSF -> pulses -> pitch/gains -> CELP
// synthesis) -> 60 ms PCM. The synthesis runs the excitation in the codec's
// float domain (gen_noise + LPC synthesis). The cross-frame predictor and
// synthesis history persist across calls because the stream is continuous.
// Ported from whatsapp-rust (MIT) wacore/src/voip/mlow/decoder.rs; MPL-2.0
// for this port.
package mlow

// opusFrameSamps is 60 ms @ 16 kHz.
const opusFrameSamps = 960

// MlowDecoder is the stateful MLow decoder. Decode consumes one RTP payload
// (a bare MLow frame, or a SplitRed packet when redundancy was negotiated)
// into a 60 ms / 960-sample PCM frame at 16 kHz, float in [-1, 1].
type MlowDecoder struct {
	lstate     SmplLsfState
	prevNLSF   []float32
	celp       *CelpDecState
	hp         *HpPostfilterState
	harm       *HarmPostfilterState
	redundancy int32
	// hadError is sticky: set whenever the inner range decoder raised its
	// error flag during any decode. Diagnostic only, never gates output.
	hadError bool
	// droppedUnsupported counts inbound frames dropped because they fall
	// outside this decoder's single operating point (16 kHz wideband,
	// low_rate=0, 60 ms); such a frame would desync the range coder.
	droppedUnsupported uint32
}

// NewMlowDecoder creates a fresh decoder.
func NewMlowDecoder() *MlowDecoder {
	return &MlowDecoder{
		celp: NewCelpDecState(),
		hp:   NewHpPostfilterState(),
		harm: NewHarmPostfilterState(),
	}
}

// HadError reports whether any decode since construction (or Reset) raised
// the range decoder's error flag. Diagnostic only.
func (d *MlowDecoder) HadError() bool { return d.hadError }

// DroppedUnsupported is the count of out-of-operating-point drops.
func (d *MlowDecoder) DroppedUnsupported() uint32 { return d.droppedUnsupported }

// SetRedundancy sets the negotiated RED redundancy level (0 = bare frames,
// the common case).
func (d *MlowDecoder) SetRedundancy(n int32) { d.redundancy = n }

// Reset clears the cross-frame state (call at a stream discontinuity).
func (d *MlowDecoder) Reset() {
	d.lstate = SmplLsfState{}
	d.prevNLSF = nil
	d.celp = NewCelpDecState()
	d.hp = NewHpPostfilterState()
	d.harm = NewHarmPostfilterState()
	d.hadError = false
}

// Decode decodes one RTP MLow payload into a 60 ms (960-sample) PCM frame.
func (d *MlowDecoder) Decode(payload []byte) []float32 {
	if len(payload) == 0 {
		return make([]float32, opusFrameSamps)
	}
	if d.redundancy > 0 {
		frames, err := DepackSplitRed(payload)
		if err != nil {
			return make([]float32, opusFrameSamps)
		}
		if len(frames) == 0 {
			return d.decodeFrame(nil)
		}
		// the main (current) frame is last.
		return d.decodeFrame(frames[len(frames)-1].Data)
	}
	return d.decodeFrame(payload)
}

func zeros(n int) []float32 { return make([]float32, n) }

func (d *MlowDecoder) decodeFrame(frame []byte) []float32 {
	if len(frame) == 0 {
		return zeros(opusFrameSamps)
	}
	toc := ParseMlowToc(frame[0])
	if toc.StdOpus {
		return zeros(int(16000 / 1000 * toc.FrameMs))
	}
	// Inactive / SID (DTX/CNG) frames carry no decodable voice and are
	// silenced without opening the range coder, so their geometry can never
	// desync. A full 60 ms slot keeps the playout cadence.
	if toc.SID || !toc.Active {
		return zeros(opusFrameSamps)
	}
	// Operating-point guard for active frames: an active frame at a different
	// internal rate, the low_rate=1 geometry, or a non-60 ms duration would
	// desync the range coder. Drop it as a lost frame so the predictor holds
	// its last good values.
	offPoint := false
	switch {
	case toc.SampleRate != 16000:
		offPoint = true
	case toc.Flag2: // the smpl TOC's low_rate bit
		offPoint = true
	case toc.FrameMs != 60:
		offPoint = true
	}
	if offPoint {
		d.droppedUnsupported++
		return zeros(opusFrameSamps)
	}
	return d.decodeActiveFrame(frame, opusFrameSamps)
}

func (d *MlowDecoder) decodeActiveFrame(frame []byte, outLen int) []float32 {
	config := int(frame[0]>>2) & 1
	built := LsfBuiltTables()
	mem := SmplMemTables()
	cc := CcTablesTables()
	dec := NewRangeDecoder(frame[1:])

	// The low_rate bit of the smpl TOC (the synth gates on it).
	lowRate := frame[0]>>2&1 != 0

	out := make([]float32, 0, 3*smplIntfLen)
	// Collect the per-40-block lags (8 per frame, 24 per packet) and the
	// average normalized bitrate for the per-packet harmonic postfilter.
	packetLags := make([]float32, 0, 3*8)
	avgNormBr := float32(0.0)
	for f := 0; f < 3; f++ {
		lsf := DecodeSmplLsf(dec, &built.Tables, &d.lstate, config, f)
		pulses := DecodeSmplPulses(dec, cc, smplIntfLen, 4, 1, int32(config), lsf.Stage1)
		voiced := lsf.Stage1 == 1
		params := CelpDecParams{
			Voiced:      voiced,
			SfPulses:    pulses.Subfr,
			TotalPulses: pulses.Subfr[0] + pulses.Subfr[1] + pulses.Subfr[2] + pulses.Subfr[3],
		}
		if voiced {
			pr := DecodeSmplPitch(dec, mem, cc, &d.lstate, smplIntfLen, 4, int32(config), pulses.Subfr)
			// lag = laginds*0.5 + SMPL_MIN_PITCH_LAG, clamped; one per
			// 40-block, 8 per frame.
			for b := 0; b < 8; b++ {
				v := float64(pr.BlockLags[b])*0.5 + 32.0
				if v > 320.0 {
					v = 320.0
				}
				params.BlockLags[b] = float32(v)
			}
			for sf := 0; sf < 4; sf++ {
				params.AcbgIdx[sf] = pr.GainIdx[sf]
				// The voiced FCB gain index is decoded in the pitch block
				// (filt_idx).
				if pr.FiltIdx[sf] > 0 {
					params.FcbgIdx[sf] = pr.FiltIdx[sf]
				}
			}
		} else {
			g := DecodeSmplGains(dec, cc, 4, pulses.Subfr)
			// The unvoiced gains decode yields gain_q (the nrgres_dbq_Q14
			// field) and nrg_res (the fcbg_idx field).
			params.NrgresDbqQ14 = g.GainQ
			params.FcbgIdx = g.NrgRes
		}
		packetLags = append(packetLags, params.BlockLags[:]...)
		avgNormBr += GetNormalizedBitrate(params.TotalPulses, smplIntfLen)

		nlsf := ReconstructNLSF(&built.Synth, int(lsf.Stage1), config, int(lsf.Grid), &lsf.Stage2, d.prevNLSF)
		var sig [smplIntfLen]float32
		d.celp.SynthFrame(nlsf, int(lsf.Extra), pulses.Pulses, &params, lowRate, smplIntfLen, sig[:], d.hp)
		d.prevNLSF = nlsf
		out = append(out, sig[:]...)
	}

	// Per-packet harmonic postfilter (the codec's final pitch comb +
	// 48-sample group delay), run once over the whole packet with the 24
	// per-40-block lags and the average normalized bitrate.
	HarmPostfilter(d.harm, out, len(out), packetLags, len(packetLags), avgNormBr/3.0)

	// The C-domain synthesis output is already float in [-1, 1]; clamp.
	for i, v := range out {
		if v < -1.0 {
			out[i] = -1.0
		} else if v > 1.0 {
			out[i] = 1.0
		}
	}
	if outLen > 0 && outLen != len(out) {
		if outLen > len(out) {
			out = append(out, make([]float32, outLen-len(out))...)
		} else {
			out = out[:outLen]
		}
	}
	if dec.Err != 0 {
		// Sticky flag; does not alter out (the frame still plays).
		d.hadError = true
	}
	return out
}
