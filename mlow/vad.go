// SILK VAD for the MLow encoder: per-internal-frame speech-activity
// probability and the coded_as_active_voice flag the bitrate controller and
// voiced/unvoiced classifier read. Fully fixed-point (i32), ported exactly
// from whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_vad.rs; MPL-2.0 for
// this port. Runs on raw int16 input PCM (no encoder HP), 320 samples per
// internal frame at 16 kHz.
package mlow

const (
	vadNBands                  = 4
	vadInternalSubframesLog2   = 2
	vadInternalSubframes       = 1 << vadInternalSubframesLog2
	vadNoiseLevelSmoothCoefQ16 = 1024
	vadNoiseLevelsBias         = 50
	vadNegativeOffsetQ5        = 128
	vadSNRFactorQ16            = 45000
	// SPEECH_ACTIVITY_DTX_THRES_Q8 = SILK_FIX_CONST(0.05, 8).
	speechActivityDTXThresQ8 = 12
	// 0xFF80_0000 interpreted as i32; masks the top 9 bits for the
	// energy-shift decision in getSaQ8.
	vadNrgHighMask = int32(-0x800000)
)

var vadTiltWeights = [vadNBands]int32{30000, 6000, -12000, -12000}

const (
	aFb120 = 3894 << 1 // i16
	aFb121 = -29322    // i16
)

type vadType uint8

const (
	vadInactive vadType = iota
	vadActive
	vadHangover
)

// SmplVadState is the persistent SILK VAD state, carried across packets.
type SmplVadState struct {
	anaState     [2]int32
	anaState1    [2]int32
	anaState2    [2]int32
	xnrgSubfr    [vadNBands]int32
	nl           [vadNBands]int32
	invNl        [vadNBands]int32
	noiseLvlBias [vadNBands]int32
	counter      int32
	hpState      int32
	// VAD tuning params (default 0 in this config).
	noiseLvlUpdateSpeed int32
	nonBinariness       int32
	highpassSharpness   int32
	// DTX hangover (per packet).
	remainingDtxHangover int32
	hangoverMs           int32
}

// VadPacketResult is the VAD output for one 60 ms packet: per-internal-frame
// speech-activity probability and the packet-level coded_as_active_voice flag.
type VadPacketResult struct {
	VadResults         [3]float32
	CodedAsActiveVoice bool
}

// NewSmplVadState initializes the VAD state.
func NewSmplVadState() *SmplVadState {
	v := &SmplVadState{
		counter:              15,
		remainingDtxHangover: 60, // SMPL_DEFAULT_HANGOVER_MS
		hangoverMs:           60,
	}
	for b := 0; b < vadNBands; b++ {
		bias := maxI32(vadNoiseLevelsBias/(int32(b)+1), 1)
		v.noiseLvlBias[b] = bias
		v.nl[b] = 100 * bias
		v.invNl[b] = silkInt32Max / v.nl[b]
	}
	return v
}

// filtHP is the first-order ARMA HP filter with zero at DC, in place over the
// first n samples.
func (v *SmplVadState) filtHP(x []int32, bQ16, aNegQ16 int32, n int) {
	for i := 0; i < n; i++ {
		inval := smulwb(bQ16, x[i])
		outval := sat16(v.hpState - inval)
		v.hpState = smlawb(inval, aNegQ16, outval)
		x[i] = outval
	}
}

// anaFiltBank1 is the 2-band split via first-order allpass filters. Writes
// the low band to outL[0:n/2] and the high band to outH[0:n/2]; s carries the
// 2-element state.
func anaFiltBank1(inp []int32, s *[2]int32, outL, outH []int32, n int) {
	n2 := n >> 1
	for k := 0; k < n2; k++ {
		in32 := inp[2*k] << 10
		y := in32 - s[0]
		xx := smlawb(y, y, aFb121)
		out1 := s[0] + xx
		s[0] = in32 + xx

		in32 = inp[2*k+1] << 10
		y = in32 - s[1]
		xx = smulwb(y, aFb120)
		out2 := s[1] + xx
		s[1] = in32 + xx

		outL[k] = sat16(rshiftRound(out2+out1, 11))
		outH[k] = sat16(rshiftRound(out2-out1, 11))
	}
}

// anaFiltBank1Inplace is the in-place 2-band split: reads x[0:n], writes the
// low band to x[0:n/2] and the high band to x[hiOff:hiOff+n/2]. Input and
// low-out overlap; in[2k]/in[2k+1] are read before out[k] is written, so a
// forward scan is safe since k <= 2k.
func anaFiltBank1Inplace(x []int32, hiOff int, s *[2]int32, n int) {
	n2 := n >> 1
	for k := 0; k < n2; k++ {
		in32 := x[2*k] << 10
		y := in32 - s[0]
		xx := smlawb(y, y, aFb121)
		out1 := s[0] + xx
		s[0] = in32 + xx

		in32 = x[2*k+1] << 10
		y = in32 - s[1]
		xx = smulwb(y, aFb120)
		out2 := s[1] + xx
		s[1] = in32 + xx

		x[hiOff+k] = sat16(rshiftRound(out2-out1, 11))
		x[k] = sat16(rshiftRound(out2+out1, 11))
	}
}

// getNoiseLevels smooths the per-band noise-level estimate toward the current
// band energies.
func (v *SmplVadState) getNoiseLevels(px *[vadNBands]int32) {
	var minCoef int32
	if v.counter < 1000 {
		minCoef = silkInt16Max / ((v.counter >> 4) + 1)
		v.counter++
	}
	for b := 0; b < vadNBands; b++ {
		nl := v.nl[b]
		nrg := addPosSat32(px[b], v.noiseLvlBias[b])
		invNrg := silkInt32Max / nrg
		var coef int32
		switch {
		case nrg > nl<<3:
			coef = vadNoiseLevelSmoothCoefQ16 >> 3
		case nrg < nl:
			coef = vadNoiseLevelSmoothCoefQ16
		default:
			coef = smulwb(smulww(invNrg, nl), vadNoiseLevelSmoothCoefQ16<<1)
		}
		coef = coef * (100 + v.noiseLvlUpdateSpeed) / 100
		coef = maxI32(coef, minCoef)
		v.invNl[b] = smlawb(v.invNl[b], invNrg-v.invNl[b], coef)
		v.nl[b] = minI32(silkInt32Max/v.invNl[b], 0x00FF_FFFF)
	}
}

// getSaQ8 returns speech_activity_Q8 for one frameLen-sample int16 frame
// (passed as int32).
func (v *SmplVadState) getSaQ8(pIn []int32, frameLen int) int32 {
	decFl1 := frameLen >> 1
	decFl2 := frameLen >> 2
	decFl3 := frameLen >> 3

	var xOffset [vadNBands]int
	xOffset[1] = decFl3 + decFl2
	xOffset[2] = xOffset[1] + decFl3
	xOffset[3] = xOffset[2] + decFl2
	xTotal := xOffset[3] + decFl1
	x := make([]int32, xTotal)

	// 0-8 kHz -> 0-4 kHz (x[0:]) and 4-8 kHz (x[offset3:]).
	s := v.anaState
	anaFiltBank1(pIn, &s, x[:xOffset[3]], x[xOffset[3]:], frameLen)
	v.anaState = s
	// 0-4 kHz -> 0-2 kHz (x[0:]) and 2-4 kHz (x[offset2:]).
	s1 := v.anaState1
	anaFiltBank1Inplace(x, xOffset[2], &s1, decFl1)
	v.anaState1 = s1
	// 0-2 kHz -> 0-1 kHz (x[0:]) and 1-2 kHz (x[offset1:]).
	s2 := v.anaState2
	anaFiltBank1Inplace(x, xOffset[1], &s2, decFl2)
	v.anaState2 = s2

	// HP filter on the lowest band, -3 dB @ 66 Hz.
	aNegQ16 := int32(53084) * (100 - v.highpassSharpness) / 100
	bQ16 := (65536 + aNegQ16) / 2
	v.filtHP(x[:decFl3], bQ16, aNegQ16, decFl3)

	// Energy in each band.
	var xnrg [vadNBands]int32
	for b := 0; b < vadNBands; b++ {
		dec := frameLen >> uint(min(vadNBands-b, vadNBands-1))
		decSubfrLen := dec >> vadInternalSubframesLog2
		decSubfrOffset := 0
		xnrg[b] = v.xnrgSubfr[b]
		var sumSquared int32
		for sf := 0; sf < vadInternalSubframes; sf++ {
			sumSquared = 0
			for i := 0; i < decSubfrLen; i++ {
				xTmp := x[xOffset[b]+i+decSubfrOffset] >> 3
				sumSquared = smlabb(sumSquared, xTmp, xTmp)
			}
			if sf < vadInternalSubframes-1 {
				xnrg[b] = addPosSat32(xnrg[b], sumSquared)
			} else {
				xnrg[b] = addPosSat32(xnrg[b], sumSquared>>1)
			}
			decSubfrOffset += decSubfrLen
		}
		v.xnrgSubfr[b] = sumSquared
	}

	v.getNoiseLevels(&xnrg)

	// Signal-plus-noise to noise ratio.
	var sumSquared, inputTilt int32
	for b := 0; b < vadNBands; b++ {
		speechNrg := xnrg[b] - v.nl[b]
		if speechNrg <= 0 {
			continue
		}
		var ratioQ8 int32
		if xnrg[b]&vadNrgHighMask == 0 {
			ratioQ8 = div32(xnrg[b]<<8, v.nl[b]+1)
		} else {
			ratioQ8 = div32(xnrg[b], (v.nl[b]>>8)+1)
		}
		snrQ7 := lin2log(ratioQ8) - 8*128
		sumSquared = smlabb(sumSquared, snrQ7, snrQ7)
		if speechNrg < 1<<20 {
			snrQ7 = smulwb(sqrtApprox(speechNrg)<<6, snrQ7)
		}
		inputTilt = smlawb(inputTilt, vadTiltWeights[b], snrQ7)
	}
	_ = inputTilt // input_tilt_Q15 unused downstream here
	sumSquared = sumSquared / vadNBands
	pSnrDbQ7 := int32(int16(3 * sqrtApprox(sumSquared)))

	vadSnrFactorQ16 := vadSNRFactorQ16 * (150 - v.nonBinariness) / 150
	saQ15 := sigmQ15(smulwb(vadSnrFactorQ16, pSnrDbQ7) - vadNegativeOffsetQ5)

	return minI32(saQ15>>7, 0xFF)
}

func div32(a, b int32) int32 { return a / b }

// ProcessPacket processes one 60 ms packet (3 internal frames of frameLen
// int16 samples). The activity input mirrors the Opus VAD NO_DECISION path,
// so the per-frame type is threshold-driven, then hangover runs.
func (v *SmplVadState) ProcessPacket(pcm []int16, frameLen int) VadPacketResult {
	const framesPerPacket = 3
	const packetMs = 60
	var res VadPacketResult
	// Reject short packets up front so the frame loop's fixed-stride
	// indexing can't panic.
	if len(pcm) < framesPerPacket*frameLen {
		return res
	}
	var vadTypes [3]vadType
	for i := 0; i < framesPerPacket; i++ {
		t := i * frameLen
		frame := make([]int32, frameLen)
		for j, smp := range pcm[t : t+frameLen] {
			frame[j] = int32(smp)
		}
		saQ8 := v.getSaQ8(frame, frameLen)
		res.VadResults[i] = float32(saQ8) / 256.0
		// activity == NO_DECISION path.
		if saQ8 > speechActivityDTXThresQ8 {
			vadTypes[i] = vadActive
		}
	}

	for i := range vadTypes {
		if vadTypes[i] == vadActive {
			v.remainingDtxHangover = v.hangoverMs
		} else if v.remainingDtxHangover > 0 {
			vadTypes[i] = vadHangover
			v.remainingDtxHangover -= packetMs / framesPerPacket
		}
		if vadTypes[i] != vadInactive {
			res.CodedAsActiveVoice = true
		}
	}
	return res
}
