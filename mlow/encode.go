// MLow entropy encoder: the exact inverse of the byte-exact decoder. Given
// the analyzed SmplFrameParams, it reproduces the same range-coder symbol
// stream the decoder consumes, against the same config=0 runtime tables in
// the same field order. Targets the active config=0 path (0x50 frames),
// p3=4, p4=1. Ported from whatsapp-rust (MIT) wacore/src/voip/mlow/encode.rs;
// MPL-2.0 for this port.
package mlow

import (
	"errors"
	"fmt"
)

const smelpEncodeBufBytes = 512

// ErrMlowBufferOverflow is the range-coder buffer-overflow encode failure.
var ErrMlowBufferOverflow = errors.New("mlow encode: range-encoder buffer overflow")

// MlowEncoder is the stateful pure-Go MLow encoder: 60 ms PCM (960 f32 @16
// kHz, ~[-1,1]) -> a wire MLow frame the WhatsApp peer decodes. Emits active
// config=0 (0x50) frames, choosing voiced (LTP) or unvoiced per internal
// frame via analysis-by-synthesis.
type MlowEncoder struct {
	state SmplEncoderState
	clean []float32
	rng   *RangeEncoder
}

// NewMlowEncoder initializes the encoder state.
func NewMlowEncoder() *MlowEncoder {
	return &MlowEncoder{
		rng: NewRangeEncoder(1 + smelpEncodeBufBytes),
	}
}

// Reset clears the cross-frame analysis history (call at a stream
// discontinuity).
func (e *MlowEncoder) Reset() {
	e.state = SmplEncoderState{}
	e.clean = e.clean[:0]
	e.rng.Reset()
}

// Encode encodes one 60 ms frame. Expects exactly 960 samples.
func (e *MlowEncoder) Encode(pcm []float32) ([]byte, error) {
	var output []byte
	if err := e.encodeInto(pcm, &output); err != nil {
		return nil, err
	}
	return output, nil
}

func (e *MlowEncoder) encodeInto(pcm []float32, output *[]byte) error {
	if len(pcm) != opusFrameSamps {
		return fmt.Errorf("mlow encode: expected %d samples (60 ms @16 kHz), got %d",
			opusFrameSamps, len(pcm))
	}
	// Sanitize: NaN -> 0, clamp to [-1,1] (the LPC analysis degenerates on
	// non-finite input).
	e.clean = e.clean[:0]
	for _, s := range pcm {
		if s != s { // NaN
			s = 0.0
		} else if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		e.clean = append(e.clean, s)
	}
	fp := SmplAnalyzeFrameSt(&e.state, e.clean)
	return encodeSmplFrameInto(&fp, e.rng, output)
}

func encodeSmplFrameInto(fp *SmplFrameParams, enc *RangeEncoder, output *[]byte) error {
	const p2, p3, p4 = int32(320), int32(4), int32(1)
	p6 := int32(fp.Config)
	tbl := &LsfBuiltTables().Tables
	mem := SmplMemTables()
	cc := CcTablesTables()
	enc.Reset()
	st := &SmplLsfState{}
	for f := 0; f < 3; f++ {
		ip := &fp.Internal[f]
		encodeSmplLsf(enc, tbl, st, fp.Config, f, &ip.Lsf)
		encodeSmplPulses(enc, cc, p2, p3, p4, p6, ip.Lsf.Stage1, &ip.Pulses)
		// Voiced internal frames emit a pitch block; unvoiced emit a gains
		// block (never both).
		if ip.Lsf.Stage1 == 1 {
			encodeSmplPitch(enc, mem, cc, st, p2, p3, p6, ip.Pulses.Subfr, &ip.Pitch)
		} else {
			encodeSmplGains(enc, cc, p3, ip.Pulses.Subfr, &ip.Gains)
		}
	}
	enc.Done()
	if enc.Err() != 0 {
		return ErrMlowBufferOverflow
	}
	n := enc.ConsumedLen()
	body := enc.Bytes()
	out := (*output)[:0]
	out = append(out, fp.Toc)
	out = append(out, body[:n]...)
	*output = out
	return nil
}

// encodeSmplLsf is the inverse of decode_smpl_lsf: mirror the selector/grid/
// 16-residual/extra reads, mutating st.
func encodeSmplLsf(enc *RangeEncoder, t *SmplTables, st *SmplLsfState, config int, intf int, lsf *SmplLsfParams) {
	sel := 0
	if intf != 0 {
		if st.PrevStage1 != 0 {
			sel = 2
		} else {
			sel = 1
		}
	}
	stage1 := lsf.Stage1
	enc.EncodeCDF(stage1, t.LsfSel[sel])

	enterMatch := intf != 0
	m := enterMatch && stage1 == st.PrevStage1
	if !m {
		st.PrevGainIdx = -1
		st.PrevFiltIdx = -1
		st.PrevLag = -1
		st.PrevFracLag = -1
		st.PrevLagblk = -1
		st.PrevLagidx = -1
	}
	st.PrevStage1 = stage1

	var gridCdf []uint16
	switch {
	case m && stage1 != 0:
		gridCdf = t.LsfGrid.Match1
	case m:
		gridCdf = t.LsfGrid.Match1Alt
	case stage1 != 0:
		gridCdf = t.LsfGrid.Match0Alt
	default:
		gridCdf = t.LsfGrid.Match0
	}
	enc.EncodeCDF(lsf.Grid, gridCdf)
	st.PrevMatch = m
	st.HavePrev = true

	st2 := t.LsfStage2[stage1][config][lsf.Grid]
	for k := 0; k < 16; k++ {
		enc.EncodeCDF(lsf.Stage2[k], st2[k])
	}
	enc.EncodeCDF(lsf.Extra, t.LsfExtra)
}

// encodeSmplPulses is the inverse of decode_smpl_pulses (config=0 NB count,
// p3=4): re-derive the count interval and the split symbols from the
// per-subframe counts, then replay the recorded magnitude/sign symbols.
func encodeSmplPulses(enc *RangeEncoder, cc *CcTables, p2, p3, p4, p6, s1 int32, pp *SmplPulseParams) {
	idx := p4 + s1
	bByte := int32(mem8Static(0xe8990 + uint32(p6*3+idx)))
	frameLen4k := bByte * p2 / 320
	subfrLen16 := frameLen4k / p3
	total := pp.Total

	// pulse COUNT (NB triangular; config=0)
	l := uint32(frameLen4k)
	triT := func(k uint32) uint32 {
		a := (k + 2) * (l + 1)
		b := (k - 1) * (k + 131070) >> 1
		return (a - b) & 0xffff
	}
	ft := triT(l)
	if ft == 0 {
		ft = 1
	}
	fl := uint32(0)
	if total > 0 {
		fl = triT(uint32(total - 1))
	}
	fh := triT(uint32(total))
	enc.Encode(fl, fh, ft)

	if total == 0 {
		return
	}

	// recursive binary SPLIT
	finalSum := pp.Subfr[0] + pp.Subfr[1]
	initSum := total - subfrLen16*2
	if initSum < 0 {
		initSum = 0
	}
	lo := total - 80
	if lo < 0 {
		lo = 0
	}
	if initSum < lo {
		return
	}
	hiBound := total - lo
	if initSum < hiBound {
		n := int(hiBound-initSum) + 2
		enc.EncodeCDFWindow(finalSum-initSum, cc.SplitCmf(total), int(initSum-lo), n)
	}
	if finalSum > 0 {
		encodeSplit3537(enc, cc, finalSum, subfrLen16, pp.Subfr[0])
	}
	if finalSum < total {
		encodeSplit3537(enc, cc, total-finalSum, subfrLen16, pp.Subfr[2])
	}

	// MAGNITUDE block: replay recorded run-length symbols through the same
	// loop.
	posPer := p2 / p3
	magIdx := 0
	for subfr := int32(0); subfr < p3; subfr++ {
		cnt := pp.Subfr[subfr]
		if cnt <= 0 {
			continue
		}
		pos := posPer
		c := cnt
		for k := int32(0); k < cnt; k++ {
			oct := (pos + 7) / 8
			bucket := cc.Runlen(oct)
			start := int(bucket.MaxSamples() - pos)
			m := pp.MagRuns[magIdx]
			magIdx++
			enc.EncodeCDFWindow(m, bucket.Cmf(c), start, int(pos+1))
			if m > 0 || k == 0 {
				pos -= m
			}
			c--
		}
	}

	// SIGN block: replay recorded raw sign symbols.
	for _, rs := range pp.SignSyms {
		enc.EncodeRawSymbol(rs.Sym, rs.Nbits)
	}
}

// encodeSplit3537 is the inverse of smpl_split_3537: encode the first-half
// count s0 against the same CDF.
func encodeSplit3537(enc *RangeEncoder, cc *CcTables, count, granularity, s0 int32) {
	lo := count
	if granularity < lo {
		lo = granularity
	}
	minSplit := count - granularity
	if minSplit < 0 {
		minSplit = 0
	}
	if lo < minSplit || minSplit == lo {
		return
	}
	n := int(lo-minSplit) + 2
	enc.EncodeCDFWindow(s0-minSplit, cc.SplitCmf(count), int(minSplit), n)
}

// encodeSmplGains is the inverse of decode_smpl_gains: encode main/delta
// gain, then per-subframe nrgres with the same gain-derived address shift.
func encodeSmplGains(enc *RangeEncoder, cc *CcTables, p3 int32, subfrCounts [4]int32, gp *SmplGainParams) {
	gainMain := gp.GainMain
	gainDelta := gp.GainDelta
	enc.EncodeCDF(gainMain, cc.NrgresGain4())
	enc.EncodeCDF(gainDelta, cc.NrgresShape4())
	cfgSel := int32(2)

	off6 := p3 * gainDelta
	base7 := gainMain*cc.NrgStep(cfgSel) - 0x154000
	var gainQ [4]int32
	for sf := int32(0); sf < p3; sf++ {
		cbv := cc.GainRecon(p3 == 4, sf+off6)
		gainQ[sf] = base7 + (cbv << 4)
	}

	for sf := int32(0); sf < p3; sf++ {
		cnt := subfrCounts[sf]
		if cnt <= 0 {
			continue
		}
		bucket := int32(3)
		if cnt < 30 {
			bucket = (cnt & 0xffff) / 10
		}
		g := (gainQ[sf] + 8192) >> 14
		if g < -85 {
			g = -85
		}
		negPart := (g >> 31) & g
		minOffset := int(-negPart)
		enc.EncodeCDF(gp.NrgRes[sf], cc.FcbgOffset(int(cfgSel), int(bucket), minOffset))
	}
}

// encodeSmplPitch is the inverse of decode_smpl_pitch: encode the LTP gains,
// primary lag (abs/delta), optional 64-fine, and the per-segment fractional
// symbols, mutating the predictor state identically.
func encodeSmplPitch(enc *RangeEncoder, mem *SmplMem, cc *CcTables, st *SmplLsfState, _p2, p3, p6 int32, subfrCounts [4]int32, pp *SmplPitchParams) {
	gp := mem.GPitch
	// The active WB path is p6==0 (HR tables, served by the logical seed);
	// the p6!=0 LR variant stays on the heap window.
	weightTab := uint32(0xe85b0)
	gainCdfBase := gp + 0x302
	if p6 != 0 {
		weightTab = 0xe8460
		gainCdfBase = gp + 0xc0
	}
	filtCdf0 := gp + 0xdc4
	filtCdf1 := gp + 0xe4c

	var gainAccum int32
	for sf := 0; sf < int(p3) && sf < 4; sf++ {
		cnt := subfrCounts[sf]
		gi := pp.GainIdx[sf]
		if p6 != 0 {
			row := uint32(int32(gainCdfBase) + st.PrevGainIdx*0x22 + 0x22)
			enc.EncodeCDF(gi, mem.CDFAt(row, 17))
		} else {
			enc.EncodeCDF(gi, cc.AcbgainRow(st.PrevGainIdx))
		}
		st.PrevGainIdx = gi
		var w0, w2 int32
		if p6 != 0 {
			w0 = int32(int16(mem.U16(weightTab + uint32(gi)*4)))
			w2 = int32(int16(mem.U16(weightTab + uint32(gi)*4 + 2)))
		} else {
			w0, w2 = cc.AcbgainWeights(gi)
		}
		gainAccum += w0 + 2*w2
		if cnt > 0 {
			fi := pp.FiltIdx[sf]
			if p6 != 0 {
				if st.PrevFiltIdx == -1 {
					enc.EncodeCDF(fi, mem.CDFAt(filtCdf0, 35))
				} else {
					enc.EncodeCDF(fi, mem.CDFAt(uint32(int32(filtCdf1)-st.PrevFiltIdx*2), 35))
				}
			} else if st.PrevFiltIdx == -1 {
				enc.EncodeCDF(fi, cc.FcbgainV())
			} else {
				enc.EncodeCDF(fi, cc.FcbgainVDelta(st.PrevFiltIdx))
			}
			st.PrevFiltIdx = fi
		}
	}
	avgGain := gainAccum / p3

	// Lag block: write the estimator's chosen contour (blockseg_idx) +
	// per-40-block lag indices (laginds) via smpl_encode_lags_wire, NOT a
	// single-lag contour-map flattening. The delta-lag CMF mode is selected
	// by the mean-ACB-gain thresholds below.
	mode := 2
	if avgGain < 10007 {
		mode = 0
	} else if avgGain < 14085 {
		mode = 1
	}
	tab := LoadPitchTables()
	SmplEncodeLagsWire(tab, enc, pp.BlocksegIdx, &pp.Laginds, st.PrevLagblk, st.PrevLagidx, mode)
	nblk, nidx := SmplLagsPredictorAfter(tab, pp.BlocksegIdx, &pp.Laginds)
	st.PrevLagblk = nblk
	st.PrevLagidx = nidx
}
