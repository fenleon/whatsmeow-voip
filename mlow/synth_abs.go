// The AbyS (analysis-by-synthesis) synthesis half of smpl_synth.rs: the
// standalone low-band core the ENCODER's analysis mirrors (func 3523/3522
// fractional LTP, func 3507 FIR16, LPC synthesis, func 3597 gain). Ported
// from whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_synth.rs:261-536;
// MPL-2.0 for this port. Note: SMPL_FIR16 here is the long-literal f32 copy —
// NOT bit-identical to the decoder's smplInterpolKernel short literals, so it
// gets its own table.
package mlow

import "math"

const (
	smplLtpHist      = 728
	smplFracStateLen = 728
	ltpHistLen       = smplLtpHist + smplIntfLen + 64
)

// The long-literal f32 copy of the 16-tap symmetric LTP interpolation kernel
// (smpl_synth.rs SMPL_FIR16); distinct from celpdec.go's short literals.
var smplSynthFIR16 = [16]float32{
	-0.000006392598606907995,
	0.00011064113641623408,
	-0.0009153038263320923,
	0.0048477197997272015,
	-0.018698347732424736,
	0.05759090930223465,
	-0.15997476875782013,
	0.617045521736145,
	0.6170454621315002,
	-0.15997475385665894,
	0.05759090557694435,
	-0.018698347732424736,
	0.0048477197997272015,
	-0.0009153038263320923,
	0.00011064114369219169,
	-0.0000063925981521606445,
}

// smplLpcSynthesis runs the LPC synthesis over ex, carrying the previous
// order outputs (f64 accumulation).
func smplLpcSynthesis(ex []float32, a []float32, out []float32, state []float32) {
	const order = smplOrder
	for n := 0; n < len(ex); n++ {
		acc := float64(ex[n])
		for j := 1; j <= order; j++ {
			var prev float64
			if n >= j {
				prev = float64(out[n-j])
			} else {
				prev = float64(state[order+n-j])
			}
			acc -= float64(a[j]) * prev
		}
		out[n] = float32(acc)
	}
	if len(out) >= order {
		copy(state[:order], out[len(out)-order:])
	}
}

// SmplGainLin is the func 3597 gain: quantized log-gain -> linear gain (fast
// pow2 bit-cast).
func SmplGainLin(gainQ int32) float64 {
	y := float32(gainQ)*6.103515625e-05*0.10000000149011612*27749388.0 + 1064866816.0
	i := int32(-2147483648)
	if y < 2147483648.0 && y > -2147483648.0 {
		i = int32(y)
	}
	f := float32(math.Float32frombits(uint32(i))) - 3.1622775509276835e-09
	if f < 0.0 {
		f = 0.0
	}
	return float64(f)
}

func smplFloorF32(x float32) float32 {
	i := int32(x)
	if float32(i) > x {
		i--
	}
	return float32(i)
}

// smplFir8 is the func 3507 8-tap symmetric FIR16 application, IN PLACE over
// sig (the WASM passes in==out, so overlapping read/write regions must see
// prior writes). f32 accumulation order matches the WASM.
func smplFir8(sig []float32, inBase, outBase, cnt int32) {
	for jj := int32(0); jj < cnt; jj++ {
		var acc float32
		for i := int32(0); i < 8; i++ {
			acc += (sig[inBase+jj+i] + sig[inBase+jj+15-i]) * smplSynthFIR16[i]
		}
		sig[outBase+jj] = acc
	}
}

// smplFracLtp is func 3523 (fractional LTP + interpolation). Reads sig
// backward from sigEnd and writes two regions per subframe into out (len
// 2*numSubfr*40); also mutates sig in place.
func smplFracLtp(lag []float32, numSubfr int32, sig []float32, sigEnd int32, stateLen int32, out []float32) {
	lb := sigEnd - (40*numSubfr - stateLen)
	for sf := int32(0); sf < numSubfr; sf++ {
		fl := smplFloorF32(lag[sf])
		intLag := int32(fl)
		if float32(intLag) == lag[sf] {
			for k := int32(0); k < 40; k++ {
				sig[lb+k] = sig[lb+k-intLag]
			}
			for k := int32(0); k < 40; k++ {
				out[sf*40+k] = sig[lb+k]
				out[(numSubfr+sf)*40+k] = sig[lb+k-intLag-1] + sig[lb+k-intLag+1]
			}
		} else {
			b := (numSubfr + sf) * 40
			for k := int32(0); k < 40; k++ {
				out[b+k] = sig[lb-intLag-1+k] + sig[lb-intLag+1+k]
			}
			var l10 float32
			for j := 0; j < 16; j++ {
				l10 += sig[lb-9-intLag+int32(j)] * smplSynthFIR16[j]
			}
			smplFir8(sig, lb-intLag-8, lb, 40)
			var l11 float32
			for j := 0; j < 16; j++ {
				l11 += sig[lb+32-intLag+int32(j)] * smplSynthFIR16[j]
			}
			for k := int32(0); k < 40; k++ {
				out[sf*40+k] = sig[lb+k]
			}
			out[b] = l10 + sig[lb+1]
			for k := int32(0); k < 38; k++ {
				out[b+1+k] = sig[lb+k] + sig[lb+2+k]
			}
			out[b+39] = l11 + sig[lb+38]
		}
		lb += 40
	}
}

// SmplExcGainState is the func 3522 (per-subframe LTP gain-apply) state,
// reset per internal frame.
type SmplExcGainState struct {
	S0, S1 float32
}

func smplExcGainApply(subLen int, input []float32, st *SmplExcGainState, out []float32, gain float32) {
	if gain != 0.0 {
		s5 := st.S1
		s6 := (s5 + s5) + st.S0
		d := st.S0 - s5
		absD := f32Abs(d)
		absS6 := f32Abs(s6)
		mn := absD + gain
		if absS6 < mn {
			mn = absS6
		}
		t := d * mn / (absD + 1e-12)
		st.S1 = (s6 - t) / 3.0
		st.S0 = (2.0*t + s6) / 3.0
	}
	if subLen == 0 {
		return
	}
	s0 := st.S0
	for n := 0; n < subLen; n++ {
		out[n] = s0 * input[n]
	}
	s1 := st.S1
	for n := 0; n < subLen; n++ {
		out[n] += s1 * input[subLen+n]
	}
}

// SmplLtpFracGain is the func 3597 offset 4342 fractional-LTP gain:
// normGain*-0.17 + 0.35.
func SmplLtpFracGain(normGain float64) float32 {
	return float32(normGain)*-0.16999998688697815 + 0.3499999940395355
}

// smplLtpSubframePred is one 80-sample subframe's fractional-LTP prediction
// (func 3523 + func 3522).
func smplLtpSubframePred(hist []float32, histPos int32, lagF, gainFrac float32, gst *SmplExcGainState, predOut []float32) {
	var fracOut [2 * 2 * 40]float32
	lags := [2]float32{lagF, lagF}
	smplFracLtp(lags[:], 2, hist, histPos-648, smplFracStateLen, fracOut[:])
	smplExcGainApply(smplSubfrLen, fracOut[:], gst, predOut, gainFrac)
}

// SmplPitchSynth carries the LTP parameters the synthesis needs for one
// internal frame.
type SmplPitchSynth struct {
	Voiced   bool
	LagSubfr [4]float64 // per-subframe func-3523 lag = intLagQ6[sf]*0.5 + 32
	NormGain float64
}

// SmplFrameSynth is the cross-internal-frame synthesis state (LPC + LTP/
// excitation history + the post-LPC band tail).
type SmplFrameSynth struct {
	lpcState [smplOrder]float32
	ltpHist  []float32
	gst      SmplExcGainState
}

// NewSmplFrameSynth initializes the cross-frame synthesis state.
func NewSmplFrameSynth() *SmplFrameSynth {
	return &SmplFrameSynth{ltpHist: make([]float32, ltpHistLen)}
}

// SynthInternalFrame turns one 20 ms internal frame's decoded parameters into
// 320 LB PCM samples (float, ~int16-scaled). Returns (signal, nlsf); nlsf
// becomes the next frame's prev_nlsf.
func SynthInternalFrame(
	t *SmplSynthTables,
	st *SmplFrameSynth,
	stage1, config, grid int,
	stage2 *[16]int32,
	prevNLSF []float32,
	pulses []int32,
	gainQ [4]int32,
	pitch *SmplPitchSynth,
) (signal []float32, nlsf []float32) {
	nlsf = ReconstructNLSF(t, stage1, config, grid, stage2, prevNLSF)
	a := NLSF2A(nlsf)

	subGain := func(sf int) float64 {
		var gq int32
		if sf < len(gainQ) {
			gq = gainQ[sf]
		}
		return SmplGainLin(gq) * float64(smplSubfrLen)
	}

	ex := make([]float32, smplIntfLen)
	for n := 0; n < smplIntfLen; n++ {
		ex[n] = float32(float64(pulses[n]) * subGain(n/smplSubfrLen))
	}
	hist := st.ltpHist

	if pitch.Voiced {
		const gLtp = float32(0.949999988079071)
		gainFrac := SmplLtpFracGain(pitch.NormGain)
		predOut := make([]float32, smplSubfrLen)
		st.gst = SmplExcGainState{}
		for sf := 0; sf < smplSubfrCount; sf++ {
			lagF := float32(pitch.LagSubfr[sf])
			intLag := int32(lagF)
			if intLag <= 0 {
				from, to := sf*smplSubfrLen, (sf+1)*smplSubfrLen
				copy(hist[smplLtpHist+from:smplLtpHist+to], ex[from:to])
				continue
			}
			exBase := sf * smplSubfrLen
			histPos := int32(smplLtpHist + exBase)
			if intLag > 0 && int(intLag) < smplSubfrLen {
				for n := int(intLag); n < smplSubfrLen; n++ {
					ex[exBase+n] += gLtp * ex[exBase+n-int(intLag)]
				}
			}
			smplLtpSubframePred(hist, histPos, lagF, gainFrac, &st.gst, predOut)
			for n := 0; n < smplSubfrLen; n++ {
				ex[exBase+n] += predOut[n]
			}
			copy(hist[histPos:histPos+smplSubfrLen], ex[exBase:exBase+smplSubfrLen])
		}
	} else {
		copy(hist[smplLtpHist:smplLtpHist+smplIntfLen], ex)
	}

	out := make([]float32, smplIntfLen)
	smplLpcSynthesis(ex, a[:], out, st.lpcState[:])

	// Roll the LTP history forward by one internal frame; clear the forward
	// margin.
	copy(hist, hist[smplIntfLen:smplLtpHist+smplIntfLen])
	for i := smplLtpHist + smplIntfLen; i < ltpHistLen; i++ {
		hist[i] = 0.0
	}
	return out, nlsf
}

const smplVoicedNormGain = float64(1.0)
