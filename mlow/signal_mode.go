// Voiced/unvoiced classifier (encoder side): folds five strengths (pitch
// correlation, VAD, spectral tilt, harmonicity, lag) plus a per-stream
// hysteresis into a single voicing_strength; the encoder codes a frame voiced
// when that is positive and the packet is coded-as-active-voice. Ported from
// whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_signal_mode.rs; MPL-2.0 for
// this port. The shared spectral-harmonicity measure lives in pitch_enc.go.
package mlow

import "math"

var smplVuvWeights = [5]float32{1.0, 0.5, 0.5, 0.7, 0.3}

const (
	smplVuvBias = float32(-0.1038)
	smplVuvHyst = float32(0.05)
	// SMPL_F_LEN / 3 (the transition index between the low/high
	// spectral-tilt bands).
	transitionIx = 257 / 3
)

// smplInvSigmoid is the inverse of gennoise.go's smplSigmoid.
func smplInvSigmoid(x float32) float32 {
	return -f32Ln((1.0 / x) - 1.0)
}

func f32Exp(x float32) float32 { return float32(math.Exp(float64(x))) }

// VuvMode is the per-stream voicing hysteresis + spectral-tilt background
// tracker. The encoder threads one instance across the whole stream.
type VuvMode struct {
	nrgLoBgn    float32
	nrgHiBgn    float32
	voicingPrev float32
	lastLagPrev float32
}

// SmplGetSignalMode combines the five voicing strengths + hysteresis into
// voicing_strength. lags is the per-lag-subframe pitch lag in samples;
// f2 is the power spectrum F2[0..257]. Mutates vuv (background tilt +
// hysteresis state).
func SmplGetSignalMode(
	pitchcorr float32,
	lags []float32,
	avgLag float32,
	harmStrength float32,
	f2 *[pitchFlen]float32,
	spActProb float32,
	vuv *VuvMode,
) float32 {
	corrStrength := smplInvSigmoid(0.1 + 0.75*clamp01(pitchcorr)) // -1.4 .. 1.4
	vadStrength := 0.04 * (1.0 - 1.04/(spActProb+0.04))           // -1 .. 0

	// spectral tilt
	var nrgLo float32
	for i := 2; i < transitionIx; i++ {
		tmp := f2[i] * float32(i+3)
		nrgLo += tmp * float32(transitionIx-i)
	}
	var nrgHi float32
	for i := transitionIx; i < pitchFlen; i++ {
		tmp := f2[i] * float32(i+3)
		nrgHi += tmp * float32(i-transitionIx)
	}
	if vadStrength < -0.1 {
		smthCoef := -0.5 * vadStrength
		vuv.nrgLoBgn += smthCoef * (nrgLo - vuv.nrgLoBgn)
		vuv.nrgHiBgn += smthCoef * (nrgHi - vuv.nrgHiBgn)
	}
	tiltLin := (maxF32(nrgLo-vuv.nrgLoBgn, 0.0) - maxF32(nrgHi-vuv.nrgHiBgn, 0.0)) /
		(nrgLo + nrgHi + 1e-9)
	tiltStrength := tiltLin * tiltLin * tiltLin // cubed to make the measure less binary
	lagStrength := -smplSigmoid(0.25 * (38.0 - avgLag))

	voicingStrength := (smplVuvWeights[0]*corrStrength+
		smplVuvWeights[1]*vadStrength+
		smplVuvWeights[2]*tiltStrength+
		smplVuvWeights[3]*harmStrength+
		smplVuvWeights[4]*lagStrength)/smplSumVec(smplVuvWeights[:], 5) +
		smplVuvBias

	// hysteresis
	if vuv.lastLagPrev > 0.0 {
		tmp := f32Log2(lags[0] / vuv.lastLagPrev)
		if tmp > 0.0 {
			tmp *= 0.5
		}
		vuv.voicingPrev /= 0.4 + tmp*tmp
	}
	voicingStrength += vuv.voicingPrev * smplVuvHyst
	vuv.voicingPrev = f32Tanh(3.0 * voicingStrength)
	vuv.lastLagPrev = lags[len(lags)-1]

	return voicingStrength
}

func clamp01(x float32) float32 {
	if x < 0.0 {
		return 0.0
	}
	if x > 1.0 {
		return 1.0
	}
	return x
}

func f32Tanh(x float32) float32 { return float32(math.Tanh(float64(x))) }
