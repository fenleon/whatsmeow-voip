// MLow harmonic postfilter: the final per-packet pitch-comb that runs on the
// full LB output after the HP postfilter. It enhances pitch harmonics by
// mixing x[-lag] + x[+lag] into the signal, low-pass filtered by a
// lag-dependent kernel, and introduces the codec's 48-sample group delay.
// Ported from whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_harm_postfilter.rs;
// MPL-2.0 for this port.
//
// The reference is built with -ffast-math -mavx, so the recursive/accumulating
// math is not IEEE-strict; this matches it to within i16 output quantization,
// not bit-for-bit.
package mlow

import (
	"math"
	"sync"
)

const (
	harmMaxFramesPerPacket         = 6
	harmMinPitchLag                = 32  // SMPL_MINPITCH_MS(2) * 16
	harmMaxPitchLag                = 320 // SMPL_MAXPITCH_MS(20) * 16
	harmMaxPitchLen                = 320
	harmFBDelay                    = 8
	harmLagSubfrLen                = 40
	harmDelay                      = harmLagSubfrLen
	harmPitchNumSubframes          = 8
	harmStateCombLen               = harmMaxPitchLen + hpFrameLen*harmMaxFramesPerPacket + harmDelay
	harmLPFiltRes                  = 2500
	harmNumLPFilt                  = (harmLPFiltRes/80 - harmLPFiltRes/harmMaxPitchLag) + 1
	harmFBStrength         float32 = 0.4734
	harmStrength           float32 = 0.6438
	harmCutoffHz           float32 = 4000.0
	harmNHarmCutoff        float32 = 6.3
	harmReductionFac       float32 = 0.0579
)

// TotPostfiltDelay is the total group delay the harmonic postfilter
// introduces (SMPL_TOT_POSTFILT_DELAY = 48), so the decoded PCM aligns at
// lag 0 with the reference.
const TotPostfiltDelay = harmFBDelay + harmDelay

func harmLagToFiltIx(lag int32) int {
	d := int32(lag + 30)
	if d < 80 {
		d = 80
	}
	return int(harmLPFiltRes/d - harmLPFiltRes/harmMaxPitchLag)
}

// harmTables is the LP-filter bank: one symmetric 17-tap kernel per quantized
// lag bucket.
type harmTables struct {
	lpFilters [harmNumLPFilt][2*harmFBDelay + 1]float32
}

var harmTablesOnce = sync.OnceValue(func() *harmTables {
	var filtWin [harmFBDelay]float32
	dOmega := (0.5 * float32(math.Pi)) / (float32(harmFBDelay) + 1.0)
	omega := dOmega
	for i := 0; i < harmFBDelay; i++ {
		filtWin[i] = float32(math.Cos(float64(omega))) / (float32(i) + 1.0)
		omega += dOmega
	}
	var t harmTables
	ixPrev := int32(-1)
	for lag := int32(harmMinPitchLag); lag <= harmMaxPitchLag; lag++ {
		ix := int32(harmLagToFiltIx(lag))
		if ix != ixPrev {
			omega0 := 2.0 * float32(math.Pi) / float32(lag)
			harmCreateLPFilter(omega0, &filtWin, &t.lpFilters[ix])
			ixPrev = ix
		}
	}
	return &t
})

func harmCreateLPFilter(omega0 float32, filtWin *[harmFBDelay]float32, blp *[2*harmFBDelay + 1]float32) {
	omegaC := omega0 * harmNHarmCutoff
	if limit := harmCutoffHz / 16000.0 * float32(math.Pi); omegaC > limit {
		omegaC = limit
	}
	var sumB float32
	omegaCSum := omegaC
	for i := 0; i < harmFBDelay; i++ {
		b := filtWin[i] * float32(math.Sin(float64(omegaCSum)))
		omegaCSum += omegaC
		blp[harmFBDelay+i+1] = b
		blp[harmFBDelay-i-1] = b
		sumB += 2.0 * b
	}
	blp[harmFBDelay] = omegaC
	sumB += omegaC
	sc := 1.0 / sumB
	for i := range blp {
		blp[i] *= sc
	}
}

// HarmPostfilterState is the persistent harm-postfilter state.
// prev_lag = 0 after init (first packet's lag).
type HarmPostfilterState struct {
	state1        [2 * harmFBDelay]float32
	lpcoefs       [2*harmFBDelay + 1]float32
	stateComb     []float32
	prevLag       int32
	prevDidFilter int32
}

// NewHarmPostfilterState creates the zero-initialized state.
func NewHarmPostfilterState() *HarmPostfilterState {
	return &HarmPostfilterState{
		stateComb: make([]float32, harmStateCombLen),
	}
}

func harmDotProd(a, b []float32, l int) float32 {
	var r float32
	for i := 0; i < l; i++ {
		r += a[i] * b[i]
	}
	return r
}

func harmNrg(x []float32, n int) float32 {
	var r float32
	for i := 0; i < n; i++ {
		r += x[i] * x[i]
	}
	return r
}

// filtMA16Sym is the 17-tap symmetric MA. buf has 16 samples of history
// before x_base.
func filtMA16Sym(buf []float32, xBase, n int, coef *[17]float32, out []float32) {
	for nn := 0; nn < n; nn++ {
		c := xBase + nn
		res := buf[c-8] * coef[8]
		for i := 0; i < 8; i++ {
			res += coef[i] * (buf[c-i] + buf[c-16+i])
		}
		out[nn] = res
	}
}

// harmPostfilterCore is the core filter for one 40-sample lag block. comb is
// the StateComb buffer; combX is this block's read pointer; out/outOff is the
// caller's y_harm destination (it doubles as scratch and holds the final
// block output). The filtered result is also fed back into
// comb[combX-FB_DELAY..], which makes the comb recursive and is the
// 48-delayed location the next packets read from.
func harmPostfilterCore(
	lpcoefs *[2*harmFBDelay + 1]float32,
	comb []float32,
	combX int,
	futureSamples int32,
	lag int32,
	diff []float32,
	diffBase int,
	out []float32,
	outOff int,
	l int,
	fbStrength float32,
	prevDidFilter *int32,
) {
	tables := harmTablesOnce()
	lagU := int(lag)
	xy := float32(0.0)
	// y_harm scratch lives in out[outOff..outOff+l].
	if lag > 0 {
		lookforward := int32(l) + lag - futureSamples
		if lookforward > 0 {
			l2 := int(maxI32(int32(l)-lookforward, 0))
			for i := 0; i < l2; i++ {
				out[outOff+i] = comb[combX+i-lagU] + comb[combX+i+lagU]
			}
			for i := 0; i < l-l2; i++ {
				out[outOff+l2+i] = comb[combX+l2+i-lagU] + comb[combX+l2+i]
			}
		} else {
			for i := 0; i < l; i++ {
				out[outOff+i] = comb[combX+i-lagU] + comb[combX+i+lagU]
			}
		}
		xy = harmDotProd(comb[combX:], out[outOff:], l)
	}
	if lag > 0 && xy > 0.0 {
		xx := harmNrg(comb[combX:], l)
		yy := 0.25 * harmNrg(out[outOff:], l)
		strength := 0.5 * xy / maxF32(yy, xx)
		highLagReduction := 1.0 -
			harmReductionFac*(float32(lag-harmMinPitchLag)/float32(harmMaxPitchLag-harmMinPitchLag))
		strength = strength * highLagReduction * harmStrength
		for i := 0; i < l; i++ {
			out[outOff+i] *= 0.5 * strength
		}
		// diff = -strength * x + y_harm
		for i := 0; i < l; i++ {
			diff[diffBase+i] = out[outOff+i] + (-strength)*comb[combX+i]
		}
		kernel := &tables.lpFilters[harmLagToFiltIx(lag)]
		for k := range lpcoefs {
			lpcoefs[k] = kernel[k] * fbStrength
		}
		var coef17 [17]float32
		copy(coef17[:], lpcoefs[:])
		// y_harm = MA(diff); then y_harm += comb[x - FB_DELAY] (the
		// 48-delayed base signal). The comb is read-only here.
		var yh [harmLagSubfrLen]float32
		filtMA16Sym(diff, diffBase, l, &coef17, yh[:])
		for i := 0; i < l; i++ {
			out[outOff+i] = yh[i] + comb[combX-harmFBDelay+i]
		}
		*prevDidFilter = 1
	} else {
		for i := 0; i < harmLagSubfrLen; i++ {
			diff[diffBase+i] = 0.0
		}
		if *prevDidFilter != 0 {
			// zero-input response of the previous filter for the first
			// 2*FB_DELAY samples, added onto the delayed base; the tail is
			// the plain 48-delayed comb.
			var coef17 [17]float32
			copy(coef17[:], lpcoefs[:])
			var yh [2 * harmFBDelay]float32
			filtMA16Sym(diff, diffBase, 2*harmFBDelay, &coef17, yh[:])
			for i := 0; i < 2*harmFBDelay; i++ {
				out[outOff+i] = yh[i] + comb[combX-harmFBDelay+i]
			}
			for i := 2 * harmFBDelay; i < l; i++ {
				out[outOff+i] = comb[combX+harmFBDelay+i-2*harmFBDelay]
			}
		} else {
			for i := 0; i < l; i++ {
				out[outOff+i] = comb[combX-harmFBDelay+i]
			}
		}
		*prevDidFilter = 0
	}
}

// HarmPostfilter applies the harmonic postfilter to a full packet in place.
// x is xLen samples; lags are the per-40-block lags (nLags = xLen/40);
// normalizedBitrate is the packet average.
func HarmPostfilter(st *HarmPostfilterState, x []float32, xLen int, lags []float32, nLags int, normalizedBitrate float32) {
	const diffPrefix = 2 * harmFBDelay
	diff := make([]float32, hpFrameLen+diffPrefix)

	lag := st.prevLag
	// StateComb layout: [history | current packet]; the current packet starts
	// at MAX_PITCH_LAG + HARM_DELAY.
	combCur := harmMaxPitchLen + harmDelay
	copy(st.stateComb[combCur:combCur+xLen], x[:xLen])

	fbStrength := 1.0 - harmFBStrength*normalizedBitrate
	offset1 := 0

	lagCtr := 0
	for lagCtr < nLags {
		offset2 := 0
		// diff[-16..0] = state1
		copy(diff[diffPrefix-16:diffPrefix], st.state1[:])
		lagCtrEnd := min(lagCtr+harmPitchNumSubframes, nLags)
		for lagCtr < lagCtrEnd {
			combX := harmMaxPitchLen + offset1
			futureSamples := int32(harmDelay) + int32(xLen) - int32(offset1)
			harmPostfilterCore(
				&st.lpcoefs,
				st.stateComb,
				combX,
				futureSamples,
				lag,
				diff,
				diffPrefix+offset2,
				x,
				offset1,
				harmLagSubfrLen,
				fbStrength,
				&st.prevDidFilter,
			)
			offset1 += harmLagSubfrLen
			offset2 += harmLagSubfrLen
			lag = int32(math.Round(float64(lags[lagCtr])))
			lagCtr++
		}
		// state1 = diff[offset2-16 .. offset2]
		copy(st.state1[:], diff[diffPrefix+offset2-16:diffPrefix+offset2])
	}

	st.prevLag = lag
	// shift StateComb left by xLen
	copy(st.stateComb, st.stateComb[xLen:xLen+combCur])
}
