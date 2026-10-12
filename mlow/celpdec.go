// Decoder-side CELP synthesis in the codec's native float domain: the
// per-subframe loop (excitation, CELP/ACB decode, gen_noise, LPC synthesis)
// plus the LSF interpolation and the FCB gain tables. Ported from
// whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_celpdec.rs; MPL-2.0 for this
// port. Output is float in [-1, 1].
package mlow

import (
	"math"
	"sync"
)

const (
	celpLagSubfrLen              = 40
	celpLTPInterpolDelay         = 8
	celpMaxPitchLag              = 320
	celpPitchSharpening  float32 = 0.9881
	celpFCBgVN                   = 34
	celpUVGainIdxLen             = 90
	celpVGainMinDB       float32 = -100.0
	celpVGainStepDB      float32 = 3.0
	celpUVGainMinDB      float32 = -90.0
	celpUVGainStepDB     float32 = 1.0
)

// ACB high-boost endpoints.
var celpDecACBHighBoost = [2]float32{0.35, 0.18}

// LSF->LPC interpolation factors per subframe, [lsf_interpol_idx][sf].
var smplLSFInterpol4 = [2][4]float32{{0.55, 0.88, 1.0, 1.0}, {0.3, 0.65, 0.95, 1.0}}

// 16-tap symmetric LTP interpolation kernel.
var smplInterpolKernel = [2 * celpLTPInterpolDelay]float32{
	-6.3925986e-6, 0.00011064114, -0.0009153038, 0.00484772, -0.018698348, 0.05759091, -0.15997477, 0.6170455,
	0.61704546, -0.15997475, 0.057590906, -0.018698348, 0.00484772, -0.0009153038, 0.000110641144, -6.392598e-6,
}

// Per-subframe ACB-gain codebooks (Q14), low-rate and high-rate.
var cbAcbgainsHRQ14 = [32]int16{
	16039, 91,
	0, 0,
	4310, 4930,
	-1431, 2862,
	2893, 0,
	8009, 4075,
	2754, 4223,
	8367, 354,
	4640, 1254,
	-176, 2734,
	-1222, 5017,
	-476, 1506,
	11351, 567,
	1243, 0,
	10601, 22,
	14088, 108,
}

var cbAcbgainsLRQ14 = [32]int16{
	2812, 2484,
	0, 0,
	-362, 2465,
	-337, 703,
	3033, 1474,
	13536, 220,
	-2630, 9226,
	6032, 3499,
	-220, 441,
	7661, 4243,
	11521, 0,
	1430, 779,
	4495, 2724,
	15535, 343,
	-779, 1559,
	480, 481,
}

type fcbGains struct {
	uv [celpUVGainIdxLen + 1]float32
	v  [celpFCBgVN]float32
}

var fcbGainsOnce = sync.OnceValue(func() *fcbGains {
	var g fcbGains
	for ix := 0; ix <= celpUVGainIdxLen; ix++ {
		g.uv[ix] = float32(math.Pow(10, float64(0.05*(float32(ix)*celpUVGainStepDB+celpUVGainMinDB))))
	}
	for ix := 0; ix < celpFCBgVN; ix++ {
		g.v[ix] = float32(math.Pow(10, float64(0.05*(float32(ix)*celpVGainStepDB+celpVGainMinDB))))
	}
	return &g
})

func smplDotProd(a, b []float32, l int) float32 {
	var r float32
	for i := 0; i < l; i++ {
		r += a[i] * b[i]
	}
	return r
}

// lpcInterpol interpolates the LSF per subframe between prevLSF and lsf, then
// NLSF->A. Writes the per-subframe A coefficients and interpolated LSF, and
// mutates prevLSF to the last interpolated LSF (carried across frames).
func lpcInterpol(lsf []float32, prevLSF *[smplOrder]float32, interpol *[4]float32, aOut *[smplSubfrCount][smplOrder + 1]float32, lsfsOut *[smplSubfrCount][smplOrder]float32) {
	if prevLSF[smplOrder-1] == 0.0 {
		copy(prevLSF[:], lsf[:smplOrder])
	}
	var ilsf [smplOrder]float32
	prevFactor := float32(-1.0)
	for j := 0; j < smplSubfrCount; j++ {
		if interpol[j] == prevFactor {
			aOut[j] = aOut[j-1]
		} else {
			if interpol[j] == 1.0 {
				copy(ilsf[:], lsf[:smplOrder])
			} else {
				for k := 0; k < smplOrder; k++ {
					ilsf[k] = prevLSF[k]*(1.0-interpol[j]) + lsf[k]*interpol[j]
				}
			}
			aOut[j] = NLSF2A(ilsf[:])
		}
		prevFactor = interpol[j]
		lsfsOut[j] = ilsf
	}
	copy(prevLSF[:], ilsf[:])
}

func acbDequant(lowRate bool, acbIdx int32, acbG *[2]float32) {
	var cb []int16
	if lowRate {
		cb = cbAcbgainsLRQ14[:]
	} else {
		cb = cbAcbgainsHRQ14[:]
	}
	sc := 1.0 / float32(int32(1)<<14)
	for m := 0; m < 2; m++ {
		acbG[m] = float32(cb[acbIdx*2+int32(m)]) * sc
	}
}

// acbSynthesize adjusts the ACB gains, then 3-tap symmetric ACB synthesis
// with high-boost applied.
func acbSynthesize(fcbSubfrLen int, acbBasis []float32, acbGIn *[2]float32, highBoost float32, acb []float32) {
	acbG := *acbGIn
	if highBoost != 0.0 {
		f0 := acbG[0] + 2.0*acbG[1]
		f1 := acbG[0] - acbG[1]
		absF2new := minF32(absF32(f1)+highBoost, absF32(f0))
		f1 = f1 * (absF2new / (absF32(f1) + 1e-12))
		acbG[0] = (f0 + 2.0*f1) / 3.0
		acbG[1] = (f0 - f1) / 3.0
	}
	for i := 0; i < fcbSubfrLen; i++ {
		acb[i] = acbG[0] * acbBasis[i]
	}
	for i := 0; i < fcbSubfrLen; i++ {
		acb[i] += acbG[1] * acbBasis[fcbSubfrLen+i]
	}
}

func pitchSharpen(x []float32, lag, l int) {
	for i := lag; i < l; i++ {
		x[i] += x[i-lag] * celpPitchSharpening
	}
}

// synLTPOperBasis builds the ACB basis from the excitation history; mutates
// state forward. state is the full ACB state; the logical start is
// state[stateLen - nLags*40].
func synLTPBasis(lags []float32, nLags int, state []float32, stateLen int, acbBasis []float32) {
	p := stateLen - nLags*celpLagSubfrLen
	for subfr := 0; subfr < nLags; subfr++ {
		iLag := int32(math.Floor(float64(lags[subfr])))
		if float32(iLag) == lags[subfr] {
			il := int(iLag)
			for i := 0; i < celpLagSubfrLen; i++ {
				state[p+i] = state[(p+i)-il]
			}
			for i := 0; i < celpLagSubfrLen; i++ {
				acbBasis[subfr*celpLagSubfrLen+i] = state[p+i]
			}
			for i := 0; i < celpLagSubfrLen; i++ {
				a := state[(p+i)-il-1]
				b := state[(p+i)-il+1]
				acbBasis[(nLags+subfr)*celpLagSubfrLen+i] = a + b
			}
		} else {
			il := int(iLag)
			baseFirst := p + (-1 - il - celpLTPInterpolDelay)
			first := smplDotProd(state[baseFirst:], smplInterpolKernel[:], 2*celpLTPInterpolDelay)
			{
				srcBase := p + (-il - celpLTPInterpolDelay)
				for nn := 0; nn < celpLagSubfrLen; nn++ {
					var ret float32
					for i := 0; i < 8; i++ {
						s0 := state[srcBase+nn+i]
						s1 := state[srcBase+nn+15-i]
						ret += (s0 + s1) * smplInterpolKernel[i]
					}
					state[p+nn] = ret
				}
			}
			// The index runs -1 -> 0 -> +SMPL_LAG_SUBFRLEN here, so the tap
			// base for the last sample is p + (40 - i_lag - delay).
			baseLast := p + (celpLagSubfrLen - il - celpLTPInterpolDelay)
			last := smplDotProd(state[baseLast:], smplInterpolKernel[:], 2*celpLTPInterpolDelay)
			for i := 0; i < celpLagSubfrLen; i++ {
				acbBasis[subfr*celpLagSubfrLen+i] = state[p+i]
			}
			b1 := (nLags + subfr) * celpLagSubfrLen
			acbBasis[b1] = first + state[p+1]
			for i := 0; i < celpLagSubfrLen-2; i++ {
				acbBasis[b1+1+i] = state[p+i] + state[p+i+2]
			}
			iLast := celpLagSubfrLen - 1
			acbBasis[b1+iLast] = state[p+iLast-1] + last
		}
		p += celpLagSubfrLen
	}
}

// celpDecode adds the voiced ACB (LTP) contribution into lpcRes, then pushes
// the subframe into the ACB state.
func celpDecode(
	acbState []float32,
	acbStateLen int,
	voiced bool,
	acbGainIdx int32,
	lags []float32,
	numLags int,
	subfrLen int,
	lowRate bool,
	normalizedBitrate float32,
	lpcRes []float32,
) {
	if voiced {
		highBoost := celpDecACBHighBoost[0] +
			(celpDecACBHighBoost[1]-celpDecACBHighBoost[0])*normalizedBitrate
		iLag := int(lags[numLags-1])
		if lowRate {
			pitchSharpen(lpcRes, iLag, subfrLen)
		}
		acbBasis := make([]float32, subfrLen*2)
		acb := make([]float32, subfrLen)
		synLTPBasis(lags, numLags, acbState, acbStateLen, acbBasis)
		var acbGain [2]float32
		acbDequant(lowRate, acbGainIdx, &acbGain)
		acbSynthesize(subfrLen, acbBasis, &acbGain, highBoost, acb)
		for i := 0; i < subfrLen; i++ {
			lpcRes[i] += acb[i]
		}
	}
	// Update ACB state: shift left by subfrLen, append this subframe's
	// excitation.
	copy(acbState[:acbStateLen-subfrLen], acbState[subfrLen:acbStateLen])
	copy(acbState[acbStateLen-2*subfrLen:acbStateLen-subfrLen], lpcRes[:subfrLen])
}

// filtAR16 is AR(16) over one subframe. ybuf holds a 16-sample history
// prefix at ybuf[base-16..base], so synthesis flows contiguously across the
// frame (cross-subframe and cross-frame history is the same buffer).
func filtAR16(x []float32, a *[smplOrder + 1]float32, ybuf []float32, base, n int) {
	for nn := 0; nn < n; nn++ {
		res := x[nn]
		for i := 0; i < smplOrder; i++ {
			res -= a[smplOrder-i] * ybuf[base+nn-smplOrder+i]
		}
		ybuf[base+nn] = res
	}
}

// CelpDecParams are the per-subframe decoded params the synthesis consumes.
type CelpDecParams struct {
	Voiced       bool
	SfPulses     [4]int32
	FcbgIdx      [4]int32
	NrgresDbqQ14 [4]int32
	AcbgIdx      [4]int32 // ACB gain index per subframe (voiced only)
	// BlockLags is the per-40-block pitch lag (codec units: float), 8 per
	// frame, 0 for unvoiced. Synthesis hands the two blocks of subframe sf to
	// the ACB/LTP basis (lags_per_subframe == 2).
	BlockLags [8]float32
	// BlockLagInt mirrors BlockLags but keeps the integer Q6 values the
	// decoder produced (unused by synthesis; kept for tests).
	TotalPulses int32
}

// CelpDecState is the persistent decoder synthesis state (float domain).
type CelpDecState struct {
	Noise       NoiseGenerator
	acbState    []float32
	acbStateLen int
	lpcSynthMem [smplOrder]float32
	lsfPrev     [smplOrder]float32
	prevNrgres  float32
	// DbgExcPre captures the per-subframe pre-noise excitation (80/subframe)
	// when enabled, for the exc_pre gate.
	DbgExcPre []float32
}

// NewCelpDecState creates zero-initialized decoder synthesis state.
func NewCelpDecState() *CelpDecState {
	acbStateLen := smplSubfrLen + 2*celpMaxPitchLag + celpLTPInterpolDelay
	return &CelpDecState{
		acbState:    make([]float32, acbStateLen),
		acbStateLen: acbStateLen,
	}
}

// HPPostfilter is applied at the end of SynthFrame; see smpl_harmcomb.go
// (SmplHarmPostfilter) — the decoder wires the persistent HpPostfilterState
// here.

// SynthFrame synthesizes one 20 ms internal frame (4 subframes) into 320
// float samples in [-1, 1]. nlsf is the reconstructed order-16 NLSF
// (radians); pulses are the signed FCB pulse magnitudes (320 positions);
// lowRate is the TOC bit. hpPost applies the post-LPC HP postfilter (nil
// skips it; the decoder wires smpl_hp_postfilter here).
func (s *CelpDecState) SynthFrame(
	nlsf []float32,
	lsfInterpolIdx int,
	pulses []int32,
	params *CelpDecParams,
	lowRate bool,
	frameLength16 int32,
	out []float32,
	hp *HpPostfilterState,
) {
	gains := fcbGainsOnce()
	// Per-subframe LPC interpolation.
	var a [smplSubfrCount][smplOrder + 1]float32
	var lsfs [smplSubfrCount][smplOrder]float32
	interpol := &smplLSFInterpol4[min(lsfInterpolIdx, 1)]
	lpcInterpol(nlsf, &s.lsfPrev, interpol, &a, &lsfs)

	normalizedBitrate := GetNormalizedBitrate(params.TotalPulses, frameLength16)

	// Excitation: sparse FCB pulses scaled by the per-subframe FCB gain.
	var lpcRes [smplIntfLen]float32
	var gainTab []float32
	if params.Voiced {
		gainTab = gains.v[:]
	} else {
		gainTab = gains.uv[:]
	}
	for pos := 0; pos < smplIntfLen; pos++ {
		if pulses[pos] != 0 {
			sf := pos / smplSubfrLen
			lpcRes[pos] = float32(pulses[pos]) * gainTab[params.FcbgIdx[sf]]
		}
	}

	const lagsPerSubfr = 2 // 80-sample subframe / 40-sample lag subframe
	// Contiguous synthesis buffer: 16-sample history prefix + 320 samples.
	var ybuf [smplOrder + smplIntfLen]float32
	copy(ybuf[:smplOrder], s.lpcSynthMem[:])
	for sf := 0; sf < smplSubfrCount; sf++ {
		base := sf * smplSubfrLen
		sfLags := []float32{params.BlockLags[2*sf], params.BlockLags[2*sf+1]}
		celpDecode(
			s.acbState, s.acbStateLen, params.Voiced, params.AcbgIdx[sf],
			sfLags, lagsPerSubfr, smplSubfrLen, lowRate, normalizedBitrate,
			lpcRes[base:base+smplSubfrLen],
		)

		s.DbgExcPre = append(s.DbgExcPre, lpcRes[base:base+smplSubfrLen]...)

		// Residual noise energy + shaped noise (the dominant unvoiced fix).
		nrgres := DecodeResnrg(params.NrgresDbqQ14[sf], smplSubfrLen)
		if !params.Voiced {
			s.prevNrgres = nrgres
		}

		var noise [160]float32
		CelpGenNoise(
			&s.Noise,
			lpcRes[base:base+smplSubfrLen],
			smplSubfrLen,
			params.Voiced,
			params.SfPulses[sf],
			nrgres,
			params.FcbgIdx[sf],
			lsfs[sf][:],
			normalizedBitrate,
			gains.uv[:],
			noise[:],
		)
		for i := 0; i < smplSubfrLen; i++ {
			lpcRes[base+i] += noise[i]
		}

		// LPC synthesis (contiguous history across subframes/frames).
		filtAR16(
			lpcRes[base:base+smplSubfrLen],
			&a[sf],
			ybuf[:],
			smplOrder+base,
			smplSubfrLen,
		)
	}
	copy(out[:smplIntfLen], ybuf[smplOrder:])
	copy(s.lpcSynthMem[:], ybuf[smplOrder+smplIntfLen-smplOrder:])

	// Post-LPC HP (pitch-harmonic) postfilter. The comb lag is the
	// energy-weighted mean of the 8 per-40-block lags (0 -> the default
	// fixed-corner curve, unvoiced).
	lag := float32(0.0)
	if params.Voiced {
		var sl, sll float32
		for _, l := range params.BlockLags {
			sl += l
			sll += l * l
		}
		if sl > 0.0 {
			lag = sll / sl
		}
	}
	var hpOut [smplIntfLen]float32
	SmplHarmcombPostfilter(hp, out[:smplIntfLen], smplIntfLen, lag, hpOut[:])
	copy(out[:smplIntfLen], hpOut[:])
}

func absF32(x float32) float32 {
	if x < 0.0 {
		return -x
	}
	return x
}
