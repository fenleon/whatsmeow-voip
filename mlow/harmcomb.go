// MLow HP (harmonic/pitch) postfilter: the post-LPC-synthesis comb that
// resonates the output at the PITCH frequency. Structure per frame:
// de-emphasis (AR1 leaky integrator {1,-0.995}) -> ARMA2 comb -> companion
// pre-emphasis (MA1 differentiator). Ported from whatsapp-rust (MIT)
// wacore/src/voip/mlow/smpl_harmcomb.rs; MPL-2.0 for this port.
package mlow

import (
	"math"
	"sync"
)

const (
	loEmph1 float32 = 1.0
	loEmph2 float32 = -0.995

	hpPitchMAF     float32 = 0.1
	hpDefMAF       float32 = 0.1
	hpDefFCornerHz float32 = 50.0

	hpFrameLen                     = 320
	hpPostfTransitionSpeed float32 = 2.0
	lagChangeThreshold     float32 = 1.25
)

var (
	hpPitchARF = [2]float32{0.608057355, 0.070939485}
	hpPitchARR = [2]float32{-2.187380512, 2.291030664}
	hpDefARF   = [2]float32{0.728508218, 0.476039848}
	hpDefARR   = [2]float32{-4.363803713, 8.441854006}
)

// HpPostfilterState is the persistent HP-postfilter state. LagOld < 0 marks a
// fresh/reset filter.
type HpPostfilterState struct {
	stateLoEmph1 float32
	stateLoEmph2 float32
	stateHP      [4]float32 // [ma2 x[-1], ma2 x[-2], ar2 y[-1], ar2 y[-2]]
	lagOld       float32
	xOld         [hpFrameLen]float32
	coefMA       [3]float32
	coefAR       [3]float32
	scratchX     [hpFrameLen]float32
	scratchYOld  [hpFrameLen]float32
	scratchYTmp  [hpFrameLen]float32
	scratchDummy [hpFrameLen]float32
}

// NewHpPostfilterState creates the reset (lag_old = -1) state.
func NewHpPostfilterState() *HpPostfilterState {
	return &HpPostfilterState{lagOld: -1.0}
}

// cosApprox is the small-angle cosine cos_approx(x) = 1 - 0.5*x^2.
func cosApprox(x float32) float32 {
	return 1.0 - 0.5*x*x
}

// SmplPfFIR3 is the 3-tap FIR with carried 2-sample input history
// (general/monic). Also reused by the excitation postfilter (comb #1).
func SmplPfFIR3(input []float32, n int, coef [3]float32, state *[2]float32, out []float32) {
	xm1 := state[0]
	xm2 := state[1]
	for i := 0; i < n; i++ {
		var p1, p2 float32
		switch {
		case i >= 1:
			p1 = input[i-1]
		default:
			p1 = xm1
		}
		switch {
		case i >= 2:
			p2 = input[i-2]
		case i == 1:
			p2 = xm1
		default:
			p2 = xm2
		}
		out[i] = coef[0]*input[i] + coef[1]*p1 + coef[2]*p2
	}
	if n >= 2 {
		state[0] = input[n-1]
		state[1] = input[n-2]
	} else if n == 1 {
		state[1] = xm1
		state[0] = input[0]
	}
}

// smplFiltAR2HP is the 2nd-order all-pole y[n] = in[n] - c1*y[n-1] - c2*y[n-2]
// (monic), state {y[-1],y[-2]}. Uses the scalar 4-wide unrolled block
// (precomputed coefficient powers) so the floating-point rounding is
// bit-exact with the scalar reference (the codec was built without NEON).
func smplFiltAR2HP(input []float32, n int, c1, c2 float32, state *[2]float32, out []float32) {
	ytmp0 := state[1]
	ytmp1 := state[0]
	ar1 := -c1
	ar2 := -c2
	ar1_2 := ar1 * ar1
	ar1_3 := ar1 * ar1_2
	ar1_4 := ar1 * ar1_3
	imp1 := ar1
	imp2 := ar1_2 + ar2
	imp3 := ar1_3 + 2.0*ar1*ar2
	imp4 := ar1_4 + ar2*ar2 + 3.0*ar1_2*ar2
	ymp1 := ar2
	ymp2 := ar2 * imp1
	ymp3 := ar2 * imp2
	ymp4 := ar2 * imp3
	nn := 0
	for nn+3 < n {
		xtmp0 := input[nn]
		xtmp1 := input[nn+1]
		xtmp2 := input[nn+2]
		out[nn+2] = xtmp2 + imp1*xtmp1 + imp2*xtmp0 + imp3*ytmp1 + ymp3*ytmp0
		xtmp3 := input[nn+3]
		out[nn+3] = xtmp3 + imp1*xtmp2 + imp2*xtmp1 + imp3*xtmp0 + imp4*ytmp1 + ymp4*ytmp0
		out[nn] = xtmp0 + imp1*ytmp1 + ymp1*ytmp0
		out[nn+1] = xtmp1 + imp1*xtmp0 + imp2*ytmp1 + ymp2*ytmp0
		ytmp0 = out[nn+2]
		ytmp1 = out[nn+3]
		nn += 4
	}
	for nn < n {
		out[nn] = input[nn] + ar1*ytmp1 + ar2*ytmp0
		ytmp0 = ytmp1
		ytmp1 = out[nn]
		nn++
	}
	state[1] = ytmp0
	state[0] = ytmp1
}

// smplFiltAR1HP is the AR1 leaky integrator y[n] = x[n] - c1*y[n-1] (here the
// de-emphasis {1,-0.995}). Uses the scalar 5-wide unrolled block for
// bit-exact rounding.
func smplFiltAR1HP(input []float32, n int, c1 float32, state *float32, out []float32) {
	ar1 := -c1
	ar1_2 := ar1 * ar1
	ar1_3 := ar1 * ar1_2
	ar1_4 := ar1 * ar1_3
	ar1_5 := ar1 * ar1_4
	ytmp := *state
	nn := 0
	for nn+4 < n {
		xtmp0 := input[nn]
		xtmp1 := input[nn+1]
		xtmp2 := input[nn+2]
		xtmp3 := input[nn+3]
		xtmp4 := input[nn+4]
		out[nn+4] = xtmp4 + ar1*xtmp3 + ar1_2*xtmp2 + ar1_3*xtmp1 + ar1_4*xtmp0 + ar1_5*ytmp
		out[nn] = xtmp0 + ar1*ytmp
		out[nn+1] = xtmp1 + ar1*xtmp0 + ar1_2*ytmp
		out[nn+2] = xtmp2 + ar1*xtmp1 + ar1_2*xtmp0 + ar1_3*ytmp
		out[nn+3] = xtmp3 + ar1*xtmp2 + ar1_2*xtmp1 + ar1_3*xtmp0 + ar1_4*ytmp
		ytmp = out[nn+4]
		nn += 5
	}
	for nn < n {
		ytmp = input[nn] + ytmp*ar1
		out[nn] = ytmp
		nn++
	}
	*state = ytmp
}

// smplFiltMA1HP is the MA1 y[n] = x[n] + c1*x[n-1] (here the companion
// pre-emphasis {1,-0.995}).
func smplFiltMA1HP(input []float32, n int, c1 float32, state *float32, out []float32) {
	prev := *state
	for i := n - 1; i >= 1; i-- {
		out[i] = input[i] + c1*input[i-1]
	}
	if n > 0 {
		out[0] = input[0] + c1*prev
		*state = input[n-1]
	}
}

// SmplGetHpCoefs builds the default fixed-corner ARMA2 biquad (also the
// encoder's input high-pass). fcornerHz clamped to [5, 1500].
func SmplGetHpCoefs(fcornerHz float32) ([3]float32, [3]float32) {
	fc := minF32(maxF32(fcornerHz, 5.0), 1500.0)
	return smplCalcHpCoefs(hpDefMAF, hpDefARF, hpDefARR, fc/16000.0)
}

// SmplFiltARMA2 is the MA2 numerator then AR2 denominator, shared 4-wide
// state {ma x[-1],x[-2], ar y[-1],y[-2]}.
func SmplFiltARMA2(input []float32, n int, coefMA, coefAR [3]float32, state *[4]float32, out []float32) {
	tmp := make([]float32, n)
	maSt := [2]float32{state[0], state[1]}
	SmplPfFIR3(input, n, coefMA, &maSt, tmp)
	state[0] = maSt[0]
	state[1] = maSt[1]
	arSt := [2]float32{state[2], state[3]}
	smplFiltAR2HP(tmp, n, coefAR[1], coefAR[2], &arSt, out)
	state[2] = arSt[0]
	state[3] = arSt[1]
}

// smplCalcHpCoefs builds the unity-numerator-DC comb biquad. The AR
// denominator is a resonance at the pitch angle 2*pi*arf*f with radius
// 1 + arr*f (arr negative -> stable), then the MA numerator is scaled so the
// comb has unity DC gain.
func smplCalcHpCoefs(maf float32, arf, arr [2]float32, f float32) ([3]float32, [3]float32) {
	coefMA := [3]float32{1.0, -2.0 * cosApprox(2.0*smplPiF32*maf*f), 1.0}
	far := arf[0]*f + arf[1]*f*f
	rar := arr[0]*f + arr[1]*f*f
	coefAR := [3]float32{
		1.0,
		-2.0 * cosApprox(2.0*smplPiF32*far) * (1.0 + rar),
		1.0 + (2.0*rar + rar*rar),
	}
	sc := (1.0 - coefAR[1] + coefAR[2]) / (1.0 - coefMA[1] + coefMA[2])
	for i := range coefMA {
		coefMA[i] *= sc
	}
	return coefMA, coefAR
}

// newCoefsInto is the voiced pitch curve when lag>0 (f=1/lag), else the
// default 50 Hz-corner curve.
func newCoefsInto(coefMA, coefAR *[3]float32, lag float32) {
	var ma, ar [3]float32
	if lag > 0.0 {
		f := 1.0 / lag
		ma, ar = smplCalcHpCoefs(hpPitchMAF, hpPitchARF, hpPitchARR, f)
	} else {
		fc := minF32(maxF32(hpDefFCornerHz, 5.0), 1500.0)
		ma, ar = smplCalcHpCoefs(hpDefMAF, hpDefARF, hpDefARR, fc/16000.0)
	}
	*coefMA = ma
	*coefAR = ar
}

// hpRampDn is the cos(omega)^2 down-ramp for the lag-change overlap-add.
// omega is accumulated by repeated addition (not d_omega*i) to stay bit-exact
// with the reference table.
var hpRampDn = sync.OnceValue(func() []float32 {
	dOmega := smplPiF32 / (2.0 * (float32(hpFrameLen) + 1.0))
	omega := dOmega
	v := make([]float32, 0, hpFrameLen)
	for i := 0; i < hpFrameLen; i++ {
		v = append(v, float32(math.Pow(float64(float32(math.Cos(float64(omega)))), float64(hpPostfTransitionSpeed))))
		omega += dOmega
	}
	return v
})

// SmplHarmcombPostfilter applies the HP (pitch-harmonic) postfilter to one
// frame's post-LPC output. lag is the frame's average pitch lag
// (sum(l^2)/sum(l) over the subframe lags), 0 for unvoiced.
func SmplHarmcombPostfilter(st *HpPostfilterState, xIn []float32, n int, lag float32, out []float32) {
	// de-emphasis (AR1) into the x scratch.
	smplFiltAR1HP(xIn, n, loEmph2, &st.stateLoEmph1, st.scratchX[:n])

	overlap := false
	if st.lagOld < 0.0 {
		newCoefsInto(&st.coefMA, &st.coefAR, lag)
		st.lagOld = lag
	} else if lag > lagChangeThreshold*st.lagOld || lagChangeThreshold*lag < st.lagOld {
		overlap = true
		SmplFiltARMA2(st.scratchX[:n], n, st.coefMA, st.coefAR, &st.stateHP, st.scratchYOld[:n])
		newCoefsInto(&st.coefMA, &st.coefAR, lag)
		st.lagOld = lag
		SmplFiltARMA2(st.xOld[:n], n, st.coefMA, st.coefAR, &st.stateHP, st.scratchDummy[:n])
	} else if lag != st.lagOld {
		newCoefsInto(&st.coefMA, &st.coefAR, lag)
		st.lagOld = lag
	}
	copy(st.xOld[:n], st.scratchX[:n])

	SmplFiltARMA2(st.scratchX[:n], n, st.coefMA, st.coefAR, &st.stateHP, st.scratchYTmp[:n])

	if overlap {
		ramp := hpRampDn()
		for i := 0; i < n; i++ {
			st.scratchYTmp[i] += (st.scratchYOld[i] - st.scratchYTmp[i]) * ramp[i]
		}
	}

	// companion pre-emphasis (MA1).
	smplFiltMA1HP(st.scratchYTmp[:n], n, loEmph2, &st.stateLoEmph2, out)
}
