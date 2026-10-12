// NLSF reconstruction (shared decoder/encoder piece of smpl_synth.rs):
// reconstruct from decoded indices, stabilize, and NLSF2A. Ported from
// whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_synth.rs; MPL-2.0 for this
// port. Float constants are byte-exact copies of the reference f32 literals.
package mlow

import "math"

const (
	smplOrder                     = 16
	smplSubfrLen                  = 80  // 5 ms @ 16 kHz
	smplIntfLen                   = 320 // 20 ms internal frame
	smplSubfrCount                = 4
	smplNLSFWeightWMax    float32 = 999.9999
	smplNLSFWeightEps     float32 = 0.0009999999
	smplPiF32             float32 = 3.1415927410125
	smplStabilizeMaxLoops         = 1000
	smplStabilizeEps      float32 = 9.5367431640625e-07
)

// smplNlsfLaroiaWeights is func 3530 (silk_NLSF_VQ_weights_laroia):
// inverse-gap weights w[k] = invgap[k] + invgap[k+1].
func smplNlsfLaroiaWeights(nlsf []float32, out *[smplOrder]float32) {
	var inv [smplOrder + 1]float32
	clamp := func(gap float32) float32 {
		if gap > smplNLSFWeightEps {
			return 1.0 / gap
		}
		return smplNLSFWeightWMax
	}
	inv[0] = clamp(nlsf[0])
	prev := nlsf[0]
	for k := 1; k < smplOrder; k++ {
		gap := nlsf[k] - prev
		inv[k] = clamp(gap)
		prev = nlsf[k]
	}
	inv[smplOrder] = clamp(smplPiF32 - nlsf[smplOrder-1])
	for k := 0; k < smplOrder; k++ {
		out[k] = inv[k] + inv[k+1]
	}
}

// smplNlsfDecorr is func 3485: out[r] = sum_c mat[c*16+r]*vec[c]
// (column-major mat).
func smplNlsfDecorr(mat []float32, vec []float32, out *[smplOrder]float32) {
	var scr [smplOrder]float32
	v0 := vec[0]
	for r := 0; r < smplOrder; r++ {
		scr[r] = v0 * mat[r]
	}
	for c := 1; c < smplOrder; c++ {
		base := c * smplOrder
		v := vec[c]
		for r := 0; r < smplOrder; r++ {
			scr[r] += mat[base+r] * v
		}
	}
	copy(out[:], scr[:])
}

// ReconstructNLSF reconstructs the order-16 NLSF (radians) from the decoded
// LSF indices.
func ReconstructNLSF(t *SmplSynthTables, stage1, config, grid int, stage2 *[16]int32, prevNLSF []float32) []float32 {
	val := t.Valtables[stage1][config][grid]
	var resid [smplOrder]float32
	for k := 0; k < smplOrder; k++ {
		sym := stage2[k]
		if sym >= 0 && int(sym) < len(val[k]) {
			resid[k] = val[k][sym]
		}
	}

	out := make([]float32, smplOrder)
	if grid == 16 {
		// grid==16: interpolate base between prevNLSF and the inverted grid16
		// base table.
		var base [smplOrder]float32
		baseTbl := &t.Grid16W[1-stage1]
		alpha := t.Grid16Alpha[stage1]
		for k := 0; k < smplOrder; k++ {
			var pv float32
			if k < len(prevNLSF) {
				pv = prevNLSF[k]
			}
			base[k] = pv + alpha*(baseTbl[k]-pv)
		}
		var w [smplOrder]float32
		smplNlsfLaroiaWeights(base[:], &w)
		for i := range w {
			w[i] = float32(math.Sqrt(float64(w[i])))
		}
		var decorr [smplOrder]float32
		mat16 := t.Grid16Matrices[stage1][config]
		smplNlsfDecorr(mat16[:], resid[:], &decorr)
		for k := 0; k < smplOrder; k++ {
			out[k] = base[k] + decorr[k]/w[k]
		}
		stabilizeNLSF(out, t.MinSpacing[stage1][:])
		return out
	}

	// matrix case (grid < 16): NLSF[r] = 2*centroid[r] + sum_c mat[c][r]*resid[c].
	cent := &t.Centroids[stage1][grid]
	mat := &t.Matrices[stage1][grid]
	for r := 0; r < smplOrder; r++ {
		acc := 2.0 * cent[r]
		for c := 0; c < smplOrder; c++ {
			acc += mat[c][r] * resid[c]
		}
		out[r] = acc
	}
	stabilizeNLSF(out, t.MinSpacing[stage1][:])
	return out
}

// stabilizeNLSF is func 3533 (silk_NLSF_stabilize): enforce minimum spacing +
// ordering in the margin domain.
func stabilizeNLSF(nlsf []float32, minSpacing []float32) {
	const pi = smplPiF32
	const l = smplOrder
	var marg [l + 1]float32
	marg[0] = nlsf[0] - minSpacing[0]
	for i := 1; i < l; i++ {
		marg[i] = nlsf[i] - nlsf[i-1] - minSpacing[i]
	}
	marg[l] = pi - nlsf[l-1] - minSpacing[l]

	argmin := func() (float32, int) {
		m := marg[0]
		idx := 0
		for i := 1; i <= l; i++ {
			if marg[i] < m {
				m = marg[i]
				idx = i
			}
		}
		return m, idx
	}
	minV, sel := argmin()
	loopN := int32(0)
	for minV < 0.0 {
		d := float32(loopN)*smplStabilizeEps - minV
		switch {
		case sel == 0:
			marg[0] += d
			marg[1] -= d
		case sel == l:
			marg[l] += d
			marg[l-1] -= d
		default:
			marg[sel] += d
			half := d * 0.5
			marg[sel-1] -= half
			marg[sel+1] -= half
		}
		minV, sel = argmin()
		if minV < 0.0 {
			loopN++
			if loopN == smplStabilizeMaxLoops {
				break
			}
		}
	}
	nlsf[0] = minSpacing[0] + marg[0]
	run := nlsf[0]
	for i := 1; i < l; i++ {
		run = run + marg[i] + minSpacing[i]
		nlsf[i] = run
	}
}

// NLSF2A is func 3513 (silk_NLSF2A): order-16 NLSF (radians) -> LPC
// coefficients a[0..16], a[0]=1.0.
func NLSF2A(nlsf []float32) [smplOrder + 1]float32 {
	order := len(nlsf)
	half := order / 2
	cosv := make([]float64, order)
	for i, x := range nlsf {
		cosv[i] = math.Cos(float64(x))
	}
	var p, q [smplOrder/2 + 1]float64
	nlsfPoly(&p, cosv, half, 0)
	nlsfPoly(&q, cosv, half, 1)

	var a [smplOrder + 1]float32
	a[0] = 1.0
	for k := 0; k < half; k++ {
		pt := p[k+1] + p[k]
		qt := q[k+1] - q[k]
		a[k+1] = float32(0.5 * (pt + qt))
		a[order-k] = float32(0.5 * (pt - qt))
	}
	return a
}

func nlsfPoly(out *[smplOrder/2 + 1]float64, cosv []float64, half int, parity int) {
	out[0] = 1.0
	out[1] = -2.0 * cosv[parity]
	for k := 1; k < half; k++ {
		c := -2.0 * cosv[2*k+parity]
		out[k+1] = 2.0*out[k-1] + c*out[k]
		for n := k; n > 1; n-- {
			out[n] += out[n-2] + c*out[n-1]
		}
		out[1] += c
	}
}
