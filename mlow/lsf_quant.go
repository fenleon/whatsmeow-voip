// MLow LSF quantizer (encoder side). Given the analysis NLSF, runs the
// VQ_temp Mahalanobis shortlist over the stage-1 centroids, then an RD beam
// (0.5*order*log2(werr)*RDw_adj + bits) with per-coeff stage-2 clamps,
// returning qi[0..16] (qi[0] = grid, qi[1..16] = stage2). Ported from
// whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_lsf_quant.rs; MPL-2.0 for
// this port. Validated 60/60 bit-exact against the reference quantizer.
package mlow

import "math"

const (
	lsfCbCentroids     = 16
	lsfQstepCondMult   = float32(0.9)
	lsfQuantOrderConst = 16 // SMPL_LPC_ORDER in this module
)

type lsfCondParams struct {
	st1Cbhalf [lsfOrder]float32
	st1CbCinv [lsfOrder]float32
	st1We     [lsfOrder][lsfOrder]float32
	st1Wie    [lsfOrder][lsfOrder]float32
}

func lsfSubVec(y, z *[lsfOrder]float32, out *[lsfOrder]float32) {
	for i := 0; i < lsfQuantOrderConst; i++ {
		out[i] = y[i] - z[i]
	}
}

func lsfDotProd(a, b *[lsfOrder]float32) float32 {
	var s float32
	for i := 0; i < lsfQuantOrderConst; i++ {
		s += a[i] * b[i]
	}
	return s
}

func lsfWerr(x, y, w *[lsfOrder]float32) float32 {
	var s float32
	for k := 0; k < lsfQuantOrderConst; k++ {
		e := x[k] - y[k]
		s += w[k] * e * e
	}
	return s
}

// getMaxiK returns the indices of the K largest values in x (descending),
// ties broken toward the lower index. Used for the VQ shortlist +
// alt-refinement; the caller re-evaluates survivors, so the survivor SET is
// the load-bearing output.
func getMaxiK(x []float32, idx []int32, k int) {
	n := len(x)
	used := make([]bool, n)
	for slot := 0; slot < k; slot++ {
		bestI := -1
		bestV := float32(math.Inf(-1))
		for i, v := range x {
			if used[i] {
				continue
			}
			if v > bestV {
				bestV = v
				bestI = i
			}
		}
		if bestI < 0 {
			idx[slot] = 0
		} else {
			used[bestI] = true
			idx[slot] = int32(bestI)
		}
	}
}

// lsfWeightsSpectral is the LSF RD weight, spectral inverse-envelope variant
// (SMPL_USE_SPEC_LSW_WEIGHT, the active build path): the weight at each LSF
// is the inverse spectral envelope magnitude 1/sqrt(|A(e^jw)|^2 * scale),
// scale = 1/min. a is the monic LPC A[0..17] (A[0]=1).
func lsfWeightsSpectral(a *[lsfOrder + 1]float32, lsf *[lsfOrder]float32) [lsfOrder]float32 {
	var lsfw [lsfOrder]float32
	for i := 0; i < lsfQuantOrderConst; i++ {
		eRe := cos32(lsf[i])
		eIm := sin32(lsf[i])
		var accRe, accIm float32 = 1.0, 0.0
		epRe, epIm := eRe, eIm
		for j := 1; j < lsfQuantOrderConst; j++ {
			accRe += epRe * a[j]
			accIm -= epIm * a[j]
			// ep *= e
			nr := epRe*eRe - epIm*eIm
			ni := epRe*eIm + epIm*eRe
			epRe, epIm = nr, ni
		}
		accRe += epRe * a[lsfQuantOrderConst]
		accIm -= epIm * a[lsfQuantOrderConst]
		lsfw[i] = accRe*accRe + accIm*accIm
	}
	minLsfw := lsfw[0]
	for _, v := range lsfw[1:] {
		if v < minLsfw {
			minLsfw = v
		}
	}
	scale := 1.0 / minLsfw
	for i := range lsfw {
		lsfw[i] = 1.0 / float32(math.Sqrt(float64(lsfw[i]*scale)))
	}
	return lsfw
}

// lsfMinDist pushes LSFs apart so consecutive spacings exceed min_dist.
func lsfMinDist(lsfs *[lsfOrder]float32, minDist *[lsfOrder + 1]float32) {
	const n = lsfQuantOrderConst
	var dlsfs [n + 1]float32
	dlsfs[0] = lsfs[0] - minDist[0]
	for i := 1; i < n; i++ {
		dlsfs[i] = (lsfs[i] - lsfs[i-1]) - minDist[i]
	}
	dlsfs[n] = (smplPiF32 - lsfs[n-1]) - minDist[n]
	findMin := func() (float32, int) {
		m := dlsfs[0]
		mi := 0
		for i := 1; i < n+1; i++ {
			if dlsfs[i] < m {
				m = dlsfs[i]
				mi = i
			}
		}
		return m, mi
	}
	dm, minIx := findMin()
	if dm > 0.0 {
		return
	}
	for k := 0; k < 1000; k++ {
		delta := float32(k)*1.0e-6 - dm
		dlsfs[minIx] += delta
		if minIx == 0 {
			dlsfs[1] -= delta
		} else if minIx == n {
			dlsfs[n-1] -= delta
		} else {
			delta *= 0.5
			dlsfs[minIx-1] -= delta
			dlsfs[minIx+1] -= delta
		}
		dm, minIx = findMin()
		if dm >= 0.0 {
			lsfs[0] = dlsfs[0] + minDist[0]
			for i := 1; i < n; i++ {
				lsfs[i] = lsfs[i-1] + (dlsfs[i] + minDist[i])
			}
			return
		}
	}
	// The reference asserts unreachable here; fall through with the
	// best-effort spacing.
}

// vqTemp is the Mahalanobis shortlist of surv stage-1 centroids (plus the
// cond centroid when present).
func vqTemp(lsf *[lsfOrder]float32, cbhalf, cbCinv *[lsfCentroids][lsfOrder]float32, cond *lsfCondParams, surv int, idxs []int32) {
	var err [lsfCbCentroids + 1]float32
	var tmp [lsfOrder]float32
	for s := 0; s < lsfCbCentroids; s++ {
		lsfSubVec(&cbhalf[s], lsf, &tmp)
		err[s] = -lsfDotProd(&tmp, &cbCinv[s])
	}
	cbCentroids := lsfCbCentroids
	if cond != nil {
		lsfSubVec(&cond.st1Cbhalf, lsf, &tmp)
		err[lsfCbCentroids] = -lsfDotProd(&tmp, &cond.st1CbCinv)
		cbCentroids++
	}
	getMaxiK(err[:cbCentroids], idxs, surv)
}

func smplSign(a float32) int32 {
	switch {
	case a > 0.0:
		return 1
	case a == 0.0:
		return 0
	default:
		return -1
	}
}

// LsfQuantResult is the result of one LSF quantization: qi[0] (=grid),
// qi[1..17] (=stage2), and the reconstructed quantized NLSF (so the caller's
// synthesis uses the SAME envelope the decoder will rebuild).
type LsfQuantResult struct {
	Qi   [lsfQuantOrderConst + 1]int32
	Qlsf [lsfQuantOrderConst]float32
}

// lsfQuantCore is the core LSF quantizer. nlsf is the analysis NLSF
// (smplA2NLSF16(A)), voiced 0/1 selects the codebook, lowRate 0/1, cond the
// conditional-coding params (non-nil => qi[0]==16 reachable), rdWAdj the RD
// weight, surv the beam width (= lsf_surv).
func lsfQuantCore(cb *LsfCb, a *[lsfOrder + 1]float32, nlsf *[lsfOrder]float32, voiced, lowRate int, cond *lsfCondParams, rdWAdj float32, surv int) LsfQuantResult {
	st1 := &cb.St1[voiced]
	st2v := &cb.St2[voiced][lowRate]
	minQi := &cb.MinQi[voiced][lowRate]
	maxQi := &cb.MaxQi[voiced][lowRate]
	minDist := &cb.MinDistUV
	if voiced == 1 {
		minDist = &cb.MinDistV
	}

	var lsf [lsfOrder]float32
	copy(lsf[:], nlsf[:lsfQuantOrderConst])
	// RD weights are the spectral inverse-envelope (SMPL_USE_SPEC_LSW_WEIGHT).
	wlsf := lsfWeightsSpectral(a, &lsf)

	qstep := cb.Qstep[voiced][lowRate]
	qstepCond := qstep * lsfQstepCondMult

	var qim1 [lsfCbCentroids + 1]int32
	vqTemp(&lsf, &st1.Cbhalf, &st1.CbCinv, cond, surv, qim1[:])

	rdBest := float32(math.MaxFloat32)
	var outQi [lsfQuantOrderConst + 1]int32
	var outQlsf [lsfQuantOrderConst]float32

	for s1 := 0; s1 < surv; s1++ {
		qi1 := int(qim1[s1])
		isCond := qi1 == lsfCbCentroids

		// lsfq1 = 2 * cbhalf[qi1] (or cond centroid).
		var lsfq1 [lsfOrder]float32
		if isCond {
			for i := 0; i < lsfQuantOrderConst; i++ {
				lsfq1[i] = cond.st1Cbhalf[i] * 2.0
			}
		} else {
			for i := 0; i < lsfQuantOrderConst; i++ {
				lsfq1[i] = st1.Cbhalf[qi1][i] * 2.0
			}
		}

		// qerr = wie^T * (lsf - lsfq1).
		var qerr_ [lsfOrder]float32
		lsfSubVec(&lsf, &lsfq1, &qerr_)
		var wie *[lsfOrder][lsfOrder]float32
		if isCond {
			wie = &cond.st1Wie
		} else {
			wie = &st1.Wie[qi1]
		}
		qerr := matrixMultTransp16(wie, &qerr_)
		invQstep := 1.0 / qstep
		if isCond {
			invQstep = 1.0 / qstepCond
		}
		for i := range qerr {
			qerr[i] *= invQstep
		}

		bits := st1.BitsCond[qi1]
		if cond == nil {
			bits = st1.Bits[qi1]
		}

		// The clamp tables are indexed by qi1 (0..16); the cond centroid uses
		// the qi1==16 row.
		var alt, qi2 [lsfOrder]int32
		var absQerr, qres [lsfOrder]float32
		st2 := &st2v[qi1]
		for i := 0; i < lsfQuantOrderConst; i++ {
			q2 := int32(f32Round(qerr[i]))
			mn := minQi[qi1][i]
			mx := maxQi[qi1][i]
			if q2 > mx {
				q2 = mx
			}
			if q2 < mn {
				q2 = mn
			}
			qerr[i] -= float32(q2)
			alt[i] = smplSign(qerr[i])
			if (q2 == mx && alt[i] > 0) || (q2 == mn && alt[i] < 0) {
				absQerr[i] = -1.0
			} else {
				absQerr[i] = f32Abs(qerr[i])
			}
			q2 -= mn
			qi2u := int(q2)
			bits += st2.NumBits[i][qi2u]
			qres[i] = st2.Qlvls[i][qi2u]
			qi2[i] = q2
		}

		var iAlt [lsfOrder]int32
		getMaxiK(absQerr[:], iAlt[:], surv)

		var we *[lsfOrder][lsfOrder]float32
		if isCond {
			we = &cond.st1We
		} else {
			we = &st1.We[qi1]
		}
		lsfq := matrixMultTransp16(we, &qres)
		for i := 0; i < lsfQuantOrderConst; i++ {
			lsfq[i] += lsfq1[i]
		}

		surv2 := surv - s1
		indChgd := -1
		bitsOrig := bits
		// The beam base lsfqBase is FIXED to the initial lsfq before the
		// loop; each refinement flips ONE coeff relative to this base,
		// undoing the previous flip.
		lsfqBase := lsfq
		curBits := bits
		for s2 := 0; s2 < surv2; s2++ {
			lsfMinDist(&lsfq, minDist)
			w := lsfWerr(&lsf, &lsfq, &wlsf)
			rd := 0.5*float32(lsfQuantOrderConst)*f32Log2(w)*rdWAdj + curBits
			if rd < rdBest {
				rdBest = rd
				outQi[0] = int32(qi1)
				copy(outQi[1:], qi2[:])
				copy(outQlsf[:], lsfq[:])
			}
			if s2 == surv2-1 || absQerr[int(iAlt[s2])] < 0.25 {
				break
			}
			if s2 > 0 {
				ic := indChgd
				qi2[ic] -= alt[ic]
			}
			indChgd = int(iAlt[s2])
			ic := indChgd
			qi2Old := int(qi2[ic])
			qi2[ic] += alt[ic]
			qi2New := int(qi2[ic])
			qlvlsDiff := st2.Qlvls[ic][qi2New] - st2.Qlvls[ic][qi2Old]
			for i := 0; i < lsfQuantOrderConst; i++ {
				lsfq[i] = lsfqBase[i] + qlvlsDiff*we[ic][i]
			}
			curBits = bitsOrig + st2.NumBits[ic][qi2New] - st2.NumBits[ic][qi2Old]
		}
	}

	return LsfQuantResult{Qi: outQi, Qlsf: outQlsf}
}

func f32Round(x float32) float32 {
	return float32(math.Round(float64(x)))
}

func f32Abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func f32Log2(x float32) float32 {
	return float32(math.Log2(float64(x)))
}

// LsfQuant is the non-conditional LSF quantization. a is the monic LPC
// A[0..17] (A[0]=1); nlsf the analysis NLSF in radians.
func LsfQuant(cb *LsfCb, a *[lsfOrder + 1]float32, nlsf *[lsfOrder]float32, voiced, lowRate int, rdWAdj float32, surv int) LsfQuantResult {
	return lsfQuantCore(cb, a, nlsf, voiced, lowRate, nil, rdWAdj, surv)
}

// LsfQuantCond is the conditional LSF quantization given the previous frame's
// quantized NLSF. a is the monic LPC A[0..17] (A[0]=1).
func LsfQuantCond(cb *LsfCb, a *[lsfOrder + 1]float32, nlsf *[lsfOrder]float32, lsfqPrev *[lsfOrder]float32, voiced, lowRate int, rdWAdj float32, surv int) LsfQuantResult {
	st1 := &cb.St1[voiced]
	cbLsfsMean := &cb.MeanUV
	if voiced == 1 {
		cbLsfsMean = &cb.MeanV
	}
	reg := cb.RegCond[voiced]
	var lsfqPrev_, st1Cbhalf [lsfOrder]float32
	for i := 0; i < lsfQuantOrderConst; i++ {
		lsfqPrev_[i] = lsfqPrev[i] + reg*(cbLsfsMean[i]-lsfqPrev[i])
		st1Cbhalf[i] = 0.5 * lsfqPrev_[i]
	}
	st1CbCinv := matrixMultTransp16(&st1.CInv, &lsfqPrev_)
	st1We, st1Wie := rotApplyWght(&st1.Rotcond[lowRate], &lsfqPrev_)
	cond := lsfCondParams{
		st1Cbhalf: st1Cbhalf,
		st1CbCinv: st1CbCinv,
		st1We:     st1We,
		st1Wie:    st1Wie,
	}
	return lsfQuantCore(cb, a, nlsf, voiced, lowRate, &cond, rdWAdj, surv)
}
