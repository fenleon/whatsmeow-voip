// MLow LPC analysis front-end (encoder side): smpl_window / smpl_lpc /
// smpl_bwe_expand plus smpl_A2NLSF_16 (wrapping silk silk_A2NLSF). Ported
// from whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_lpc.rs; MPL-2.0 for
// this port.
//
// The autocorrelation is derived by a 512-pt real FFT of the windowed signal
// -> power spectrum -> cosine transform (bruteDct) -> reflection coefficients
// (ac2rcDbl) -> A (rc2a), then bandwidth expansion (A[i] *= 0.9999^i).
package mlow

import (
	"math"
	"sync"
)

const (
	smplLpcOrder = 16
	smplLpcNfft  = 512
	smplLpcReg   = float32(5e-7)
	smplLpcBwe   = float32(0.9999)

	smplLpcWin1_20msLen = 264
	smplWin3LongLen     = 64
	smplWin3ShortLen    = 32
	// 20 ms LPC analysis buffer length (lpcbuf_len for 60 ms packets).
	smplLpcBufLen = 448

	nfft4 = smplLpcNfft / 4 // 128

	smplSubfrs  = 4
	maxRcStable = float32(0.9995)
)

// smplFLen is the number of power-spectrum bins smpl_pitch /
// smpl_get_signal_mode read (SMPL_F_LEN).
const smplFLen = smplLpcNfft/2 + 1

var smplPiF64 = float64(smplPi)

// genSinWin: win[i] = sinf((i+1)/(N+1) * PI/2). The reference computes the
// window values with f32::sin (libm sinf); the float64-backed sin32 differs
// from sinf on ~1% of inputs by 1 ulp, which propagates through the windowed
// FFT chains and flips rare quantizer ties.
func genSinWin(n int) []float32 {
	w := make([]float32, n)
	for i := 0; i < n; i++ {
		w[i] = f32Sin((float32(i) + 1.0) / (float32(n) + 1.0) * smplPi / 2.0)
	}
	return w
}

// genCosWin: win[i] = cosf((i+1)/(N+1) * PI/2). See genSinWin for the sinf note.
func genCosWin(n int) []float32 {
	w := make([]float32, n)
	for i := 0; i < n; i++ {
		w[i] = f32Cos((float32(i) + 1.0) / (float32(n) + 1.0) * smplPi / 2.0)
	}
	return w
}

// smplWindowLpc20 is smpl_window for the LPC 20 ms path (use_lpc_win=true,
// frame_ms=20, len=448). useLongWin selects the 64-tap vs 32-tap trailing
// cosine window; the short variant zeros the trailing 32 samples. Produces
// the windowed buffer the FFT consumes.
func smplWindowLpc20(input []float32, useLongWin bool) []float32 {
	win1 := genSinWin(smplLpcWin1_20msLen)
	win3len := smplWin3LongLen
	if !useLongWin {
		win3len = smplWin3ShortLen
	}
	win3 := genCosWin(win3len)
	out := make([]float32, smplLpcBufLen)
	copy(out, input)
	for i := 0; i < smplLpcWin1_20msLen; i++ {
		out[i] = input[i] * win1[i]
	}
	// Untouched middle section copies straight through (no window taper);
	// trailing window region below.
	base := smplLpcBufLen - smplWin3LongLen
	for i := 0; i < win3len; i++ {
		out[base+i] = input[base+i] * win3[i]
	}
	if !useLongWin {
		for i := base + smplWin3ShortLen; i < base+smplWin3LongLen; i++ {
			out[i] = 0
		}
	}
	return out
}

// genCosRow: row[k] = cos(omega)*scale with omega accumulated via
// fmod(omega+domega, 2pi) in f64 (NOT cos(k*domega); the running fmod differs
// in float rounding).
func genCosRow(domega, scale float64) []float64 {
	row := make([]float64, nfft4)
	omega := 0.0
	twoPi := 2.0 * smplPiF64
	for k := 0; k < nfft4; k++ {
		row[k] = math.Cos(omega) * scale
		omega = math.Mod(omega+domega, twoPi)
	}
	return row
}

type dctTables struct {
	cdif     [][]float64 // smplLpcOrder/2 rows
	csumdiff [][]float64 // smplLpcOrder/4 rows
	csumsum  [][]float64 // smplLpcOrder/4 rows
}

func buildDctTables() *dctTables {
	twoPi := 2.0 * smplPiF64
	t := &dctTables{}
	for j := 0; j < smplLpcOrder/2; j++ {
		t.cdif = append(t.cdif, genCosRow(
			float64(1+j*2)*twoPi/float64(smplLpcNfft),
			2.0/float64(smplLpcNfft),
		))
	}
	for j := 0; j < smplLpcOrder/4; j++ {
		t.csumdiff = append(t.csumdiff, genCosRow(
			float64(2+j*4)*twoPi/float64(smplLpcNfft),
			1.0/float64(smplLpcNfft),
		))
	}
	for j := 0; j < smplLpcOrder/4; j++ {
		t.csumsum = append(t.csumsum, genCosRow(
			float64(4+j*4)*twoPi/float64(smplLpcNfft),
			1.0/float64(smplLpcNfft),
		))
	}
	return t
}

var dctTablesOnce struct {
	sync.Once
	t *dctTables
}

func dctTablesShared() *dctTables {
	dctTablesOnce.Do(func() { dctTablesOnce.t = buildDctTables() })
	return dctTablesOnce.t
}

// bruteDct derives the autocorrelation r[0:order] from the power spectrum f2
// via the precomputed cosine sums. All accumulation in f64 (f2 is cast to
// double first).
func bruteDct(t *dctTables, f2 []float64, order int, r []float64) {
	half := smplLpcNfft / 2 // 256
	f2sum := 0.0
	f2Dif := make([]float64, nfft4)
	f2Sumsum := make([]float64, nfft4)
	f2Sumdif := make([]float64, nfft4)
	for n := 0; n < nfft4; n++ {
		f2sum += f2[n] + f2[nfft4+n]
		f2Dif[n] = f2[n] - f2[half-n]
		f2Sumsum[n] = f2[n] + f2[half-n] + f2[nfft4+n] + f2[nfft4-n]
		f2Sumdif[n] = f2[n] + f2[half-n] - f2[nfft4+n] - f2[nfft4-n]
	}
	f2Dif[0] *= 0.5
	r[0] = (2.0*f2sum - f2[0] + f2[half]) / float64(smplLpcNfft)

	for j := 0; j < order/2; j++ {
		rtmp := 0.0
		row := t.cdif[j]
		for k := 0; k < nfft4; k++ {
			rtmp += row[k] * f2Dif[k]
		}
		r[1+j*2] = rtmp
	}
	for j := 0; j < order/4; j++ {
		rtmp := 0.0
		row := t.csumdiff[j]
		for k := 0; k < nfft4; k++ {
			rtmp += row[k] * f2Sumdif[k]
		}
		r[2+j*4] = rtmp
	}
	for j := 0; j < order/4; j++ {
		rtmp := 0.0
		row := t.csumsum[j]
		for k := 0; k < nfft4; k++ {
			rtmp += row[k] * f2Sumsum[k]
		}
		r[4+j*4] = rtmp
	}
}

// ac2rcDbl: autocorrelation corr[0:order] -> reflection coefficients (Schur),
// with c0[0] *= (1+reg). rc[k] is truncated to f32 each step (load-bearing
// for bit-faithfulness).
func ac2rcDbl(corr []float64, order int, reg float32, rc []float32) {
	c0 := make([]float64, order+1)
	c1 := make([]float64, order+1)
	copy(c0, corr[:order+1])
	c0[0] *= float64(1.0 + reg)
	copy(c1, c0)
	for k := 0; k < order; k++ {
		if c0[k+1] > c1[0] {
			rc[k] = -1.0
			break
		}
		if c0[k+1] < -c1[0] {
			rc[k] = 1.0
			break
		}
		if c1[0] == 0.0 {
			break
		}
		rcTmp := -c0[k+1] / c1[0]
		rc[k] = float32(rcTmp)
		for n := 0; n < order-k; n++ {
			ctmp1 := c0[n+k+1]
			ctmp2 := c1[n]
			c0[n+k+1] = ctmp1 + ctmp2*rcTmp
			c1[n] = ctmp2 + ctmp1*rcTmp
		}
	}
}

// rc2a: reflection coefficients -> monic LPC a[0:order+1] (a[0]=1). All f32.
func rc2a(rc []float32, order int, a []float32) {
	for i := 1; i < order+1; i++ {
		a[i] = 0
	}
	a[0] = 1.0
	for k := 0; k < order; k++ {
		rcTmp := rc[k]
		for n := 0; n < (k+1)/2; n++ {
			tmp1 := a[n+1]
			tmp2 := a[k-n]
			a[n+1] = tmp1 + tmp2*rcTmp
			a[k-n] = tmp2 + tmp1*rcTmp
		}
		a[k+1] = rcTmp
	}
}

// bweExpand (the bwe>0 path): a[i] *= bwe^i for i in 1..=order.
func bweExpand(a []float32, order int, bwe float32) {
	c := bwe
	for i := 1; i <= order; i++ {
		a[i] *= c
		c *= bwe
	}
}

// smplLpcAnalyzeWithF2 runs the full smpl_lpc + smpl_bwe_expand over a 20 ms
// windowed LPC buffer (smplLpcBufLen), returning the post-BWE monic LPC
// a[0:17] (a[0]=1) and the power spectrum f2[0:257] (lpcbuf_F2) that
// smpl_pitch (harm_strength) and smpl_get_signal_mode (spectral tilt) consume.
func smplLpcAnalyzeWithF2(windowed []float32, fft *fftScratch) (a [smplLpcOrder + 1]float32, f2 [smplFLen]float32) {
	// Zero-pad to NFFT and forward real FFT (pffft ordered layout).
	xbuf := make([]float32, smplLpcNfft)
	copy(xbuf, windowed)
	f := make([]float32, smplLpcNfft)
	rfftForwardOrderedSC(xbuf, f, fft)

	// Power spectrum f2 (float); pffft packs DC in f[0], Nyquist in f[1].
	f2[0] = f[0] * f[0]
	f2[smplLpcNfft/2] = f[1] * f[1]
	for i := 1; i < smplLpcNfft/2; i++ {
		f2[i] = f[2*i]*f[2*i] + f[2*i+1]*f[2*i+1]
	}
	f2d := make([]float64, smplFLen)
	for i, v := range f2 {
		f2d[i] = float64(v)
	}

	var r [smplLpcOrder + 1]float64
	bruteDct(dctTablesShared(), f2d[:], smplLpcOrder, r[:])

	var rc [smplLpcOrder]float32
	ac2rcDbl(r[:], smplLpcOrder, smplLpcReg, rc[:])
	rc2a(rc[:], smplLpcOrder, a[:])
	bweExpand(a[:], smplLpcOrder, smplLpcBwe)
	return a, f2
}

// smplLsfInterpol4: per-subframe interpolation weights (idx 0 and the
// alternative idx 1 that lsf_interpol_search may pick when it lowers the
// residual energy).
var smplLsfInterpol4Tbl = [2][smplSubfrs]float32{
	{0.55, 0.88, 1.0, 1.0},
	{0.3, 0.65, 0.95, 1.0},
}

// lpcIsStable reports whether the monic LPC a[0:17] (a[0]=1) is a stable
// all-pole filter.
func lpcIsStable(a []float32) bool {
	order := smplLpcOrder
	if a[order]*a[order] > maxRcStable {
		return false
	}
	var a0, a1 [smplLpcOrder]float64
	for i := 0; i < order; i++ {
		a0[i] = float64(a[i+1])
	}
	m := order - 1
	for {
		den := 1.0 - a0[m]*a0[m]
		if den == 0.0 {
			return false
		}
		inv := 1.0 / den
		for k := 0; k < m; k++ {
			a1[k] = (a0[k] - a0[m]*a0[m-k-1]) * inv
		}
		if a1[m-1]*a1[m-1] > float64(maxRcStable) {
			return false
		}
		if m == 1 {
			return true
		}
		m--
		den = 1.0 - a1[m]*a1[m]
		if den == 0.0 {
			return false
		}
		inv = 1.0 / den
		for k := 0; k < m; k++ {
			a0[k] = (a1[k] - a1[m]*a1[m-k-1]) * inv
		}
		if a0[m-1]*a0[m-1] > float64(maxRcStable) {
			return false
		}
		if m == 1 {
			return true
		}
		m--
	}
}

// lpcStabilize bandwidth-expands until stable.
func lpcStabilize(a []float32) {
	if lpcIsStable(a) {
		return
	}
	for iter := 1; ; iter++ {
		bweExpand(a, smplLpcOrder, 1.0-float32(iter)*0.001)
		if lpcIsStable(a) {
			return
		}
	}
}

// smplLpcInterpol returns the per-subframe interpolated LPC predcoefs
// (lsf_interpol_idx 0), and the updated ilsf (the last subframe's
// interpolated NLSF, carried to the next internal frame). On a reset
// (prevLsf all zero) lsf is copied into the effective prev.
func smplLpcInterpol(lsf, prevLsf []float32) ([smplSubfrs][smplLpcOrder + 1]float32, [smplLpcOrder]float32) {
	return smplLpcInterpolIdx(lsf, prevLsf, 0)
}

// smplLpcInterpolIdx is smplLpcInterpol for an explicit lsf_interpol_idx (the
// smpl_lsf_interpol_4 row).
func smplLpcInterpolIdx(lsf, prevLsf []float32, interpolIdx int) ([smplSubfrs][smplLpcOrder + 1]float32, [smplLpcOrder]float32) {
	interp := &smplLsfInterpol4Tbl[min(interpolIdx, 1)]
	var prev [smplLpcOrder]float32
	if len(prevLsf) == smplLpcOrder && prevLsf[smplLpcOrder-1] != 0.0 {
		copy(prev[:], prevLsf)
	} else {
		copy(prev[:], lsf[:smplLpcOrder])
	}
	var predcoefs [smplSubfrs][smplLpcOrder + 1]float32
	var ilsf [smplLpcOrder]float32
	for j := 0; j < smplSubfrs; j++ {
		w := interp[j]
		if w == 1.0 {
			copy(ilsf[:], lsf[:smplLpcOrder])
		} else {
			for k := 0; k < smplLpcOrder; k++ {
				ilsf[k] = (1.0-w)*prev[k] + w*lsf[k]
			}
		}
		a := NLSF2A(ilsf[:])
		pc := a
		pc[0] = 1.0
		lpcStabilize(pc[:])
		predcoefs[j] = pc
	}
	return predcoefs, ilsf
}

// --- silk_A2NLSF (fixed-point forward A -> NLSF) ---

const (
	lsfCosTabSzFix         = 128
	binDivStepsA2nlsfFix   = 3
	maxIterationsA2nlsfFix = 16
)

// silk_LSFCosTab_FIX_Q12 (Q12, 129 entries) for the A2NLSF root search.
var silkLsfCosTabFixQ12 = [lsfCosTabSzFix + 1]int32{
	8192, 8190, 8182, 8170, 8152, 8130, 8104, 8072,
	8034, 7994, 7946, 7896, 7840, 7778, 7714, 7644,
	7568, 7490, 7406, 7318, 7226, 7128, 7026, 6922,
	6812, 6698, 6580, 6458, 6332, 6204, 6070, 5934,
	5792, 5648, 5502, 5352, 5198, 5040, 4880, 4718,
	4552, 4382, 4212, 4038, 3862, 3684, 3502, 3320,
	3136, 2948, 2760, 2570, 2378, 2186, 1990, 1794,
	1598, 1400, 1202, 1002, 802, 602, 402, 202,
	0, -202, -402, -602, -802, -1002, -1202, -1400,
	-1598, -1794, -1990, -2186, -2378, -2570, -2760, -2948,
	-3136, -3320, -3502, -3684, -3862, -4038, -4212, -4382,
	-4552, -4718, -4880, -5040, -5198, -5352, -5502, -5648,
	-5792, -5934, -6070, -6204, -6332, -6458, -6580, -6698,
	-6812, -6922, -7026, -7128, -7226, -7318, -7406, -7490,
	-7568, -7644, -7714, -7778, -7840, -7896, -7946, -7994,
	-8034, -8072, -8104, -8130, -8152, -8170, -8182, -8190,
	-8192,
}

func silkRshiftRound(a, shift int32) int32 {
	if shift == 1 {
		return (a >> 1) + (a & 1)
	}
	return ((a >> uint(shift-1)) + 1) >> 1
}

// silkSmlaww: a32 + ((i64)b32 * c32 >> 16).
func silkSmlaww(a32, b32, c32 int32) int32 {
	return int32(int64(a32) + ((int64(b32) * int64(c32)) >> 16))
}

// silkBwexpander32 chirps (bandwidth expands) LPC coefficients in Q16 in
// place: chirp_Q16 += RSHIFT_ROUND(MUL(chirp, chirp-65536), 16) with MUL a
// wrapping i32 multiply.
func silkBwexpander32(ar []int32, d int, chirpQ16 int32) {
	chirp := chirpQ16
	chirpMinusOne := chirpQ16 - 65536
	for i := 0; i < d-1; i++ {
		ar[i] = int32((int64(chirp) * int64(ar[i])) >> 16) // silk_SMULWW
		mul := chirp * chirpMinusOne                       // silk_MUL (i32 wrap)
		chirp += silkRshiftRound(mul, 16)
	}
	ar[d-1] = int32((int64(chirp) * int64(ar[d-1])) >> 16)
}

func silkA2nlsfTransPoly(p []int32, dd int) {
	for k := 2; k <= dd; k++ {
		for n := dd; n >= k+1; n-- {
			p[n-2] -= p[n]
		}
		p[k-2] -= p[k] << 1
	}
}

func silkA2nlsfEvalPoly(p []int32, x int32, dd int) int32 {
	xQ16 := x << 4
	y32 := p[dd]
	for n := dd - 1; n >= 0; n-- {
		y32 = silkSmlaww(p[n], y32, xQ16)
	}
	return y32
}

func silkA2nlsfInit(aQ16 []int32, p, q []int32, dd int) {
	p[dd] = 1 << 16
	q[dd] = 1 << 16
	for k := 0; k < dd; k++ {
		p[k] = -aQ16[dd-k-1] - aQ16[dd+k]
		q[k] = -aQ16[dd-k-1] + aQ16[dd+k]
	}
	for k := dd; k >= 1; k-- {
		p[k-1] -= p[k]
		q[k-1] += q[k]
	}
	silkA2nlsfTransPoly(p, dd)
	silkA2nlsfTransPoly(q, dd)
}

// silkA2nlsf: monic whitening coefficients (Q16) -> NLSF (Q15, 0..2^15-1).
// Mutates aQ16 (bandwidth expansion on non-convergence). d is the filter
// order (even).
func silkA2nlsf(nlsf []int32, aQ16 []int32, d int) {
	dd := d >> 1
	p := make([]int32, dd+1)
	q := make([]int32, dd+1)

	silkA2nlsfInit(aQ16, p, q, dd)

	// useQ: false => P, true => Q
	useQ := false
	xlo := silkLsfCosTabFixQ12[0]
	ylo := silkA2nlsfEvalPoly(p, xlo, dd)

	rootIx := 0
	if ylo < 0 {
		nlsf[0] = 0
		useQ = true
		ylo = silkA2nlsfEvalPoly(q, xlo, dd)
		rootIx = 1
	}
	k := 1
	i := 0
	thr := int32(0)
	for {
		xhi := silkLsfCosTabFixQ12[k]
		poly := p
		if useQ {
			poly = q
		}
		yhi := silkA2nlsfEvalPoly(poly, xhi, dd)

		if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
			thr = 0
			if yhi == 0 {
				thr = 1
			}
			xloL, yloL, xhiL := xlo, ylo, xhi
			ffrac := int32(-256)
			for m := 0; m < binDivStepsA2nlsfFix; m++ {
				xmid := silkRshiftRound(xloL+xhiL, 1)
				ymid := silkA2nlsfEvalPoly(poly, xmid, dd)
				if (yloL <= 0 && ymid >= 0) || (yloL >= 0 && ymid <= 0) {
					xhiL = xmid
					yhi = ymid
				} else {
					xloL = xmid
					yloL = ymid
					ffrac += 128 >> uint(m)
				}
			}
			if absI32(yloL) < 65536 {
				den := yloL - yhi
				nom := (yloL << uint(8-binDivStepsA2nlsfFix)) + (den >> 1)
				if den != 0 {
					ffrac += nom / den
				}
			} else {
				ffrac += yloL / ((yloL - yhi) >> uint(8-binDivStepsA2nlsfFix))
			}
			nlsf[rootIx] = minI32((int32(k)<<8)+ffrac, silkInt16Max)

			rootIx++
			if rootIx >= d {
				break
			}
			useQ = rootIx&1 != 0
			xlo = silkLsfCosTabFixQ12[k-1]
			ylo = (1 - int32(rootIx&2)) << 12
		} else {
			k++
			xlo = xhi
			ylo = yhi
			thr = 0
			if k > lsfCosTabSzFix {
				i++
				if i > maxIterationsA2nlsfFix {
					nlsf[0] = (1 << 15) / int32(d+1)
					for kk := 1; kk < d; kk++ {
						nlsf[kk] = nlsf[kk-1] + nlsf[0]
					}
					return
				}
				silkBwexpander32(aQ16, d, 65536-(1<<uint(i)))
				silkA2nlsfInit(aQ16, p, q, dd)
				useQ = false
				xlo = silkLsfCosTabFixQ12[0]
				ylo = silkA2nlsfEvalPoly(p, xlo, dd)
				if ylo < 0 {
					nlsf[0] = 0
					useQ = true
					ylo = silkA2nlsfEvalPoly(q, xlo, dd)
					rootIx = 1
				} else {
					rootIx = 0
				}
				k = 1
			}
		}
	}
}

func absI32(a int32) int32 {
	if a < 0 {
		return -a
	}
	return a
}

// smplA2NLSF16: post-BWE float a[0:17] (a[0]=1) -> analysis NLSF in radians
// (0..pi).
func smplA2NLSF16(a []float32) [smplLpcOrder]float32 {
	var aQ16 [smplLpcOrder]int32
	for i := 0; i < smplLpcOrder; i++ {
		aQ16[i] = int32(math.Round(float64(-a[i+1] * 65536.0)))
	}
	var lsfQ15 [smplLpcOrder]int32
	silkA2nlsf(lsfQ15[:], aQ16[:], smplLpcOrder)
	var nlsf [smplLpcOrder]float32
	for i := 0; i < smplLpcOrder; i++ {
		nlsf[i] = float32(lsfQ15[i]) / 32768.0 * smplPi
	}
	return nlsf
}
