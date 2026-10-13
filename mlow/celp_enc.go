// MLow (smpl_audio_codec) CELP excitation encoder plus the unvoiced
// residual-energy quantizer. The encoder builds the perceptually weighted
// impulse response (Phi), an LTP/ACB basis for voiced frames, runs the FCB
// pulse search (greedy + delayed-decision beam), and RD-quantizes the
// adaptive- and fixed-codebook gains against entropy tables built once at
// init. Ported from whatsapp-rust (MIT)
// wacore/src/voip/mlow/smpl_celp.rs + smpl_nrgres.rs; MPL-2.0 for this port.
//
// Self-contained on purpose (as upstream): small leaf helpers whose decoder
// counterparts differ (filt_ma/filt_ma9/ar16 history layout, symtoepl2, the
// vector ops) are duplicated here rather than shared. Existing package
// helpers are reused where the arithmetic is identical: smplDotProd, smplNrg,
// getMaxi/getMaxiK, dcmfToCmf (same truncating-int expansion), synLTPBasis,
// acbDequant, acbSynthesize (highBoost=0 is the encoder's no-boost path),
// pitchSharpen, the interpolation kernel, the fcb gain magnitude tables
// (fcbGainsOnce) and the ACB-gain Q14 codebooks.
package mlow

import (
	"math"
	"sync"
)

const (
	// celpMaxPitchLag (320, celpdec.go) covers both SMPL_MAXPITCH_LEN and
	// SMPL_MAX_PITCH_LAG; smplMaxSFLen, acbgN/acbgRows, fcbgVN/fcbgVDeltaN,
	// smplMaxPulsesPerSF and smplLpcOrder likewise reuse existing constants.
	smplMaxLResp    = 33 // 32 + 1
	smplPercRespLen = 32

	acbgM = 2

	smplGAcbRdMu float32 = 0.014999999664723873

	smplVGainMaxDB  float32 = 0.0
	smplUVGainMaxDB float32 = 0.0

	smplRateAcbScale float32 = 0.9
	smplFcbSrvMax            = 4
	celpMaxNumSurv           = 8

	celpIdxFEC   = 0
	celpIdxMain  = 1
	celpMaxRates = 2

	nGainSteps = 2
)

// Raw constant codec tables.

// smplAcbgainsDcmfHR is the high-rate ACB-gain DCMF entropy table
// (SMPL_ACBGAINS_DCMF_HR); the low-rate table is smplAcbgainsDcmfLR in
// cc_tables.go.
var smplAcbgainsDcmfHR = [acbgRows * acbgN]byte{
	254, 105, 212, 26, 110, 255, 202, 93, 152, 121, 110, 43, 150, 20, 81, 176,
	255, 28, 100, 5, 26, 184, 61, 29, 36, 26, 28, 9, 61, 4, 27, 116,
	121, 255, 161, 39, 195, 215, 191, 75, 186, 178, 119, 82, 68, 41, 43, 56,
	188, 65, 243, 15, 74, 255, 205, 79, 123, 84, 95, 26, 139, 13, 67, 154,
	81, 219, 173, 70, 219, 165, 234, 102, 231, 255, 191, 119, 87, 60, 62, 59,
	106, 255, 182, 49, 242, 196, 233, 95, 247, 228, 152, 96, 81, 45, 54, 61,
	236, 55, 178, 10, 56, 255, 131, 54, 85, 58, 59, 18, 93, 9, 43, 133,
	123, 95, 224, 24, 113, 202, 255, 105, 186, 134, 135, 38, 141, 18, 82, 111,
	126, 97, 204, 34, 126, 186, 255, 141, 210, 147, 149, 46, 165, 22, 113, 122,
	96, 156, 185, 42, 188, 178, 255, 116, 248, 199, 157, 66, 109, 29, 69, 75,
	102, 207, 194, 57, 224, 193, 255, 107, 253, 242, 180, 95, 97, 44, 60, 64,
	105, 119, 202, 39, 140, 189, 255, 110, 207, 173, 165, 54, 119, 24, 75, 85,
	74, 255, 142, 59, 214, 150, 182, 76, 194, 215, 138, 122, 61, 56, 41, 45,
	200, 53, 255, 17, 66, 238, 222, 109, 129, 78, 101, 21, 227, 11, 110, 243,
	74, 255, 128, 50, 187, 149, 154, 63, 165, 184, 115, 101, 52, 47, 37, 34,
	159, 66, 232, 26, 86, 196, 255, 146, 171, 113, 134, 31, 245, 16, 145, 190,
	255, 29, 182, 7, 33, 235, 115, 55, 59, 37, 47, 11, 139, 6, 60, 234,
}

var smplFcbgVDcmf = [fcbgVN]byte{
	107, 12, 17, 25, 31, 41, 52, 65, 83, 103, 122, 146, 169, 191, 210, 227,
	240, 249, 255, 253, 246, 229, 200, 161, 120, 82, 51, 29, 14, 6, 2, 2,
	2, 2,
}

var smplFcbgVDeltaDcmf = [fcbgVDeltaN]byte{
	1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 6, 8, 10, 12, 12,
	12, 13, 14, 14, 14, 13, 12, 11, 10, 9, 8, 9, 15, 33, 65, 119,
	196, 255, 220, 144, 90, 57, 36, 23, 17, 14, 12, 12, 12, 13, 12, 12,
	12, 12, 12, 11, 11, 10, 9, 7, 6, 4, 3, 2, 1, 1, 1, 1,
	1, 1, 1,
}

// Tier 1: leaf math helpers.

// smplSubVecInplace: x -= y. Scaffolding kept from the Rust module (currently
// uncalled there too).
func smplSubVecInplace(y, x []float32, l int) {
	for i := 0; i < l; i++ {
		x[i] -= y[i]
	}
}

// smplSubVec: x = y - z.
func smplSubVec(y, z, x []float32, l int) {
	for i := 0; i < l; i++ {
		x[i] = y[i] - z[i]
	}
}

func smplAddVecInplace(y, x []float32, l int) {
	for i := 0; i < l; i++ {
		x[i] += y[i]
	}
}

func smplScaleVecInplace(x []float32, l int, g float32) {
	for i := 0; i < l; i++ {
		x[i] *= g
	}
}

func smplScaleVec(x, y []float32, l int, g float32) {
	for i := 0; i < l; i++ {
		y[i] = x[i] * g
	}
}

// smplAddScaleVecInplace: y += g * x.
func smplAddScaleVecInplace(x, y []float32, l int, g float32) {
	for i := 0; i < l; i++ {
		y[i] += g * x[i]
	}
}

// smplAddScaleVec: y = x0 + g * x1.
func smplAddScaleVec(x0, x1, y []float32, l int, g float32) {
	for i := 0; i < l; i++ {
		y[i] = x0[i] + g*x1[i]
	}
}

func smplMulVecInplace(x, y []float32, l int) {
	for i := 0; i < l; i++ {
		y[i] *= x[i]
	}
}

func smplCelpQ(num, den, q []float32, l int) {
	for i := 0; i < l; i++ {
		q[i] = (num[i] * num[i]) / den[i]
	}
}

func smplReverse(x []float32, l int) {
	for i := 0; i < l/2; i++ {
		x[i], x[l-i-1] = x[l-i-1], x[i]
	}
}

// smplMultSymtoepl2 is `y[n] = sum C[.] * x[.]` for a symmetric Toeplitz
// multiply. c must carry the trailing zero at index 2*lResp-1 and x must be
// readable up to n + lResp (zero padded past n): the inner loop deliberately
// reads one extra zero element for SIMD friendliness.
func smplMultSymtoepl2(c []float32, lResp int, x, y []float32, n int) {
	dotLen := lResp
	nn := 0
	for nn < lResp-1 {
		y[nn] = smplDotProd(c[lResp-1-nn:], x, dotLen)
		dotLen++
		nn++
	}
	dotLen = 2 * lResp
	for nn < n-lResp {
		// nn >= lResp-1 here, so (nn + 1 - lResp) stays non-negative.
		y[nn] = smplDotProd(c, x[nn+1-lResp:], dotLen)
		nn++
	}
	for nn < n {
		dotLen--
		y[nn] = smplDotProd(c, x[nn+1-lResp:], dotLen)
		nn++
	}
}

// smplFiltAR16 is a 16th-order AR filter. The 16-sample state sits in
// y[yBase-16 .. yBase]; x/y are slices whose index 0 is the logical sample 0
// (so callers pass an offset slice for the history).
func smplFiltAR16(x []float32, n int, coef []float32, yBase int, y []float32) {
	for nn := 0; nn < n; nn++ {
		res := x[nn]
		for i := 0; i < 16; i++ {
			res -= coef[16-i] * y[yBase+nn-16+i]
		}
		y[yBase+nn] = res
	}
}

// smplFiltMA is a MA filter: the (coefLen-1) history samples sit before
// x[xBase] (caller passes an offset). x != y.
func smplFiltMA(x []float32, xBase, n int, coef []float32, coefLen int, y []float32) {
	var i int
	if coef[0] == 1.0 {
		// y[k] = x[k] + coef[1]*x[k-1]
		for k := 0; k < n; k++ {
			y[k] = x[xBase+k] + coef[1]*x[xBase+k-1]
		}
		i = 2
	} else {
		for k := 0; k < n; k++ {
			y[k] = coef[0] * x[xBase+k]
		}
		i = 1
	}
	for i < coefLen {
		for k := 0; k < n; k++ {
			y[k] += coef[i] * x[xBase+k-i]
		}
		i++
	}
}

// smplFiltMA9 is a 9th-order MA. The 9-sample history sits before x[xBase]
// (caller passes an offset).
func smplFiltMA9(x []float32, xBase, n int, coef []float32, y []float32) {
	for nn := 0; nn < n; nn++ {
		var res float32
		for i := 0; i < 10; i++ {
			res += coef[i] * x[xBase+nn-i]
		}
		y[nn] = res
	}
}

// smplCmfToBits is -log2(p) per symbol from a cumulative u16 CDF.
func smplCmfToBits(cmf []uint16, bits []float32) {
	for i := 0; i < len(cmf)-1; i++ {
		bits[i] = -f32Log2(float32(cmf[i+1]-cmf[i]) / float32(cmf[len(cmf)-1]))
	}
}

// Tier 2: entropy/gain table builder.

type celpEncTables struct {
	acbgainsCmfLR     [acbgRows * acbgRows]uint16
	acbgainsCmfHR     [acbgRows * acbgRows]uint16
	acbgInvProbLR     [acbgRows * acbgN]float32
	acbgInvProbHR     [acbgRows * acbgN]float32
	fcbgainsVCmf      [fcbgVN + 1]uint16
	fcbgainsVDeltaCmf [fcbgVDeltaN + 1]uint16
	fcbgVInvProb      [fcbgVN]float32
	fcbgVDeltaInvProb [fcbgVDeltaN]float32
}

var celpTablesOnce = sync.OnceValue(func() *celpEncTables {
	return buildCelpTables()
})

func celpTables() *celpEncTables {
	return celpTablesOnce()
}

func buildCelpTables() *celpEncTables {
	t := &celpEncTables{}

	// (a) acbgains cmf, 17 rows of 16-dcmf -> 17-cmf. dcmfToCmf keeps the
	// truncating integer normalization, which the RD winner depends on
	// bit-exactly.
	for i := 0; i < acbgRows; i++ {
		copy(t.acbgainsCmfLR[i*acbgRows:(i+1)*acbgRows], dcmfToCmf(smplAcbgainsDcmfLR[i*acbgN:(i+1)*acbgN]))
		copy(t.acbgainsCmfHR[i*acbgRows:(i+1)*acbgRows], dcmfToCmf(smplAcbgainsDcmfHR[i*acbgN:(i+1)*acbgN]))
	}

	// (b) acbg inv-prob: per row, cmf_to_bits(17) -> bits[16], then 2^(bits*mu)
	for i := 0; i < acbgRows; i++ {
		smplCmfToBits(t.acbgainsCmfLR[i*acbgRows:(i+1)*acbgRows], t.acbgInvProbLR[i*acbgN:(i+1)*acbgN])
		smplCmfToBits(t.acbgainsCmfHR[i*acbgRows:(i+1)*acbgRows], t.acbgInvProbHR[i*acbgN:(i+1)*acbgN])
		for j := 0; j < acbgN; j++ {
			t.acbgInvProbLR[i*acbgN+j] = float32(math.Pow(2, float64(t.acbgInvProbLR[i*acbgN+j]*smplGAcbRdMu)))
			t.acbgInvProbHR[i*acbgN+j] = float32(math.Pow(2, float64(t.acbgInvProbHR[i*acbgN+j]*smplGAcbRdMu)))
		}
	}

	// (c) fcb voiced gain cmfs + inv-prob
	copy(t.fcbgainsVCmf[:], dcmfToCmf(smplFcbgVDcmf[:]))
	copy(t.fcbgainsVDeltaCmf[:], dcmfToCmf(smplFcbgVDeltaDcmf[:]))
	smplCmfToBits(t.fcbgainsVCmf[:], t.fcbgVInvProb[:])
	for i := 0; i < fcbgVN; i++ {
		t.fcbgVInvProb[i] = float32(math.Pow(2, float64(t.fcbgVInvProb[i]*smplGAcbRdMu)))
	}
	smplCmfToBits(t.fcbgainsVDeltaCmf[:], t.fcbgVDeltaInvProb[:])
	for i := 0; i < fcbgVDeltaN; i++ {
		t.fcbgVDeltaInvProb[i] = float32(math.Pow(2, float64(t.fcbgVDeltaInvProb[i]*smplGAcbRdMu)))
	}

	// (d) gain magnitude tables: the fcbGains tables built by fcbGainsOnce in
	// celpdec.go compute the identical 10^(0.05*db) values, so they are reused.

	return t
}

// Tier 3: LTP / ACB synthesis is shared with the decoder (synLTPBasis,
// acbDequant, acbSynthesize, pitchSharpen in celpdec.go); the encoder passes
// highBoost=0, which makes acbSynthesize the plain 3-tap symmetric basis
// combination.

// Tier 4: FCB search.

type fcb struct {
	wnrg        float32
	nPulses     int32
	posNew      int32
	signNew     float32
	sgntr       uint64
	fcbStateIdx int
}

// fcbState uses fixed-length (smplMaxSFLen) scratch indexed by subframe
// position; never resized, so per-subframe copies in the survivor search are
// memmoves instead of allocations.
type fcbState struct {
	pulsePositions [smplMaxSFLen]int32
	pulseSigns     [smplMaxSFLen]float32
	num            [smplMaxSFLen]float32
	den            [smplMaxSFLen]float32
}

func calcDAbsAndSign(d []float32, l int, dAbs, dSign []float32) {
	for i := 0; i < l; i++ {
		if d[i] > 0.0 {
			dAbs[i] = d[i]
			dSign[i] = 1.0
		} else {
			dAbs[i] = -d[i]
			dSign[i] = -1.0
		}
	}
}

func checkIfBetter(wnrg float32, nrgThr *float32, wnrgPerPulse float32) bool {
	*nrgThr += wnrgPerPulse
	if wnrg > *nrgThr {
		*nrgThr = wnrg
		return true
	}
	return false
}

// phiColOffset returns the start offset into phiFlip for the PhiFlip column of
// col: the column starts at phiFlip[smplMaxSFLen-col] and is read at arbitrary
// non-negative indices.
func phiColOffset(col int32) int32 {
	return smplMaxSFLen - col
}

func nonZeroRange(col int32, percRespLen, fcbSubfrLen int) (int, int) {
	lo := col - int32(percRespLen) + 1
	if lo < 0 {
		lo = 0
	}
	hi := col + int32(percRespLen)
	if hi > int32(fcbSubfrLen) {
		hi = int32(fcbSubfrLen)
	}
	return int(lo), int(hi)
}

// Public output of the per-subframe encoder.

type CelpSubframeOut struct {
	Pulses  [celpMaxRates][]int16
	NPulses [celpMaxRates]int16
	AcbIdx  [celpMaxRates]int16
	GainIdx [celpMaxRates]int16
	ExcLPC  []float32
}

// ACBG params, carried within a single EncodeSubframe call.

type acbgParams struct {
	werrIn      float32
	phiAcb      [acbgM * acbgM]float32
	dAcbLPC     [acbgM]float32
	acbBasisPhi []float32 // acbgM * fcbSubfrLen
}

// Persistent encoder state.

type CelpEncoder struct {
	// stateWghtBuf has smplLpcOrder samples of history before the "logical"
	// start; logical index 0 == stateWghtBuf[smplLpcOrder].
	stateWghtBuf   []float32 // smplMaxSFLen + smplLpcOrder
	stateErrLPCSyn [smplLpcOrder]float32
	hanningWin     []float32 // percRespLen
	// sgntrs are random u64, only used to dedup identical pulse SETS in the
	// deldec beam. A deterministic distinct-per-position sequence is
	// correctness-equivalent (a different beam tie-break can pick a different
	// but equally valid candidate; bit-exact reproduction of the reference's
	// RNG is not required for a correct encoder). Fixed LCG for reproducibility.
	sgntrs         []uint64 // smplMaxSFLen
	acbState       []float32
	acbStateLen    int
	prevAcbIdx     [celpMaxRates]int32
	prevFcbIdx     [celpMaxRates]int32
	subfrCnt       int32
	subfrPerPacket int32
	fcbSubfrLen    int
	percRespLen    int
	lowRate        bool
	ignoreZir      bool
	fcbgain        float32
	useMA9         bool

	// Scratch reused across calls.
	impLPCBuf []float32 // smplMaxSFLen + smplLpcOrder, logical start at smplLpcOrder
	phi       []float32 // smplMaxSFLen
	phiFlip   []float32 // 2 * smplMaxSFLen
	// Greedy-FCB-search scratch (per subframe + per pulse), each fully
	// rewritten over [0..fcbSubfrLen] before it is read.
	fcbGreedy fcbGreedyScratch
	// Delayed-decision beam-search scratch (the heavier path), reset per
	// search; buffers are overwritten before they are read.
	fcbSearch fcbSearchScratch
	// Per-subframe working buffers for EncodeSubframe; each is zeroed to
	// exactly the needed length at the start of each call (zeroed()).
	sf subframeScratch
}

// subframeScratch pools the per-EncodeSubframe buffers. Each buffer is sized
// on first use and zeroed/reused thereafter.
type subframeScratch struct {
	impLPCRev []float32
	resLPCPad []float32
	dLPC      []float32
	zirLPC    []float32
	acbBasis  []float32
	acb       []float32
	dLtp      []float32
	wtgtTmp   []float32
	wtgt      []float32
	excFcb    []float32
}

// zeroed reuses (or allocates) a pooled buffer holding exactly n zeros — the
// counterpart of Rust's clear()+resize(n, 0.0) scratch pooling.
func zeroed(buf *[]float32, n int) []float32 {
	if cap(*buf) < n {
		*buf = make([]float32, n)
	}
	b := (*buf)[:n]
	for i := range b {
		b[i] = 0
	}
	return b
}

// fcbGreedyScratch holds the per-call working buffers for the greedy
// smplFcbSearch, hoisted off the per-frame/per-pulse hot path.
type fcbGreedyScratch struct {
	dAbs  [smplMaxSFLen]float32
	dSign [smplMaxSFLen]float32
	num   [smplMaxSFLen]float32
	den   [smplMaxSFLen]float32
	q     [smplMaxSFLen]float32
}

func NewCelpEncoder(lowRate bool, percRespLen, fcbSubfrLen, subfrPerPacket int) *CelpEncoder {
	// Rust debug_asserts: perc_resp_len <= SMPL_MAX_L_RESP,
	// fcb_subfrlen <= SMPL_MAX_SF_LEN (compiled out in release builds).

	// Force table build at first construction.
	_ = celpTables()

	acbStateLen := fcbSubfrLen + celpMaxPitchLag + celpLTPInterpolDelay

	sgntrs := make([]uint64, smplMaxSFLen)
	s := uint64(0x9E3779B97F4A7C15)
	for i := range sgntrs {
		s = s*6364136223846793005 + 1442695040888963407
		sgntrs[i] = s
	}

	// The Rust computes this with f32 sinf at init; sin32 is the matching
	// wrapper (f32libm.go).
	hanningWin := make([]float32, percRespLen)
	scale := 1.0 / float32(2*smplPercRespLen+1)
	for i := 0; i < percRespLen; i++ {
		hanningWin[i] = sin32(smplPi * float32(percRespLen+i+1) * scale)
	}
	useMA9 := percRespLen == 10

	return &CelpEncoder{
		stateWghtBuf:   make([]float32, smplMaxSFLen+smplLpcOrder),
		stateErrLPCSyn: [smplLpcOrder]float32{},
		hanningWin:     hanningWin,
		sgntrs:         sgntrs,
		acbState:       make([]float32, celpMaxPitchLag+smplMaxSFLen+celpLTPInterpolDelay),
		acbStateLen:    acbStateLen,
		prevAcbIdx:     [celpMaxRates]int32{-1, -1},
		prevFcbIdx:     [celpMaxRates]int32{-1, -1},
		subfrCnt:       0,
		subfrPerPacket: int32(subfrPerPacket),
		fcbSubfrLen:    fcbSubfrLen,
		percRespLen:    percRespLen,
		lowRate:        lowRate,
		ignoreZir:      false,
		fcbgain:        0.0,
		useMA9:         useMA9,
		impLPCBuf:      make([]float32, smplMaxSFLen+smplLpcOrder),
		phi:            make([]float32, smplMaxSFLen),
		phiFlip:        make([]float32, 2*smplMaxSFLen),
		fcbGreedy:      fcbGreedyScratch{},
		fcbSearch:      newFcbSearchScratch(),
	}
}

func (e *CelpEncoder) percFiltMA(x []float32, xBase, n int, coef []float32, coefLen int, y []float32) {
	if e.useMA9 {
		smplFiltMA9(x, xBase, n, coef, y)
	} else {
		smplFiltMA(x, xBase, n, coef, coefLen, y)
	}
}

// fcbSearchScratch holds the deldec-relevant working buffers.

type fcbSearchScratch struct {
	// Double-buffered candidate states (read/write ping-pong), each
	// celpMaxNumSurv entries.
	fcbStates         [2][]fcbState
	readIdx           int
	writeIdx          int
	fcbs              []fcb // celpMaxNumSurv
	fcbsSize          int
	fcbCandidates     []fcb // up to celpMaxNumSurv^2
	fcbCandidatesSize int
	uniqueSgntr       []uint64 // up to celpMaxNumSurv^2
	uniqueSgntrSize   int
	// Per-addPulse working buffers. q is fully written by smplCelpQ before it
	// is read; ddDen accumulates so it is zeroed per call.
	q     []float32 // smplMaxSFLen
	ddDen []float32 // smplMaxSFLen
	// The pitch-sharpened deldec target. Fully overwritten over the live
	// region before it is read.
	dNew []float32 // smplMaxSFLen
}

func newFcbSearchScratch() fcbSearchScratch {
	mk := func() []fcbState { return make([]fcbState, celpMaxNumSurv) }
	return fcbSearchScratch{
		fcbStates:     [2][]fcbState{mk(), mk()},
		readIdx:       0,
		writeIdx:      1,
		fcbs:          make([]fcb, celpMaxNumSurv),
		fcbCandidates: make([]fcb, celpMaxNumSurv*celpMaxNumSurv),
		uniqueSgntr:   make([]uint64, celpMaxNumSurv*celpMaxNumSurv),
		q:             make([]float32, smplMaxSFLen),
		ddDen:         make([]float32, smplMaxSFLen),
		dNew:          make([]float32, smplMaxSFLen),
	}
}

// reset puts the bookkeeping back to a fresh-search state (the buffers are
// overwritten before they are read, so only the indices/sizes need clearing).
func (sc *fcbSearchScratch) reset() {
	sc.readIdx = 0
	sc.writeIdx = 1
	sc.fcbsSize = 0
	sc.fcbCandidatesSize = 0
	sc.uniqueSgntrSize = 0
}

func (sc *fcbSearchScratch) swapRW() {
	sc.readIdx, sc.writeIdx = sc.writeIdx, sc.readIdx
}

func (sc *fcbSearchScratch) isUnique(sgntr uint64) bool {
	for i := 0; i < sc.uniqueSgntrSize; i++ {
		if sc.uniqueSgntr[i] == sgntr {
			return false
		}
	}
	return true
}

// Greedy FCB search.
func (e *CelpEncoder) smplFcbSearch(
	d []float32,
	wnrgPerPulse *[celpMaxRates]float32,
	fcbPulsesMax *[celpMaxRates]int16,
	pulses *[celpMaxRates][smplMaxPulsesPerSF]int16,
	nPulses *[celpMaxRates]int16,
	wnrg *[celpMaxRates]float32,
	gainFromSearch *[celpMaxRates]float32,
	fcbWnrg *[celpMaxRates]float32,
) {
	fcbSubfrLen := e.fcbSubfrLen
	percRespLen := e.percRespLen
	*nPulses = [celpMaxRates]int16{}

	phi := e.phi
	phiFlip := e.phiFlip
	g := &e.fcbGreedy
	dAbs, dSign, num, den, q := g.dAbs[:], g.dSign[:], g.num[:], g.den[:], g.q[:]

	var positions [smplMaxPulsesPerSF]int32
	phi0 := phi[0]
	calcDAbsAndSign(d, fcbSubfrLen, dAbs, dSign)

	for i := 0; i < fcbSubfrLen; i++ {
		den[i] = phi0 + 1e-16
	}
	copy(num[:fcbSubfrLen], dAbs[:fcbSubfrLen])
	positions[0] = int32(getMaxi(num[:fcbSubfrLen]))
	var nrgThr [celpMaxRates]float32
	p0 := positions[0]
	ratio := num[p0] / den[p0]
	wnrg0 := num[p0] * ratio
	if checkIfBetter(wnrg0, &nrgThr[celpIdxMain], wnrgPerPulse[celpIdxMain]) {
		nPulses[celpIdxMain] = 1
		wnrg[celpIdxMain] = wnrg0
		wnrg[celpIdxFEC] = wnrg0
		gainFromSearch[celpIdxMain] = ratio
		gainFromSearch[celpIdxFEC] = ratio
		fcbWnrg[celpIdxMain] = den[p0]
		fcbWnrg[celpIdxFEC] = den[p0]
		if fcbPulsesMax[celpIdxFEC] > 0 {
			nPulses[celpIdxFEC] = nPulses[celpIdxMain]
			wnrg[celpIdxFEC] = wnrg[celpIdxMain]
			gainFromSearch[celpIdxFEC] = gainFromSearch[celpIdxMain]
			fcbWnrg[celpIdxFEC] = fcbWnrg[celpIdxMain]
		}
	}

	for pulseNr := 1; pulseNr < int(fcbPulsesMax[celpIdxMain]); pulseNr++ {
		position := positions[pulseNr-1]
		sgn := dSign[position]
		for i := 0; i < fcbSubfrLen; i++ {
			num[i] += dAbs[position]
		}
		nz0, nz1 := nonZeroRange(position, percRespLen, fcbSubfrLen)
		colOff := phiColOffset(position)
		var dDen float32
		for i := 0; i < pulseNr-1; i++ {
			pi := positions[i]
			dDen += phiFlip[colOff+pi] * dSign[pi]
		}
		dDen *= 2.0 * sgn
		dDen += phiFlip[colOff+position]
		for i := 0; i < fcbSubfrLen; i++ {
			den[i] += dDen
		}
		for i := nz0; i < nz1; i++ {
			den[i] += 2.0 * sgn * dSign[i] * phiFlip[colOff+int32(i)]
		}
		smplCelpQ(num, den, q, fcbSubfrLen)
		positions[pulseNr] = int32(getMaxi(q[:fcbSubfrLen]))
		pp := positions[pulseNr]
		if checkIfBetter(q[pp], &nrgThr[celpIdxMain], wnrgPerPulse[celpIdxMain]) {
			nPulses[celpIdxMain] = int16(pulseNr + 1)
			wnrg[celpIdxMain] = q[pp]
			gainFromSearch[celpIdxMain] = num[pp] / den[pp]
			fcbWnrg[celpIdxMain] = den[pp]
		}
		if int(fcbPulsesMax[celpIdxFEC]) >= pulseNr &&
			checkIfBetter(q[pp], &nrgThr[celpIdxFEC], wnrgPerPulse[celpIdxFEC]) {
			nPulses[celpIdxFEC] = int16(pulseNr + 1)
			wnrg[celpIdxFEC] = q[pp]
			gainFromSearch[celpIdxFEC] = num[pp] / den[pp]
			fcbWnrg[celpIdxFEC] = den[pp]
		}
	}

	for r := celpIdxFEC; r <= celpIdxMain; r++ {
		if nrgThr[r] > 0.0 {
			for i := 0; i < int(nPulses[r]); i++ {
				position := positions[i]
				p := int16(1 + position)
				if dSign[position] <= 0.0 {
					p = -p
				}
				pulses[r][i] = p
			}
		} else {
			wnrg[r] = 0.0
			gainFromSearch[r] = 0.0
			fcbWnrg[r] = 0.0
			nPulses[r] = 0
		}
	}
}

// addPulse is the deldec beam helper: extend candidate fcbIdxIn (in
// sc.fcbs) with one more pulse and emit up to numsurv new candidates.
func (e *CelpEncoder) addPulse(
	fcbIdxIn int,
	dAbs, dSign []float32,
	numsurv, idx int,
	lag int32,
	pitchSharp float32,
) {
	fcbSubfrLen := e.fcbSubfrLen
	percRespLen := e.percRespLen
	sc := &e.fcbSearch

	// Snapshot the read-side fcb (sc.fcbs[fcbIdxIn] is the mutable candidate).
	fcbPosNew := sc.fcbs[fcbIdxIn].posNew
	fcbSignNew := sc.fcbs[fcbIdxIn].signNew
	fcbNPulses := sc.fcbs[fcbIdxIn].nPulses
	fcbStateIdx := sc.fcbs[fcbIdxIn].fcbStateIdx
	fcbSgntrBase := sc.fcbs[fcbIdxIn].sgntr

	readIdx := sc.readIdx
	writeIdx := sc.writeIdx

	rs := &sc.fcbStates[readIdx][fcbStateIdx]
	ws := &sc.fcbStates[writeIdx][idx]

	// num_w[i] = num_r[i] + dAbs[posNew]
	add := dAbs[fcbPosNew]
	for i := 0; i < fcbSubfrLen; i++ {
		ws.num[i] = rs.num[i] + add
	}
	// den_w = copy(den_r)
	for i := 0; i < fcbSubfrLen; i++ {
		ws.den[i] = rs.den[i]
	}

	if pitchSharp == 0.0 {
		nz0, nz1 := nonZeroRange(fcbPosNew, percRespLen, fcbSubfrLen)
		colOff := phiColOffset(fcbPosNew)
		var dDen float32
		for i := 0; i < int(fcbNPulses); i++ {
			pos := rs.pulsePositions[i]
			sgn := rs.pulseSigns[i]
			dDen += e.phiFlip[colOff+pos] * sgn
		}
		dDen *= 2.0 * fcbSignNew
		dDen += e.phiFlip[colOff+fcbPosNew]
		for i := 0; i < fcbSubfrLen; i++ {
			ws.den[i] += dDen
		}
		for i := nz0; i < nz1; i++ {
			ws.den[i] += 2.0 * fcbSignNew * dSign[i] * e.phiFlip[colOff+int32(i)]
		}
	} else {
		// Pitch-sharpened cross terms.
		var g1 float32
		var dDen float32
		// cross term: new pulse train vs each previous pulse train
		{
			g1 = 1.0
			pos := fcbPosNew
			for pos < int32(fcbSubfrLen) {
				colOff := phiColOffset(pos)
				for i := 0; i < int(fcbNPulses); i++ {
					g2 := g1
					pulsePos := rs.pulsePositions[i]
					pulseSgn := rs.pulseSigns[i]
					pos2 := pulsePos
					for pos2 < int32(fcbSubfrLen) {
						dDen += g2 * e.phiFlip[colOff+pos2] * pulseSgn
						g2 *= pitchSharp
						pos2 += lag
					}
				}
				g1 *= pitchSharp
				pos += lag
			}
		}
		dDen *= 2.0 * fcbSignNew
		// self term: new pulse train vs itself
		{
			g1 = 1.0
			pos1 := fcbPosNew
			for pos1 < int32(fcbSubfrLen) {
				colOff := phiColOffset(pos1)
				g2 := g1
				pos2 := fcbPosNew
				for pos2 < int32(fcbSubfrLen) {
					dDen += g2 * e.phiFlip[colOff+pos2]
					g2 *= pitchSharp
					pos2 += lag
				}
				g1 *= pitchSharp
				pos1 += lag
			}
		}
		for i := 0; i < fcbSubfrLen; i++ {
			ws.den[i] += dDen
		}
		// ddDen (accumulated, so zero the live region first).
		for i := 0; i < fcbSubfrLen; i++ {
			sc.ddDen[i] = 0.0
		}
		g1 = 1.0
		pos := fcbPosNew
		for pos < int32(fcbSubfrLen) {
			nz0, nz1 := nonZeroRange(pos, percRespLen, fcbSubfrLen)
			colOff := phiColOffset(pos)
			g2 := g1
			k := int32(0)
			for k < int32(fcbSubfrLen) {
				startI := int32(nz0) - k
				if startI < 0 {
					startI = 0
				}
				endI := int32(fcbSubfrLen) - k
				if int32(nz1)-k < endI {
					endI = int32(nz1) - k
				}
				for i := startI; i < endI; i++ {
					sc.ddDen[i] += g2 * e.phiFlip[colOff+i+k]
				}
				g2 *= pitchSharp
				k += lag
			}
			g1 *= pitchSharp
			pos += lag
		}
		for i := 0; i < fcbSubfrLen; i++ {
			ws.den[i] += 2.0 * fcbSignNew * dSign[i] * sc.ddDen[i]
		}
	}

	// Append the new pulse to the state.
	for i := 0; i < int(fcbNPulses); i++ {
		ws.pulsePositions[i] = rs.pulsePositions[i]
		ws.pulseSigns[i] = rs.pulseSigns[i]
	}
	ws.pulsePositions[fcbNPulses] = fcbPosNew
	ws.pulseSigns[fcbNPulses] = fcbSignNew

	// fcb->n_pulses++, fcb->fcb_state_idx = idx
	newNPulses := fcbNPulses + 1

	// Q = num^2/den; top-numsurv -> candidates with unique-signature dedup.
	smplCelpQ(ws.num[:], ws.den[:], sc.q, fcbSubfrLen)
	var sortIx [celpMaxNumSurv]int32
	getMaxiK(sc.q[:fcbSubfrLen], sortIx[:], numsurv)

	for i := 0; i < numsurv; i++ {
		pos := sortIx[i]
		sgntr := fcbSgntrBase + e.sgntrs[pos]
		if sc.isUnique(sgntr) {
			sc.fcbCandidates[sc.fcbCandidatesSize] = fcb{
				wnrg:        sc.q[pos],
				nPulses:     newNPulses,
				posNew:      pos,
				signNew:     dSign[pos],
				sgntr:       sgntr,
				fcbStateIdx: idx,
			}
			sc.fcbCandidatesSize++
			sc.uniqueSgntr[sc.uniqueSgntrSize] = sgntr
			sc.uniqueSgntrSize++
		}
	}
}

// Delayed-decision beam FCB search.
func (e *CelpEncoder) smplFcbSearchDeldec(
	d []float32,
	pitchSharp float32,
	lag int32,
	wnrgPerPulse *[celpMaxRates]float32,
	fcbPulsesMax *[celpMaxRates]int16,
	surv []int16,
	pulses *[celpMaxRates][smplMaxPulsesPerSF]int16,
	nPulses *[celpMaxRates]int16,
	wnrg *[celpMaxRates]float32,
	gainFromSearch *[celpMaxRates]float32,
	fcbWnrg *[celpMaxRates]float32,
) {
	fcbSubfrLen := e.fcbSubfrLen
	sc := &e.fcbSearch
	sc.reset()

	// dAbs/dSign are locals (passed to addPulse alongside the scratch);
	// dNew is pooled.
	dAbs := make([]float32, smplMaxSFLen)
	dSign := make([]float32, smplMaxSFLen)
	phi0 := e.phi[0]

	if pitchSharp != 0.0 && lag > 0 && lag < int32(fcbSubfrLen) {
		copy(sc.dNew[:fcbSubfrLen], d[:fcbSubfrLen])
		for j := 0; j < fcbSubfrLen; j++ {
			g := pitchSharp
			i := lag + int32(j)
			for i < int32(fcbSubfrLen) {
				sc.dNew[j] += g * d[i]
				g *= pitchSharp
				i += lag
			}
		}
		calcDAbsAndSign(sc.dNew, fcbSubfrLen, dAbs, dSign)
	} else {
		calcDAbsAndSign(d, fcbSubfrLen, dAbs, dSign)
		pitchSharp = 0.0
	}

	// readIdx/writeIdx already set by sc.reset() above.
	var bestFcb [celpMaxRates]fcb
	var bestFcbState [celpMaxRates]fcbState
	var nrgThr [celpMaxRates]float32

	// Initialize the first state buffer (writeIdx slot 0).
	{
		wi := sc.writeIdx
		copy(sc.fcbStates[wi][0].num[:fcbSubfrLen], dAbs[:fcbSubfrLen])
		if pitchSharp == 0.0 {
			for i := 0; i < fcbSubfrLen; i++ {
				sc.fcbStates[wi][0].den[i] = phi0 + 1e-16
			}
		} else {
			offset := int32(fcbSubfrLen - 1)
			i := int32(fcbSubfrLen - 1)
			for i >= 0 {
				res := float32(1e-16)
				g1 := float32(1.0)
				j := i
				for j < int32(fcbSubfrLen) {
					colOff := phiColOffset(j)
					g2 := float32(1.0)
					k := i
					for k < int32(fcbSubfrLen) {
						res += g1 * g2 * e.phiFlip[colOff+k]
						g2 *= pitchSharp
						k += lag
					}
					g1 *= pitchSharp
					j += lag
				}
				l := lag
				if offset+1 < l {
					l = offset + 1
				}
				for jj := int32(0); jj < l; jj++ {
					sc.fcbStates[wi][0].den[offset-jj] = res
				}
				offset -= l
				i -= lag
			}
		}
	}

	sc.swapRW()
	{
		// fcbState is the slot just written: after the swap, readIdx points
		// back at fcbStates[old writeIdx][0].
		ri := sc.readIdx
		st := &sc.fcbStates[ri][0]
		if pitchSharp == 0.0 {
			copy(sc.q[:fcbSubfrLen], st.num[:fcbSubfrLen])
		} else {
			smplCelpQ(st.num[:], st.den[:], sc.q, fcbSubfrLen)
		}
	}

	var sortIx [celpMaxNumSurv]int32
	getMaxiK(sc.q[:fcbSubfrLen], sortIx[:], int(surv[0]))
	sc.fcbsSize = 0
	{
		ri := sc.readIdx
		for i := 0; i < int(surv[0]); i++ {
			pos := sortIx[i]
			st := &sc.fcbStates[ri][0]
			sc.fcbs[sc.fcbsSize] = fcb{
				sgntr:       e.sgntrs[pos],
				posNew:      pos,
				signNew:     dSign[pos],
				wnrg:        (st.num[pos] * st.num[pos]) / st.den[pos],
				nPulses:     0,
				fcbStateIdx: 0,
			}
			sc.fcbsSize++
		}
	}

	e.checkIfBetterDeldec(sc, 0, &bestFcb[celpIdxMain], &bestFcbState[celpIdxMain], &nrgThr[celpIdxMain], wnrgPerPulse[celpIdxMain])
	if fcbPulsesMax[celpIdxFEC] > 0 {
		e.checkIfBetterDeldec(sc, 0, &bestFcb[celpIdxFEC], &bestFcbState[celpIdxFEC], &nrgThr[celpIdxFEC], wnrgPerPulse[celpIdxFEC])
	}

	if fcbPulsesMax[celpIdxMain] > 1 {
		for pulseNr := 2; pulseNr < int(fcbPulsesMax[celpIdxMain]); pulseNr++ {
			sc.fcbCandidatesSize = 0
			sc.uniqueSgntrSize = 0
			fcbsSize := sc.fcbsSize
			// idx increments lockstep with the loop counter, so idx == i.
			for i := 0; i < fcbsSize; i++ {
				e.addPulse(i, dAbs, dSign, int(surv[pulseNr-1]), i, lag, pitchSharp)
			}
			sc.swapRW()
			// Sort candidates by wnrg.
			candSize := sc.fcbCandidatesSize
			for i := 0; i < candSize; i++ {
				sc.q[i] = sc.fcbCandidates[i].wnrg
			}
			getMaxiK(sc.q[:candSize], sortIx[:], int(surv[pulseNr-1]))
			sc.fcbsSize = 0
			for i := 0; i < int(surv[pulseNr-1]); i++ {
				c := sc.fcbCandidates[sortIx[i]]
				sc.fcbs[sc.fcbsSize] = c
				sc.fcbsSize++
			}
			e.checkIfBetterDeldec(sc, 0, &bestFcb[celpIdxMain], &bestFcbState[celpIdxMain], &nrgThr[celpIdxMain], wnrgPerPulse[celpIdxMain])
			if int(fcbPulsesMax[celpIdxFEC]) >= pulseNr {
				e.checkIfBetterDeldec(sc, 0, &bestFcb[celpIdxFEC], &bestFcbState[celpIdxFEC], &nrgThr[celpIdxFEC], wnrgPerPulse[celpIdxFEC])
			}
		}
		// Last pulse (surv=1, MAIN only).
		sc.fcbCandidatesSize = 0
		sc.uniqueSgntrSize = 0
		fcbsSize := sc.fcbsSize
		for i := 0; i < fcbsSize; i++ {
			e.addPulse(i, dAbs, dSign, 1, i, lag, pitchSharp)
		}
		sc.swapRW()
		bestIdx := 0
		maxWnrg := sc.fcbCandidates[0].wnrg
		for i := 1; i < sc.fcbCandidatesSize; i++ {
			if sc.fcbCandidates[i].wnrg > maxWnrg {
				maxWnrg = sc.fcbCandidates[i].wnrg
				bestIdx = i
			}
		}
		// check_if_better_deldec on a candidate (not in fcbs); its
		// fcb_state_idx points into the current read buffer.
		e.checkIfBetterDeldecCand(sc, bestIdx, &bestFcb[celpIdxMain], &bestFcbState[celpIdxMain], &nrgThr[celpIdxMain], wnrgPerPulse[celpIdxMain])
	}

	for r := celpIdxFEC; r <= celpIdxMain; r++ {
		for i := 0; i < int(bestFcb[r].nPulses); i++ {
			p := int16(1 + bestFcbState[r].pulsePositions[i])
			if bestFcbState[r].pulseSigns[i] <= 0.0 {
				p = -p
			}
			pulses[r][i] = p
		}
		p := int16(1 + bestFcb[r].posNew)
		if bestFcb[r].signNew <= 0.0 {
			p = -p
		}
		pulses[r][bestFcb[r].nPulses] = p

		if bestFcb[r].wnrg > 0.0 {
			wnrg[r] = bestFcb[r].wnrg
			pn := bestFcb[r].posNew
			gainFromSearch[r] = bestFcbState[r].num[pn] / bestFcbState[r].den[pn]
			fcbWnrg[r] = bestFcbState[r].den[pn]
			nPulses[r] = int16(bestFcb[r].nPulses) + 1
		} else {
			wnrg[r] = 0.0
			gainFromSearch[r] = 0.0
			fcbWnrg[r] = 0.0
			nPulses[r] = 0
		}
	}
}

// checkIfBetterDeldec is the check_if_better variant where the fcb is
// sc.fcbs[fcbsIdx]; copies the read-side state.
func (e *CelpEncoder) checkIfBetterDeldec(
	sc *fcbSearchScratch,
	fcbsIdx int,
	bestFcb *fcb,
	bestFcbState *fcbState,
	nrgThr *float32,
	wnrgPerPulse float32,
) {
	*nrgThr += wnrgPerPulse
	f := &sc.fcbs[fcbsIdx]
	if f.wnrg > *nrgThr {
		*nrgThr = f.wnrg
		*bestFcb = *f
		*bestFcbState = sc.fcbStates[sc.readIdx][f.fcbStateIdx]
	}
}

// checkIfBetterDeldecCand is the variant where the fcb is a candidate in
// sc.fcbCandidates.
func (e *CelpEncoder) checkIfBetterDeldecCand(
	sc *fcbSearchScratch,
	candIdx int,
	bestFcb *fcb,
	bestFcbState *fcbState,
	nrgThr *float32,
	wnrgPerPulse float32,
) {
	*nrgThr += wnrgPerPulse
	f := &sc.fcbCandidates[candIdx]
	if f.wnrg > *nrgThr {
		*nrgThr = f.wnrg
		*bestFcb = *f
		*bestFcbState = sc.fcbStates[sc.readIdx][f.fcbStateIdx]
	}
}

// Tier 5: gain quantization.

func smplWnrg2(c, x []float32) float32 {
	return x[0]*(c[0]*x[0]+c[1]*x[1]) + x[1]*(c[2]*x[0]+c[3]*x[1])
}

func smplWnrg3(c, x []float32) float32 {
	return x[0]*(c[0]*x[0]+c[1]*x[1]+c[2]*x[2]) +
		x[1]*(c[3]*x[0]+c[4]*x[1]+c[5]*x[2]) +
		x[2]*(c[6]*x[0]+c[7]*x[1]+c[8]*x[2])
}

func quantGainUV(gainFromSearch float32) int16 {
	gainDB := 20.0 * f32Log10(gainFromSearch+1.0e-16)
	gainDB = maxF32(gainDB, celpUVGainMinDB)
	gainDB = minF32(gainDB, smplUVGainMaxDB)
	return int16(f32Round((gainDB - celpUVGainMinDB) / celpUVGainStepDB))
}

func fcbSynthesize(fcbSubfrLen int, pulses []int16, nPulses int, fcb []float32) {
	for i := 0; i < fcbSubfrLen; i++ {
		fcb[i] = 0.0
	}
	for n := 0; n < nPulses; n++ {
		// sign = 1 + 2*(p>>15); pos = p*sign - 1 (arithmetic shift on the
		// signed pulse).
		sign := 1 + 2*(int32(pulses[n])>>15)
		pos := int32(pulses[n])*sign - 1
		fcb[pos] += float32(sign)
	}
}

// calcAcbGain returns the best acbg idx; fills acbg and dLtp.
func (e *CelpEncoder) calcAcbGain(
	lResp int,
	acbBasis []float32,
	dLPC []float32,
	acbg *acbgParams,
	dLtp []float32,
) int32 {
	tbl := celpTables()
	fcbSubfrLen := e.fcbSubfrLen

	for m := 0; m < acbgM; m++ {
		// acbBasisPhi[m] = symtoepl2(PhiFlip + MAX_SF_LEN - L_resp + 1, ...)
		cOff := smplMaxSFLen - lResp + 1
		tmp := make([]float32, fcbSubfrLen)
		smplMultSymtoepl2(e.phiFlip[cOff:], lResp, acbBasis[m*fcbSubfrLen:], tmp, fcbSubfrLen)
		copy(acbg.acbBasisPhi[m*fcbSubfrLen:m*fcbSubfrLen+fcbSubfrLen], tmp)
		for i := 0; i < acbgM; i++ {
			acbg.phiAcb[m*acbgM+i] = smplDotProd(acbBasis[i*fcbSubfrLen:], acbg.acbBasisPhi[m*fcbSubfrLen:], fcbSubfrLen)
		}
		acbg.dAcbLPC[m] = smplDotProd(acbBasis[m*fcbSubfrLen:], dLPC, fcbSubfrLen)
	}

	bestRD := float32(1e30)
	var bestAcbgIdx int32
	var transitionIdx int32
	if e.prevAcbIdx[celpIdxMain] == -1 {
		transitionIdx = 0
	} else {
		transitionIdx = e.prevAcbIdx[celpIdxMain] + 1
	}
	var acbgInvProbFull []float32
	if e.lowRate {
		acbgInvProbFull = tbl.acbgInvProbLR[:]
	} else {
		acbgInvProbFull = tbl.acbgInvProbHR[:]
	}
	acbgInvProb := acbgInvProbFull[int(transitionIdx)*acbgN:]
	var cb []int16
	if e.lowRate {
		cb = cbAcbgainsLRQ14[:]
	} else {
		cb = cbAcbgainsHRQ14[:]
	}
	scQ14 := 1.0 / float32(int32(1)<<14)
	var acbGains [acbgM]float32
	for n := 0; n < acbgN; n++ {
		for m := 0; m < acbgM; m++ {
			acbGains[m] = float32(cb[n*acbgM+m]) * scQ14
		}
		werrOut := acbg.werrIn + smplWnrg2(acbg.phiAcb[:], acbGains[:]) -
			2.0*(acbg.dAcbLPC[0]*acbGains[0]+acbg.dAcbLPC[1]*acbGains[1])
		rd := werrOut * acbgInvProb[n]
		if rd < bestRD {
			bestRD = rd
			bestAcbgIdx = int32(n)
		}
	}

	// fcb target signal
	g0 := -float32(cb[int(bestAcbgIdx)*acbgM]) * scQ14
	smplAddScaleVec(dLPC, acbg.acbBasisPhi, dLtp, fcbSubfrLen, g0)
	g1 := -float32(cb[int(bestAcbgIdx)*acbgM+1]) * scQ14
	smplAddScaleVecInplace(acbg.acbBasisPhi[fcbSubfrLen:], dLtp, fcbSubfrLen, g1)

	return bestAcbgIdx
}

// calcGainsV (g_acb_rd_mu > 0 always -> RD branch). Returns the chosen voiced
// fcb gain.
func (e *CelpEncoder) calcGainsV(
	fcbWnrg float32,
	gainFromSearch float32,
	excFcb []float32,
	dLPC []float32,
	acbg *acbgParams,
	rateIdx int,
	acbIdx *[celpMaxRates]int16,
	fcbIdx *[celpMaxRates]int16,
) float32 {
	tbl := celpTables()
	fg := fcbGainsOnce()
	fcbSubfrLen := e.fcbSubfrLen

	fcbgain := maxF32(gainFromSearch, 0.0)
	gainDB := 20.0 * f32Log10(fcbgain+1.0e-16)
	gainDB = maxF32(gainDB, celpVGainMinDB)
	gainDB = minF32(gainDB, smplVGainMaxDB)
	maxGainIdx := int32(f32Round((smplVGainMaxDB - celpVGainMinDB) / celpVGainStepDB))

	// g_acb_rd_mu > 0 -> always RD branch
	var bestAcbgIdx int32
	var bestFcbgIdx int32

	var acbFcb [acbgM]float32
	for i := 0; i < acbgM; i++ {
		acbFcb[i] = smplDotProd(acbg.acbBasisPhi[i*fcbSubfrLen:], excFcb, fcbSubfrLen)
	}
	// phiAll is (M+1)x(M+1) row-major.
	var phiAll [(acbgM + 1) * (acbgM + 1)]float32
	stride := acbgM + 1
	for i := 0; i < acbgM; i++ {
		for j := 0; j < acbgM; j++ {
			phiAll[i*stride+j] = acbg.phiAcb[i*acbgM+j]
		}
	}
	for i := 0; i < acbgM; i++ {
		phiAll[i*stride+acbgM] = acbFcb[i]
		phiAll[acbgM*stride+i] = acbFcb[i]
	}
	phiAll[acbgM*stride+acbgM] = fcbWnrg

	var dall [acbgM + 1]float32
	dall[0] = acbg.dAcbLPC[0]
	dall[1] = acbg.dAcbLPC[1]
	dall[acbgM] = smplDotProd(dLPC, excFcb, fcbSubfrLen)

	var gainIdxs [nGainSteps]int32
	var fcbgains [nGainSteps]float32
	var fcbgInvProb [nGainSteps]float32
	firstGainIdx := int32(f32RoundDown((gainDB-celpVGainMinDB)/celpVGainStepDB)) - (nGainSteps-1)/2
	if firstGainIdx < 0 {
		firstGainIdx = 0
	}
	if firstGainIdx > maxGainIdx-1 {
		firstGainIdx = maxGainIdx - 1
	}
	offset := int32(f32RoundDown((celpVGainMinDB - smplVGainMaxDB) / celpVGainStepDB))
	for i := 0; i < nGainSteps; i++ {
		gainIdxs[i] = firstGainIdx + int32(i)
		fcbgains[i] = fg.v[gainIdxs[i]]
		if e.prevFcbIdx[rateIdx] == -1 {
			fcbgInvProb[i] = tbl.fcbgVInvProb[gainIdxs[i]]
		} else {
			delta := e.prevFcbIdx[rateIdx] - gainIdxs[i]
			cmfIdx := delta - offset
			fcbgInvProb[i] = tbl.fcbgVDeltaInvProb[cmfIdx]
		}
	}

	bestRD := float32(1e30)
	var transitionIdx int32
	if e.prevAcbIdx[rateIdx] == -1 {
		transitionIdx = 0
	} else {
		transitionIdx = e.prevAcbIdx[rateIdx] + 1
	}
	var cb []int16
	if e.lowRate {
		cb = cbAcbgainsLRQ14[:]
	} else {
		cb = cbAcbgainsHRQ14[:]
	}
	var acbgInvProbFull []float32
	if e.lowRate {
		acbgInvProbFull = tbl.acbgInvProbLR[:]
	} else {
		acbgInvProbFull = tbl.acbgInvProbHR[:]
	}
	acbgInvProb := acbgInvProbFull[int(transitionIdx)*acbgN:]
	scQ14 := 1.0 / float32(int32(1)<<14)
	for n := 0; n < acbgN; n++ {
		var gains [acbgM + 1]float32
		for m := 0; m < acbgM; m++ {
			gains[m] = float32(cb[n*acbgM+m]) * scQ14
		}
		for i := 0; i < nGainSteps; i++ {
			gains[acbgM] = fcbgains[i]
			werrOut := acbg.werrIn + smplWnrg3(phiAll[:], gains[:]) -
				2.0*(dall[0]*gains[0]+dall[1]*gains[1]+dall[2]*gains[2])
			rd := werrOut * fcbgInvProb[i] * acbgInvProb[n]
			if rd < bestRD {
				bestRD = rd
				bestAcbgIdx = int32(n)
				bestFcbgIdx = gainIdxs[i]
			}
		}
	}
	acbIdx[rateIdx] = int16(bestAcbgIdx)
	fcbIdx[rateIdx] = int16(bestFcbgIdx)

	if fcbIdx[rateIdx] < 0 {
		fcbIdx[rateIdx] = 0
	}
	if fcbIdx[rateIdx] > int16(maxGainIdx) {
		fcbIdx[rateIdx] = int16(maxGainIdx)
	}
	return fg.v[fcbIdx[rateIdx]]
}

// Tier 6: main per-subframe encoder.

func (e *CelpEncoder) EncodeSubframe(
	resLPC []float32,
	predcoef *[17]float32,
	percWghtResp []float32,
	lags []float32,
	subfrImportance [celpMaxRates]float32,
	fcbPulsesMax [celpMaxRates]int16,
	surv []int16,
) CelpSubframeOut {
	lResp := e.percRespLen
	fcbSubfrLen := e.fcbSubfrLen
	voiced := lags[1] > 0.0

	// impLPC = filt_ar16(perc_wght_resp, L_resp, predcoef) * hanning_win.
	// The 16-sample AR lead (impLPCBuf[0..smplLpcOrder]) is zeroed once at
	// construction and never written by the filter (it only writes indices
	// >= smplLpcOrder), so it stays zero across calls.
	smplFiltAR16(percWghtResp, lResp, predcoef[:], smplLpcOrder, e.impLPCBuf)
	{
		imp := e.impLPCBuf[smplLpcOrder:]
		smplMulVecInplace(e.hanningWin, imp, lResp)
	}

	// impLPCRev: buffer of 2*MAX_L_RESP-1, logical ptr at MAX_L_RESP-1.
	impLPCRev := zeroed(&e.sf.impLPCRev, 2*smplMaxLResp-1)
	revBase := smplMaxLResp - 1
	{
		imp := e.impLPCBuf[smplLpcOrder:]
		for i := 0; i < lResp; i++ {
			impLPCRev[revBase+i] = imp[lResp-i-1]
		}
	}
	// Zero the (L_resp-1) lead before revBase; already zero, kept explicit.
	for i := 0; i < lResp-1; i++ {
		impLPCRev[revBase-(lResp-1)+i] = 0.0
	}
	// Phi = perc_filt_ma(imp_lpc_rev, L_resp, imp_lpc, L_resp)
	{
		phi := make([]float32, smplMaxSFLen)
		e.percFiltMA(impLPCRev, revBase, lResp, e.impLPCBuf[smplLpcOrder:smplLpcOrder+lResp], lResp, phi)
		smplReverse(phi, lResp)
		for i := lResp; i < fcbSubfrLen; i++ {
			phi[i] = 0.0
		}
		copy(e.phi, phi)
	}
	// PhiFlip
	for i := range e.phiFlip {
		e.phiFlip[i] = 0.0
	}
	e.phiFlip[smplMaxSFLen] = e.phi[0]
	for i := 0; i <= lResp; i++ {
		e.phiFlip[smplMaxSFLen-i] = e.phi[i]
		e.phiFlip[smplMaxSFLen+i] = e.phi[i]
	}

	// dLPC = symtoepl2(PhiFlip + MAX_SF_LEN - L_resp + 1, L_resp, res_lpc,
	// fcb_subfrlen). resLPC must be readable up to fcbSubfrLen + L_resp (zero
	// padded); build a padded copy.
	resLPCPad := zeroed(&e.sf.resLPCPad, fcbSubfrLen+lResp+1)
	copy(resLPCPad[:fcbSubfrLen], resLPC[:fcbSubfrLen])
	dLPC := zeroed(&e.sf.dLPC, smplMaxSFLen)
	{
		cOff := smplMaxSFLen - lResp + 1
		smplMultSymtoepl2(e.phiFlip[cOff:], lResp, resLPCPad, dLPC, fcbSubfrLen)
	}

	acbg := acbgParams{
		acbBasisPhi: make([]float32, acbgM*fcbSubfrLen),
	}
	zirLPC := zeroed(&e.sf.zirLPC, smplMaxSFLen)

	if !e.ignoreZir {
		// zirTmp buffer: MAX_SF_LEN + MAX_L_RESP - 1, logical ptr at
		// MAX_L_RESP-1
		zirTmp := make([]float32, smplMaxSFLen+smplMaxLResp-1)
		zt := smplMaxLResp - 1
		// htZir buffer: 2*MAX_L_RESP-1, ptr at MAX_L_RESP-1
		htZir := make([]float32, 2*smplMaxLResp-1)
		ht := smplMaxLResp - 1

		stateLen := maxInt(smplLpcOrder, lResp-1)
		// Copy state_wght[fcbSubfrLen-stateLen .. fcbSubfrLen] into the
		// stateLen samples before zirTmp[zt] (logical start at
		// stateWghtBuf[smplLpcOrder]).
		for i := 0; i < stateLen; i++ {
			zirTmp[zt-stateLen+i] = e.stateWghtBuf[smplLpcOrder+(fcbSubfrLen-stateLen)+i]
		}
		// In-place AR over zirTmp at base zt (x==y). The input term x[n] for
		// n in 0..L_resp is the zeroed region, read before being overwritten,
		// while y[n-16..n-1] supply the just-copied history; writing y[n]
		// after reading x[n] keeps the aliasing safe.
		for nn := 0; nn < lResp; nn++ {
			res := zirTmp[zt+nn]
			for i := 0; i < 16; i++ {
				res -= predcoef[16-i] * zirTmp[zt+nn-16+i]
			}
			zirTmp[zt+nn] = res
		}
		// zirLPC = perc_filt_ma(zir_lpc_tmp, L_resp, perc_wght_resp, L_resp)
		e.percFiltMA(zirTmp, zt, lResp, percWghtResp, lResp, zirLPC)

		for i := 0; i < lResp; i++ {
			zirTmp[zt+i] = zirLPC[lResp-i-1]
		}
		for i := 0; i < lResp-1; i++ {
			zirTmp[zt-(lResp-1)+i] = 0.0
		}
		// htZir = perc_filt_ma(zir_lpc_tmp, L_resp, imp_lpc, L_resp)
		e.percFiltMA(zirTmp, zt, lResp, e.impLPCBuf[smplLpcOrder:smplLpcOrder+lResp], lResp, htZir[ht:])
		smplReverse(htZir[ht:], lResp)

		if voiced {
			acbg.werrIn = smplDotProd(dLPC, resLPC, fcbSubfrLen) +
				2.0*smplDotProd(htZir[ht:], resLPC, lResp) +
				smplNrg(zirLPC[:lResp])
		} else {
			acbg.werrIn = 0.0
		}
		// dLPC[0..L_resp] += htZir
		for i := 0; i < lResp; i++ {
			dLPC[i] += htZir[ht+i]
		}
	} else {
		for i := 0; i < lResp; i++ {
			zirLPC[i] = 0.0
		}
		if voiced {
			acbg.werrIn = smplDotProd(dLPC, resLPC, fcbSubfrLen)
		} else {
			acbg.werrIn = 0.0
		}
	}

	// ACB
	acbBasis := zeroed(&e.sf.acbBasis, smplMaxSFLen*acbgM)
	acb := zeroed(&e.sf.acb, smplMaxSFLen)
	dLtp := zeroed(&e.sf.dLtp, smplMaxSFLen)
	acbIdx := [celpMaxRates]int16{-1, -1}

	if voiced {
		synLTPBasis(lags, fcbSubfrLen/celpLagSubfrLen, e.acbState, e.acbStateLen, acbBasis)
		idx := e.calcAcbGain(lResp, acbBasis, dLPC, &acbg, dLtp)
		acbIdx[celpIdxMain] = int16(idx)
		var acbGain [acbgM]float32
		acbDequant(e.lowRate, int32(acbIdx[celpIdxMain]), &acbGain)
		acbSynthesize(fcbSubfrLen, acbBasis, &acbGain, 0, acb)
		acbIdx[celpIdxFEC] = acbIdx[celpIdxMain]
	}

	// Weighted target + thresholds.
	// wtgtTmp buffer: MAX_SF_LEN + 2*MAX_L_RESP - 1, ptr at MAX_L_RESP-1
	wtgtTmp := zeroed(&e.sf.wtgtTmp, smplMaxSFLen+2*smplMaxLResp-1)
	wt := smplMaxLResp - 1
	wtgt := zeroed(&e.sf.wtgt, smplMaxSFLen+smplMaxLResp)
	copy(wtgtTmp[wt:wt+fcbSubfrLen], resLPC[:fcbSubfrLen])
	if voiced {
		// wtgtTmp += -RATE_ACB_SCALE * acb
		for i := 0; i < fcbSubfrLen; i++ {
			wtgtTmp[wt+i] += -smplRateAcbScale * acb[i]
		}
	}
	for i := 0; i < lResp; i++ {
		wtgtTmp[wt+fcbSubfrLen+i] = 0.0
	}
	for i := 0; i < lResp-1; i++ {
		wtgtTmp[wt-(lResp-1)+i] = 0.0
	}
	e.percFiltMA(wtgtTmp, wt, fcbSubfrLen+lResp, e.impLPCBuf[smplLpcOrder:smplLpcOrder+lResp], lResp, wtgt)
	for i := 0; i < lResp; i++ {
		wtgt[i] += zirLPC[i]
	}
	nrgWtgt := smplNrg(wtgt[:fcbSubfrLen+lResp])
	var wnrgPerPulse [celpMaxRates]float32
	for r := 0; r < celpMaxRates; r++ {
		wnrgPerPulse[r] = nrgWtgt / (subfrImportance[r] + 1.0e-3)
	}
	iLag := int32(lags[fcbSubfrLen/celpLagSubfrLen-1])

	var nPulses [celpMaxRates]int16
	var gainFromSearch [celpMaxRates]float32
	var fcbWnrg [celpMaxRates]float32
	var wnrg [celpMaxRates]float32
	var pulses [celpMaxRates][smplMaxPulsesPerSF]int16

	if fcbPulsesMax[celpIdxMain] > 0 {
		target := dLPC
		if voiced {
			target = dLtp
		}
		useGreedy := fcbPulsesMax[celpIdxMain]-1 > 0 &&
			surv[fcbPulsesMax[celpIdxMain]-2] == 1 &&
			!e.lowRate
		if useGreedy {
			e.smplFcbSearch(target, &wnrgPerPulse, &fcbPulsesMax, &pulses, &nPulses, &wnrg, &gainFromSearch, &fcbWnrg)
		} else {
			var psScale float32
			if e.lowRate {
				psScale = 1.0
			}
			ps := celpPitchSharpening * psScale
			e.smplFcbSearchDeldec(target, ps, iLag, &wnrgPerPulse, &fcbPulsesMax, surv, &pulses, &nPulses, &wnrg, &gainFromSearch, &fcbWnrg)
		}
	}

	// Per-rate gain quant + excFcb. fcbgain carries the last assigned (MAIN)
	// value.
	gainIdx := [celpMaxRates]int16{-1, -1}
	var fcbgain float32
	excFcb := zeroed(&e.sf.excFcb, smplMaxSFLen)
	fg := fcbGainsOnce()
	for r := 0; r < celpMaxRates; r++ {
		excFcbRaw := make([]float32, smplMaxSFLen)
		fcbSynthesize(fcbSubfrLen, pulses[r][:], int(nPulses[r]), excFcbRaw)
		copy(excFcb[:fcbSubfrLen], excFcbRaw[:fcbSubfrLen])
		if nPulses[r] > 0 {
			if voiced {
				if e.lowRate {
					pitchSharpen(excFcb, int(iLag), fcbSubfrLen)
				}
				fcbgain = e.calcGainsV(fcbWnrg[r], gainFromSearch[r], excFcb, dLPC, &acbg, r, &acbIdx, &gainIdx)
			} else {
				gainIdx[r] = quantGainUV(gainFromSearch[r])
				fcbgain = fg.uv[gainIdx[r]]
			}
			smplScaleVecInplace(excFcb, fcbSubfrLen, fcbgain)
		}
	}

	// excLPC = excFcb (last r = MAIN); for voiced add the ACB contribution.
	// The reference's trailing smpl_sub_vec_inplace(acb, res_ltp) mutates only
	// the throwaway plotting buffer res_ltp, NOT exc_lpc, so the excitation
	// fed back into the ACB state is exactly exc_fcb + acb.
	excLPC := make([]float32, fcbSubfrLen)
	copy(excLPC, excFcb[:fcbSubfrLen])
	if voiced {
		var acbGain [acbgM]float32
		acbDequant(e.lowRate, int32(acbIdx[celpIdxMain]), &acbGain)
		acbSynthesize(fcbSubfrLen, acbBasis, &acbGain, 0, acb)
		smplAddVecInplace(acb, excLPC, fcbSubfrLen)
	}

	// Update adaptive codebook state: shift down by fcbSubfrLen (keeping
	// acbStateLen - 2*fcbSubfrLen elements) and write excLPC at
	// acbStateLen - 2*fcbSubfrLen, leaving the trailing fcbSubfrLen untouched
	// (an in-place shift, not an append-at-end).
	copy(e.acbState, e.acbState[fcbSubfrLen:e.acbStateLen-fcbSubfrLen])
	writeOff := e.acbStateLen - 2*fcbSubfrLen
	copy(e.acbState[writeOff:writeOff+fcbSubfrLen], excLPC[:fcbSubfrLen])

	// Update ZIR state.
	if !e.ignoreZir {
		lpcResErr := make([]float32, smplMaxSFLen)
		smplSubVec(resLPC, excLPC, lpcResErr, fcbSubfrLen)
		// stateWght[-16..0] = stateErrLPCSyn (stateWghtBuf[0..16])
		for i := 0; i < smplLpcOrder; i++ {
			e.stateWghtBuf[i] = e.stateErrLPCSyn[i]
		}
		smplFiltAR16(lpcResErr, fcbSubfrLen, predcoef[:], smplLpcOrder, e.stateWghtBuf)
		// stateErrLPCSyn = stateWght[fcbSubfrLen-16 .. fcbSubfrLen]
		for i := 0; i < smplLpcOrder; i++ {
			e.stateErrLPCSyn[i] = e.stateWghtBuf[smplLpcOrder+(fcbSubfrLen-smplLpcOrder)+i]
		}
	}

	// Packet-boundary / prev-idx state.
	e.subfrCnt++
	if e.subfrCnt == e.subfrPerPacket {
		for r := 0; r < celpMaxRates; r++ {
			e.prevAcbIdx[r] = -1
			e.prevFcbIdx[r] = -1
		}
		e.subfrCnt = 0
	} else {
		for r := 0; r < celpMaxRates; r++ {
			if voiced {
				e.prevAcbIdx[r] = int32(acbIdx[r])
				e.prevFcbIdx[r] = int32(gainIdx[r])
			} else {
				e.prevAcbIdx[r] = -1
				e.prevFcbIdx[r] = -1
			}
		}
	}
	e.fcbgain = fcbgain

	// Materialize the per-rate pulse vectors trimmed to nPulses.
	npFec := nPulses[celpIdxFEC]
	if npFec < 0 {
		npFec = 0
	}
	npMain := nPulses[celpIdxMain]
	if npMain < 0 {
		npMain = 0
	}
	pulsesFEC := append([]int16(nil), pulses[celpIdxFEC][:npFec]...)
	pulsesMain := append([]int16(nil), pulses[celpIdxMain][:npMain]...)

	return CelpSubframeOut{
		Pulses:  [celpMaxRates][]int16{pulsesFEC, pulsesMain},
		NPulses: nPulses,
		AcbIdx:  acbIdx,
		GainIdx: gainIdx,
		ExcLPC:  excLPC,
	}
}

// Tier 7: survivor distribution.

func SmplDistributeFcbSurv(numsurv []int16, maxPulses, totSurv int32) {
	if maxPulses <= 1 {
		numsurv[0] = 1
		return
	}
	for i := 0; i < int(maxPulses); i++ {
		numsurv[i] = 1
	}
	sumSurv := maxPulses
	extraSurv := totSurv - maxPulses
	extra := extraSurv / (maxPulses - 1)
	if extra > smplFcbSrvMax-1 {
		extra = smplFcbSrvMax - 1
	}
	for i := 0; i < int(maxPulses-1); i++ {
		numsurv[i] += int16(extra)
	}
	sumSurv += extra * (maxPulses - 1)
	ix := maxPulses - 2
	for sumSurv < totSurv {
		if int32(numsurv[ix]) < smplFcbSrvMax {
			numsurv[ix]++
			sumSurv++
		}
		ix--
		if ix < 0 {
			break
		}
	}
}

// ---------------------------------------------------------------------------
// smpl_nrgres: residual-energy quantizer (num_subfr == 4). The unvoiced
// excitation LEVEL is carried entirely by the per-subframe quantized
// residual-energy floor (nrgres_dbq_Q14): a frame-mean scalar quant plus a
// shape VQ. The decoder reads this floor back as gain_q (validated bit-exact
// in param_decode_match), so the encoder must produce the same
// nrgres_dbq_Q14 for the round-trip level to be right. The wire gain block IS
// the nrgres layout (gain_main==nrgres_frame_qi, gain_delta==nrgres_shape_qi,
// the gain table == the shape codebook, cb1 == the frame step), so
// gain_q[sf] decodes back as nrgres_dbq_Q14.
// ---------------------------------------------------------------------------

const (
	smplResNrgMinDB float32 = -85.0
	smplResNrgMaxDB float32 = 0.0
	// smpl_nrg_step_db_Q14[2] (the 4-subframe table index).
	smplNrgStepDBQ144 int32 = 16686
)

// nrgresShapeCB4Q10 is nrgres_shape_CB_4_Q10 (98 vectors x 4 subframes),
// stored verbatim. nrgresShape4N (98) is shared with cc_tables.go.
var nrgresShapeCB4Q10 = [nrgresShape4N * 4]int16{
	-2515, -2238, 2632, 2121, 790, 3973, -2872, -1891, -533, 2847, 1453, -3767, -6174, -402, 2668, 3908,
	-1623, -1458, 153, 2928, -1254, 3197, -476, -1467, 1803, -1086, 270, -987, 1952, -66, -1257, -629,
	161, 19, -85, -96, 4833, 3147, -105, -7875, -1320, 1377, -1156, 1099, 3398, -2247, 1485, -2637,
	-3031, 2756, 1841, -1566, -1487, 2202, -2668, 1954, 5518, -5344, 522, -696, 8400, -3123, -6235, 958,
	5152, -2444, -2811, 102, 2513, -82, 1181, -3612, -561, -197, -1074, 1832, -294, -1250, -1839, 3383,
	5126, 522, -782, -4866, -7760, -5178, -1840, 14779, -1119, 6007, -1489, -3399, -4567, -2543, 1855, 5255,
	53, -1626, 67, 1506, -12256, -7706, -1982, 21943, 3549, -969, -1096, -1484, -10824, 2981, 2204, 5639,
	-229, 1106, 945, -1821, -9237, 10157, 1616, -2537, 4916, -199, -2177, -2540, 6673, 984, -3355, -4302,
	-7130, -4677, 8925, 2882, 445, 2762, -348, -2859, -196, -1859, 1761, 294, 2725, -2093, -966, 334,
	-3908, -308, 3675, 541, 735, 890, -2516, 891, 504, 1631, -1157, -977, -17817, 2119, 7104, 8594,
	-2056, 1897, -198, 356, 292, -4544, -287, 4538, -1455, -304, 603, 1156, -18259, -12643, 15247, 15655,
	4177, 1778, -1815, -4140, 1425, 576, -294, -1707, -1301, 5132, 2838, -6669, -4727, -3148, -905, 8781,
	-650, 152, -4654, 5152, 13746, 2320, -6259, -9807, -1356, 396, 3789, -2829, 2337, 1947, -29, -4256,
	6033, 820, -5730, -1123, -1795, 1091, 1080, -377, 2208, -1921, -3314, 3027, 9688, 5218, -3754, -11152,
	3814, -3941, -6183, 6310, -1017, -2391, 4393, -984, 10944, -1182, -5011, -4751, -4640, 7201, -218, -2343,
	-1278, 4720, -4212, 770, 2777, 1333, -5944, 1833, -16066, 8107, 5165, 2795, 2530, -5020, 6073, -3582,
	-2111, -7534, 4575, 5070, -8702, -3762, 4050, 8414, 1335, -997, -1567, 1229, 9348, 1534, -3959, -6922,
	2440, 1153, -2175, -1418, -2715, -4538, -4478, 11730, 569, -885, 2032, -1716, 3529, -91, -3218, -219,
	2157, -4121, 191, 1772, -2123, -1968, -1355, 5446, 1475, -354, 3651, -4772, 1654, -3521, 2726, -859,
	2393, 6820, -2958, -6255, -3861, 1365, 1177, 1319, 7614, -1638, -2789, -3187, -3628, -2635, 6902, -639,
	1925, 2295, -1451, -2769, -3683, 4517, -981, 147, -1260, -529, 2339, -550, 3013, 639, -1050, -2602,
	3651, 1959, -3218, -2391, 6267, 3124, -2926, -6464, -8180, 3900, 4191, 89, -3372, -611, 1042, 2941,
	-2510, 856, -925, 2579, -11667, -8436, 10605, 9498, 6427, -2733, 1887, -5581, 1581, -1722, -328, 469,
	2011, 1989, -3606, -394, -1014, 2197, -1200, 17, 1544, -2555, 765, 247, 1188, -183, 1966, -2972,
	-6057, 3480, -2284, 4860, -25659, 8466, 8891, 8303,
}

// NrgResQuant is the result of QuantNrgRes4 for the 4-subframe path.
type NrgResQuant struct {
	FrameQi int32
	ShapeQi int32
	// Per-subframe quantized residual-energy floor (Q14); the decoder reads
	// this as gain_q.
	DbqQ14 [4]int32
}

// QuantNrgRes4 is the residual-energy quantizer for num_subfr == 4. nrgres[sf]
// is the per-subframe residual energy nrg(reslpc_sf)/subfrlen in the SAME
// (int16-scaled) domain reslpc lives in.
func QuantNrgRes4(nrgres *[4]float32) NrgResQuant {
	var nrgresDB [4]float32
	var frameDB float32
	for i := 0; i < 4; i++ {
		nrgresDB[i] = minF32(10.0*f32Log10(nrgres[i]+smplResNrgBias), smplResNrgMaxDB)
		frameDB += nrgresDB[i]
	}
	frameDB /= 4.0
	scQ14 := 1.0 / float32(int32(1)<<14)
	frameQi := int32(f32Round((frameDB - smplResNrgMinDB) / (scQ14 * float32(smplNrgStepDBQ144))))
	frameDBqQ14 := frameQi * smplNrgStepDBQ144
	frameDBqQ14 += int32(smplResNrgMinDB) * (1 << 14)
	for i := 0; i < 4; i++ {
		nrgresDB[i] -= float32(frameDBqQ14) * scQ14
	}
	// Shape VQ (min RD) over the 98 codebook vectors.
	scQ10 := 1.0 / float32(int32(1)<<10)
	bestRD := float32(1e30)
	qi := 0
	for n := 0; n < nrgresShape4N; n++ {
		var rd float32
		for i := 0; i < 4; i++ {
			d := nrgresDB[i] - float32(nrgresShapeCB4Q10[n*4+i])*scQ10
			rd += d * d
		}
		if rd < bestRD {
			qi = n
			bestRD = rd
		}
	}
	var dbqQ14 [4]int32
	for i := 0; i < 4; i++ {
		dbqQ14[i] = frameDBqQ14 + int32(nrgresShapeCB4Q10[qi*4+i])*16
	}
	return NrgResQuant{
		FrameQi: frameQi,
		ShapeQi: int32(qi),
		DbqQ14:  dbqQ14,
	}
}
