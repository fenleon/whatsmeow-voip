// SILK-style multi-stage pitch estimator (encoder side). HP-filters and 2x-
// downsamples the perceptually-weighted ltp_buf, runs an open-loop block-
// track survivor search at the coarse resolution, refines per-block at full
// resolution around the survivors, and folds in rate/prev-lag/spectral-
// harmonicity biases. Ported from whatsapp-rust (MIT)
// wacore/src/voip/mlow/smpl_pitch_enc.rs; MPL-2.0 for this port. Only the
// 20 ms / 8-subframe config is supported (the active MLow 1:1 path).
package mlow

import "math"

const (
	fsKhz           = 16
	stage1FsKhz     = 8
	coarseFsKhz     = 16
	minpitchMs      = 2
	maxpitchMs      = 20
	minpitchLen     = minpitchMs * fsKhz                      // 32
	maxpitchLen     = maxpitchMs * fsKhz                      // 320
	minpitchSt1     = minpitchMs*stage1FsKhz - totInterpDelay // 10
	maxpitchSt1     = maxpitchMs*stage1FsKhz + totInterpDelay // 166
	pitchDeltawght  = float32(0.1439)
	pitchShortwght1 = float32(0.04)
	specHarmBias    = float32(2.5)
	prevwght        = float32(0.7981)
	prevwghtSpan    = float32(0.15)
	ratewghtHR      = float32(0.022)
	lagSubfrlen     = 40
	lagSubfrlenSt1  = stage1FsKhz * lagSubfrlen / fsKhz // 20
	pitchblockMs    = 2
	pitchLookahead  = 7
	maxLtpBufLen    = 659
	pitchFlen       = 257

	pitchDownsampDelay  = 7
	pitchInterpolDelayC = 4

	pitchNumBlocksEnc = pitchNumBlocks                          // 9
	pitchblock        = pitchblockMs * fsKhz                    // 32
	numLagsStage1     = maxpitchSt1 - minpitchSt1 + 1           // 157
	numlagsCoarse     = coarseFsKhz * (maxpitchMs - minpitchMs) // 288
	numlagsFs         = fsKhz * (maxpitchMs - minpitchMs)       // 288

	// numstates1 block-track survivors at complexity 5-8 (pitch_numstates1
	// = 24); overrides the per-stream init default of 8.
	numstates1 = 24
	// complexity-8 is NOT low-complexity (numstates1 > 4).
	lowComplexity = false
	// 20 kbps is the HIGH-rate path.
	lowRate = false
)

const totInterpDelay = 6

// PitchEstState is the per-stream estimator state (the PitchEstimator
// non-scratch fields). prevLagblk/prevLagidx are reset to -1 at frame
// boundaries by the encoder (reset_cond).
type PitchEstState struct {
	PrevLag       float32
	PrevPitchCorr float32
	PrevLagblk    int32
	PrevLagidx    int32
}

// ResetCond clears the cross-frame lag-block predictor (called after the last
// frame of a packet and after any unvoiced frame, so cond-coding restarts).
func (st *PitchEstState) ResetCond() {
	st.PrevLagblk = -1
	st.PrevLagidx = -1
}

// PitchResult is the pitch estimator result for one internal frame.
type PitchResult struct {
	Pitchcorr    float32
	Lags         [pitchNumSubframes]float32
	Laginds      [pitchNumSubframes]int32
	AvgLag       float32
	HarmStrength float32
	BlocksegIdx  int
}

// pitchHPFilter is the ARMA1 with pitch_hp_b={1,-1}, pitch_hp_a={1,-0.96},
// zero state at call start. MA1 (ma[n] = x[n]-x[n-1], ma[0] = x[0]) then AR1
// in a 5-sample unrolled form using precomputed powers of ar1 = 0.96, so the
// float accumulation order is stable (normative).
func pitchHPFilter(x []float32, out []float32) {
	n := len(x)
	// MA1 into out (state_ma = x[-1] = 0).
	var stateMA float32
	for i := 0; i < n; i++ {
		out[i] = x[i] - stateMA
		stateMA = x[i]
	}
	// AR1: y[n] = ma[n] + 0.96*y[n-1], unrolled by 5.
	ar1 := float32(0.96)
	ar12 := ar1 * ar1
	ar13 := ar1 * ar12
	ar14 := ar1 * ar13
	ar15 := ar1 * ar14
	var ytmp float32 // y[-1]
	idx := 0
	for idx+4 < n {
		x0, x1, x2, x3, x4 := out[idx], out[idx+1], out[idx+2], out[idx+3], out[idx+4]
		out[idx+4] = x4 + ar1*x3 + ar12*x2 + ar13*x1 + ar14*x0 + ar15*ytmp
		out[idx] = x0 + ar1*ytmp
		out[idx+1] = x1 + ar1*x0 + ar12*ytmp
		out[idx+2] = x2 + ar1*x1 + ar12*x0 + ar13*ytmp
		out[idx+3] = x3 + ar1*x2 + ar12*x1 + ar13*x0 + ar14*ytmp
		ytmp = out[idx+4]
		idx += 5
	}
	for idx < n {
		ytmp = out[idx] + ytmp*ar1
		out[idx] = ytmp
		idx++
	}
}

var downsampFilt = [2*pitchDownsampDelay + 1]float32{
	-0.045472838,
	0.0,
	0.06366198,
	0.0,
	-0.10610329,
	0.0,
	0.31830987,
	0.5,
	0.31830987,
	0.0,
	-0.10610329,
	0.0,
	0.06366198,
	0.0,
	-0.045472838,
}

// pitchDownsample is the 2x decimating FIR. ptrIn has
// pitchDownsampDelay lead samples (offset) already written into ptrOut; the
// output length is (L-2*delay)/2.
func pitchDownsample(ptrIn []float32, l int, ptrOut []float32) int {
	const d = pitchDownsampDelay
	n := (l - 2*d) / 2
	for j := 0; j < n; j++ {
		tmp := ptrIn[2*j+d] * downsampFilt[d]
		for i := 0; i < d; i += 2 {
			tmp += (ptrIn[2*j+i] + ptrIn[2*j+2*d-i]) * downsampFilt[i]
		}
		ptrOut[j] = tmp
	}
	return n
}

var interpolFiltC = [2 * pitchInterpolDelayC]float32{
	-0.0024414062,
	0.023925781,
	-0.119628906,
	0.59814453,
	0.59814453,
	-0.119628906,
	0.023925783,
	-0.0024414062,
}

// upsampECore writes 2*len samples backwards from y using x (read backwards).
// Even taps copy x, odd taps average adjacent. xEnd/yEnd are the indices of
// the LAST written/read.
func upsampECore(buf []float32, xEnd, yEnd, length int) {
	xi, yi := xEnd, yEnd
	for i := 0; i < length; i++ {
		v := (buf[xi] + buf[xi+1]) * 0.5
		buf[yi] = v
		yi--
		buf[yi] = buf[xi]
		yi--
		xi--
	}
}

// upsampCCore is like upsampECore but the interpolated sample uses the 8-tap
// INTERPOL_FILT_C.
func upsampCCore(buf []float32, xEnd, yEnd, length int) {
	xi, yi := xEnd, yEnd
	for i := 0; i < length; i++ {
		var tmp float32
		for j := 0; j < pitchInterpolDelayC; j++ {
			a := buf[xi+j-(pitchInterpolDelayC-1)]
			b := buf[xi+pitchInterpolDelayC-j]
			tmp += (a + b) * interpolFiltC[j]
		}
		buf[yi] = tmp
		yi--
		buf[yi] = buf[xi]
		yi--
		xi--
	}
}

// getMaxi returns the argmax; ties resolve to the FIRST index. A simple
// strict-> scan (lowest index wins) matches the reference tree-reduction
// (validated by a tie-probe harness on this data).
func getMaxi(x []float32) int {
	bi := 0
	best := x[0]
	for n := 1; n < len(x); n++ {
		if x[n] > best {
			best = x[n]
			bi = n
		}
	}
	return bi
}

// getMaxiKOrder returns the K highest-value indices in selection order
// (descending value). The ascending masked-max scan (strict >,
// lowest-index-wins) is the validated equivalent of the reference tree
// selection.
func getMaxiKOrder(x []float32, k int) []int {
	taken := make([]bool, len(x))
	out := make([]int, 0, k)
	for i := 0; i < k; i++ {
		bi := -1
		var best float32
		for n, v := range x {
			if !taken[n] && (bi < 0 || v > best) {
				best = v
				bi = n
			}
		}
		if bi < 0 {
			break
		}
		taken[bi] = true
		out = append(out, bi)
	}
	return out
}

// calcE1Inner is the running energy of lag_subfrlen-length windows ending
// just before lag minpitch, for each of numlags lags. t is the window-start
// anchor in ltpbuf.
func calcE1Inner(e1 []float32, ltpbuf []float32, t, minpitch, maxpitch, lagSubfrlen int) {
	numlags := maxpitch - minpitch + 1
	reg0 := t - minpitch // absolute index of reg[0]; reg[-i] for i in 0..numlags valid
	e1[0] = maxF32(smplNrg(ltpbuf[reg0:reg0+lagSubfrlen]), 1e-9)
	for i := 1; i < numlags; i++ {
		rm := ltpbuf[reg0-i]
		rs := ltpbuf[reg0+lagSubfrlen-i]
		e1[i] = maxF32(e1[i-1]+rm*rm-rs*rs, 1e-9)
	}
}

// calcE1 computes the per-subframe E1 by computing an extended E1_ once then
// offsetting per subframe.
func calcE1(e1 []float32, ltpbuf []float32, ltpbufLen, numsubfrs, minpitch, maxpitch, lagSubfrlen int) {
	numlags := maxpitch - minpitch + 1
	maxpitchX := maxpitch + (numsubfrs-1)*lagSubfrlen
	numlagsX := maxpitchX - minpitch + 1
	t := ltpbufLen - lagSubfrlen
	e1Ext := make([]float32, numlagsX)
	calcE1Inner(e1Ext, ltpbuf, t, minpitch, maxpitchX, lagSubfrlen)
	offset := numlagsX - numlags
	for sf := 0; sf < numsubfrs; sf++ {
		copy(e1[sf*numlags:sf*numlags+numlags], e1Ext[offset:offset+numlags])
		offset -= lagSubfrlen
	}
}

func dotProdF32(a, b []float32, n int) float32 {
	var r float32
	for i := 0; i < n; i++ {
		r += a[i] * b[i]
	}
	return r
}

// calcCE2 computes the stage-1 cross-correlation C (8-sample dot,
// NUM_LAGS_STAGE1 lags/subframe) and per-subframe target energy E2.
func calcCE2(c, e2 []float32, ltpbuf []float32, ltpbufLen, numsubfrs int) {
	t := ltpbufLen - lagSubfrlenSt1*numsubfrs
	for sf := 0; sf < numsubfrs; sf++ {
		tgt := ltpbuf[t : t+20]
		reg0 := t - minpitchSt1
		for i := 0; i < numLagsStage1; i++ {
			// dot_prod_20(tgt, &reg[-i]) where reg = &ltpbuf[reg0]
			r := ltpbuf[reg0-i : reg0-i+20]
			c[sf*numLagsStage1+i] = dotProdF32(tgt, r, 20)
		}
		t += lagSubfrlenSt1
		e2[sf] = maxF32(dotProdF32(tgt, tgt, 20), 1e-9)
	}
}

// upsampEFast is the in-place 2x upsample of a per-subframe E array, high
// subframe first.
func upsampEFast(buf []float32, numsubfrs int, minpitch *int, numlags *int) {
	nin := *numlags
	nout := (nin - 1) * 2
	for sf := numsubfrs - 1; sf >= 0; sf-- {
		xEnd := sf*nin + nin - 2
		yEnd := sf*nout + nout - 1
		upsampECore(buf, xEnd, yEnd, nin-1)
	}
	*numlags = nout
	*minpitch *= 2
}

// upsampCFast is the in-place 2x upsample of a per-subframe C array using the
// interpolation filter.
func upsampCFast(buf []float32, numsubfrs int, minpitch *int, numlags *int) {
	nin := *numlags
	nout := (nin - pitchInterpolDelayC) * 2
	for sf := numsubfrs - 1; sf >= 0; sf-- {
		xEnd := sf*nin + nin - 1 - pitchInterpolDelayC
		yEnd := sf*nout + nout - 1
		upsampCCore(buf, xEnd, yEnd, nin-(pitchInterpolDelayC*2-1))
	}
	*numlags = nout
	*minpitch *= 2
}

func dotProd40(a, b []float32) float32 {
	var r float32
	for i := 0; i < 40; i++ {
		r += a[i] * b[i]
	}
	return r
}

func sumdeltas(laginds []int32, numsubfrs int) int32 {
	var ret int32
	for i := 1; i < numsubfrs; i++ {
		d := laginds[i] - laginds[i-1]
		if d < 0 {
			d = -d
		}
		ret += d
	}
	return ret
}

// encodeLagsBits is the rate (bits) the lag indices would cost, used as a
// survivor bias. Accumulates n_bits with no entropy-coding side-effects (the
// cost-only path).
func encodeLagsBits(tab *PitchTables, blocksegsIx int, laginds []int32, prevLagblk, prevLagidx int32, mode int) float32 {
	var nBits float32
	ixJulia := int32(tab.Blocksegs2idx[blocksegsIx])
	const blocksize = pitchblockMs * fsKhz * 2 // 64
	pblockseg := &tab.Blocksegs[blocksegsIx]

	if prevLagblk < 0 {
		cmf := tab.BlocksegIdxCmf
		nBits += ecEncodeBits(cmf[ixJulia-1], cmf[ixJulia], cmf[len(tab.Blocksegs)])
	} else {
		cmf := tab.BlockTransitionCmf[prevLagblk]
		b0 := pblockseg.Blocks[0]
		nBits += ecEncodeBits(cmf[b0], cmf[b0+1], cmf[pitchNumBlocksEnc])
		startIx := int32(tab.FirstblockRange[b0][0])
		cmfLen := int32(tab.FirstblockRange[b0][1] - tab.FirstblockRange[b0][0] + 1)
		subCmf := tab.BlocksegIdxCmf[startIx:]
		lo := ixJulia - startIx - 1
		hi := ixJulia - startIx
		nBits += ecEncodeBits(subCmf[lo]-subCmf[0], subCmf[hi]-subCmf[0], subCmf[cmfLen]-subCmf[0])
	}

	blk := int32(pblockseg.Blocks[0])
	deltaBlk := blk - prevLagblk
	startSeg := 0
	lagindsIx := 0
	if !(prevLagblk > -1 && deltaBlk >= -1 && deltaBlk <= 2) {
		nBits += 6.0 // uniform first-lag cost (log2 blocksize)
		prevLagblk = blk
		prevLagidx = laginds[lagindsIx]
		lagindsIx += pblockseg.Seglens[0]
		startSeg = 1
	}
	deltaLagCmf := tab.DeltaLagCmfs[mode]
	for k := startSeg; k < pblockseg.Nblocks; k++ {
		blk = int32(pblockseg.Blocks[k])
		idx := laginds[lagindsIx]
		lagindsIx += pblockseg.Seglens[k]
		deltaBlk = blk - prevLagblk
		deltaIdx := idx - prevLagidx
		prevLagidxMod := prevLagidx - prevLagblk*blocksize
		deltaRangeStart := -prevLagidxMod + deltaBlk*blocksize
		cmfBase := deltaRangeStart + 2*blocksize - 1
		ix := deltaIdx - deltaRangeStart
		p0 := deltaLagCmf[cmfBase]
		nBits += ecEncodeBits(
			deltaLagCmf[cmfBase+ix]-p0,
			deltaLagCmf[cmfBase+ix+1]-p0,
			deltaLagCmf[cmfBase+blocksize]-p0,
		)
		prevLagblk = blk
		prevLagidx = idx
	}
	return nBits
}

// ecEncodeBits returns the symbol's bit cost -log2((fh-fl)/ft) without coding
// it.
func ecEncodeBits(fl, fh, ft uint32) float32 {
	p := (float32(fh) - float32(fl)) / float32(ft)
	if p <= 0.0 {
		return 0.0
	}
	return -f32Log2(p)
}

// SmplEncodeLagsWire writes the blockseg selector + the per-40-block lag
// indices (laginds) to the range stream. This IS the voiced lag wire encode,
// the inverse of decode_smpl_pitch's contour reconstruction. prevLagblk /
// prevLagidx are the lag predictor (-1 on the first frame of a packet / after
// a no-match); mode selects the delta-lag CMF (0/1/2 by the mean ACB gain).
// The decoder rebuilds the exact per-block contour from these bits, so its
// voiced ACB basis matches the encoder's.
func SmplEncodeLagsWire(tab *PitchTables, enc *RangeEncoder, blocksegsIx int, laginds *[pitchNumSubframes]int32, prevLagblk, prevLagidx int32, mode int) {
	ixJulia := int32(tab.Blocksegs2idx[blocksegsIx])
	const blocksize = pitchblockMs * fsKhz * 2 // 64
	pblockseg := &tab.Blocksegs[blocksegsIx]

	// Blockseg selector: absolute (CMF over all blocksegs) when no
	// predictor, else block-transition CMF for blocks[0] followed by the
	// first-block-range CMF.
	if prevLagblk < 0 {
		cmf := tab.BlocksegIdxCmf
		enc.Encode(cmf[ixJulia-1], cmf[ixJulia], cmf[len(tab.Blocksegs)])
	} else {
		cmf := tab.BlockTransitionCmf[prevLagblk]
		b0 := pblockseg.Blocks[0]
		enc.Encode(cmf[b0], cmf[b0+1], cmf[pitchNumBlocksEnc])
		startIx := int32(tab.FirstblockRange[b0][0])
		cmfLen := int32(tab.FirstblockRange[b0][1] - tab.FirstblockRange[b0][0] + 1)
		subCmf := tab.BlocksegIdxCmf[startIx:]
		lo := ixJulia - startIx - 1
		hi := ixJulia - startIx
		enc.Encode(subCmf[lo]-subCmf[0], subCmf[hi]-subCmf[0], subCmf[cmfLen]-subCmf[0])
	}

	blk := int32(pblockseg.Blocks[0])
	deltaBlk := blk - prevLagblk
	startSeg := 0
	lagindsIx := 0
	if !(prevLagblk > -1 && deltaBlk >= -1 && deltaBlk <= 2) {
		// First lag uniform-coded over blocksize.
		idxMod := uint32(laginds[lagindsIx] - blk*blocksize)
		enc.Encode(idxMod, idxMod+1, blocksize)
		prevLagblk = blk
		prevLagidx = laginds[lagindsIx]
		lagindsIx += pblockseg.Seglens[0]
		startSeg = 1
	}
	deltaLagCmf := tab.DeltaLagCmfs[mode]
	for k := startSeg; k < pblockseg.Nblocks; k++ {
		blk = int32(pblockseg.Blocks[k])
		idx := laginds[lagindsIx]
		lagindsIx += pblockseg.Seglens[k]
		deltaBlk = blk - prevLagblk
		deltaIdx := idx - prevLagidx
		prevLagidxMod := prevLagidx - prevLagblk*blocksize
		deltaRangeStart := -prevLagidxMod + deltaBlk*blocksize
		cmfBase := deltaRangeStart + 2*blocksize - 1
		ix := deltaIdx - deltaRangeStart
		p0 := deltaLagCmf[cmfBase]
		enc.Encode(
			deltaLagCmf[cmfBase+ix]-p0,
			deltaLagCmf[cmfBase+ix+1]-p0,
			deltaLagCmf[cmfBase+blocksize]-p0,
		)
		prevLagblk = blk
		prevLagidx = idx
	}
}

// SmplLagsPredictorAfter returns the lag predictor after a voiced encode:
// prevLagblk = blocks[nblocks-1], prevLagidx = laginds[numsubframes-1].
// Exposed so the analysis advances its mirror identically.
func SmplLagsPredictorAfter(tab *PitchTables, blocksegsIx int, laginds *[pitchNumSubframes]int32) (int32, int32) {
	pblockseg := &tab.Blocksegs[blocksegsIx]
	lastBlk := int32(pblockseg.Blocks[pblockseg.Nblocks-1])
	return lastBlk, laginds[pitchNumSubframes-1]
}

const harmonicityUndef = float32(-10000.0)

// buildF2w weights the power spectrum for the harmonicity/tilt measures.
func buildF2w(f2 *[pitchFlen]float32) [pitchFlen]float32 {
	var f2w [pitchFlen]float32
	for i := 2; i < pitchFlen; i++ {
		f2w[i] = f2[i] * float32(i+3)
	}
	return f2w
}

// HarmStrengthAt recomputes the spectral harmonicity for a single value (the
// classifier's recompute path, always-reset cache).
func HarmStrengthAt(avgLag float32, f2w *[pitchFlen]float32) float32 {
	var cache [50]float32
	return spectralHarmonicity(avgLag, f2w, cache[:], true)
}

// spectralHarmonicity is the cached harmonicity measure shared with the
// classifier.
func spectralHarmonicity(avgLag float32, f2w *[pitchFlen]float32, cache []float32, reset bool) float32 {
	if reset {
		for i := range cache {
			cache[i] = harmonicityUndef
		}
	}
	invF2StepHz := 2.0 * float32(pitchFlen-1) / 16000.0
	harmHz := 16000.0 / avgLag
	harmIx := int32(f32Round(harmHz * 2.0 * invF2StepHz))
	cacheLen := int32(len(cache))
	if harmIx >= cacheLen {
		// The reference asserts this never happens; guard defensively and
		// fall through to recompute.
		return recomputeHarmonicity(harmHz, invF2StepHz, f2w)
	}
	if cache[harmIx] > harmonicityUndef {
		return cache[harmIx]
	}
	hs := recomputeHarmonicity(harmHz, invF2StepHz, f2w)
	cache[harmIx] = hs
	return hs
}

const numHarms = 4

func recomputeHarmonicity(harmHz, invF2StepHz float32, f2w *[pitchFlen]float32) float32 {
	harmWidth := harmHz * invF2StepHz
	harmStrength := float32(0.1)
	if harmWidth > 1.97 {
		var peakValleyMags [2*numHarms + 1]float32
		for numHarm := range peakValleyMags {
			ixStart := 0.5 * float32(numHarm) * harmWidth
			ixEnd := ixStart + harmWidth
			idxStart := int32(f32RoundUp(ixStart))
			idxEnd := int32(f32RoundDown(ixEnd))
			weightsLen := maxInt(int(idxEnd-idxStart+1), 0)
			var weights [20]float32
			invHarmWidth := 1.0 / harmWidth
			for i := 0; i < weightsLen; i++ {
				tmp := (float32(idxStart) - ixStart + float32(i)) * invHarmWidth
				tmp -= tmp * tmp
				weights[i] = tmp * tmp
			}
			base := maxInt(int(idxStart), 0)
			if base > len(f2w) {
				base = len(f2w)
			}
			// The reference assumes the harmonic window stays within F2w;
			// clamp defensively so a degenerate (too-short) lag can't read
			// past the spectrum.
			avail := minInt(len(f2w)-base, weightsLen)
			peakValleyNrg := dotProdF32(f2w[base:], weights[:], avail) / smplSumVec(weights[:], weightsLen)
			peakValleyMags[numHarm] = f32Sqrt(peakValleyNrg + 1e-30)
		}
		var magRatiosLog, magWeights [numHarms]float32
		magPeakWeights := [3]float32{1.0, 10.0, 1.0}
		magValleyWeights := [3]float32{5.0, 2.0, 5.0}
		for numHarm := 0; numHarm < numHarms; numHarm++ {
			magPeak := magPeakWeights[0]*peakValleyMags[2*numHarm] +
				magPeakWeights[1]*peakValleyMags[2*numHarm+1] +
				magPeakWeights[2]*peakValleyMags[2*numHarm+2]
			magValley := magValleyWeights[0]*peakValleyMags[2*numHarm] +
				magValleyWeights[1]*peakValleyMags[2*numHarm+1] +
				magValleyWeights[2]*peakValleyMags[2*numHarm+2]
			magRatiosLog[numHarm] = f32Ln(magPeak / magValley)
			magWeights[numHarm] = f32Sqrt(magPeak + magValley + 1e-30)
		}
		harmStrength = dotProdF32(magWeights[:], magRatiosLog[:], numHarms) / smplSumVec(magWeights[:], numHarms)
	}
	return harmStrength
}

func smplSumVec(x []float32, l int) float32 {
	var s float32
	for i := 0; i < l; i++ {
		s += x[i]
	}
	return s
}

// spectralHarmonicityCached is the harmonicity with a per-survivor cache
// (keyed by harmonic bin), used by the estimator's survivor loop.
func spectralHarmonicityCached(avgLag float32, f2w *[pitchFlen]float32, cache []float32, reset bool) float32 {
	if reset {
		for i := range cache {
			cache[i] = harmonicityUndef
		}
	}
	invF2StepHz := 2.0 * float32(pitchFlen-1) / 16000.0
	harmHz := 16000.0 / avgLag
	harmIx := int32(f32Round(harmHz * 2.0 * invF2StepHz))
	// The classifier's recompute is the single source of truth: an
	// out-of-range bin only arises from a degenerate lag and is handled there
	// by a clamped recompute.
	if harmIx < 0 || int(harmIx) >= len(cache) {
		return HarmStrengthAt(avgLag, f2w)
	}
	if cache[harmIx] > harmonicityUndef {
		return cache[harmIx]
	}
	hs := HarmStrengthAt(avgLag, f2w)
	cache[harmIx] = hs
	return hs
}

// SmplPitch is the full estimator. ltpBuf is the perceptually-weighted speech
// of length maxLtpBufLen (the last pitchLookahead samples are lookahead); f2
// is the LPC power spectrum. codedAsActiveVoice gates the search (false ->
// unvoiced defaults). Mutates the cross-frame predictor in st.
func SmplPitch(st *PitchEstState, ltpBuf []float32, f2 *[pitchFlen]float32, codedAsActiveVoice bool) PitchResult {
	tab := LoadPitchTables()
	numsubfrs := pitchNumSubframes
	l := maxLtpBufLen
	look := pitchLookahead

	if !codedAsActiveVoice {
		minLag := float32(minpitchMs * fsKhz)
		st.PrevLag = 0.0
		st.PrevPitchCorr = 0.0
		st.PrevLagblk = -1
		st.PrevLagidx = -1
		var lags [pitchNumSubframes]float32
		for i := range lags {
			lags[i] = minLag
		}
		return PitchResult{AvgLag: minLag, Lags: lags}
	}

	// HP filter into stage1[offset..], where offset = PITCH_DOWNSAMP_DELAY
	// leading zeros.
	offset := pitchDownsampDelay
	stage1 := make([]float32, l+offset+64) // small slack
	pitchHPFilter(ltpBuf, stage1[offset:offset+l])
	// ltp_buf_hp = stage1[offset .. offset + (L - look)]
	hpLen := l - look
	ltpBufHP := stage1[offset : offset+hpLen]

	// Downsample stage1[0 .. L+offset] -> stage1_ds.
	stage1Ds := make([]float32, (l+offset)/2+8)
	stage1Len := pitchDownsample(stage1, l+offset, stage1Ds)

	numlags0 := numLagsStage1
	e1 := make([]float32, numlags0*numsubfrs+16)
	calcE1(e1, stage1Ds, stage1Len, numsubfrs, minpitchSt1, maxpitchSt1, lagSubfrlenSt1)
	var e2 [pitchNumSubframes]float32
	// C / E arrays are over-allocated: the upsample stages expand them in
	// place to full-res widths.
	cap := (2*fsKhz/stage1FsKhz)*numLagsStage1*numsubfrs + 64
	c := make([]float32, cap)
	e := make([]float32, cap)
	cStage1 := make([]float32, numlags0*numsubfrs)
	calcCE2(cStage1, e2[:], stage1Ds, stage1Len, numsubfrs)
	copy(c, cStage1)

	// E from sqrt-energy blend (stage 1).
	numlags := numlags0
	for sf := 0; sf < numsubfrs; sf++ {
		sqrtE2 := f32Sqrt(e2[sf] + 1e-30)
		for i := 0; i < numlags; i++ {
			sqrtE1 := f32Sqrt(e1[sf*numlags+i] + 1e-30)
			tmp := 0.5 * (sqrtE1 + sqrtE2)
			e[sf*numlags+i] = tmp * tmp
		}
	}

	// Upsample to coarse (16 kHz) resolution.
	minpitchC := minpitchSt1
	numlagsC := numlags
	minpitchE := minpitchSt1
	numlagsE := numlags
	if lowComplexity {
		upsampEFast(c, numsubfrs, &minpitchC, &numlagsC)
	} else {
		upsampCFast(c, numsubfrs, &minpitchC, &numlagsC)
	}
	upsampEFast(e, numsubfrs, &minpitchE, &numlagsE)

	minpitchCoarse := coarseFsKhz * minpitchMs
	offsetC0 := minpitchCoarse - minpitchC
	offsetE0 := minpitchCoarse - minpitchE

	// H (coarse). (The reference also snapshots c_coarse/e_coarse here; those
	// copies are never read downstream, so they are not ported.)
	numlagsCoarseC := numlagsCoarse
	h := make([]float32, numlagsCoarseC*numsubfrs*2+64)
	for sf := 0; sf < numsubfrs; sf++ {
		for i := 0; i < numlagsCoarseC; i++ {
			cv := c[sf*numlagsC+offsetC0+i]
			ev := e[sf*numlagsE+offsetE0+i]
			h[sf*numlagsCoarseC+i] = cv / ev
		}
	}

	// Per-block coarse maxima -> Hblk.
	pitchblockCoarse := pitchblockMs * coarseFsKhz // 32
	var hblk [pitchNumSubframes][pitchNumBlocksEnc]float32
	for sf := 0; sf < numsubfrs; sf++ {
		for block := 0; block < pitchNumBlocksEnc; block++ {
			base := sf*numlagsCoarseC + block*pitchblockCoarse
			hblk[sf][block] = smplMaximum(h[base : base+pitchblockCoarse])
		}
	}

	// Block-track survivor selection.
	blocksizeFs := pitchblock * 2 // BLOCKSIZE = 64
	reductionFactor := float32(0.7)
	pitchDeltawghtC := pitchDeltawght / float32(blocksizeFs)
	var sfWght [pitchNumSubframes]float32
	{
		var sumE2 float32
		for _, v := range e2 {
			sumE2 += v
		}
		for sf := 0; sf < numsubfrs; sf++ {
			sfWght[sf] = e2[sf] / sumE2
		}
	}
	numBlocktracks := len(tab.Blocktracks)
	utils := make([]float32, numBlocktracks)
	for i := 0; i < numBlocktracks; i++ {
		bt := &tab.Blocktracks[i]
		var corr float32
		for sf := 0; sf < numsubfrs; sf++ {
			corr += hblk[sf][bt.Track[sf]] * sfWght[sf]
		}
		shortlagbias1 := (float32(maxpitchLen)/((bt.Meanblock+1.5)*float32(pitchblock)) - 1.0) * pitchShortwght1
		utils[i] = 1.0/(1.1-corr) - reductionFactor*float32(pitchblock)*pitchDeltawghtC*bt.Trackdeltas + shortlagbias1
	}
	trackIdx := getMaxiKOrder(utils, numstates1)

	// Recompute full-res E1 over the HP signal.
	e1Fs := make([]float32, numlagsE*numsubfrs+16)
	calcE1(e1Fs, ltpBufHP, l-look, numsubfrs, minpitchE, minpitchE+numlagsE-1, lagSubfrlen)

	// uniqueblocks bitmask per subframe from the survivor tracks.
	var uniqueblocks [pitchNumSubframes]uint16
	for _, ti := range trackIdx {
		track := &tab.Blocktracks[ti].Track
		for sf := 0; sf < numsubfrs; sf++ {
			uniqueblocks[sf] |= 1 << uint(track[sf])
		}
	}

	hThres := float32(0.25)
	if lowComplexity {
		hThres = 0.0
	}
	offsetC := minpitchMs*fsKhz - minpitchC
	offsetE := minpitchMs*fsKhz - minpitchE
	// Update C and E around survivor block peaks at full resolution.
	for sf := 0; sf < numsubfrs; sf++ {
		mask := uint16(1)
		cPtr := offsetC + sf*numlagsC
		ePtr := offsetE + sf*numlagsE
		e1Ptr := offsetE + sf*numlagsE
		hPtr := sf * numlagsFs
		// ltp_buf_ptr = &ltp_buf_hp[L - look + (sf - numsubfrs)*LAG_SUBFRLEN]
		ltpOff := (l - look) + (sf-numsubfrs)*lagSubfrlen
		e2Sf := maxF32(dotProd40(ltpBufHP[ltpOff:], ltpBufHP[ltpOff:]), 1e-9)
		e2[sf] = e2Sf
		sqrtE2 := f32Sqrt(e2Sf + 1e-30)
		for block := 0; block < pitchNumBlocksEnc; block++ {
			if uniqueblocks[sf]&mask != 0 {
				var sqrtE1 [pitchblock + 1]float32
				for i := 0; i < pitchblock+1; i++ {
					sqrtE1[i] = f32Sqrt(e1Fs[e1Ptr+block*pitchblock+i] + 1e-30)
				}
				for i := 0; i < pitchblock+1; i++ {
					tmp := 0.5 * (sqrtE1[i] + sqrtE2)
					e[ePtr+block*pitchblock+i] = 0.5 * tmp * tmp
				}
				for i := 0; i < pitchblock; i++ {
					if h[hPtr+block*pitchblock+i] > hThres {
						lag := minpitchLen + block*pitchblock + i
						a := ltpBufHP[ltpOff:]
						b := ltpBufHP[ltpOff-lag:]
						c[cPtr+block*pitchblock+i] = 0.5 * dotProd40(a, b)
					}
				}
			}
			mask <<= 1
		}
	}

	// Upsample C and E around survivor peaks to half-sample resolution and
	// compute H (high to low).
	strideC := pitchNumBlocksEnc*2*pitchblock + offsetC // per-subframe frac stride
	strideE := pitchNumBlocksEnc*2*pitchblock + offsetE
	for sf := numsubfrs - 1; sf >= 0; sf-- {
		cPtr := offsetC + sf*numlagsC
		cPtrFrac := offsetC + sf*strideC
		ePtr := offsetE + sf*numlagsE
		ePtrFrac := offsetE + sf*strideE
		hPtr := sf * 2 * pitchblock * pitchNumBlocksEnc
		mask := uint16(1) << (pitchNumBlocksEnc - 1)
		for block := pitchNumBlocksEnc - 1; block >= 0; block-- {
			if uniqueblocks[sf]&mask != 0 {
				ein := ePtr + block*pitchblock
				eout := ePtrFrac + block*2*pitchblock
				upsampECore(e, ein+pitchblock-1, eout+2*pitchblock-1, pitchblock)
				cin := cPtr + block*pitchblock
				cout := cPtrFrac + block*2*pitchblock
				if lowComplexity {
					upsampECore(c, cin+pitchblock-1, cout+2*pitchblock-1, pitchblock)
				} else {
					upsampCCore(c, cin+pitchblock-1, cout+2*pitchblock-1, pitchblock)
				}
				for i := 0; i < 2*pitchblock; i++ {
					h[hPtr+block*2*pitchblock+i] = c[cout+i] / e[eout+i]
				}
			}
			mask >>= 1
		}
	}

	// Fine search: per survivor, per blockseg, per block: combine H over the
	// seg's subframes, argmax.
	var lagindsSurv [][pitchNumSubframes]int32
	var blocksegsIxList []int
	var hComb [2 * pitchblock]float32
	lagindCache := make(map[int32]int32)
	for _, idx := range trackIdx {
		rng := tab.BlocksegsIx[idx]
		for j := 0; j < rng[1]; j++ {
			bsx := rng[0] + j
			pblockseg := &tab.Blocksegs[bsx]
			var lagindsRow [pitchNumSubframes]int32
			startSf := 0
			for n := 0; n < pblockseg.Nblocks; n++ {
				lookupKey := ((int32(startSf)<<3)+int32(pblockseg.Seglens[n]))<<4 |
					int32(pblockseg.Blocks[n])
				bestI, ok := lagindCache[lookupKey]
				if !ok {
					for i := range hComb {
						hComb[i] = 0
					}
					for sf := startSf; sf < startSf+pblockseg.Seglens[n]; sf++ {
						hPtr := sf*2*pitchblock*pitchNumBlocksEnc + pblockseg.Blocks[n]*2*pitchblock
						for i := 0; i < 2*pitchblock; i++ {
							hComb[i] += h[hPtr+i] * e2[sf]
						}
					}
					bestI = int32(getMaxi(hComb[:]))
					lagindCache[lookupKey] = bestI
				}
				for sf := startSf; sf < startSf+pblockseg.Seglens[n]; sf++ {
					lagindsRow[sf] = bestI + int32(pblockseg.Blocks[n]*2*pitchblock)
				}
				startSf += pblockseg.Seglens[n]
			}
			lagindsSurv = append(lagindsSurv, lagindsRow)
			blocksegsIxList = append(blocksegsIxList, bsx)
		}
	}
	nlaginds := len(lagindsSurv)

	// Final search.
	pitchRatewght := ratewghtHR
	if lowRate {
		pitchRatewght = 0.028
	}
	f2w := buildF2w(f2)
	maxIx := getMaxi(sfWght[:numsubfrs])
	var spectralHarmCache [50]float32

	bestUtil := float32(0)
	bestPitchcorr := float32(0)
	bestSurv := 0
	pitchDeltawghtFs := pitchDeltawght / float32(blocksizeFs)

	for surv := 0; surv < nlaginds; surv++ {
		var sumC, sumE float32
		for sf := 0; sf < numsubfrs; sf++ {
			cBase := offsetC + sf*strideC
			eBase := offsetE + sf*strideE
			li := int(lagindsSurv[surv][sf])
			sumC += c[cBase+li]
			sumE += e[eBase+li]
		}
		rateBias := encodeLagsBits(tab, blocksegsIxList[surv], lagindsSurv[surv][:],
			st.PrevLagblk, st.PrevLagidx, 1) * pitchRatewght
		meanLag := float32(lagindsSurv[surv][maxIx])*0.5 + float32(minpitchLen)
		pitchcorr := sumC / sumE
		firstLag := 0.5*float32(lagindsSurv[surv][0]) + float32(minpitchLen)
		prevLagBias := getPrevLagBias(st, firstLag)
		spectralHarmBias := specHarmBias * spectralHarmonicityCached(meanLag, &f2w, spectralHarmCache[:], surv == 0)
		util := 1.0/(1.1-pitchcorr) -
			pitchDeltawghtFs*float32(sumdeltas(lagindsSurv[surv][:], numsubfrs)) +
			spectralHarmBias + prevLagBias - rateBias
		if surv == 0 || util > bestUtil {
			bestUtil = util
			bestSurv = surv
		}
		if surv == 0 || pitchcorr > bestPitchcorr {
			bestPitchcorr = pitchcorr
		}
	}

	var lags [pitchNumSubframes]float32
	var lagindsOut [pitchNumSubframes]int32
	for sf := 0; sf < numsubfrs; sf++ {
		lags[sf] = float32(lagindsSurv[bestSurv][sf])*0.5 + float32(minpitchLen)
		lagindsOut[sf] = lagindsSurv[bestSurv][sf]
	}
	avgLag := float32(lagindsSurv[bestSurv][maxIx])*0.5 + float32(minpitchLen)
	// With SMPL_PITCH_SPEC_HARM_BIAS defined, the final harmonicity reuses
	// the survivor-loop cache.
	harmStrength := spectralHarmonicityCached(avgLag, &f2w, spectralHarmCache[:], false)

	st.PrevLag = lags[numsubfrs-1]
	st.PrevPitchCorr = bestPitchcorr
	st.PrevLagidx = lagindsSurv[bestSurv][numsubfrs-1]
	st.PrevLagblk = st.PrevLagidx / (2 * pitchblock)

	return PitchResult{
		Pitchcorr:    bestPitchcorr,
		Lags:         lags,
		Laginds:      lagindsOut,
		AvgLag:       avgLag,
		HarmStrength: harmStrength,
		BlocksegIdx:  blocksegsIxList[bestSurv],
	}
}

func getPrevLagBias(st *PitchEstState, lag float32) float32 {
	lagDiff := f32Abs(lag - st.PrevLag)
	diffThres := prevwghtSpan * st.PrevLag
	if lagDiff < diffThres {
		return st.PrevPitchCorr * (1.0 - lagDiff/diffThres) * prevwght
	}
	return 0.0
}

func f32Sqrt(x float32) float32      { return float32(math.Sqrt(float64(x))) }
func f32Ln(x float32) float32        { return float32(math.Log(float64(x))) }
func f32RoundUp(x float32) float32   { return float32(math.Ceil(float64(x))) }
func f32RoundDown(x float32) float32 { return float32(math.Floor(float64(x))) }
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
