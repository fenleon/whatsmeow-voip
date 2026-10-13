// MLow encoder analysis: PCM -> SmplFrameParams. Per internal frame the LPC
// front-end windows the 20 ms lpcbuf, FFT-autocorrelates it, derives the
// bandwidth-expanded A and its NLSF, and feeds the bit-exact LSF quantizer
// (with the conditional-coding path); the excitation comes from the CELP
// encoder over the per-subframe interpolated LPC residual. The UNVOICED level
// is the bit-exact nrg_res floor; the VOICED (LTP, stage1=1) path runs the
// real CELP ACB/LTP encode. Ported from whatsapp-rust (MIT)
// wacore/src/voip/mlow/analysis.rs + params.rs; MPL-2.0 for this port.
package mlow

import (
	"math"
)

// params.rs structures.

// SmplLsfParams is the encoded LSF block (grid + 16 stage-2 symbols + extra).
type SmplLsfParams struct {
	Stage1 int32
	Grid   int32
	Stage2 [16]int32
	Extra  int32
}

// SmplRawSym is one uniform raw-symbol write (encode(sym, sym+1, 1<<nbits)).
type SmplRawSym struct {
	Sym   uint32
	Nbits uint32
}

// SmplPulseParams carries the structured pulse counts plus the raw entropy
// symbols the encoder replays (the counts alone are lossy w.r.t. the exact
// bitstream).
type SmplPulseParams struct {
	Total    int32
	Subfr    [4]int32
	MagRuns  []int32
	SignSyms []SmplRawSym
}

// SmplGainParams is the unvoiced gains block (the wire gain block IS the
// nrgres layout).
type SmplGainParams struct {
	GainMain  int32
	GainDelta int32
	NrgRes    [4]int32
}

// SmplPitchParams carries the estimator's chosen contour (blockseg_idx +
// laginds); these ARE the wire pitch encoding.
type SmplPitchParams struct {
	GainIdx     [4]int32
	FiltIdx     [4]int32
	BlocksegIdx int
	Laginds     [8]int32
}

// SmplInternalParams is one internal frame's parameter block.
type SmplInternalParams struct {
	Lsf    SmplLsfParams
	Pulses SmplPulseParams
	Pitch  SmplPitchParams
	Gains  SmplGainParams
}

// SmplFrameParams is the full analyzed parameter set for one 60 ms MLow frame.
type SmplFrameParams struct {
	Toc      uint8
	Config   int
	Internal [3]SmplInternalParams
}

// analysis.rs constants.

const (
	// HP-history samples to carry for the LPC window buffer (full lpc_buf_mem,
	// safe and exact).
	smplLpcHistLen = 144
	// lpcbuf starts 96 samples before each internal frame.
	smplLpcPre = 96
	// surv = lsf_surv for complexity 8 (update_complexity_setting).
	smplLsfSurv = 6
	// RDw_adj = sqrt(mainBitRate / 14000) for the HIGH-rate path at 20 kbps.
	smplLsfRdwAdj = float32(1.1952286)

	smplMainBitRate = 20000
	smplComplexity  = 8

	smelpCelpLowRate      = false
	smelpCelpPercRespLen  = 32
	smelpCelpFcbSubfrLen  = 80
	smelpCelpSubfrPerPack = 12
	// perc_resp_len + SMPL_PERC_EMPH_V_LEN - 1 (= 33 = SMPL_MAX_L_RESP).
	smelpPercRLen = smelpCelpPercRespLen + 1
	// smpl_fcb_tot_surv_20ms_max for complexity 5-8. Drives tot_surv.
	smelpFcbTotSurv20msMax = 100

	// Encoder input high-pass 3 dB corner (SMPL_ENC_HP_FCORNER_3DB_HZ).
	smelpEncHpFcornerHz = float32(35.0)

	// The perceptual emphasis the pitch weighting uses.
	smelpPercEmphPitch = float32(-0.82)
	// pitch_perc_resp_len for complexity 5-8 (the 17-tap monic MA weighting).
	smelpPitchPercRespLen = 17
	// Pitch search history span in samples (SMPL_MAXPITCH_LEN).
	smelpPitchLagMax = 320
	// Pitch estimator lookahead (SMPL_PITCH_LOOKAHEAD_LEN).
	smelpPitchLookaheadLen = 7

	smelpMaxPulsesPerSf = int32(40)
)

// SmplEncoderState is the cross-frame analysis state: only the LPC-analysis
// input history persists (the decoder rebuilds synthesis state per 60 ms
// frame).
type SmplEncoderState struct {
	hist        []float64
	vadPcm      []int16
	hp          []float32
	x           []float64
	xn          []float32
	hpFull      []float32
	hpCoefs     *[2][3]float32
	hpState     [4]float32
	celp        *CelpEncoder
	perc        *PercModelState
	percPrev    []float32
	bitrate     *BitrateController
	lpcHist     []float32
	prevLsfq    []float32
	prevVoiced  bool
	vad         *SmplVadState
	vuv         VuvMode
	hpPitchHist []float32
	ltpBuf      []float32
	pitchEst    PitchEstState
	lpcFft      *fftScratch

	// DbgCaptures, when non-nil, receives one entry per internal frame with
	// the analysis-internal values the stage dump vectors pin — the
	// encoder-side analogue of the decoder's Dbg fields.
	DbgCaptures *[]SmplEncDbg
}

// SmplEncDbg is one internal frame's analysis-internal debug capture.
type SmplEncDbg struct {
	LtpBuf      []float32
	F2          []float32
	Pitchcorr   float32
	AvgLag      float32
	Harm        float32
	Vstr        float32
	Laginds     [8]int32
	BlocksegIdx int
}

func unvoicedPitch() SmplPitchSynth {
	return SmplPitchSynth{Voiced: false, NormGain: 0.0}
}

type candCandidate struct {
	ip       SmplInternalParams
	stage1   int32
	grid     int32
	qsym     [16]int32
	pulseVec []int32
	// Per-subframe excitation gainQ used by the synthesis (rate-control gain
	// for unvoiced, 0 for voiced). Must match what commitCandidate feeds the
	// shadow synth (warm history).
	gainQ  [4]int32
	pitch  SmplPitchSynth
	silent bool
}

// celpFrameCtx is the borrowed CELP/perceptual state for one internal frame's
// excitation analysis.
type celpFrameCtx struct {
	celp     *CelpEncoder
	perc     *PercModelState
	percPrev []float32
	bitrate  *BitrateController
	// Full normalized HP frame (960 samples, [-1,1]).
	hpN                []float32
	intf               int
	spActProb          float32
	codedAsActiveVoice bool
	f2                 [smplFLen]float32
	voicingStrength    float32
	vuv                *VuvMode
	hpPitchHist        []float32
	ltpBuf             []float32
	pitchEst           *PitchEstState
	// Per-subframe perceptual autocorrelation (shared CELP + pitch input).
	percCorrs [][]float32
	dbg       *[]SmplEncDbg
	// Decoder-reconstructed per-block pitch lags (2 per subframe) for the
	// voiced CELP ACB.
	blockLags [smplSubfrCount][2]float32
}

// frontEndLsf is the front-end LPC/NLSF analysis result for one internal
// frame, plus the conditional-coding context.
type frontEndLsf struct {
	a          [smplLpcOrder + 1]float32
	nlsf       [smplLpcOrder]float32
	prevLsfq   []float32
	prevVoiced bool
	intf       int
}

// quantize runs the bit-exact LSF quantizer for voiced and the cond-coding
// condition, returning the wire grid + stage2 + the committed
// (decoder-reconstructed) NLSF + the quantized predcoef.
func (fe *frontEndLsf) quantize(synthT *SmplSynthTables, voiced int, prevNLSF []float32) (int32, [16]int32, []float32, [17]float32) {
	cond := (fe.prevVoiced == (voiced != 0)) && fe.intf > 0
	var res LsfQuantResult
	if cond && len(fe.prevLsfq) == smplLpcOrder {
		res = LsfQuantCond(&LsfBuiltTables().Cb, &fe.a, &fe.nlsf, pad16(fe.prevLsfq), voiced, 0, smplLsfRdwAdj, smplLsfSurv)
	} else {
		res = LsfQuant(&LsfBuiltTables().Cb, &fe.a, &fe.nlsf, voiced, 0, smplLsfRdwAdj, smplLsfSurv)
	}
	grid := res.Qi[0]
	var stage2 [16]int32
	copy(stage2[:], res.Qi[1:])
	// Committed NLSF = the envelope the decoder rebuilds from the wire.
	committed := ReconstructNLSF(synthT, voiced, 0, int(grid), &stage2, prevNLSF)
	aVq := NLSF2A(committed)
	var predcoef [17]float32
	copy(predcoef[:], aVq[:])
	predcoef[0] = 1.0
	return grid, stage2, committed, predcoef
}

// pad16 copies prevLsfq into a fixed array for LsfQuantCond.
func pad16(v []float32) *[lsfOrder]float32 {
	var out [lsfOrder]float32
	copy(out[:], v)
	return &out
}

func commitCandidate(synthT *SmplSynthTables, st *SmplFrameSynth, cand *candCandidate, prevNLSF []float32) []float32 {
	if cand.silent {
		nlsf := ReconstructNLSF(synthT, 0, 0, int(cand.ip.Lsf.Grid), &cand.ip.Lsf.Stage2, prevNLSF)
		pulseVec := make([]int32, smplIntfLen)
		SynthInternalFrame(synthT, st, 0, 0, int(cand.ip.Lsf.Grid), &cand.ip.Lsf.Stage2,
			prevNLSF, pulseVec, cand.gainQ, &cand.pitch)
		return nlsf
	}
	_, nlsf := SynthInternalFrame(synthT, st, int(cand.stage1), 0, int(cand.grid), &cand.qsym,
		prevNLSF, cand.pulseVec, cand.gainQ, &cand.pitch)
	return nlsf
}

func smplUnvoicedCandidate(synthT *SmplSynthTables, _st *SmplFrameSynth, win []float64, winN []float32, prevNLSF []float32, fe *frontEndLsf, cs *celpFrameCtx) candCandidate {
	frame := win[smplOrder:]

	r0 := smplAutocorr(frame, 0)[0]
	if r0 <= 0.0 {
		// Silent frame: still advance the CELP excitation state (zeros) so it
		// stays in sync.
		var flat [smplSubfrCount][17]float32
		for p := range flat {
			flat[p][0] = 1.0
		}
		runCelpSubframes(cs, &flat, make([]float32, smplIntfLen), nil, cs.percCorrs,
			smplPercEmphUV, 0)
		return smplSilentInternal(synthT)
	}

	// LSF: bit-exact C quantizer fed the faithful front-end NLSF.
	bgrid, bsym, brec, _ := fe.quantize(synthT, 0, prevNLSF)

	// Per-subframe interpolated LPC; the interpolation search tries idx 1 too
	// and keeps it when it lowers the residual energy.
	predcoefs, resLpc, interpolIdx := smplLsfInterpolSearch(brec, fe.prevLsfq, winN)

	// Run the CELP excitation encoder per subframe.
	celpOut := runCelpSubframes(cs, &predcoefs, resLpc, nil, cs.percCorrs, smplPercEmphUV, 0)

	// Map CELP pulses -> per-position pulse train; collect the per-subframe
	// FCB gain index (= the wire nrg_res symbol).
	const main = 1
	pulseVec := make([]int32, smplIntfLen)
	var fcbgIdx [4]int32
	for sf := 0; sf < smplSubfrCount; sf++ {
		out := &celpOut[sf]
		for _, v := range out.Pulses[main] {
			// sign = 1 + 2*(v>>15); pos = v*sign - 1; pPulses[pos] += sign.
			sign := 1 + 2*(int32(v)>>15)
			pos := int32(v)*sign - 1
			if pos >= 0 && pos < smplSubfrLen {
				pulseVec[sf*smplSubfrLen+int(pos)] += sign
			}
		}
		fcbgIdx[sf] = int32(out.GainIdx[main])
	}

	// Unvoiced LEVEL (nrgres): bit-exact smpl_quant_nrg_res on the
	// per-subframe residual energy.
	var nrgres [4]float32
	for sf := range nrgres {
		res := resLpc[sf*smplSubfrLen : (sf+1)*smplSubfrLen]
		var e float32
		for _, v := range res {
			e += v * v
		}
		nrgres[sf] = e / float32(smplSubfrLen)
	}
	nq := QuantNrgRes4(&nrgres)
	gm := nq.FrameQi
	gd := nq.ShapeQi
	// Synthesis gain_q[sf] = the reconstructed per-subframe nrgres floor.
	gainQ := nq.DbqQ14

	pp := smplBuildPulseParams(pulseVec)
	gains := SmplGainParams{GainMain: gm, GainDelta: gd, NrgRes: [4]int32{-1, -1, -1, -1}}
	for sf := 0; sf < 4; sf++ {
		// The wire writes a per-subframe nrg_res (= fcbg_idx) only where
		// pulses exist.
		if pp.Subfr[sf] > 0 {
			gains.NrgRes[sf] = fcbgIdx[sf]
		}
	}

	return candCandidate{
		ip: SmplInternalParams{
			Lsf: SmplLsfParams{
				Stage1: 0,
				Grid:   bgrid,
				Stage2: bsym,
				// lsf_interpol_idx: the decoder interpolates the
				// per-subframe envelope with this, so it must match the
				// index the residual was whitened under.
				Extra: interpolIdx,
			},
			Pulses: pp,
			Gains:  gains,
		},
		stage1:   0,
		grid:     bgrid,
		qsym:     bsym,
		pulseVec: pulseVec,
		gainQ:    gainQ,
		pitch:    unvoicedPitch(),
	}
}

// runCelpSubframes is the per-subframe perceptual weighting + CELP excitation
// for one internal frame (4 subframes of 80). Returns the per-subframe CELP
// outputs; mutates the CELP/perc state so it stays in sync. blockLags may be
// nil for unvoiced (all-zero lags). emph selects the perceptual emphasis (UV
// vs V) and voiced drives the bitrate controller.
func runCelpSubframes(cs *celpFrameCtx, predcoefs *[smplSubfrCount][17]float32, resLpc []float32, blockLags *[smplSubfrCount][2]float32, percCorrs [][]float32, emph [2]float32, voiced int32) []CelpSubframeOut {
	percWght := percCorrsToWght(percCorrs, emph, smelpCelpPercRespLen)
	outs := make([]CelpSubframeOut, 0, smplSubfrCount)

	// Per-subframe weighted target energy (the bitrate controller's wnrg).
	wnrgs := make([]float32, smplSubfrCount)
	for sf := 0; sf < smplSubfrCount; sf++ {
		res := resLpc[sf*smplSubfrLen : (sf+1)*smplSubfrLen]
		const scale = float32(32768.0)
		var s float32
		for _, v := range res {
			s += (v * scale) * (v * scale)
		}
		wnrgs[sf] = s
	}

	enc := BitrateControllerInputs{
		InternalSampleRate:       16000,
		PayloadSizeMs:            60,
		FecBitRate:               0,
		MainBitRate:              smplMainBitRate,
		Complexity:               smplComplexity,
		UseFecRateCompensation:   0,
		UseDtx:                   0,
		SubFrameImportanceFactor: 1.0,
	}

	for sf := 0; sf < smplSubfrCount; sf++ {
		wnrg := wnrgs[sf]
		wnrgNext := wnrgs[sf]
		if sf+1 < smplSubfrCount {
			wnrgNext = wnrgs[sf+1]
		}
		nonflatness := float32(2.0)
		if voiced != 0 {
			nonflatness = 0.0
		}
		// Real classifier voicing_strength, negative for unvoiced.
		voicingStrength := cs.voicingStrength
		cav := int32(0)
		if cs.codedAsActiveVoice {
			cav = 1
		}
		maxPulses, importance := cs.bitrate.Control(&enc, 0, cav,
			cs.spActProb, nonflatness, voicingStrength, voiced, wnrg, wnrgNext,
			0, 320, 80)
		var numsurv [smplMaxPulsesPerSF]int16
		for i := range numsurv {
			numsurv[i] = 1
		}
		totSurv := int32(1000 * (smelpFcbTotSurv20msMax * smelpCelpFcbSubfrLen) / (20 * 16000))
		SmplDistributeFcbSurv(numsurv[:], int32(maxPulses[1]), totSurv)

		// The two 40-sample sub-blocks of this subframe carry their own
		// (decoder-reconstructed) lags; index 2 is read by the encoder as the
		// trailing lag (lags[n_lags-1]).
		var lags [3]float32
		if blockLags != nil {
			lags = [3]float32{blockLags[sf][0], blockLags[sf][1], blockLags[sf][1]}
		}

		res := resLpc[sf*smplSubfrLen : (sf+1)*smplSubfrLen]
		out := cs.celp.EncodeSubframe(res, &predcoefs[sf], percWght[sf], lags[:],
			importance, maxPulses, numsurv[:])
		outs = append(outs, out)
	}
	return outs
}

// computePercCorrs is the per-subframe perceptual autocorrelation
// (perc_corrs_buf, length SMPL_PERC_R_LEN), the shared input to BOTH the CELP
// weighting and the pitch-perceptual weighting. Advances the perc-model
// state, so it must run EXACTLY ONCE per internal frame.
func computePercCorrs(cs *celpFrameCtx) [smplSubfrCount][]float32 {
	const frameMs = 20
	const shorter = 32 // SMPL_WINNEXT_WB_LONG_LEN - SMPL_WINNEXT_WB_LEN
	var corrs [smplSubfrCount][]float32
	for sf := 1; sf < smplSubfrCount; sf += 2 {
		start := cs.intf*smplIntfLen + (sf-1)*smplSubfrLen
		xlen := 2*smplSubfrLen + shorter
		xsubfr := make([]float32, xlen)
		for i := 0; i < xlen; i++ {
			idx := start + i
			if idx < len(cs.hpN) {
				xsubfr[i] = cs.hpN[idx]
			}
		}
		isLast := int32(0)
		if cs.intf == 2 && sf == smplSubfrCount-1 {
			isLast = 1
		}
		r := SmplPercModel(cs.perc, xsubfr, xlen, frameMs, isLast, smelpPercRLen)
		even := make([]float32, smelpPercRLen)
		for i := 0; i < smelpPercRLen; i++ {
			var prev float32
			if i < len(cs.percPrev) {
				prev = cs.percPrev[i]
			}
			even[i] = 0.5 * (r[i] + prev)
		}
		corrs[sf-1] = even
		// Refresh the persistent prev-pair buffer in place.
		cs.percPrev = append(cs.percPrev[:0], r...)
		corrs[sf] = r
	}
	return corrs
}

// percCorrsToWght derives the per-subframe perc_wght_resp (length resp_len)
// from precomputed percCorrs for the given emphasis (smpl_perc_ac2a). Pure.
func percCorrsToWght(corrs [][]float32, emph [2]float32, respLen int) [][]float32 {
	emphSel := emph[0]
	if smelpCelpLowRate {
		emphSel = emph[1]
	}
	out := make([][]float32, len(corrs))
	for i, c := range corrs {
		out[i] = SmplPercAc2a(c, smelpPercRLen, emphSel, respLen, smplPercReg)
	}
	return out
}

// smplLsfInterpolSearch is the per-subframe residual + interpolated predcoef
// for lsf_interpol_idx 0, and the alternative idx 1 when it lowers the summed
// per-subframe residual RMS by the 0.998 margin. Returns (predcoefs,
// residual, chosen idx). At complexity 5-8 this search runs for every active
// frame.
func smplLsfInterpolSearch(brec []float32, prevLsfq []float32, winN []float32) ([smplSubfrCount][17]float32, []float32, int32) {
	residualFor := func(idx int) ([smplSubfrCount][17]float32, []float32, float32) {
		predcoefs, _ := smplLpcInterpolIdx(brec, prevLsfq, idx)
		res := make([]float32, smplIntfLen)
		var sumRms float32
		for sf := 0; sf < smplSubfrCount; sf++ {
			r := smplAnalysisResidualSubfr(&predcoefs[sf], winN, sf)
			var nrg float32
			for _, v := range r {
				nrg += v * v
			}
			sumRms += f32Sqrt(nrg + 1e-30)
			copy(res[sf*smplSubfrLen:(sf+1)*smplSubfrLen], r[:])
		}
		return predcoefs, res, sumRms
	}

	pc0, res0, rms0 := residualFor(0)
	// The alt interpolation runs whenever lsf_interpol_search && active &&
	// numsubfrs>1.
	pc1, res1, rms1 := residualFor(1)
	if rms1 < rms0*0.998 {
		return pc1, res1, 1
	}
	return pc0, res0, 0
}

// smplAnalysisResidualSubfr is the one-subframe residual under that
// subframe's interpolated predcoef (smpl_filt_ma16_monic over the sf-th
// 80-sample block of winN, which carries SMPL_ORDER lead history).
func smplAnalysisResidualSubfr(aSyn *[17]float32, winN []float32, sf int) [smplSubfrLen]float32 {
	var res [smplSubfrLen]float32
	for n := range res {
		idx := smplOrder + sf*smplSubfrLen + n
		acc := winN[idx]
		for j := 1; j <= smplOrder; j++ {
			acc += aSyn[j] * winN[idx-j]
		}
		res[n] = acc
	}
	return res
}

func smplSilentInternal(synthT *SmplSynthTables) candCandidate {
	var sym [16]int32
	for k := range sym {
		sym[k] = int32(len(synthT.Valtables[0][0][0][k]) / 2)
	}
	// Silent frame: lowest encodable gain (no pulses, so the exact value is
	// immaterial).
	gm, gd, _ := smplRateControlGains(0.0)
	return candCandidate{
		ip: SmplInternalParams{
			Lsf: SmplLsfParams{Stage1: 0, Grid: 0, Stage2: sym, Extra: 0},
			Gains: SmplGainParams{
				GainMain:  gm,
				GainDelta: gd,
				NrgRes:    [4]int32{-1, -1, -1, -1},
			},
		},
		grid:     0,
		qsym:     sym,
		pulseVec: make([]int32, smplIntfLen),
		pitch:    unvoicedPitch(),
		silent:   true,
	}
}

func smplAutocorr(x []float64, order int) []float64 {
	n := len(x)
	r := make([]float64, order+1)
	for lag := 0; lag <= order; lag++ {
		var s float64
		for i := lag; i < n; i++ {
			s += x[i] * x[i-lag]
		}
		r[lag] = s
	}
	return r
}

func smplBuildPulseParams(pulse []int32) SmplPulseParams {
	const p3 = 4
	posPer := smplIntfLen / p3 // 80
	var pp SmplPulseParams
	for sf := 0; sf < p3; sf++ {
		var s int32
		for n := sf * posPer; n < (sf+1)*posPer; n++ {
			v := pulse[n]
			if v < 0 {
				v = -v
			}
			s += v
		}
		pp.Subfr[sf] = s
	}
	for _, v := range pp.Subfr {
		pp.Total += v
	}

	var magRuns, signs []int32
	for sf := 0; sf < p3; sf++ {
		if pp.Subfr[sf] <= 0 {
			continue
		}
		basePos := posPer * sf
		type posMag struct {
			p   int
			mag int32
		}
		var positions []posMag
		for n := basePos; n < basePos+posPer; n++ {
			if pulse[n] != 0 {
				positions = append(positions, posMag{n, pulse[n]})
			}
		}
		runPos := int32(basePos)
		first := true
		for _, pm := range positions {
			magv := pm.mag
			mag := magv
			if mag < 0 {
				mag = -mag
			}
			var m int32
			if first {
				m = int32(pm.p) - int32(basePos)
			} else {
				m = int32(pm.p) - runPos
			}
			magRuns = append(magRuns, m)
			runPos = int32(pm.p)
			if mag > 1 {
				for i := int32(0); i < mag-1; i++ {
					magRuns = append(magRuns, 0)
				}
			}
			if magv < 0 {
				signs = append(signs, -1)
			} else {
				signs = append(signs, 1)
			}
			first = false
		}
	}
	pp.MagRuns = magRuns

	// SIGN block: batch signs into raw symbols (<=15 bits each, MSB-first).
	numPos := len(signs)
	var signSyms []SmplRawSym
	p := 0
	for p < numPos {
		nbits := numPos - p
		if nbits > 15 {
			nbits = 15
		}
		var sym uint32
		for q := 0; q < nbits; q++ {
			var bit uint32
			if signs[p+q] > 0 {
				bit = 1
			}
			sym |= bit << uint(nbits-1-q)
		}
		signSyms = append(signSyms, SmplRawSym{Sym: sym, Nbits: uint32(nbits)})
		p += nbits
	}
	pp.SignSyms = signSyms
	return pp
}

// smplRateControlGains finds the (gainMain, gainDelta, reconstructed gainQ)
// whose linear gain is closest to targetLinear.
func smplRateControlGains(targetLinear float64) (int32, int32, int32) {
	cc := CcTablesTables()
	cfgSel := int32(2)
	cb1 := cc.NrgStep(cfgSel)
	bestD := math.Inf(1)
	var bgm, bgd, bgq int32
	for gm := int32(0); gm < 84; gm++ {
		base7 := gm*cb1 - 0x154000
		for gd := int32(0); gd < 98; gd++ {
			cbv := cc.GainRecon(true, 4*gd)
			gq := base7 + (cbv << 4)
			d := math.Abs(SmplGainLin(gq) - targetLinear)
			if d < bestD {
				bestD = d
				bgm, bgd, bgq = gm, gd, gq
			}
		}
	}
	return bgm, bgd, bgq
}

// buildLtpBuf rolls the persistent perceptually-weighted speech buffer
// (ltp_buf) and writes this internal frame's weighted speech into its tail:
// shift left by framelen, then per CELP subframe i apply the 17-tap monic
// perceptual MA of the HP frame under resp_pitch[i], plus the lookahead tail
// under resp_pitch[3].
func buildLtpBuf(cs *celpFrameCtx, percCorrs [][]float32) {
	respPitch := percCorrsToWght(percCorrs,
		[2]float32{smelpPercEmphPitch, smelpPercEmphPitch}, smelpPitchPercRespLen)
	maxLen := maxLtpBufLen // 659
	look := smelpPitchLookaheadLen
	framelen := smplIntfLen
	// Shift existing weighted speech left by one internal frame.
	keep := maxLen - framelen - look
	copy(cs.ltpBuf, cs.ltpBuf[framelen:framelen+keep])
	// HP sample at internal-frame-relative index idx (xhp_frame origin),
	// reaching into the previous packet's tail (hp_pitch_hist) for idx<0.
	frameStart := int(cs.intf)*smplIntfLen - smplWinNextWbLen
	hist := smelpPitchLagMax
	sample := func(rel int) float32 {
		idx := frameStart + rel
		if idx >= 0 {
			if idx < len(cs.hpN) {
				return cs.hpN[idx]
			}
			return 0.0
		}
		if len(cs.hpPitchHist) == hist {
			k := idx + hist
			if k >= 0 {
				return cs.hpPitchHist[k]
			}
		}
		return 0.0
	}
	// w_speech write origin in ltp_buf.
	wOrigin := maxLen - smplSubfrCount*smplSubfrLen - look // 332
	for i := 0; i < smplSubfrCount; i++ {
		coef := respPitch[i]
		for n := 0; n < smplSubfrLen; n++ {
			pos := i*smplSubfrLen + n
			res := sample(pos) // monic coef[0]==1
			for j := 1; j < smelpPitchPercRespLen; j++ {
				res += coef[j] * sample(pos-j)
			}
			cs.ltpBuf[wOrigin+i*smplSubfrLen+n] = res
		}
	}
	// Lookahead tail under the last subframe's response.
	coef := respPitch[smplSubfrCount-1]
	for n := 0; n < look; n++ {
		pos := framelen + n
		res := sample(pos)
		for j := 1; j < smelpPitchPercRespLen; j++ {
			res += coef[j] * sample(pos-j)
		}
		cs.ltpBuf[maxLen-look+n] = res
	}
}

// smplAnalyzeInternal analyzes one internal frame: compute the shared
// perceptual autocorrelation, build the perceptually-weighted ltp_buf, run
// the faithful multi-stage pitch estimator + the voicing classifier, then
// build the voiced (LTP) or unvoiced candidate, commit it to the shadow synth
// st, and advance the entropy predictor mirror.
func smplAnalyzeInternal(
	synthT *SmplSynthTables,
	st *SmplFrameSynth,
	lstate *SmplLsfState,
	intf int,
	win []float64,
	winN []float32,
	prevNLSF []float32,
	fe *frontEndLsf,
	cs *celpFrameCtx,
) (SmplInternalParams, []float32, bool) {
	mem := SmplMemTables()

	// Shared perceptual autocorrelation (advances perc state EXACTLY ONCE per
	// frame); both the pitch weighting and the CELP weighting derive from it.
	corrs := computePercCorrs(cs)
	cs.percCorrs = corrs[:]

	// Roll the persistent perceptually-weighted speech buffer, then run the
	// faithful multi-stage pitch estimator.
	buildLtpBuf(cs, cs.percCorrs)
	f2 := cs.f2
	pr := SmplPitch(cs.pitchEst, cs.ltpBuf, &f2, cs.codedAsActiveVoice)
	pitchcorr := pr.Pitchcorr
	avgLag := pr.AvgLag
	harm := pr.HarmStrength
	lags8 := pr.Lags
	// The single representative lag the voiced encode path uses; the C's wire
	// contour is anchored on the first-subframe lag.
	lagSamples := pr.Lags[0]
	sp := cs.spActProb
	vstr := SmplGetSignalMode(pitchcorr, lags8[:], avgLag, harm, &f2, sp, cs.vuv)
	cs.voicingStrength = vstr
	if cs.dbg != nil {
		ltpCopy := make([]float32, len(cs.ltpBuf))
		copy(ltpCopy, cs.ltpBuf)
		f2Copy := make([]float32, len(f2))
		copy(f2Copy, f2[:])
		*cs.dbg = append(*cs.dbg, SmplEncDbg{
			LtpBuf: ltpCopy, F2: f2Copy, Pitchcorr: pitchcorr, AvgLag: avgLag,
			Harm: harm, Vstr: vstr, Laginds: pr.Laginds, BlocksegIdx: pr.BlocksegIdx,
		})
	}
	isVoicedDecision := vstr > 0.0 && cs.codedAsActiveVoice
	if isVoicedDecision {
		lstate.PrevLagSamples = lagSamples
	} else {
		lstate.PrevLagSamples = 0.0
	}
	// The C resets the lag-block predictor after an unvoiced frame (and after
	// each packet's last frame, handled at the call site).
	if !isVoicedDecision {
		cs.pitchEst.ResetCond()
		lags8 = [8]float32{}
	}

	// The CELP excitation encoder advances its per-subframe acb/zir/prev-idx
	// state, so it must run EXACTLY ONCE per internal frame with the lags of
	// the committed decision.
	voicedLstate := *lstate
	AdvanceLsfState(&voicedLstate, intf, 1)
	var vd *voicedDecision
	if isVoicedDecision {
		vd = smplVoicedDecisionForLag(pr.BlocksegIdx, &pr.Laginds, cs, &lags8)
	}

	var chosen candCandidate
	var chosenLstate *SmplLsfState
	var isVoiced bool
	if vd != nil {
		chosen = smplVoicedCandidate(synthT, winN, prevNLSF, fe, cs, vd)
		chosenLstate = &voicedLstate
		isVoiced = true
	} else {
		chosen = smplUnvoicedCandidate(synthT, st, win, winN, prevNLSF, fe, cs)
	}
	committedNlsf := commitCandidate(synthT, st, &chosen, prevNLSF)
	if chosen.stage1 == 1 {
		*lstate = *chosenLstate
		smplReplayPitchState(mem, lstate, 4, chosen.ip.Pulses.Subfr, &chosen.ip.Pitch)
	} else {
		AdvanceLsfState(lstate, intf, chosen.stage1)
	}
	return chosen.ip, committedNlsf, isVoiced
}

// smplReplayPitchState advances the predictor mirror exactly as
// encode_smpl_pitch does, without entropy coding, so the analysis predicts
// the lag/gain predictor for the next internal frame.
func smplReplayPitchState(_mem *SmplMem, st *SmplLsfState, p3 int32, subfrCounts [4]int32, pp *SmplPitchParams) {
	for sf := 0; sf < int(p3) && sf < 4; sf++ {
		st.PrevGainIdx = pp.GainIdx[sf]
		if subfrCounts[sf] > 0 {
			st.PrevFiltIdx = pp.FiltIdx[sf]
		}
	}
	tab := LoadPitchTables()
	nblk, nidx := SmplLagsPredictorAfter(tab, pp.BlocksegIdx, &pp.Laginds)
	st.PrevLagblk = nblk
	st.PrevLagidx = nidx
}

// voicedDecision is the committed voiced decision for one internal frame: the
// encodable pitch params and the per-subframe synthesis lag carried in pitch.
type voicedDecision struct {
	pp    SmplPitchParams
	pitch SmplPitchSynth
}

// smplVoicedDecisionForLag carries the estimator's full per-block contour
// (blockseg_idx + laginds) into the voiced decision: the wire pitch encode
// writes them straight through smpl_encode_lags, and the CELP ACB basis uses
// the SAME per-block lags (lag = laginds*0.5 + 32) so the encoder/decoder LTP
// contributions agree. The gain/filter indices here are placeholders; the
// voiced candidate overwrites them with the real CELP acb_idx/fcb_idx per
// subframe.
func smplVoicedDecisionForLag(blocksegIdx int, laginds *[8]int32, cs *celpFrameCtx, lags8 *[8]float32) *voicedDecision {
	// The decoder maps each 40-block lag index lag = laginds*0.5 +
	// SMPL_MIN_PITCH_LAG, clamped <= 320.
	var blockLags8 [8]float32
	for b := 0; b < 8; b++ {
		v := float32(laginds[b])*0.5 + 32.0
		if v > 320.0 {
			v = 320.0
		}
		blockLags8[b] = v
	}
	*lags8 = blockLags8
	for sf := 0; sf < smplSubfrCount; sf++ {
		cs.blockLags[sf] = [2]float32{blockLags8[2*sf], blockLags8[2*sf+1]}
	}
	var meanLag float32
	for _, v := range blockLags8 {
		meanLag += v
	}
	meanLag /= 8.0

	pp := SmplPitchParams{
		GainIdx:     [4]int32{5, 5, 5, 5},
		BlocksegIdx: blocksegIdx,
		Laginds:     *laginds,
	}
	pitch := SmplPitchSynth{
		Voiced:   true,
		LagSubfr: [4]float64{float64(meanLag), float64(meanLag), float64(meanLag), float64(meanLag)},
		NormGain: smplVoicedNormGain,
	}
	return &voicedDecision{pp: pp, pitch: pitch}
}

// smplVoicedCandidate builds the voiced (stage1=1 + LTP) candidate for one
// internal frame. The real CELP voiced encoder runs with the decoder-
// reconstructed per-block lags (so its ACB basis matches the decoder's), and
// its outputs drive the wire: pulses[MAIN] -> the pulse train,
// acb_idx[MAIN] -> the wire gain_idx (ACB/LTP gain), gain_idx[MAIN] -> the
// wire filt_idx (voiced FCB gain).
func smplVoicedCandidate(synthT *SmplSynthTables, winN []float32, prevNLSF []float32, fe *frontEndLsf, cs *celpFrameCtx, vd *voicedDecision) candCandidate {
	gainQ := [4]int32{0, 0, 0, 0} // voiced synthesis uses the ACB+FCB excitation

	// Voiced-grid LSF: bit-exact C quantizer fed the faithful front-end NLSF.
	bgrid, bsym, brec, _ := fe.quantize(synthT, 1, prevNLSF)

	// Per-subframe interpolated LPC (same as the unvoiced path).
	predcoefs, _ := smplLpcInterpol(brec, fe.prevLsfq)
	resLpc := make([]float32, smplIntfLen)
	for sf := 0; sf < smplSubfrCount; sf++ {
		r := smplAnalysisResidualSubfr(&predcoefs[sf], winN, sf)
		copy(resLpc[sf*smplSubfrLen:(sf+1)*smplSubfrLen], r[:])
	}

	// Real voiced CELP: with nonzero lags the encoder runs the ACB/LTP path.
	blockLags := cs.blockLags
	celpOut := runCelpSubframes(cs, &predcoefs, resLpc, &blockLags, cs.percCorrs, smplPercEmphV, 1)

	// Unpack the MAIN-rate pulses into a per-position train; collect acb/fcb
	// indices per subframe.
	const main = 1
	pulseVec := make([]int32, smplIntfLen)
	var acbg, fcbg [4]int32
	for sf := 0; sf < smplSubfrCount; sf++ {
		out := &celpOut[sf]
		for _, v := range out.Pulses[main] {
			sign := 1 + 2*(int32(v)>>15)
			pos := int32(v)*sign - 1
			if pos >= 0 && pos < smplSubfrLen {
				pulseVec[sf*smplSubfrLen+int(pos)] += sign
			}
		}
		// acb_idx is always coded; fcb (filt) only where pulses exist. Clamp
		// to the wire ranges.
		acbg[sf] = clampI32(int32(out.AcbIdx[main]), 0, 15)
		fcbg[sf] = maxI32(int32(out.GainIdx[main]), 0)
	}
	ppPulses := smplBuildPulseParams(pulseVec)
	subfr := ppPulses.Subfr
	pp := vd.pp
	pp.GainIdx = acbg
	for sf := 0; sf < 4; sf++ {
		if subfr[sf] > 0 {
			pp.FiltIdx[sf] = fcbg[sf]
		} else {
			pp.FiltIdx[sf] = -1
		}
	}

	return candCandidate{
		ip: SmplInternalParams{
			Lsf: SmplLsfParams{
				Stage1: 1,
				Grid:   bgrid,
				Stage2: bsym,
				Extra:  0,
			},
			Pulses: ppPulses,
			Pitch:  pp,
		},
		stage1:   1,
		grid:     bgrid,
		qsym:     bsym,
		pulseVec: pulseVec,
		gainQ:    gainQ,
		pitch:    vd.pitch,
	}
}

// SmplAnalyzeFrameSt turns one 60 ms PCM frame (960 f32 @16 kHz, ~[-1,1])
// into params, advancing es.
func SmplAnalyzeFrameSt(es *SmplEncoderState, pcm []float32) SmplFrameParams {
	need := smplIntfLen * 3
	if len(pcm) < need {
		padded := make([]float32, need)
		copy(padded, pcm)
		pcm = padded
	}
	synthT := &LsfBuiltTables().Synth

	// SILK VAD on the int16 input PCM (runs on the raw API samples, before
	// the encoder HP).
	es.vadPcm = es.vadPcm[:0]
	for _, s := range pcm[:need] {
		v := f32Round(s * 32768.0)
		if v < -32768.0 {
			v = -32768.0
		} else if v > 32767.0 {
			v = 32767.0
		}
		es.vadPcm = append(es.vadPcm, int16(v))
	}
	if es.vad == nil {
		es.vad = NewSmplVadState()
	}
	vad := es.vad.ProcessPacket(es.vadPcm, smplIntfLen)
	spActProb := vad.VadResults
	codedAsActiveVoice := vad.CodedAsActiveVoice

	// Encoder input high-pass (ARMA2, fcorner 35 Hz), matching the real
	// encoder.
	if es.hpCoefs == nil {
		ma, ar := SmplGetHpCoefs(smelpEncHpFcornerHz)
		es.hpCoefs = &[2][3]float32{ma, ar}
	}
	hpMA, hpAR := es.hpCoefs[0], es.hpCoefs[1]
	es.hp = make([]float32, need)
	SmplFiltARMA2(pcm[:need], need, hpMA, hpAR, &es.hpState, es.hp)

	// int16-scaled input with smplOrder lead samples of history.
	es.x = make([]float64, smplOrder+need)
	if len(es.hist) >= smplOrder {
		copy(es.x[:smplOrder], es.hist[len(es.hist)-smplOrder:])
	}
	for i := 0; i < need; i++ {
		es.x[smplOrder+i] = float64(es.hp[i]) * 32768.0
	}

	shadow := NewSmplFrameSynth()
	var prevNLSF []float32
	// Predictor mirror, fresh per 60 ms frame (mirrors encode_smpl_frame's
	// fresh SmplLsfState), threaded across the 3 internal frames.
	lstate := &SmplLsfState{}

	// Lazily build the persistent CELP encoder + perceptual model.
	if es.celp == nil {
		es.celp = NewCelpEncoder(smelpCelpLowRate, smelpCelpPercRespLen, smelpCelpFcbSubfrLen, smelpCelpSubfrPerPack)
	}
	if es.perc == nil {
		es.perc = NewPercModelState()
	}
	if es.bitrate == nil {
		es.bitrate = NewBitrateController()
	}
	if len(es.percPrev) != smelpPercRLen {
		es.percPrev = make([]float32, smelpPercRLen)
	}

	// Normalized HP input for the CELP residual (the real encoder works in
	// [-1,1], not int16). xhp_frame for internal frame 0 starts
	// SMPL_WINNEXT_WB_LEN (32) samples BEFORE the packet's first sample.
	resLead := smplOrder + smplWinNextWbLen
	es.xn = make([]float32, resLead+need)
	if len(es.hist) >= resLead {
		for i := 0; i < resLead; i++ {
			es.xn[i] = float32(es.hist[len(es.hist)-resLead+i] / 32768.0)
		}
	}
	copy(es.xn[resLead:resLead+need], es.hp[:need])

	// Full HP-domain buffer that lpcbuf indexes:
	// [history(144)] ++ [current 960 HP] ++ [32 zeros].
	es.hpFull = make([]float32, smplLpcHistLen+need+smplWinNextWbLen)
	if len(es.lpcHist) == smplLpcHistLen {
		copy(es.hpFull[:smplLpcHistLen], es.lpcHist)
	}
	copy(es.hpFull[smplLpcHistLen:smplLpcHistLen+need], es.hp[:need])

	// Snapshot the previous packet's HP tail (pitch history for this packet's
	// intf=0), then refresh it from this packet's tail for the next call.
	hpPitchHist := es.hpPitchHist
	if len(hpPitchHist) != smelpPitchLagMax {
		hpPitchHist = make([]float32, smelpPitchLagMax)
	}

	// Lazily size the persistent perceptually-weighted speech buffer.
	if len(es.ltpBuf) != maxLtpBufLen {
		es.ltpBuf = make([]float32, maxLtpBufLen)
	}

	prevLsfq := es.prevLsfq
	prevVoiced := es.prevVoiced

	var internal [3]SmplInternalParams
	for f := 0; f < 3; f++ {
		base := smplOrder + f*smplIntfLen
		win := es.x[base-smplOrder : base+smplIntfLen]
		// win_n carries res_lead samples before the internal frame so the
		// residual can start res_pre samples early.
		nbase := resLead + f*smplIntfLen
		winN := es.xn[nbase-resLead : nbase+smplIntfLen]

		// Front-end LPC analysis: window lpcbuf (448 samples starting 96
		// before this frame), FFT-autocorrelate it, and derive A/NLSF.
		// use_long_win is true except the last frame.
		lpcStart := smplLpcHistLen - smplLpcPre + f*smplIntfLen
		var lpcbuf [smplLpcBufLen]float32
		copy(lpcbuf[:], es.hpFull[lpcStart:lpcStart+smplLpcBufLen])
		windowed := smplWindowLpc20(lpcbuf[:], f < 2)
		if es.lpcFft == nil {
			es.lpcFft = newFftScratch(smplLpcNfft)
		}
		a, f2 := smplLpcAnalyzeWithF2(windowed, es.lpcFft)
		nlsf := smplA2NLSF16(a[:])

		cs := &celpFrameCtx{
			celp:               es.celp,
			perc:               es.perc,
			percPrev:           es.percPrev,
			bitrate:            es.bitrate,
			hpN:                es.hp,
			intf:               f,
			spActProb:          spActProb[f],
			codedAsActiveVoice: codedAsActiveVoice,
			f2:                 f2,
			vuv:                &es.vuv,
			hpPitchHist:        hpPitchHist,
			ltpBuf:             es.ltpBuf,
			pitchEst:           &es.pitchEst,
			dbg:                es.DbgCaptures,
		}
		fe := &frontEndLsf{
			a:          a,
			nlsf:       nlsf,
			prevLsfq:   prevLsfq,
			prevVoiced: prevVoiced,
			intf:       f,
		}
		ip, nlsfOut, voicedOut := smplAnalyzeInternal(synthT, shadow, lstate, f,
			win, winN, prevNLSF, fe, cs)
		prevNLSF = nlsfOut
		prevLsfq = nlsfOut
		prevVoiced = voicedOut
		internal[f] = ip
		// The C resets the lag-block predictor after the last internal frame
		// of each packet (and after any unvoiced frame, handled in
		// smpl_analyze_internal), so cond-coding restarts per packet.
		if f == 2 {
			cs.pitchEst.ResetCond()
		}
	}

	// Carry SMPL_ORDER + SMPL_WINNEXT_WB_LEN history so the next packet's
	// residual lead is filled.
	es.hist = append(es.hist[:0], es.x[len(es.x)-(smplOrder+smplWinNextWbLen):]...)
	// Carry the last 144 HP samples as next packet's LPC window history
	// (mirrors lpc_buf_mem).
	es.lpcHist = append(es.lpcHist[:0], es.hp[need-smplLpcHistLen:need]...)
	copy(hpPitchHist, es.hp[need-smelpPitchLagMax:need])
	es.hpPitchHist = hpPitchHist
	es.prevLsfq = prevLsfq
	es.prevVoiced = prevVoiced
	return SmplFrameParams{
		Toc:      0x50,
		Config:   0,
		Internal: internal,
	}
}
