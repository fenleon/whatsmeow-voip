// MLow perceptual-weighting front-end: FFT-based perceptual autocorrelation
// (SmplPercModel), the emphasis -> LPC response conversion (SmplPercAc2a), and
// the per-subframe pulse budget / subframe importance from the bitrate
// controller. Ported from whatsapp-rust (MIT)
// wacore/src/voip/mlow/smpl_perc.rs; MPL-2.0 for this port.
//
// The mixed-radix FFT this module uses lives in fft.go (same source file, same
// arithmetic). The two smpl_harmcomb.rs helpers (smpl_get_hp_coefs /
// smpl_filt_arma2) were ported earlier into harmcomb.go as SmplGetHpCoefs /
// SmplFiltARMA2 and are reused from here, not duplicated.
//
// The spectrum uses the ordered real layout documented in fft.go: f[0]=DC,
// f[1]=Nyquist, then interleaved [re,im] for bins 1..N/2-1, so smthFilt and
// the inverse index identical bins. PERCW_NFFT is 576 = 2^6 * 3^2, NOT a power
// of two, hence the mixed-radix DFT.
//
// Small leaf helpers (smplNrg, smplFiltMA2, genSinWin/genCosWin, smplSigmoid)
// already exist in gennoise.go / lpc.go with identical arithmetic and are
// reused.
package mlow

import "math"

const (
	percwNfft  = 512 + 64 // 576
	percwFsKhz = 16.0

	smplPercMaskSmth = 0.1158
	smplPercMelFcHz  = 320.0

	smplWinNextWbLen     = 16 * 2  // 32
	smplWinNextWbLongLen = 16 * 4  // 64
	smplWinPrevPercLen   = 16 * 12 // 192
	smplPercWin1_10MsLen = 192
	smplPercWin1_20MsLen = 352

	smplTrue  = 1
	smplFalse = 0

	smplPercReg float32 = 1e-3

	// Bitrate-controller constants
	smplRateContScale float32 = 26.0
	smplE             float32 = 2.7182818284590

	// SmplFrameTypes
	frameBackgroundNoise = 0
	frameUnvoiced        = 1
	frameVoiced          = 2
)

// Perceptual emphasis coefficients, exposed for the caller selecting voiced
// vs unvoiced emphasis.
var (
	smplPercEmphV  = [2]float32{-0.72, -0.77}
	smplPercEmphUV = [2]float32{-0.55, -0.6}
)

// [lowRate][BACKGROUND_NOISE/UNVOICED/VOICED]
var smplMaxPulsesPerFrame = [2][3]uint8{{80, 160, 160}, {16, 32, 32}}

// [framelenidx][lowrate][8]
var smplRateControlModelComp5 = [4][2][8]float32{
	{
		{
			5.166876656946171,
			-8.981699804753452,
			0.07280811614105594,
			0.1301196310618402,
			-0.01597680442864421,
			1.7601470147884113,
			-3.8161195433141755,
			0.3038629198331684,
		},
		{
			-71.71229978402292,
			14.197572549553076,
			-0.9863630205846172,
			0.032124893286072924,
			-0.0003538411576874928,
			1.803705259861388e-11,
			10.0,
			1.2454667523627154,
		},
	},
	{
		{
			32.5371190670542,
			-41.270234279452104,
			10.490270829170875,
			-1.102121269442237,
			0.03848319274046071,
			3.405326741403831,
			-5.102658181889428,
			0.2141935195026695,
		},
		{
			-177.10486363500775,
			43.952329593498376,
			-3.7049735533247454,
			0.14239771116996938,
			-0.001919963993993193,
			7.953695588409639e-6,
			5.220317075476664,
			0.6435364076926223,
		},
	},
	{
		{
			-79.2663194911617,
			45.00981883522089,
			-10.063311543498518,
			1.2311531056576501,
			-0.06023559069137118,
			0.059204788212259364,
			3.033961466462233,
			1.0111383197827808,
		},
		{
			-122.04861900525415,
			31.62096398905459,
			-2.613237037423586,
			0.10050433143234094,
			-0.0013233009240188039,
			2.14859438836692e-7,
			1.9077791307787761,
			0.7059420500333776,
		},
	},
	{
		{
			-182.64255084224325,
			122.90780796179816,
			-31.308790671748525,
			3.7850563849431462,
			-0.1750480676903051,
			0.05399618467364628,
			3.009451055091342,
			1.1243365512229038,
		},
		{
			-132.4565456943888,
			34.361297004632966,
			-2.7956546289118887,
			0.10428149547078584,
			-0.001322667891395693,
			2.678747426340249e-6,
			6.9940208056381925,
			0.7551244069345737,
		},
	},
}

// [framelenidx][lowrate]
var smplRateControlThrsComp5 = [4][2]uint16{
	{7500, 10000}, {4500, 5750}, {4000, 5000}, {4000, 4750},
}

// f32Pow is f32 powf, computed in double and rounded, matching the package's
// other f32 libm wrappers (f32Sqrt, f32Ln, ...).
func f32Pow(x, y float32) float32 { return float32(math.Pow(float64(x), float64(y))) }

// percWindows holds the perc-relevant window tables, built once per state.
type percWindows struct {
	percWin1_10ms []float32 // genSinWin, len 192
	percWin1_20ms []float32 // genSinWin, len 352
	win3Short     []float32 // genCosWin, len 32
	win3Long      []float32 // genCosWin, len 64
}

func newPercWindows() *percWindows {
	return &percWindows{
		percWin1_10ms: genSinWin(smplPercWin1_10MsLen),
		percWin1_20ms: genSinWin(smplPercWin1_20MsLen),
		win3Short:     genCosWin(smplWin3ShortLen),
		win3Long:      genCosWin(smplWin3LongLen),
	}
}

// smplWindowPerc is the windowing for the perc case (use_lpc_win == FALSE).
func smplWindowPerc(win *percWindows, input, out []float32, n int, frameMs int32, useLongWin bool) {
	win1len := smplPercWin1_20MsLen
	win1 := win.percWin1_20ms
	if frameMs == 10 {
		win1len = smplPercWin1_10MsLen
		win1 = win.percWin1_10ms
	}
	win3len := smplWin3ShortLen
	win3 := win.win3Short
	if useLongWin {
		win3len = smplWin3LongLen
		win3 = win.win3Long
	}

	for i := 0; i < win1len; i++ {
		out[i] = win1[i] * input[i]
	}
	mid := n - win1len - smplWin3LongLen
	copy(out[win1len:win1len+mid], input[win1len:win1len+mid])
	for i := 0; i < win3len; i++ {
		out[n-smplWin3LongLen+i] = win3[i] * input[n-smplWin3LongLen+i]
	}
	if !useLongWin {
		start := n - smplWin3LongLen + smplWin3ShortLen
		for i := start; i < n; i++ {
			out[i] = 0.0
		}
	}
}

// smthFilt is the bidirectional masking smooth across the power spectrum.
func smthFilt(f []float32, smthcoef []float32) {
	half := percwNfft / 2
	f2smth := f[0]
	for i := 1; i < half; i++ {
		f2new := f[2*i]
		f2smth = f2new + smthcoef[i]*(f2smth-f2new)
		f[2*i] = f2smth
	}
	f[1] = f[1] + smthcoef[half]*(f2smth-f[1])
	f2smth = f[1]
	for i := half - 1; i > 0; i-- {
		f2new := f[2*i]
		f2smth = f2new + smthcoef[i]*(f2smth-f2new)
		f[2*i] = f2smth
	}
	f[0] = f[0] + smthcoef[0]*(f2smth-f[0])
}

// PercModelState carries the buf history (length PERCW_NFFT) across
// SmplPercModel calls, plus the reusable per-call FFT/windowing scratch so the
// perceptual model allocates nothing per frame.
type PercModelState struct {
	buf      [percwNfft]float32
	smthcoef []float32 // length percwNfft/2 + 1
	windows  *percWindows
	fft      *fftScratch
	bufWin   []float32 // length percwNfft, windowed time-domain frame
	f        []float32 // length percwNfft, ordered REAL spectrum / power
}

func NewPercModelState() *PercModelState {
	// Per-bin mel-width smoothing coefficients.
	fsStep := (percwFsKhz * 1000.0) / float32(percwNfft)
	smthcoef := make([]float32, percwNfft/2+1)
	for i := range smthcoef {
		percWidthPerBin := smplPercMaskSmth * (fsStep*float32(i) + smplPercMelFcHz) / fsStep
		smthcoef[i] = percWidthPerBin / (percWidthPerBin + 1.0)
	}
	return &PercModelState{
		smthcoef: smthcoef,
		windows:  newPercWindows(),
		fft:      newFftScratch(percwNfft),
		bufWin:   make([]float32, percwNfft),
		f:        make([]float32, percwNfft),
	}
}

// SmplPercModel: windowed power spectrum -> bidirectional masking smooth ->
// inverse -> 1/NFFT scale. Returns the first rLen autocorrelation lags. The
// buf history advances by dropping the consumed front, keeping the tail, and
// appending the new subframe.
func SmplPercModel(sc *PercModelState, xsubfr []float32, xlen int, frameMs int32, isLast int32, rLen int) []float32 {
	// xsubfr_len <= PERCW_NFFT (debug_assert in the Rust).
	// Advance history: drop the consumed front, keep the (NFFT - xlen) tail,
	// then append xsubfr. Source offset is xlen - (WB_LONG_LEN - WB_LEN) = xlen - 32.
	srcOff := xlen - (smplWinNextWbLongLen - smplWinNextWbLen)
	keep := percwNfft - xlen
	copy(sc.buf[:keep], sc.buf[srcOff:srcOff+keep])
	copy(sc.buf[keep:keep+xlen], xsubfr[:xlen])

	winlen := smplWinPrevPercLen + int(frameMs)*16 + smplWin3LongLen
	skipSamples := percwNfft - winlen
	// frame_ms == 10 || skip_samples == 0 (debug_assert in the Rust).

	// Reuse the persistent scratch; re-zero bufWin so the windowed frame
	// matches a fresh zeroed buffer (the skip region plus any window bins the
	// windowing leaves untouched).
	for i := range sc.bufWin {
		sc.bufWin[i] = 0.0
	}
	smplWindowPerc(sc.windows, sc.buf[skipSamples:], sc.bufWin[skipSamples:], winlen, frameMs, isLast == smplFalse)

	rfftForwardOrderedSC(sc.bufWin, sc.f, sc.fft)
	f := sc.f
	f[0] = f[0] * f[0] // DC
	f[1] = f[1] * f[1] // Nyquist
	for i := 1; i < percwNfft/2; i++ {
		f[2*i] = f[2*i]*f[2*i] + f[2*i+1]*f[2*i+1]
		f[2*i+1] = 0.0
	}
	smthFilt(f, sc.smthcoef)
	rfftBackwardOrderedSC(sc.f, sc.bufWin, sc.fft)

	r := make([]float32, rLen)
	scale := 1.0 / float32(percwNfft)
	for i := 0; i < rLen; i++ {
		r[i] = sc.bufWin[i] * scale
	}
	return r
}

// smplPercAc2RcDbl: autocorrelation -> reflection coefficients (Levinson,
// double precision). Same algorithm as lpc.go's ac2rcDbl but with the
// regularization arithmetic of smpl_perc.rs: reg is widened to f64 BEFORE the
// (1.0 + reg) multiply, where smpl_lpc.rs computes 1.0 + reg in f32 first.
func smplPercAc2RcDbl(corr []float64, order int, reg float64, rc []float32) {
	c0 := make([]float64, order+1)
	c1 := make([]float64, order+1)
	copy(c0, corr[:order+1])
	c0[0] *= 1.0 + reg
	copy(c1, c0)
	for i := 0; i < order; i++ {
		rc[i] = 0.0
	}
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

// smplAc2Rc promotes to double precision before Levinson.
func smplAc2Rc(corr []float32, order int, reg float32, rc []float32) {
	corrDbl := make([]float64, order+1)
	for i := 0; i <= order; i++ {
		corrDbl[i] = float64(corr[i])
	}
	smplPercAc2RcDbl(corrDbl, order, float64(reg), rc)
}

// SmplPercAc2a: ma2 (b={pe, 1+pe^2, pe}) on R[1..] then Levinson + rc2a ->
// A[0..percRespLen].
func SmplPercAc2a(r []float32, rLen int, percEmph float32, percRespLen int, reg float32) []float32 {
	// rLen >= percRespLen+1 and smplMaxLResp >= percRespLen (debug_asserts).
	b := [3]float32{percEmph, 1.0 + percEmph*percEmph, percEmph}
	state := [2]float32{r[0], r[1]}
	r_ := make([]float32, smplMaxLResp)
	smplFiltMA2(r[1:], percRespLen, &b, &state, r_)

	rc := make([]float32, smplMaxLResp)
	smplAc2Rc(r_, percRespLen-1, reg, rc)

	a := make([]float32, percRespLen)
	rc2a(rc, percRespLen-1, a)
	return a
}

// Bitrate-controller

func bitrate2pulses(rateKbps float32, coeff *[8]float32) float32 {
	return coeff[0] +
		coeff[1]*rateKbps +
		coeff[2]*rateKbps*rateKbps +
		coeff[3]*f32Pow(rateKbps, 3.0) +
		coeff[4]*f32Pow(rateKbps, 4.0) +
		coeff[5]*f32Pow(smplE, (rateKbps-coeff[6])*coeff[7])
}

func bitrate2pulsesHrFec(rateKbps float32, coeff *[8]float32, onePulseRateBps float32) float32 {
	const rateThresKbps = 9.0 // only compensate below this value
	if rateKbps >= rateThresKbps {
		return bitrate2pulses(rateKbps, coeff)
	} else if onePulseRateBps >= rateThresKbps*1000.0 {
		return 1.0
	}
	pulsesThres := bitrate2pulses(rateThresKbps, coeff)
	sc := (rateThresKbps - rateKbps) / (rateThresKbps - onePulseRateBps/1000.0)
	return pulsesThres - sc*(pulsesThres-1.0)
}

// BitrateControllerInputs are the encoder-control inputs the controller reads.
type BitrateControllerInputs struct {
	InternalSampleRate       int32
	PayloadSizeMs            int32
	FecBitRate               int32
	MainBitRate              int32
	Complexity               int32
	UseFecRateCompensation   int32
	UseDtx                   int32
	SubFrameImportanceFactor float32
}

// BitrateController state carried across frames.
type BitrateController struct {
	prevVoiced           int32
	rateContWnrgSmth     float32
	rateContBitrateScale [celpMaxRates]float32
	bitrateDeltaSmth     [celpMaxRates]float32 // parity scaffolding, not read yet
	rateContBitrate      [celpMaxRates]float32
	adjustmentFactor     [celpMaxRates]float32
}

// NewBitrateController: controller init plus zeroed state.
func NewBitrateController() *BitrateController {
	return &BitrateController{
		adjustmentFactor: [celpMaxRates]float32{1.0, 1.0},
	}
}

// Control returns (max_pulses_per_subfr, subfr_importance).
func (b *BitrateController) Control(
	enc *BitrateControllerInputs,
	dtxSidFrame int32,
	codedAsActiveVoice int32,
	spActProb float32,
	nonflatness float32,
	voicingStrength float32,
	voiced int32,
	wnrg float32,
	wnrgNext float32,
	lowRate int32,
	framelen int32,
	subfrlen int32,
) (maxPulsesPerSubfr [celpMaxRates]int16, subfrImportance [celpMaxRates]float32) {
	// sp_act_prob in [0, 1] (debug_assert in the Rust).

	bweBitrate := int32(0)
	if enc.InternalSampleRate > 16000 {
		if lowRate != 0 {
			bweBitrate += 450
		} else {
			bweBitrate += 750
		}
		if enc.PayloadSizeMs == 10 {
			bweBitrate += 450
		}
	}

	b.rateContWnrgSmth += 0.6 * (wnrg - b.rateContWnrgSmth)

	framelenIdx := 3
	switch enc.PayloadSizeMs {
	case 10:
		framelenIdx = 0
	case 20:
		framelenIdx = 1
	case 60:
		framelenIdx = 2
	}

	// Operator precedence: `SMPL_CELP_IDX_FEC + (a == 0) || (b == c)` parses as
	// `(SMPL_CELP_IDX_FEC + (a == 0)) || (b == c)` (a `||` whose value is 0 or 1).
	fecEq0 := int32(0)
	if enc.FecBitRate == 0 {
		fecEq0 = 1
	}
	startR := 0
	if celpIdxFEC+int(fecEq0) != 0 || enc.FecBitRate == enc.MainBitRate {
		startR = 1
	}

	lrIdx := 1
	if lowRate != 0 {
		lrIdx = 0
	}

	for r := startR; r <= celpIdxMain; r++ {
		bitRate := float32(enc.MainBitRate)
		if r == celpIdxFEC {
			bitRate = float32(enc.FecBitRate)
		}
		bitRate = minF32(bitRate, 30000.0) // don't extrapolate pulses_per_20ms_target_max curves
		rateKbps := (bitRate - float32(bweBitrate)) / 1000.0
		if lowRate == 0 {
			switch enc.Complexity {
			case 1, 2:
				rateKbps *= 0.9900990
			case 3, 4:
				rateKbps *= 1.0101010
			}
		}

		var pulsesPer20msTargetMax float32
		rateControlThrs := float32(smplRateControlThrsComp5[framelenIdx][lrIdx])
		if (bitRate - float32(bweBitrate)) < rateControlThrs {
			pulsesPer20msTargetMax = 1.0
		} else {
			coeff := &smplRateControlModelComp5[framelenIdx][lrIdx]
			if r == celpIdxFEC && lowRate == 0 && enc.UseFecRateCompensation != 0 {
				pulsesPer20msTargetMax = maxF32(bitrate2pulsesHrFec(rateKbps, coeff, rateControlThrs), 1.0)
			} else {
				pulsesPer20msTargetMax = maxF32(bitrate2pulses(rateKbps, coeff), 1.0)
			}
		}

		relPulserate := pulsesPer20msTargetMax / 16.0 * (320.0 / float32(framelen))
		relPulserateLog := f32Ln(relPulserate)
		if b.rateContBitrate[r] != bitRate {
			bitrateScale := smplRateContScale * relPulserate * (1.0 + 0.4*relPulserateLog*relPulserateLog)
			b.rateContBitrateScale[r] = bitrateScale
			b.rateContBitrate[r] = bitRate
		}

		numsubfrs := framelen / subfrlen
		mpps := 1 + int32(f32Round(pulsesPer20msTargetMax*(1.0+0.5)/float32(numsubfrs)))
		if enc.UseDtx != 0 && dtxSidFrame != 0 {
			mpps = 0
		} else {
			mpps = int32(f32Round(float32(mpps) * (0.5 + 0.5*f32Sqrt(spActProb+1e-12))))
			frameType := frameBackgroundNoise
			if codedAsActiveVoice != 0 {
				frameType = frameUnvoiced
				if voiced == 1 {
					frameType = frameVoiced
				}
			}
			// smpl_max_pulses_per_frame is indexed by the raw low_rate flag
			// (0=high-rate row), unlike the model/thrs tables which use
			// `low_rate ? 0 : 1`.
			maxPulses := int32(smplMaxPulsesPerFrame[lowRate][frameType]) * framelen / 320
			mpps = minI32(mpps, maxPulses/numsubfrs) // don't overshoot the PDF
		}
		// mpps <= SMPL_MAX_PULSES_PER_SF (debug_assert in the Rust).
		maxPulsesPerSubfr[r] = int16(mpps)

		imp := (wnrg + 0.01*wnrgNext) / (b.rateContWnrgSmth + 0.02*wnrgNext + 1e-12)
		if voiced != 0 {
			if bitRate <= 9000.0 {
				imp = f32Sqrt(imp + 1e-12)
			}
		} else {
			imp *= 0.9 + 0.3*smplSigmoid(nonflatness-2.0)
			imp *= 0.8
		}
		if voiced != b.prevVoiced {
			imp *= 1.1
		}
		imp *= 0.9 + 0.3*1.0/(1.0+25.0*voicingStrength*voicingStrength)

		// Speech-activity bitrate allocation shaping.
		impFactor := enc.SubFrameImportanceFactor
		if impFactor <= 1.0 {
			imp *= (1.0 - impFactor) + impFactor*f32Sqrt(spActProb+1e-12)
		} else if impFactor <= 2.0 {
			impFactor -= 1.0
			imp *= (1.0 - impFactor) + impFactor*spActProb
		} else {
			// imp_factor <= 3.0 (debug_assert in the Rust).
			impFactor -= 2.0
			imp *= (1.0 - impFactor) + impFactor*spActProb*spActProb
		}
		imp *= b.adjustmentFactor[r] * b.rateContBitrateScale[r]
		subfrImportance[r] = imp
		b.prevVoiced = voiced
	}

	return maxPulsesPerSubfr, subfrImportance
}
