// Logical tables for the nrgres/gains (Group A/E) and LTP gain (Group C)
// decode, built from the cc_seed.bin DCMF seed. Ported from whatsapp-rust
// (MIT) wacore/src/voip/mlow/smpl_cc_tables.rs (+ the LR constants from
// smpl_celp.rs); MPL-2.0 for this port.
//
// CDFs are u16 (the integer cmf fits 16 bits) to feed DecodeCDF/EncodeCDF
// directly. The pulse (Group B) and pitch lag/contour (Group D) tables ride
// the same struct; the pitch contour window itself lives in the heap window
// (SmplMem).
package mlow

import (
	"encoding/binary"
	"sync"
)

const (
	log2Exp1Q15 = 47274
	log2_2piQ14 = 43442
	oneQ31      = int64(1) << 31

	nrgresGain4N      = 84
	nrgresShape4N     = 98
	fcbgOffsetSteps   = 176 // dcmf len; cmf row len is +1
	fcbgOffsetBuckets = 4
	acbgN             = 16
	acbgRows          = acbgN + 1
	fcbgVN            = 34
	fcbgVDeltaN       = 67

	// Pulse-coding (Group B) shape constants.
	smplMaxPulsesPerSF = 40
	runlengthStep      = 8
	numRunlenCmfs      = 20 // SMPL_MAX_SF_LEN(160) / RUNLENGTH_STEP
	splitNumTables     = smplMaxPulsesPerSF*4 - 1
)

// pdfToCmf (`smpl_pdf_to_CMF`, maxval == -1 path): truncating-int normalize
// into a cumulative u16 CDF of length len(pdf)+1.
func pdfToCmf(pdf []int32) []uint16 {
	n := int64(len(pdf))
	const maxval int64 = 32767
	var sump int64
	for _, x := range pdf {
		sump += int64(x)
	}
	cmf := make([]uint16, len(pdf)+1)
	for i := 0; i < len(pdf); i++ {
		p := (int64(pdf[i])*(maxval-n))/sump + 1
		cmf[i+1] = uint16(int32(cmf[i]) + int32(p))
	}
	return cmf
}

// stirling is the Stirling-approximation log-factorial term used by the
// split-probability model.
func stirling(n int32) int32 {
	if n == 0 {
		return 0
	}
	ret := ((n << 1) + 1) * (lin2log(n) << 7)
	ret -= log2Exp1Q15 * n
	ret += log2_2piQ14
	return ret + log2Exp1Q15/(12*n)
}

// probSplitFast is `smpl_prob_split_fast`: 2^(stirling(N)-stirling(k)-
// stirling(N-k)-N), as a Q-scaled int.
func probSplitFast(k, n int32) int32 {
	tmp := stirling(n) - stirling(k) - stirling(n-k) - n*(1<<15)
	if tmp == 0 {
		return 1 << 30
	}
	ret := log2lin((-tmp) >> 8)
	return (1 << 30) / ret
}

// createSplitCmfs builds split_CMFs[num_pulses-1] for num_pulses in
// 1..SMPL_MAX_PULSES_PER_SF*4.
func createSplitCmfs() [][]uint16 {
	out := make([][]uint16, 0, splitNumTables)
	for numPulses := int32(1); numPulses <= splitNumTables; numPulses++ {
		minSplit := maxI32(numPulses-int32(smplMaxPulsesPerSF)*2, 0)
		maxSplit := numPulses - minSplit
		p := make([]int32, 0, maxSplit-minSplit+1)
		for k := minSplit; k <= maxSplit; k++ {
			p = append(p, probSplitFast(k, numPulses))
		}
		out = append(out, pdfToCmf(p))
	}
	return out
}

// createRunlenTable builds the per-`nump` magnitude CDFs (full length
// max_samples + 1) for one octave bucket, Q31 sigmoid/log2lin geometric model.
func createRunlenTable(maxSamples int32) RunlenCmfs {
	ms := maxSamples
	cmfs := make([][]uint16, 0, smplMaxPulsesPerSF)
	for nump := int32(1); nump <= int32(smplMaxPulsesPerSF); nump++ {
		plongerQ31 := oneQ31
		p := make([]int32, ms)
		for nums := int32(1); nums <= ms; nums++ {
			tmp := oneQ31 - oneQ31/int64(ms-nums+1)
			p1Q31 := tmp
			for i := int32(0); i < nump-1; i++ {
				p1Q31 = (p1Q31 * tmp) >> 31
			}
			p1Q31 = oneQ31 - p1Q31
			if p1Q31 > 2147376274 { // 0.99995
				p1Q31 = 2147376274
			}
			var logOutQ7 int32
			if nump > ms {
				logOutQ7 = lin2log((nump<<10)/ms) - 10*128
			} else {
				logOutQ7 = -(lin2log((ms<<10)/nump) - 10*128)
			}
			const sigmBiasQ5 = 146
			const scaleMaxQ15 = 36000
			const scaleMinQ15 = 26000
			scaleFacQ15 := scaleMaxQ15 -
				(((scaleMaxQ15 - scaleMinQ15) * sigmQ15((logOutQ7>>2)+sigmBiasQ5)) >> 15)
			p1Q31 = oneQ31 -
				int64(log2lin(((scaleFacQ15*(lin2log(int32(oneQ31-p1Q31))-31*128))>>15)+31*128))
			if p1Q31 > 2147376274 {
				p1Q31 = 2147376274
			}
			p[nums-1] = int32((plongerQ31 * p1Q31) >> 31)
			plongerQ31 = (plongerQ31 * (oneQ31 - p1Q31)) >> 31
		}
		cmfs = append(cmfs, pdfToCmf(p))
	}
	return RunlenCmfs{maxSamples: ms, cmfs: cmfs}
}

// LR (p6!=0) ACB-gain constants (smpl_celp.rs; no seed bytes).
var smplAcbgainsDcmfLR = [acbgRows * acbgN]byte{
	103, 70, 48, 3, 122, 135, 47, 192, 2, 255, 99, 96, 186, 194, 4, 28,
	161, 90, 76, 3, 181, 60, 37, 219, 2, 132, 81, 146, 255, 43, 3, 36,
	114, 222, 55, 6, 203, 34, 42, 154, 6, 255, 33, 209, 225, 78, 6, 45,
	198, 161, 110, 8, 239, 26, 35, 162, 4, 117, 42, 214, 255, 33, 6, 72,
	55, 255, 124, 55, 124, 55, 55, 55, 55, 78, 55, 215, 111, 55, 55, 167,
	154, 136, 77, 4, 220, 33, 38, 166, 2, 144, 50, 196, 255, 43, 4, 41,
	56, 21, 19, 3, 48, 255, 38, 220, 2, 225, 107, 31, 122, 227, 2, 11,
	63, 38, 23, 4, 77, 85, 58, 190, 4, 255, 53, 53, 145, 138, 4, 14,
	95, 47, 33, 2, 110, 146, 53, 255, 2, 219, 79, 73, 198, 122, 2, 15,
	84, 255, 84, 84, 147, 84, 84, 84, 84, 120, 84, 120, 84, 84, 84, 84,
	73, 58, 25, 1, 95, 99, 52, 175, 1, 255, 48, 69, 151, 184, 1, 15,
	105, 32, 43, 2, 84, 225, 34, 255, 2, 156, 129, 49, 189, 124, 3, 19,
	152, 230, 89, 6, 253, 28, 40, 153, 2, 195, 31, 255, 249, 58, 5, 61,
	138, 84, 54, 3, 173, 96, 45, 247, 2, 176, 83, 128, 255, 69, 2, 26,
	22, 17, 8, 1, 23, 106, 26, 88, 1, 182, 37, 18, 50, 255, 1, 6,
	218, 174, 228, 65, 186, 65, 65, 92, 65, 65, 65, 255, 174, 65, 65, 174,
	117, 255, 101, 16, 180, 20, 33, 94, 10, 131, 20, 222, 143, 38, 15, 105,
}

var smplCbAcbgainsLRQ14 = [acbgN * 2]int16{
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

// RunlenCmfs is the run-length CDF bucket for one octave
// (max_samples = oct * RUNLENGTH_STEP). cmfs[c-1] is the full cumulative CDF
// for c pulses-left, length max_samples + 1.
type RunlenCmfs struct {
	maxSamples int32
	cmfs       [][]uint16
}

// MaxSamples is the bucket's max_samples (the per-row base value the call
// site uses to shift into cmf).
func (r *RunlenCmfs) MaxSamples() int32 { return r.maxSamples }

// Cmf is the full magnitude CDF for c pulses-left (cmfs[c-1]). The caller
// windows it by (max_samples - pos) and reads pos+1 entries.
func (r *RunlenCmfs) Cmf(c int32) []uint16 { return r.cmfs[c-1] }

// CcTables is the runtime gains/pulses CDF table set expanded from cc_seed.
type CcTables struct {
	nrgresGain4     []uint16   // [85]
	nrgresShape4    []uint16   // [99]
	fcbgOffset      [][]uint16 // [3*4][177]
	acbgainsHr      []uint16   // [17*17] row-major rows of 17
	acbgainsLr      []uint16   // [17*17] LR variant (p6!=0)
	fcbgainsV       []uint16   // [35]
	fcbgainsVDelta  []uint16   // [68]
	acbgainsCbHrQ14 []int16    // [16*2]
	acbgainsCbLrQ14 []int16    // [16*2] LR weight table (p6!=0)
	gainRecon       []int16    // rodata, addressed from gainReconBase
	gainReconBase   uint32
	nPulseCmfs      [3][]uint16 // pulse-count CDFs [bgn,uv,v]
	splitCmfs       [][]uint16  // [SPLIT_NUM_TABLES]; row i = split_CMFs[i] (num_pulses = i+1)
	runlen          []RunlenCmfs
}

func buildCcTables(seed *CcSeed) *CcTables {
	nrgresGain4 := dcmfToCmf(seed.NrgresGain4Dcmf)
	nrgresShape4 := dcmfToCmf(seed.NrgresShape4Dcmf)
	var fcbgOffset [][]uint16
	for i := 0; i < len(seed.FcbgOffsetDcmf); i += fcbgOffsetSteps {
		fcbgOffset = append(fcbgOffset, dcmfToCmf(seed.FcbgOffsetDcmf[i:i+fcbgOffsetSteps]))
	}
	var acbgainsHr []uint16
	for i := 0; i < len(seed.AcbgainsHrDcmf); i += acbgN {
		acbgainsHr = append(acbgainsHr, dcmfToCmf(seed.AcbgainsHrDcmf[i:i+acbgN])...)
	}
	var acbgainsLr []uint16
	for i := 0; i < len(smplAcbgainsDcmfLR); i += acbgN {
		acbgainsLr = append(acbgainsLr, dcmfToCmf(smplAcbgainsDcmfLR[i:i+acbgN])...)
	}
	fcbgainsV := dcmfToCmf(seed.FcbgainsVDcmf)
	fcbgainsVDelta := dcmfToCmf(seed.FcbgainsVDeltaDcmf)
	acbgainsCbHrQ14 := make([]int16, len(seed.AcbgainsCbHrQ14))
	for i, x := range seed.AcbgainsCbHrQ14 {
		acbgainsCbHrQ14[i] = int16(x)
	}
	gainRecon := make([]int16, len(seed.GainRecon)/2)
	for i := range gainRecon {
		gainRecon[i] = int16(binary.LittleEndian.Uint16(seed.GainRecon[i*2:]))
	}

	nPulseCmfs := [3][]uint16{
		dcmfToCmf(seed.NPulsesDcmfBgn),
		dcmfToCmf(seed.NPulsesDcmfUV),
		dcmfToCmf(seed.NPulsesDcmfV),
	}
	splitCmfs := createSplitCmfs()
	runlen := make([]RunlenCmfs, 0, numRunlenCmfs)
	for oct := int32(1); oct <= numRunlenCmfs; oct++ {
		runlen = append(runlen, createRunlenTable(oct*runlengthStep))
	}

	return &CcTables{
		nrgresGain4:     nrgresGain4,
		nrgresShape4:    nrgresShape4,
		fcbgOffset:      fcbgOffset,
		acbgainsHr:      acbgainsHr,
		acbgainsLr:      acbgainsLr,
		fcbgainsV:       fcbgainsV,
		fcbgainsVDelta:  fcbgainsVDelta,
		acbgainsCbHrQ14: acbgainsCbHrQ14,
		acbgainsCbLrQ14: smplCbAcbgainsLRQ14[:],
		gainRecon:       gainRecon,
		gainReconBase:   seed.GainReconBase,
		nPulseCmfs:      nPulseCmfs,
		splitCmfs:       splitCmfs,
		runlen:          runlen,
	}
}

var (
	ccOnce  sync.Once
	ccValue *CcTables
	ccErr   error
)

// LoadCcTables decodes the cc_seed blob and expands the runtime tables once.
func LoadCcTables(blob []byte) (*CcTables, error) {
	ccOnce.Do(func() {
		seed, err := LoadCcSeed(blob)
		if err != nil {
			ccErr = err
			return
		}
		ccValue = buildCcTables(seed)
	})
	return ccValue, ccErr
}

// NrgresGain4 is the nrgres main-gain CDF (n=85).
func (c *CcTables) NrgresGain4() []uint16 { return c.nrgresGain4 }

// NrgresShape4 is the nrgres delta-gain / shape CDF (n=99).
func (c *CcTables) NrgresShape4() []uint16 { return c.nrgresShape4 }

// FcbgOffset is the fcbg-offset CDF row [tableIx][bucket] sliced at
// minOffset (n=92). minOffset is the non-negative entry shift the old
// -2*neg_part pointer offset encoded.
func (c *CcTables) FcbgOffset(tableIx, bucket, minOffset int) []uint16 {
	row := c.fcbgOffset[tableIx*fcbgOffsetBuckets+bucket]
	return row[minOffset : minOffset+92]
}

// AcbgainRow is the HR ACB-gain CDF for predictor prev (row prev+1), n=17.
func (c *CcTables) AcbgainRow(prev int32) []uint16 {
	r := prev + 1
	base := int(r) * (acbgN + 1)
	return c.acbgainsHr[base : base+acbgN+1]
}

// AcbgainRowLr is the LR ACB-gain CDF for predictor prev (the p6!=0
// variant), n=17.
func (c *CcTables) AcbgainRowLr(prev int32) []uint16 {
	r := prev + 1
	base := int(r) * (acbgN + 1)
	return c.acbgainsLr[base : base+acbgN+1]
}

// AcbgainWeightsLr is the LR ACB-gain weights (w0, w2) for gain index gi.
func (c *CcTables) AcbgainWeightsLr(gi int32) (int32, int32) {
	i := gi * 2
	return int32(c.acbgainsCbLrQ14[i]), int32(c.acbgainsCbLrQ14[i+1])
}

// FcbgainV is the FCB voiced-gain CDF (prev_filt == -1), n=35.
func (c *CcTables) FcbgainV() []uint16 { return c.fcbgainsV }

// FcbgainVDelta is the FCB voiced-gain delta CDF sliced at
// [FCBG_V_N-1 - prevFilt] (n=35).
func (c *CcTables) FcbgainVDelta(prevFilt int32) []uint16 {
	start := fcbgVN - 1 - prevFilt
	return c.fcbgainsVDelta[start : start+35]
}

// AcbgainWeights is the HR ACB-gain weights (w0, w2) for gain index gi.
func (c *CcTables) AcbgainWeights(gi int32) (int32, int32) {
	i := gi * 2
	return int32(c.acbgainsCbHrQ14[i]), int32(c.acbgainsCbHrQ14[i+1])
}

// NrgStep is the nrg-step const for config cfg (gain-reconstruction base
// scale).
func (c *CcTables) NrgStep(cfg int32) int32 {
	return c.gainReconAt(c.gainReconBase + uint32(cfg)*2)
}

// GainRecon is the unvoiced gain-reconstruction codebook entry. p4 selects
// the p3==4 table; idx is the (possibly out-of-the-visible-array) rodata
// index the WASM reads as adjacent constants.
func (c *CcTables) GainRecon(p4 bool, idx int32) int32 {
	var base uint32
	if p4 {
		base = 0xf35f0
	} else {
		base = 0xf3970
	}
	return c.gainReconAt(base + uint32(idx)*2)
}

// gainReconAt reads an int16 from the gain-reconstruction rodata at WASM
// address addr. Out-of-region reads fall back to 0, matching the old heap
// window.
func (c *CcTables) gainReconAt(addr uint32) int32 {
	off := int(addr - c.gainReconBase)
	if off >= 0 && off%2 == 0 && off/2 < len(c.gainRecon) {
		return int32(c.gainRecon[off/2])
	}
	return 0
}

// NPulseCount is the pulse-count CDF for table idx (0 = background,
// 1 = unvoiced, 2 = voiced).
func (c *CcTables) NPulseCount(idx int32) []uint16 { return c.nPulseCmfs[idx] }

// SplitCmf is the split CDF base for total pulses (split_CMFs[total-1]); the
// caller windows it at its min_split offset.
func (c *CcTables) SplitCmf(total int32) []uint16 {
	i := int(total - 1)
	if i < 0 || i >= len(c.splitCmfs) {
		return nil
	}
	return c.splitCmfs[i]
}

// Runlen is the run-length CDF bucket for octave oct.
func (c *CcTables) Runlen(oct int32) *RunlenCmfs { return &c.runlen[oct-1] }

func maxI32(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}
