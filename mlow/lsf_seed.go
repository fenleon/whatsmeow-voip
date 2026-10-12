// MLow LSF runtime tables, expanded from the lsf_seed.bin ROM. Ported from
// whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_lsf_seed.rs; MPL-2.0 for this
// port.
//
// The float op order here is load-bearing (matmul accumulation, sqrt-then-
// reciprocal in rotApplyWght, integer truncation in dcmfToCmf, scalar
// unpack8) so the rebuilt tables are bit-faithful. Explicit indexed loops are
// intentional: the accumulation order is normative.
package mlow

import (
	"math"
	"sync"
)

const (
	lsfOrder     = 16 // SMPL_LPC_ORDER
	lsfCentroids = 16 // LSF_CB_CENTROIDS
	cinvLen      = lsfOrder * (lsfOrder + 1) / 2
	st2Len       = 9593 // LSF_ST2_ALL_QLVLS_LEN
)

// Per-voiced (index 0 = unvoiced, 1 = voiced) scale/min constants.
var (
	cinvMin      = [2]float32{-3.5960955e-5, -2.778548e-5}
	cinvScale    = [2]float32{1.8589316e-9, 1.2180106e-9}
	cbMin        = [2]float32{-0.5873778, -0.24721986}
	cbScale      = [2]float32{1.3145164e-5, 7.226229e-6}
	rotMin       = [2]float32{-0.9124832, -0.8455929}
	rotScale     = [2]float32{0.006554049, 0.0069253775}
	rotCondMin   = [2]float32{-0.67291605, -0.8248211}
	rotCondScale = [2]float32{0.0052386564, 0.0064186584}
)

const (
	st2QlvlsMin   float32 = -0.45
	st2QlvlsScale float32 = 0.0034478905
	// LSF_QSTEP_COND_MULT
	qstepCondMult float32 = 0.9
)

// LsfGrid holds the stage-1 grid CDFs keyed on match × stage1.
type LsfGrid struct {
	Match1    []uint16 // CMF_cond_v
	Match1Alt []uint16 // CMF_cond_uv
	Match0    []uint16 // CMF_uv
	Match0Alt []uint16 // CMF_v
}

// SmplTables is the runtime CDF table set the LSF/pulse/gains decoders read
// (the LSF block of func 3559's output), expanded from the LSF seed ROM.
type SmplTables struct {
	LsfSel  [][]uint16 // [3][3]
	LsfGrid LsfGrid
	// LsfStage2[stage1][config][grid][coeff] -> cumulative CDF.
	LsfStage2 [][][][][]uint16
	LsfExtra  []uint16 // [3]
}

// St1 is one voiced-ness entry of the stage-1 VQ codebook (encoder-side).
type St1 struct {
	Cbhalf   [lsfCentroids][lsfOrder]float32
	CInv     [lsfOrder][lsfOrder]float32
	BitsCond []float32 // [17]
	Rotcond  [2][lsfOrder][lsfOrder]float32
	CbCinv   [lsfCentroids][lsfOrder]float32
	We       [lsfCentroids][lsfOrder][lsfOrder]float32
	Bits     []float32 // [16]
	Wie      [lsfCentroids][lsfOrder][lsfOrder]float32
}

// St2 is one [voiced][lowRate][centroid] stage-2 residual codebook.
type St2 struct {
	NumQlvls [lsfOrder]int32
	Qlvls    [lsfOrder][]float32
	NumBits  [lsfOrder][]float32
}

// LsfCb is the stage-1/stage-2 LSF VQ codebook (encoder-side; the decoder
// uses its valtables subset).
type LsfCb struct {
	St1       [2]St1
	St2       [2][2][lsfCentroids + 1]St2
	MinQi     [2][2][lsfCentroids + 1][lsfOrder]int32
	MaxQi     [2][2][lsfCentroids + 1][lsfOrder]int32
	Qstep     [2][2]float32
	MeanV     [lsfOrder]float32
	MeanUV    [lsfOrder]float32
	RegCond   [2]float32
	MinDistV  [lsfCentroids + 1]float32
	MinDistUV [lsfCentroids + 1]float32
}

// SmplSynthTables carries the decoder-side synthesis tables expanded from the
// same seed.
type SmplSynthTables struct {
	// Valtables[voiced][config][grid][coeff] -> residual value table.
	Valtables [][][][][]float32
	// Centroids[voiced][grid] / Matrices[voiced][grid] (grid < 16).
	Centroids [2][lsfCentroids][lsfOrder]float32
	Matrices  [2][lsfCentroids][lsfOrder][lsfOrder]float32
	// MinSpacing[v] = min_dist[1-v] (the index swap, not separate ROM).
	MinSpacing [2][lsfCentroids + 1]float32
	// grid16_w[v] = mean[1-v] (the swap bakes in the synth's INVERTED
	// selection); grid16_alpha = reg_cond; grid16_matrices = unpack8(rot_cond_8)
	// flattened row-major.
	Grid16W        [2][lsfOrder]float32
	Grid16Alpha    [2]float32
	Grid16Matrices [2][2][lsfOrder * lsfOrder]float32
}

// LsfBuilt is the three runtime structs rebuilt from one seed.
type LsfBuilt struct {
	Synth  SmplSynthTables
	Tables SmplTables
	Cb     LsfCb
}

// matrixMultTransp16: y[i] = sum_j C[j][i]*x[j], seeded with j=0 then += for
// j>0 (accumulation order is load-bearing).
func matrixMultTransp16(c *[lsfOrder][lsfOrder]float32, x *[lsfOrder]float32) [lsfOrder]float32 {
	var y [lsfOrder]float32
	x0 := x[0]
	for i := 0; i < lsfOrder; i++ {
		y[i] = c[0][i] * x0
	}
	for j := 1; j < lsfOrder; j++ {
		xj := x[j]
		for i := 0; i < lsfOrder; i++ {
			y[i] += c[j][i] * xj
		}
	}
	return y
}

// laroia is the Laroia inverse-gap LSF weights, with the gap floored at 1e-3.
func laroia(lsf *[lsfOrder]float32) [lsfOrder]float32 {
	const (
		pi      float32 = 3.14159265358979323846
		minDist float32 = 1e-3
	)
	var inv [lsfOrder + 1]float32
	inv[0] = 1.0 / maxF32(lsf[0], minDist)
	for i := 1; i < lsfOrder; i++ {
		inv[i] = 1.0 / maxF32(lsf[i]-lsf[i-1], minDist)
	}
	inv[lsfOrder] = 1.0 / maxF32(pi-lsf[lsfOrder-1], minDist)
	var w [lsfOrder]float32
	for i := 0; i < lsfOrder; i++ {
		w[i] = inv[i] + inv[i+1]
	}
	return w
}

// rotApplyWght applies the Laroia weights to the rotation:
// lsfw = sqrt(laroia(lsf)), we[i][j]=rot[i][j]/lsfw[j], wie[j][i]=rot[i][j]*lsfw[j].
func rotApplyWght(rot *[lsfOrder][lsfOrder]float32, lsf *[lsfOrder]float32) (we, wie [lsfOrder][lsfOrder]float32) {
	lsfw := laroia(lsf)
	for i := range lsfw {
		lsfw[i] = float32(math.Sqrt(float64(lsfw[i])))
	}
	var lsfwInv [lsfOrder]float32
	for i := 0; i < lsfOrder; i++ {
		lsfwInv[i] = 1.0 / lsfw[i]
	}
	for i := 0; i < lsfOrder; i++ {
		for j := 0; j < lsfOrder; j++ {
			we[i][j] = rot[i][j] * lsfwInv[j]
			wie[j][i] = rot[i][j] * lsfw[j]
		}
	}
	return we, wie
}

// cmfToBits: bits[i] = -log2f((cmf[i+1]-cmf[i]) / (f32)cmf[len-1]).
func cmfToBits(cmf []uint16) []float32 {
	den := float32(cmf[len(cmf)-1])
	bits := make([]float32, 0, len(cmf)-1)
	for i := 0; i < len(cmf)-1; i++ {
		num := float32(int32(cmf[i+1]) - int32(cmf[i]))
		bits = append(bits, -float32(math.Log2(float64(num/den))))
	}
	return bits
}

// dcmfToCmf is the integer expansion of a delta-CMF to a cumulative u16 CDF
// of length len+1 (truncating division is load-bearing).
func dcmfToCmf(dcmf []byte) []uint16 {
	dcmfLen := len(dcmf)
	cmf := make([]uint16, dcmfLen+1)
	var sum int64
	for n := 0; n < dcmfLen; n++ {
		tmp := int32(dcmf[n]) + 1
		tmp *= tmp
		if tmp > 65535 {
			tmp = 65535
		}
		cmf[n+1] = uint16(tmp)
		sum += int64(tmp)
	}
	cmf[0] = 0
	for n := 1; n < dcmfLen+1; n++ {
		prev := int64(cmf[n-1])
		add := (int64(cmf[n])*(32767-int64(dcmfLen)))/sum + 1
		cmf[n] = uint16(prev + add)
	}
	return cmf
}

func unpack8_16x16(packed []byte, scale, min float32) [lsfOrder][lsfOrder]float32 {
	var out [lsfOrder][lsfOrder]float32
	p := 0
	for i := 0; i < lsfOrder; i++ {
		for j := 0; j < lsfOrder; j++ {
			out[i][j] = min + float32(packed[p])*scale
			p++
		}
	}
	return out
}

func maxF32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// BuildLsfTables runs the LSF codebook expansion to produce the three runtime
// structs from the decoded seed ROM.
func BuildLsfTables(seed *LsfSeed) *LsfBuilt {
	var built LsfBuilt
	var st1 [2]St1
	synthCentroids := &[2][lsfCentroids][lsfOrder]float32{}
	synthMatrices := &[2][lsfCentroids][lsfOrder][lsfOrder]float32{}
	var synthGrid16Matrices [2][2][lsfOrder * lsfOrder]float32

	// Flat seed views.
	u16At := func(v []uint16, i int) uint16 { return v[i] }

	for voiced := 0; voiced < 2; voiced++ {
		// cInv (symmetric lower-triangular fill).
		var cInv [lsfOrder][lsfOrder]float32
		p := 0
		for i := 0; i < lsfOrder; i++ {
			for j := 0; j <= i; j++ {
				v := cinvMin[voiced] + cinvScale[voiced]*float32(seed.Cinv16[voiced*cinvLen+p])
				cInv[i][j] = v
				cInv[j][i] = v
				p++
			}
		}

		var cbhalf, cbCinv [lsfCentroids][lsfOrder]float32
		var we, wie [lsfCentroids][lsfOrder][lsfOrder]float32
		for c := 0; c < lsfCentroids; c++ {
			var lsfCb [lsfOrder]float32
			for i := 0; i < lsfOrder; i++ {
				lsfCb[i] = cbMin[voiced] +
					float32(u16At(seed.Cb16, (voiced*lsfCentroids+c)*lsfOrder+i))*cbScale[voiced] +
					seed.Mean[voiced*lsfOrder+i]
				cbhalf[c][i] = lsfCb[i] * 0.5
			}
			cbCinv[c] = matrixMultTransp16(&cInv, &lsfCb)
			rot := unpack8_16x16(seed.Rot8[voiced*lsfCentroids*lsfOrder*lsfOrder+c*lsfOrder*lsfOrder:], rotScale[voiced], rotMin[voiced])
			weC, wieC := rotApplyWght(&rot, &lsfCb)
			we[c] = weC
			wie[c] = wieC
		}

		// Rotcond[lowRate] = unpack8(rot_cond_8[lowRate]).
		var rotcond [2][lsfOrder][lsfOrder]float32
		for lr := 0; lr < 2; lr++ {
			rotcond[lr] = unpack8_16x16(seed.RotCond8[voiced*2*lsfOrder*lsfOrder+lr*lsfOrder*lsfOrder:], rotCondScale[voiced], rotCondMin[voiced])
		}

		bits := cmfToBits(seed.Cmf[voiced*17 : voiced*17+17])
		bitsCond := cmfToBits(seed.CmfCond[voiced*18 : voiced*18+18])

		st1[voiced] = St1{
			Cbhalf:   cbhalf,
			CInv:     cInv,
			BitsCond: bitsCond,
			Rotcond:  rotcond,
			CbCinv:   cbCinv,
			We:       we,
			Bits:     bits,
			Wie:      wie,
		}
		built.Cb.St1[voiced] = st1[voiced]

		for g := 0; g < lsfCentroids; g++ {
			synthCentroids[voiced][g] = cbhalf[g]
			synthMatrices[voiced][g] = we[g]
		}
		for lr := 0; lr < 2; lr++ {
			flatten(rotcond[lr], &synthGrid16Matrices[voiced][lr])
		}
	}

	// Stage 2: the flat QlvlsTable / cmfTable / numBitsTable walks.
	var qlvlsFlat [st2Len]float32
	var numqlvls [2][2][lsfCentroids + 1][lsfOrder]int32
	var qoff [2][2][lsfCentroids + 1][lsfOrder]int32
	var cmfSlices [2][2][lsfCentroids + 1][lsfOrder][]uint16
	var numbits [2][2][lsfCentroids + 1][lsfOrder][]float32

	qPtr, q8Ptr, dcmfPtr := 0, 0, 0
	for voiced := 0; voiced < 2; voiced++ {
		for lr := 0; lr < 2; lr++ {
			for c := 0; c < lsfCentroids+1; c++ {
				qstep := seed.Qstep[voiced*2+lr]
				if c == lsfCentroids {
					qstep *= qstepCondMult
				}
				for i := 0; i < lsfOrder; i++ {
					qiIdx := (((voiced*2+lr)*(lsfCentroids+1))+c)*lsfOrder + i
					minQi := int32(int8(seed.St2MinQi[qiIdx]))
					maxQi := int32(int8(seed.St2MaxQi[qiIdx]))
					numQlvls := int(maxQi - minQi + 1)
					numqlvls[voiced][lr][c][i] = int32(numQlvls)
					qoff[voiced][lr][c][i] = int32(qPtr)
					for lvl := 0; lvl < numQlvls; lvl++ {
						q8 := float32(seed.St2AllQlvls8[q8Ptr])
						qlvlsFlat[qPtr] = (st2QlvlsMin + st2QlvlsScale*q8 + float32(lvl) + float32(minQi)) * qstep
						qPtr++
						q8Ptr++
					}
					dcmf := seed.St2AllQlvlDcmfs[dcmfPtr : dcmfPtr+numQlvls]
					cmf := dcmfToCmf(dcmf)
					nb := cmfToBits(cmf)
					dcmfPtr += numQlvls
					cmfSlices[voiced][lr][c][i] = cmf
					numbits[voiced][lr][c][i] = nb
				}
			}
		}
	}
	if qPtr != st2Len || q8Ptr != st2Len || dcmfPtr != st2Len {
		panic("mlow lsf seed: stage-2 flat walks did not consume exactly ST2_LEN")
	}

	// Cb.St2 with Qlvls sliced [off..off+numQlvls]; Valtables/LsfStage2 likewise.
	for voiced := 0; voiced < 2; voiced++ {
		for lr := 0; lr < 2; lr++ {
			for c := 0; c < lsfCentroids+1; c++ {
				st2c := &built.Cb.St2[voiced][lr][c]
				vtC := make([][]float32, lsfOrder)
				s2C := make([][]uint16, lsfOrder)
				for i := 0; i < lsfOrder; i++ {
					n := int(numqlvls[voiced][lr][c][i])
					off := int(qoff[voiced][lr][c][i])
					qv := make([]float32, n)
					copy(qv, qlvlsFlat[off:off+n])
					st2c.NumQlvls[i] = numqlvls[voiced][lr][c][i]
					st2c.Qlvls[i] = qv
					st2c.NumBits[i] = numbits[voiced][lr][c][i]
					vtC[i] = qv
					s2C[i] = cmfSlices[voiced][lr][c][i]
					qiIdx := (((voiced*2+lr)*(lsfCentroids+1))+c)*lsfOrder + i
					built.Cb.MinQi[voiced][lr][c][i] = int32(int8(seed.St2MinQi[qiIdx]))
					built.Cb.MaxQi[voiced][lr][c][i] = int32(int8(seed.St2MaxQi[qiIdx]))
				}
				valtablesSet(&built.Synth.Valtables, voiced, lr, c, vtC)
				lsfStage2Set(&built.Tables.LsfStage2, voiced, lr, c, s2C)
			}
		}
	}

	for v := 0; v < 2; v++ {
		for i := 0; i < lsfOrder; i++ {
			built.Cb.MeanV[i] = seed.Mean[lsfOrder+i]
			built.Cb.MeanUV[i] = seed.Mean[i]
		}
		built.Cb.RegCond[v] = seed.RegCond[v]
		for i := 0; i < lsfCentroids+1; i++ {
			built.Cb.MinDistV[i] = seed.MinDist[lsfCentroids+1+i]
			built.Cb.MinDistUV[i] = seed.MinDist[i]
		}
		built.Cb.Qstep[v][0] = seed.Qstep[v*2]
		built.Cb.Qstep[v][1] = seed.Qstep[v*2+1]

		built.Synth.Grid16Alpha[v] = seed.RegCond[v]
		for i := 0; i < lsfOrder; i++ {
			built.Synth.Grid16W[v][i] = seed.Mean[(1-v)*lsfOrder+i]
			built.Synth.MinSpacing[v][i] = seed.MinDist[(1-v)*(lsfCentroids+1)+i]
		}
		built.Synth.MinSpacing[v][lsfCentroids] = seed.MinDist[(1-v)*(lsfCentroids+1)+lsfCentroids]
	}

	// SmplTables static CDF sections.
	built.Synth.Centroids = *synthCentroids
	built.Synth.Matrices = *synthMatrices
	built.Synth.Grid16Matrices = synthGrid16Matrices
	built.Tables.LsfSel = [][]uint16{
		seed.LsfSel[0 : 0+3],
		seed.LsfSel[3 : 3+3],
		seed.LsfSel[6 : 6+3],
	}
	built.Tables.LsfGrid = LsfGrid{
		Match1:    cloneU16(seed.CmfCond[18:36]),
		Match1Alt: cloneU16(seed.CmfCond[0:18]),
		Match0:    cloneU16(seed.Cmf[0:17]),
		Match0Alt: cloneU16(seed.Cmf[17:34]),
	}
	built.Tables.LsfExtra = cloneU16(seed.LsfExtra)

	return &built
}

func valtablesSet(dst *[][][][][]float32, voiced, lr, c int, vt [][]float32) {
	// Grow [voiced][lr][c].
	for len(*dst) <= voiced {
		*dst = append(*dst, nil)
	}
	for len((*dst)[voiced]) <= lr {
		(*dst)[voiced] = append((*dst)[voiced], nil)
	}
	for len((*dst)[voiced][lr]) <= c {
		(*dst)[voiced][lr] = append((*dst)[voiced][lr], nil)
	}
	(*dst)[voiced][lr][c] = vt
}

func lsfStage2Set(dst *[][][][][]uint16, voiced, lr, c int, s2 [][]uint16) {
	for len(*dst) <= voiced {
		*dst = append(*dst, nil)
	}
	for len((*dst)[voiced]) <= lr {
		(*dst)[voiced] = append((*dst)[voiced], nil)
	}
	for len((*dst)[voiced][lr]) <= c {
		(*dst)[voiced][lr] = append((*dst)[voiced][lr], nil)
	}
	(*dst)[voiced][lr][c] = s2
}

func flatten(mat [lsfOrder][lsfOrder]float32, out *[lsfOrder * lsfOrder]float32) {
	p := 0
	for i := 0; i < lsfOrder; i++ {
		for j := 0; j < lsfOrder; j++ {
			out[p] = mat[i][j]
			p++
		}
	}
}

func cloneU16(v []uint16) []uint16 {
	out := make([]uint16, len(v))
	copy(out, v)
	return out
}

var (
	lsfBuiltOnce sync.Once
	lsfBuiltVal  *LsfBuilt
	lsfBuiltErr  error
)

// LsfBuilt loads the LSF seed ROM and builds all three runtime structs once.
// The blob bytes come from vectors/mlow/lsf_seed.bin (the reference embeds
// the same file).
func LoadLsfBuilt(blob []byte) (*LsfBuilt, error) {
	lsfBuiltOnce.Do(func() {
		seed, err := LoadLsfSeed(blob)
		if err != nil {
			lsfBuiltErr = err
			return
		}
		lsfBuiltVal = BuildLsfTables(seed)
	})
	return lsfBuiltVal, lsfBuiltErr
}
