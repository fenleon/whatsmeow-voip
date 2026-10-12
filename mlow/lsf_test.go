package mlow

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// loadBuiltTable loads a table blob and expands it, cached across tests.
var builtTables = func(t *testing.T) *LsfBuilt {
	t.Helper()
	blob, err := os.ReadFile("../vectors/mlow/lsf_seed.bin")
	require.NoError(t, err)
	built, err := LoadLsfBuilt(blob)
	require.NoError(t, err)
	return built
}

// f32bits exposes the IEEE bit pattern of a float32 for golden comparisons.
func f32bits(f float32) uint32 {
	return math.Float32bits(f)
}

// TestLsfSeedBuildGoldenChecksums pins the expansion output against the Rust
// reference's committed golden constants (smpl_lsf_seed.rs tests).
func TestLsfSeedBuildGoldenChecksums(t *testing.T) {
	built := builtTables(t)
	st1v := &built.Cb.St1[1]
	assert.Equal(t, uint32(0x3d93b440), f32bits(st1v.Cbhalf[0][0]), "cbhalf[1][0][0]")
	assert.Equal(t, uint32(0x3e0ff885), f32bits(st1v.We[0][0][0]), "we[1][0][0][0]")
	assert.Equal(t, uint32(0x4062b10f), f32bits(st1v.Wie[0][0][0]), "wie[1][0][0][0]")
	assert.Equal(t, uint32(0x40883c1d), f32bits(st1v.Bits[0]), "bits[1][0]")
	assert.EqualValues(t, 6, built.Cb.St2[1][0][0].NumQlvls[0], "numQlvls[1][0][0][0]")
	assert.Equal(t, uint32(0xbf2c0e76), f32bits(built.Cb.St2[1][0][0].Qlvls[0][0]), "Qlvls[1][0][0][0][0]")
	assert.Equal(t, []uint16{0, 33, 140, 932, 6942, 28552, 32763},
		built.Tables.LsfStage2[1][0][0][0], "lsf_stage2[1][0][0][0]")
	assert.Len(t, built.Synth.Valtables[1][0][0][0], 6, "valtables width = numQlvls")
}

func TestLsfSeedGrid16Derivation(t *testing.T) {
	blob, err := os.ReadFile("../vectors/mlow/lsf_seed.bin")
	require.NoError(t, err)
	seed, err := LoadLsfSeed(blob)
	require.NoError(t, err)
	built := BuildLsfTables(seed)

	assert.Equal(t, seed.RegCond, []float32{built.Synth.Grid16Alpha[0], built.Synth.Grid16Alpha[1]},
		"grid16_alpha == reg_cond")
	for v := 0; v < 2; v++ {
		for i := 0; i < lsfOrder; i++ {
			assert.Equal(t, seed.Mean[(1-v)*lsfOrder+i], built.Synth.Grid16W[v][i],
				"grid16_w[%d] == mean[1-v]", v)
		}
		for lr := 0; lr < 2; lr++ {
			want := unpack8_16x16(seed.RotCond8[(v*2+lr)*lsfOrder*lsfOrder:], rotCondScale[v], rotCondMin[v])
			var flat [lsfOrder * lsfOrder]float32
			flatten(want, &flat)
			assert.Equal(t, flat[:], built.Synth.Grid16Matrices[v][lr][:],
				"grid16_matrices[%d][%d] == unpack8(rot_cond_8)", v, lr)
		}
	}
}

type lsfVector struct {
	Frame  string  `json:"frame"`
	Stage1 int32   `json:"stage1"`
	Grid   int32   `json:"grid"`
	Extra  int32   `json:"extra"`
	Stage2 []int32 `json:"stage2"`
}

// TestLsfDecodeFrame0MatchesVectors validates the first-internal-frame LSF
// parse (range coder at body start, the reliable validation point) against
// the captured vectors over every active frame.
func TestLsfDecodeFrame0MatchesVectors(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/lsf_vectors.json")
	require.NoError(t, err)
	var vectors []lsfVector
	require.NoError(t, json.Unmarshal(data, &vectors))
	require.NotEmpty(t, vectors)
	built := builtTables(t)
	tables := &built.Tables

	for _, rec := range vectors {
		rec := rec
		t.Run("frame", func(t *testing.T) {
			frame, err := hex.DecodeString(rec.Frame)
			require.NoError(t, err)
			var st SmplLsfState
			dec := NewRangeDecoder(frame[1:])
			idx := DecodeSmplLsf(dec, tables, &st, 0, 0)
			assert.Equal(t, rec.Stage1, idx.Stage1, "stage1")
			assert.Equal(t, rec.Grid, idx.Grid, "grid")
			assert.Equal(t, rec.Extra, idx.Extra, "extra")
			assert.Equal(t, rec.Stage2, idx.Stage2[:], "stage2")
			assert.EqualValues(t, 0, dec.Err, "no decode error")
		})
	}
}
