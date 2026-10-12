package mlow

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type pulseEntry struct {
	Pos int32 `json:"pos"`
	Val int32 `json:"val"`
}

type pulseVector struct {
	Frame  string       `json:"frame"`
	Subfr  []int32      `json:"subfr"`
	Pulses []pulseEntry `json:"pulses"`
}

type gainsVector struct {
	Frame  string  `json:"frame"`
	GainQ  []int32 `json:"gain_q"`
	NrgRes []int32 `json:"nrg_res"`
}

// ccBuilt loads and expands the cc seed, cached across tests.
var ccBuilt = func(t *testing.T) *CcTables {
	t.Helper()
	blob, err := os.ReadFile("../vectors/mlow/cc_seed.bin")
	require.NoError(t, err)
	cc, err := LoadCcTables(blob)
	require.NoError(t, err)
	return cc
}

// decodeFrame0LSFAndPulses runs the shared LSF+pulse prefix on one captured
// frame (first internal frame, config 0).
func decodeFrame0LSFAndPulses(t *testing.T, frameHex string, cc *CcTables, tables *SmplTables) (*RangeDecoder, SmplLsfIndices, *SmplPulseResult) {
	t.Helper()
	frame, err := hex.DecodeString(frameHex)
	require.NoError(t, err)
	var st SmplLsfState
	dec := NewRangeDecoder(frame[1:])
	lsf := DecodeSmplLsf(dec, tables, &st, 0, 0)
	pr := DecodeSmplPulses(dec, cc, 320, 4, 1, 0, lsf.Stage1)
	return dec, lsf, pr
}

func TestPulsesMatchVectors(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/pulse_vectors.json")
	require.NoError(t, err)
	var vectors []pulseVector
	require.NoError(t, json.Unmarshal(data, &vectors))
	require.NotEmpty(t, vectors)
	built := builtTables(t)
	cc := ccBuilt(t)

	for _, rec := range vectors {
		rec := rec
		t.Run("frame", func(t *testing.T) {
			dec, _, pr := decodeFrame0LSFAndPulses(t, rec.Frame, cc, &built.Tables)
			assert.Equal(t, rec.Subfr, pr.Subfr[:], "subfr")

			want := make([]int32, len(pr.Pulses))
			for _, pu := range rec.Pulses {
				want[pu.Pos] = pu.Val
			}
			assert.Equal(t, want, pr.Pulses, "pulse vector")
			assert.EqualValues(t, 0, dec.Err, "no decode error")
		})
	}
}

func TestGainsMatchVectors(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/gains_vectors.json")
	require.NoError(t, err)
	var vectors []gainsVector
	require.NoError(t, json.Unmarshal(data, &vectors))
	require.NotEmpty(t, vectors)
	built := builtTables(t)
	cc := ccBuilt(t)

	for _, rec := range vectors {
		rec := rec
		t.Run("frame", func(t *testing.T) {
			dec, _, pr := decodeFrame0LSFAndPulses(t, rec.Frame, cc, &built.Tables)
			g := DecodeSmplGains(dec, cc, 4, pr.Subfr)
			assert.Equal(t, rec.GainQ, g.GainQ[:], "gain_q")
			assert.Equal(t, rec.NrgRes, g.NrgRes[:], "nrg_res")
			assert.EqualValues(t, 0, dec.Err, "no decode error")
		})
	}
}

func TestCorruptSplitZeroesSubframes(t *testing.T) {
	cc := ccBuilt(t)
	dec := NewRangeDecoder(make([]byte, 8))
	// count=10 > 2*granularity=2 hits the lo < min_split corrupt branch.
	assert.Equal(t, int32(-1), smplSplit3537(dec, cc, 10, 1))
}
