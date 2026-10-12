package mlow

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type pitchVector struct {
	Frame    string  `json:"frame"`
	Lag      int32   `json:"lag"`
	Contour  int32   `json:"contour"`
	GainIdx  []int32 `json:"gain_idx"`
	FiltIdx  []int32 `json:"filt_idx"`
	IntLagQ6 []int32 `json:"int_lag_q6"`
}

// memBuilt builds the pitch heap window once for the test binary.
var memBuilt = func(t *testing.T) *SmplMem {
	t.Helper()
	blob, err := os.ReadFile("../vectors/mlow/pitch_seed.bin")
	require.NoError(t, err)
	mem, err := LoadSmplMem(blob)
	require.NoError(t, err)
	return mem
}

// decodeFrame0Pipeline runs LSF → pulses → pitch on one captured frame
// (first internal frame, config 0).
func decodeFrame0Pipeline(t *testing.T, frameHex string, cc *CcTables, tables *SmplTables, mem *SmplMem) (*RangeDecoder, *SmplPitchResult) {
	t.Helper()
	frame, err := hex.DecodeString(frameHex)
	require.NoError(t, err)
	var st SmplLsfState
	dec := NewRangeDecoder(frame[1:])
	lsf := DecodeSmplLsf(dec, tables, &st, 0, 0)
	pulses := DecodeSmplPulses(dec, cc, 320, 4, 1, 0, lsf.Stage1)
	pr := DecodeSmplPitch(dec, mem, cc, &st, 320, 4, 0, pulses.Subfr)
	return dec, pr
}

func TestPitchMatchVectors(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/pitch_vectors.json")
	require.NoError(t, err)
	var vectors []pitchVector
	require.NoError(t, json.Unmarshal(data, &vectors))
	require.NotEmpty(t, vectors)
	built := builtTables(t)
	cc := ccBuilt(t)
	mem := memBuilt(t)

	for _, rec := range vectors {
		rec := rec
		t.Run("frame", func(t *testing.T) {
			dec, pr := decodeFrame0Pipeline(t, rec.Frame, cc, &built.Tables, mem)
			assert.Equal(t, rec.Lag, pr.Lag, "lag")
			assert.Equal(t, rec.Contour, pr.Contour, "contour")
			assert.Equal(t, rec.GainIdx, pr.GainIdx[:], "gain_idx")
			assert.Equal(t, rec.FiltIdx, pr.FiltIdx[:], "filt_idx")
			assert.Equal(t, rec.IntLagQ6, pr.IntLagQ6[:], "int_lag_q6")
			assert.EqualValues(t, 0, dec.Err, "no decode error")
		})
	}
}
