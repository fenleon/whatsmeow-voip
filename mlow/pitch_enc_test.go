package mlow

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The seed-build tripwire: the expanded PitchTables must match the reference
// bit-for-bit (smpl_pitch_enc.rs:117).
func TestPitchTablesDebugChecksum(t *testing.T) {
	tab := LoadPitchTables()
	assert.Equal(t, uint64(0x21ce965d809c8048), tab.DebugChecksum())
}

// Feed the reference encoder's exact per-frame ltp_buf + F2 into the
// estimator (threading the cross-frame predictor + frame-boundary reset) and
// require the outputs to converge to the reference: pitchcorr (tight float
// tol), avg_lag, per-subframe laginds, blockseg_idx, harm_strength
// (cache-aliasing tol). Mirrors smpl_pitch_enc.rs:1097.
func TestPitchEstimatorMatchesGroundTruth(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/pitchio_ground_truth.json")
	require.NoError(t, err)
	var recs []struct {
		Frame       int       `json:"frame"`
		Cav         int       `json:"cav"`
		Pitchcorr   float64   `json:"pitchcorr"`
		AvgLag      float64   `json:"avg_lag"`
		Harm        float64   `json:"harm"`
		BlocksegIdx int       `json:"blockseg_idx"`
		PrevLagblk  int       `json:"prev_lagblk"`
		PrevLagidx  int       `json:"prev_lagidx"`
		Laginds     []int32   `json:"laginds"`
		LtpBuf      []float32 `json:"ltp_buf"`
		F2          []float32 `json:"F2"`
	}
	require.NoError(t, json.Unmarshal(data, &recs))
	require.GreaterOrEqual(t, len(recs), 30)

	// Thread prev_lag/prev_pitch_corr across frames (the estimator carries
	// these), but seed prev_lagblk/prev_lagidx from the reference dump per
	// frame so the rate-bias survivor selection uses the exact predictor (its
	// reset timing depends on the voiced decision, out of scope for the
	// isolated estimator test).
	st := &PitchEstState{}
	var maxPcErr, maxAvglagErr, maxHarmErr float32
	lagindMismatch, bsxMismatch, checked := 0, 0, 0
	for n, rec := range recs {
		cav := rec.Cav != 0
		st.PrevLagblk = int32(rec.PrevLagblk)
		st.PrevLagidx = int32(rec.PrevLagidx)
		require.Len(t, rec.LtpBuf, maxLtpBufLen)
		var f2 [pitchFlen]float32
		copy(f2[:], rec.F2)

		res := SmplPitch(st, rec.LtpBuf, &f2, cav)

		if !cav {
			continue
		}
		maxPcErr = maxF32(maxPcErr, f32Abs(res.Pitchcorr-float32(rec.Pitchcorr)))
		maxAvglagErr = maxF32(maxAvglagErr, f32Abs(res.AvgLag-float32(rec.AvgLag)))
		maxHarmErr = maxF32(maxHarmErr, f32Abs(res.HarmStrength-float32(rec.Harm)))
		for sf := 0; sf < pitchNumSubframes; sf++ {
			if res.Laginds[sf] != rec.Laginds[sf] {
				lagindMismatch++
				t.Logf("rec %d laginds[%d] got %d want %d", n, sf, res.Laginds[sf], rec.Laginds[sf])
				break
			}
		}
		if res.BlocksegIdx != rec.BlocksegIdx {
			bsxMismatch++
			t.Logf("rec %d blockseg_idx got %d want %d", n, res.BlocksegIdx, rec.BlocksegIdx)
		}
		checked++
	}
	assert.GreaterOrEqual(t, checked, 20, "too few active frames checked")
	assert.Less(t, maxPcErr, float32(1e-3), "pitchcorr diverges from reference")
	assert.Less(t, maxAvglagErr, float32(1e-3), "avg_lag diverges from reference")
	assert.Zero(t, lagindMismatch, "per-subframe laginds diverge from reference")
	assert.Zero(t, bsxMismatch, "blockseg_idx diverges from reference")
	assert.Less(t, maxHarmErr, float32(0.05), "harm_strength diverges beyond cache-aliasing tol")
}
