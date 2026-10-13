package mlow

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Feed the reference encoder's exact per-frame pitchcorr/avg_lag/harm/lags/
// F2/sp_act_prob (in stream order, threading one VuvMode) and require our
// voicing_strength + voiced decision to match the reference output. Isolates
// the classifier from the pitch estimator. Mirrors smpl_signal_mode.rs:222.
func TestSignalModeMatchesGroundTruth(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/sigmode_ground_truth.json")
	require.NoError(t, err)
	var recs []struct {
		Frame     int       `json:"frame"`
		Pitchcorr float64   `json:"pitchcorr"`
		AvgLag    float64   `json:"avg_lag"`
		Harm      float64   `json:"harm"`
		SpActProb float64   `json:"sp_act_prob"`
		Vstr      float64   `json:"vstr"`
		Voiced    int       `json:"voiced"`
		Lags      []float32 `json:"lags"`
		F2        []float32 `json:"F2"`
	}
	require.NoError(t, json.Unmarshal(data, &recs))
	require.GreaterOrEqual(t, len(recs), 12)

	vuv := &VuvMode{}
	var maxErr, maxHarmErr float32
	for _, rec := range recs {
		var f2 [pitchFlen]float32
		copy(f2[:], rec.F2)

		// Validate HarmStrengthAt against the reference harm on frames where
		// the pitch search ran. On inactive frames the pitch search
		// early-returns (lag clamped to the 32-sample floor) and never
		// computes harmonicity, leaving harm at its 0.0 init, not a
		// recompute target.
		avgLag := float32(rec.AvgLag)
		if avgLag > 33.0 {
			f2w := buildF2w(&f2)
			harmRs := HarmStrengthAt(avgLag, &f2w)
			maxHarmErr = maxF32(maxHarmErr, f32Abs(harmRs-float32(rec.Harm)))
		}

		vstrRs := SmplGetSignalMode(float32(rec.Pitchcorr), rec.Lags, avgLag,
			float32(rec.Harm), &f2, float32(rec.SpActProb), vuv)
		maxErr = maxF32(maxErr, f32Abs(vstrRs-float32(rec.Vstr)))
		// Voiced decision (all dump frames are coded_as_active_voice).
		assert.Equal(t, float32(rec.Vstr) > 0.0, vstrRs > 0.0,
			"voiced flip frame %d vstr_rs=%v vstr_c=%v", rec.Frame, vstrRs, rec.Vstr)
	}
	assert.Less(t, maxErr, float32(1e-4),
		"voicing_strength diverges from reference: max_err=%g", maxErr)
	// HarmStrengthAt is exact PER call, but the reference reuses a
	// survivor-loop cache keyed by a quantized harm bin, so its FINAL value
	// can be one computed at a different (bin-sharing) survivor lag. A
	// fresh-cache recompute is therefore close but not bit-exact without the
	// full pitch survivor sequence; bound the residual.
	assert.Less(t, maxHarmErr, float32(0.05),
		"harm_strength diverges from reference beyond cache-aliasing tolerance: %g", maxHarmErr)
}
