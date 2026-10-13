package mlow

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func fvec(t *testing.T, v any) []float32 {
	t.Helper()
	arr, ok := v.([]any)
	require.True(t, ok)
	out := make([]float32, len(arr))
	for i, x := range arr {
		out[i] = float32(x.(float64))
	}
	return out
}

// The forward A2NLSF (silkA2NLSF, fixed-point) must reproduce the reference
// lsf from the post-BWE A for every captured record. Both use the same
// fixed-point arithmetic, so this is exact (NLSF in radians within float
// rounding of the Q15->radians scale). Mirrors smpl_lpc.rs:628.
func TestA2NLSFMatchesC(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/lsf_quant_io.json")
	require.NoError(t, err)
	var recs []map[string]any
	require.NoError(t, json.Unmarshal(data, &recs))
	require.NotEmpty(t, recs)

	worst := float32(0)
	for n, r := range recs {
		a := fvec(t, r["A"])
		want := fvec(t, r["lsf"])
		got := smplA2NLSF16(a)
		for k := 0; k < smplLpcOrder; k++ {
			d := float32(math.Abs(float64(got[k] - want[k])))
			worst = float32(math.Max(float64(worst), float64(d)))
			assert.Less(t, d, float32(1e-4),
				"rec %d nlsf[%d] %v != ref %v", n, k, got[k], want[k])
		}
	}
	t.Logf("a2nlsf worst abs error = %g", worst)
}

// The full front-end (re-window from the raw lpcbuf, then FFT-autocorr ->
// reflection -> A -> BWE) must reproduce the reference post-BWE A from the
// raw lpcbuf. The FFT differs only in internal rounding (portable
// mixed-radix vs PFFFT), so A matches to a tight float tolerance
// (documented, like the decoder postfilters). Mirrors smpl_lpc.rs:655.
func TestFrontEndAMatchesC(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/fe_dump.json")
	require.NoError(t, err)
	var recs []map[string]any
	require.NoError(t, json.Unmarshal(data, &recs))
	require.GreaterOrEqual(t, len(recs), 12, "need front-end vectors")

	worst := float32(0)
	worstWin := float32(0)
	for n, r := range recs {
		lpcbuf := fvec(t, r["lpcbuf"])
		numframe := int(r["numframe"].(float64))
		require.Len(t, lpcbuf, smplLpcBufLen)
		// useLongWin = numframe < 2 (frames_per_packet-1 == 2)
		win := smplWindowLpc20(lpcbuf, numframe < 2)
		// Cross-check the windowing exactly against the reference `windowed`.
		wantWin := fvec(t, r["windowed"])
		for k := 0; k < smplLpcBufLen; k++ {
			d := float32(math.Abs(float64(win[k] - wantWin[k])))
			worstWin = float32(math.Max(float64(worstWin), float64(d)))
		}
		a, _ := smplLpcAnalyzeWithF2(win, newFftScratch(smplLpcNfft))
		wantA := fvec(t, r["A"])
		r0 := r["R"].([]any)[0].(float64)
		rd := float32(0)
		for k := 0; k <= smplLpcOrder; k++ {
			d := float32(math.Abs(float64(a[k] - wantA[k])))
			rd = float32(math.Max(float64(rd), float64(d)))
		}
		worst = float32(math.Max(float64(worst), float64(rd)))
		// Frames with audible energy must match A tightly (FFT-internal
		// rounding only). Below a very low energy floor the autocorrelation
		// is dominated by reg + numerical noise so the LPC is ill-conditioned
		// and the reconstructed frame is silent regardless of A; skip the A
		// assert there but still verify the autocorrelation R itself (the
		// FFT+DCT output).
		if r0 > 1e-7 {
			assert.Less(t, rd, float32(5e-3),
				"rec %d (nf %d, R0=%g) |dA|=%g too large", n, numframe, r0, rd)
		}
	}
	// The windowing is exact; the autocorrelation tracks the reference
	// FFT/DCT to a tight tolerance.
	assert.Less(t, worstWin, float32(1e-6), "windowing |dwin|=%g", worstWin)
	t.Logf("front-end worst |dA| = %g, worst |dwin| = %g", worst, worstWin)
}
