package mlow

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// lsfQuantBuilt loads and expands the LSF seed (codebook included), cached
// across tests.
var lsfQuantBuilt = func(t *testing.T) *LsfBuilt {
	t.Helper()
	lsfOnce.Do(func() {
		blob, err := os.ReadFile("../vectors/mlow/lsf_seed.bin")
		require.NoError(t, err)
		built, err := LoadLsfBuilt(blob)
		require.NoError(t, err)
		lsfBuiltVal = built
	})
	return lsfBuiltVal
}

var (
	lsfOnce     sync.Once
	lsfBuiltVal *LsfBuilt
)

// Tier-B: feed the reference's post-A2NLSF lsf + LPC A + state to our
// quantizer; both the indices qi[] AND the reconstructed qlsf must match the
// reference bit-for-bit (qi exact, qlsf within 1e-4). Mirrors
// smpl_lsf_quant.rs:596.
func TestLsfQuantMatchesC(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/lsf_quant_io.json")
	require.NoError(t, err)
	var recs []map[string]any
	require.NoError(t, json.Unmarshal(data, &recs))
	require.NotEmpty(t, recs)
	built := lsfQuantBuilt(t)

	for n, r := range recs {
		n := n
		r := r
		t.Run(quantRecName(n, r), func(t *testing.T) {
			lsf := fvec(t, r["lsf"])
			a := fvec(t, r["A"])
			voiced := int(r["voiced"].(float64))
			lowRate := int(r["lowRate"].(float64))
			surv := int(r["surv"].(float64))
			rdW := float32(r["RDw_adj"].(float64))
			cond := r["cond_coding"].(float64) != 0
			prevLsf := fvec(t, r["prev_lsf"])
			wantQi := ivec(t, r["qi"])
			wantQlsf := fvec(t, r["qlsf"])

			var aArr [lsfOrder + 1]float32
			copy(aArr[:], a)
			var lsfArr, prevArr [lsfOrder]float32
			copy(lsfArr[:], lsf)
			copy(prevArr[:], prevLsf)

			var res LsfQuantResult
			if cond {
				res = LsfQuantCond(&built.Cb, &aArr, &lsfArr, &prevArr, voiced, lowRate, rdW, surv)
			} else {
				res = LsfQuant(&built.Cb, &aArr, &lsfArr, voiced, lowRate, rdW, surv)
			}
			require.Equal(t, wantQi, ivecFrom17(res.Qi[:]), "qi mismatch vs reference")
			for k := 0; k < lsfOrder; k++ {
				assert.InDelta(t, wantQlsf[k], res.Qlsf[k], 1e-4, "qlsf[%d]", k)
			}
		})
	}
}

func quantRecName(n int, r map[string]any) string {
	return fmt.Sprintf("rec%02d_v%d_lr%d_cond%v", n,
		int(r["voiced"].(float64)), int(r["lowRate"].(float64)),
		r["cond_coding"].(float64) != 0)
}

func ivec(t *testing.T, v any) []int32 {
	t.Helper()
	arr := v.([]any)
	out := make([]int32, len(arr))
	for i, x := range arr {
		out[i] = int32(x.(float64))
	}
	return out
}

func ivecFrom17(v []int32) []int32 {
	out := make([]int32, len(v))
	copy(out, v)
	return out
}
