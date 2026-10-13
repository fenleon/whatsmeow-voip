package mlow

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type vadGroundTruthEntry struct {
	Pkt   int     `json:"pkt"`
	Frame int     `json:"frame"`
	Cav   int     `json:"cav"`
	Spact float64 `json:"spact"`
}

// loadSynthMic reads the synthetic mic capture as little-endian int16 samples.
func loadSynthMic(t *testing.T) []int16 {
	t.Helper()
	raw, err := os.ReadFile("../vectors/mlow/synth_mic.raw")
	require.NoError(t, err)
	samples := make([]int16, len(raw)/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return samples
}

// runVadOverCapture processes the capture in 960-sample (60 ms) packets and
// returns the flat per-internal-frame results (3 per packet).
func runVadOverCapture(t *testing.T, samples []int16) [][2]any {
	t.Helper()
	vad := NewSmplVadState()
	var results [][2]any
	for off := 0; off+960 <= len(samples); off += 960 {
		r := vad.ProcessPacket(samples[off:off+960], 320)
		for f := 0; f < 3; f++ {
			cav := 0
			if r.CodedAsActiveVoice {
				cav = 1
			}
			results = append(results, [2]any{r.VadResults[f], cav})
		}
	}
	return results
}

func TestVadMatchesGroundTruth(t *testing.T) {
	samples := loadSynthMic(t)
	data, err := os.ReadFile("../vectors/mlow/vad_ground_truth.json")
	require.NoError(t, err)
	var gt []vadGroundTruthEntry
	require.NoError(t, json.Unmarshal(data, &gt))
	require.NotEmpty(t, gt)

	results := runVadOverCapture(t, samples)

	// The ground truth covers the prefix of packets where the carried
	// noise-level state stays bit-exact (smpl_vad.rs:503-506); compare only
	// the recorded prefix.
	last := gt[len(gt)-1]
	require.Less(t, last.Pkt*3+last.Frame, len(results), "capture too short for ground truth prefix")

	for _, rec := range gt {
		rec := rec
		t.Run(fmt.Sprintf("pkt%d_frame%d", rec.Pkt, rec.Frame), func(t *testing.T) {
			got := results[rec.Pkt*3+rec.Frame]
			assert.InDelta(t, rec.Spact, float64(got[0].(float32)), 1e-4, "spact")
			assert.Equal(t, rec.Cav, got[1].(int), "cav")
		})
	}
}

func TestVadBehavioral(t *testing.T) {
	t.Run("silence gives low activity", func(t *testing.T) {
		vad := NewSmplVadState()
		r := vad.ProcessPacket(make([]int16, 960), 320)
		for _, v := range r.VadResults {
			assert.LessOrEqual(t, v, float32(0.05))
		}
	})

	t.Run("speechlike onset is active", func(t *testing.T) {
		// The SILK VAD adapts its noise floor toward a sustained signal, so
		// settle the floor on quiet input, then present a loud broadband
		// onset and check the activity rises on that onset before the floor
		// catches up (mirrors smpl_vad.rs:471-495).
		vad := NewSmplVadState()
		quiet := make([]int16, 960)
		for i := 0; i < 30; i++ {
			vad.ProcessPacket(quiet, 320)
		}
		loud := make([]int16, 960)
		seed := uint32(0x12345678)
		for i := range loud {
			tt := float64(i) / 16000.0
			tone := math.Sin(2*math.Pi*200*tt) +
				math.Sin(2*math.Pi*900*tt) +
				math.Sin(2*math.Pi*2500*tt)
			seed = seed*1664525 + 1013904223
			noise := float32(seed>>16)/65536.0 - 0.5
			loud[i] = int16(6000 * (tone/3 + 0.2*float64(noise)))
		}
		r := vad.ProcessPacket(loud, 320)
		onset := r.VadResults[0]
		if r.VadResults[1] > onset {
			onset = r.VadResults[1]
		}
		if r.VadResults[2] > onset {
			onset = r.VadResults[2]
		}
		assert.Greater(t, onset, float32(0.5), "speech-like onset activity")
	})
}
