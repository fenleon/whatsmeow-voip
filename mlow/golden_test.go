//go:build cgo

package mlow

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Self-contained no-regression guard for the mlow codec: the encode->decode
// pipeline over a deterministic synthetic signal (voiced tone, unvoiced
// noise, voiced+noise, silence) must reproduce the committed golden digests.
// Mirrors wacore/src/voip/mlow/golden.rs.
const (
	goldenFrame   = 960
	goldenPackets = 24

	goldenSamples        = 23040
	goldenRms            = float32(0.154699)
	goldenPeak           = float32(0.458405)
	goldenClip           = float32(0.0)
	goldenChecksum       = uint64(0x8782cb8cd7b5f262)
	goldenFramesChecksum = uint64(0xb1793f1072d15a7d)
)

// goldenSynthInput cycles voiced tone / unvoiced noise / voiced+noise /
// silence, so the round-trip exercises the voiced (LTP) and unvoiced (gains)
// encode paths from code alone. The silence cycle is encoded as a low-energy
// active frame, not DTX (the encoder has no DTX/SID path).
func goldenSynthInput() []float32 {
	out := make([]float32, 0, goldenPackets*goldenFrame)
	seed := uint32(0x9e3779b9)
	for i := 0; i < goldenPackets*goldenFrame; i++ {
		t := float32(i) / 16000.0
		seed = seed*1664525 + 1013904223
		noise := float32(seed>>8)/8388608.0 - 1.0 // [-1, 1)
		var v float32
		switch (i / goldenFrame) % 4 {
		case 0:
			v = 0.30*f32Sin(2*smplPi*150*t) + 0.10*f32Sin(2*smplPi*300*t)
		case 1:
			v = 0.18 * noise
		case 2:
			v = 0.25*f32Sin(2*smplPi*120*t) + 0.05*noise
		}
		out = append(out, v)
	}
	return out
}

func fnv1aGo(bytes []byte) uint64 {
	h := uint64(0xcbf29ce484222325)
	for _, b := range bytes {
		h ^= uint64(b)
		h *= 0x100000001b3
	}
	return h
}

func TestGoldenRoundtripNoRegression(t *testing.T) {
	input := goldenSynthInput()
	enc := NewMlowEncoder()
	dec := NewMlowDecoder()
	var out []int16
	var pktBytes []byte
	for off := 0; off+goldenFrame <= len(input); off += goldenFrame {
		pkt, err := enc.Encode(input[off : off+goldenFrame])
		require.NoError(t, err)
		pktBytes = append(pktBytes, pkt...)
		for _, s := range dec.Decode(pkt) {
			v := s * 32767.0
			if v < -32768.0 {
				v = -32768.0
			} else if v > 32767.0 {
				v = 32767.0
			}
			out = append(out, int16(v))
		}
		// The clean synthetic corpus must never make the range decoder raise
		// its error flag.
		assert.False(t, dec.HadError(), "clean corpus raised the range decoder error flag")
		if dec.HadError() {
			break
		}
	}

	require.Len(t, out, goldenSamples, "output length drifted")
	// Digest.
	var sq float64
	peak := int32(0)
	clip := 0
	var bytes []byte
	for _, s := range out {
		a := int32(s)
		if a < 0 {
			a = -a
		}
		sq += float64(s) * float64(s)
		if a > peak {
			peak = a
		}
		if a >= 32767 {
			clip++
		}
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(s))
		bytes = append(bytes, b[:]...)
	}
	rms := float32(math.Sqrt(sq/float64(len(out))) / 32768.0)
	peakF := float32(peak) / 32768.0
	clipPct := 100.0 * float32(clip) / float32(len(out))
	checksum := fnv1aGo(bytes)
	framesChecksum := fnv1aGo(pktBytes)

	assert.Less(t, math.Abs(float64(rms-goldenRms)), 1e-3, "RMS drifted %v vs golden %v", rms, goldenRms)
	assert.Less(t, math.Abs(float64(peakF-goldenPeak)), 1e-3, "peak drifted %v vs golden %v", peakF, goldenPeak)
	assert.Less(t, math.Abs(float64(clipPct-goldenClip)), 0.1, "clip drifted %v vs golden %v", clipPct, goldenClip)
	assert.Equal(t, goldenChecksum, checksum,
		"output bytes drifted (codec regression). scalars rms=%v peak=%v clip=%v", rms, peakF, clipPct)
	assert.Equal(t, goldenFramesChecksum, framesChecksum,
		"encoder bitstream drifted (encoder regression)")
}
