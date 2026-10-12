package mlow

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestHarmPostfilterMatchesVectors validates the harmonic postfilter against
// the instrumented reference decoder, processing the full active packet
// sequence in order (the filter carries StateComb/state1/prev_lag across
// packets). The reference is -ffast-math: steady-state must match within the
// i16 quantization step; the only larger residual is the first 48 samples of
// a silence packet immediately following voiced (the comb's zero-input
// response).
func TestHarmPostfilterMatchesVectors(t *testing.T) {
	var i16LSB float32 = 1.0 / 32768.0
	const transitionTol float32 = 6.0e-4
	data, err := os.ReadFile("../vectors/mlow/harm_postfilter_vectors.raw")
	require.NoError(t, err)
	o := 0
	count := ri32(data, &o)
	require.Greater(t, count, int32(0))
	st := NewHarmPostfilterState()
	var worst, worstSteady float32
	for i := int32(0); i < count; i++ {
		ri32(data, &o) // packet
		plen := int(ri32(data, &o))
		nlags := int(ri32(data, &o))
		nbr := rf32(data, &o)
		lags := make([]float32, nlags)
		for j := range lags {
			lags[j] = rf32(data, &o)
		}
		inp := make([]float32, plen)
		for j := range inp {
			inp[j] = rf32(data, &o)
		}
		cout := make([]float32, plen)
		for j := range cout {
			cout[j] = rf32(data, &o)
		}
		transition := lags[0] == 0.0
		HarmPostfilter(st, inp, plen, lags, nlags, nbr)
		for j := 0; j < plen; j++ {
			d := inp[j] - cout[j]
			if d < 0 {
				d = -d
			}
			if d > worst {
				worst = d
			}
			if !(transition && j < TotPostfiltDelay) && d > worstSteady {
				worstSteady = d
			}
		}
	}
	assert.Less(t, worstSteady, i16LSB,
		"harm_postfilter steady-state diverges from reference by %e", worstSteady)
	assert.Less(t, worst, transitionTol,
		"harm_postfilter transition residual %e exceeds tolerance %e", worst, transitionTol)
}

func loadCaptureFrames(t *testing.T) []string {
	t.Helper()
	data, err := os.ReadFile("../vectors/mlow/inbound_capture_frames.json")
	require.NoError(t, err)
	var frames []string
	require.NoError(t, json.Unmarshal(data, &frames))
	return frames
}

// TestE2EDecodeMatchesUseSmpl decodes the whole capture and compares against
// the reference output (ref_usesmpl_expected.raw), aligned at lag 0 (the
// harmonic postfilter emits the 48-sample group delay).
func TestE2EDecodeMatchesUseSmpl(t *testing.T) {
	frames := loadCaptureFrames(t)
	raw, err := os.ReadFile("../vectors/mlow/ref_usesmpl_expected.raw")
	require.NoError(t, err)
	refp := make([]float32, len(raw)/2)
	for i := range refp {
		refp[i] = float32(int16(binary.LittleEndian.Uint16(raw[i*2:]))) / 32768.0
	}

	dec := NewMlowDecoder()
	var out []float32
	for _, hexFrame := range frames {
		frame, err := hex.DecodeString(hexFrame)
		require.NoError(t, err)
		out = append(out, dec.Decode(frame)...)
	}
	require.Equal(t, len(refp), len(out), "decode length vs reference")

	const lag = 0
	n := len(refp) - lag
	r, o := refp[lag:lag+n], out[:n]
	var mr, mo float64
	for i := 0; i < n; i++ {
		mr += float64(r[i])
		mo += float64(o[i])
	}
	mr /= float64(n)
	mo /= float64(n)
	var sxy, sxx, syy float64
	for i := 0; i < n; i++ {
		dr := float64(r[i]) - mr
		dz := float64(o[i]) - mo
		sxy += dr * dz
		sxx += dr * dr
		syy += dz * dz
	}
	corr := sxy / math.Sqrt(sxx*syy)
	assert.Greater(t, corr, 0.95, "lag-0 corr %0.4f vs reference", corr)
}

// TestFuzzDecodeNoPanicBoundedOutput: the decoder is fed adversarial inputs
// and must neither panic nor over-emit. Corpus: a deterministic LCG of random
// byte vectors, plus every capture frame with each single byte flipped and
// each prefix truncation.
func TestFuzzDecodeNoPanicBoundedOutput(t *testing.T) {
	const maxSamps = 32 * 120 // max sample_rate(kHz) * max frame_ms across all TOCs
	dec := NewMlowDecoder()
	check := func(input []byte) {
		out := dec.Decode(input)
		assert.LessOrEqual(t, len(out), maxSamps,
			"decode emitted %d samples for input len %d", len(out), len(input))
	}

	// Deterministic LCG (numerical-recipes constants) over thousands of
	// random-length buffers.
	var seed uint32 = 0x12345678
	next := func() uint32 {
		seed = seed*1664525 + 1013904223
		return seed
	}
	for i := 0; i < 8000; i++ {
		length := int(next() % 400)
		buf := make([]byte, length)
		for j := range buf {
			buf[j] = byte(next() >> 24)
		}
		check(buf)
	}

	// Mutations of the real capture frames: every single-byte flip and every
	// truncation.
	for _, hexFrame := range loadCaptureFrames(t) {
		frame, err := hex.DecodeString(hexFrame)
		require.NoError(t, err)
		for i := 0; i < len(frame); i++ {
			for bit := 0; bit < 8; bit++ {
				m := append([]byte(nil), frame...)
				m[i] ^= 1 << bit
				check(m)
			}
			check(frame[:i]) // truncation at every prefix length
		}
		check(frame)
	}
}

func TestUnsupportedFramesDropCleanWithoutDesync(t *testing.T) {
	real, err := hex.DecodeString(loadCaptureFrames(t)[0])
	require.NoError(t, err)
	dec := NewMlowDecoder()

	// A real 0x50 (16kHz, low_rate=0) frame decodes to audio — the guards
	// don't false-positive.
	normal := dec.Decode(real)
	require.Len(t, normal, 960)
	assert.True(t, anyNonZero(normal), "a real low_rate=0 frame must decode to audio")

	// A 32kHz/fullband TOC (bit5=1, e.g. 0x70) -> dropped to 60ms silence.
	out32k := dec.Decode([]byte{0x70, 0xAA, 0xBB, 0xCC})
	assert.Len(t, out32k, 960)
	assert.False(t, anyNonZero(out32k), "32kHz frame must drop to silence")

	// A low_rate=1 TOC -> dropped to 60ms silence (0x54: 16kHz, 60ms, bit2=1).
	outLR := dec.Decode([]byte{0x54, 0xAA, 0xBB, 0xCC})
	assert.Len(t, outLR, 960)
	assert.False(t, anyNonZero(outLR), "low_rate=1 frame must drop to silence")

	// A non-60ms ACTIVE 16kHz/low_rate=0 TOC (20ms, e.g. 0x48) must also
	// drop: the active decode hardcodes the 3x20ms / 60ms geometry.
	out20ms := dec.Decode([]byte{0x48, 0xAA, 0xBB, 0xCC})
	assert.Len(t, out20ms, 960)
	assert.False(t, anyNonZero(out20ms), "a non-60ms active frame must drop to silence")

	assert.False(t, dec.HadError(), "dropped frames must not touch the range decoder")
	after := dec.Decode(real)
	assert.True(t, anyNonZero(after),
		"a real frame after the drops must still decode (no poisoned predictor)")
}

func TestInactiveOffPointFrameIsSilencedNotDropped(t *testing.T) {
	real, err := hex.DecodeString(loadCaptureFrames(t)[0])
	require.NoError(t, err)
	dec := NewMlowDecoder()

	assert.True(t, anyNonZero(dec.Decode(real)),
		"a real low_rate=0 frame must decode to audio")

	// TOC 0x00 -> inactive: silenced via the DTX/SID path, not the
	// operating-point drop.
	inactive := dec.Decode([]byte{0x00, 0xAA, 0xBB, 0xCC})
	assert.Len(t, inactive, 960, "an inactive frame still fills a 60ms slot")
	assert.False(t, anyNonZero(inactive), "an inactive frame must be silence")
	assert.Equal(t, uint32(0), dec.DroppedUnsupported(),
		"an inactive frame is the DTX/silence path, not an operating-point drop")
	assert.False(t, dec.HadError(), "the inactive path must not open the range decoder")

	// Contrast: an active off-point frame (0x48 = vad=true, 20ms) IS counted.
	dec.Decode([]byte{0x48, 0xAA, 0xBB, 0xCC})
	assert.Equal(t, uint32(1), dec.DroppedUnsupported(),
		"an active off-point frame must count as a drop")

	// After both, a real frame still decodes: no desync, no poisoned predictor.
	assert.True(t, anyNonZero(dec.Decode(real)),
		"a real frame must still decode after an inactive frame and a drop")
}

func anyNonZero(v []float32) bool {
	for _, s := range v {
		if s != 0.0 {
			return true
		}
	}
	return false
}

func TestRedEnvelopeDecodesToBareMain(t *testing.T) {
	bare, err := hex.DecodeString(loadCaptureFrames(t)[0])
	require.NoError(t, err)

	bareDec := NewMlowDecoder()
	bareOut := bareDec.Decode(bare)

	// SplitRed N=1: red_hdr [0x80|tc, size], main_marker (high bit clear),
	// red payload, main.
	redPayload := []byte{0xAA, 0xBB}
	env := []byte{0x80, byte(len(redPayload)), 0x00}
	env = append(env, redPayload...)
	env = append(env, bare...)

	redDec := NewMlowDecoder()
	redDec.SetRedundancy(1)
	redOut := redDec.Decode(env)

	assert.Equal(t, bareOut, redOut, "RED-wrapped main differs from bare decode")
	assert.False(t, redDec.HadError(), "RED decode raised the range decoder error flag")
}

func TestDepackSplitRed(t *testing.T) {
	t.Run("one redundant plus main", func(t *testing.T) {
		p := []byte{0x85, 0x03, 0x00, 0xAA, 0xBB, 0xCC, 0x50, 0x11, 0x22, 0x33}
		frames, err := DepackSplitRed(p)
		require.NoError(t, err)
		require.Len(t, frames, 2)
		assert.Equal(t, MlowFrame{Data: []byte{0xAA, 0xBB, 0xCC}, TimeCode: 5}, frames[0])
		assert.Equal(t, MlowFrame{Data: []byte{0x50, 0x11, 0x22, 0x33}, TimeCode: 0, IsMain: true}, frames[1])
	})
	t.Run("no redundant just main", func(t *testing.T) {
		p := []byte{0x00, 0x50, 0x11, 0x22}
		frames, err := DepackSplitRed(p)
		require.NoError(t, err)
		require.Len(t, frames, 1)
		assert.True(t, frames[0].IsMain)
		assert.Equal(t, []byte{0x50, 0x11, 0x22}, frames[0].Data)
	})
	t.Run("empty packet errors", func(t *testing.T) {
		_, err := DepackSplitRed(nil)
		assert.ErrorIs(t, err, ErrRedPktSizeZero)
	})
	t.Run("bare frame is rejected", func(t *testing.T) {
		_, err := DepackSplitRed([]byte{0x90, 0x01, 0x02})
		assert.Error(t, err)
	})
}
