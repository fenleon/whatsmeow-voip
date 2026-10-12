package mlow

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type tocVector struct {
	B      byte  `json:"b"`
	Std    bool  `json:"std"`
	SID    bool  `json:"sid"`
	VAD    bool  `json:"vad"`
	SR     int32 `json:"sr"`
	Ms     int32 `json:"ms"`
	Voiced bool  `json:"voiced"`
	Active bool  `json:"active"`
	F2     bool  `json:"f2"`
	F0     bool  `json:"f0"`
}

func loadTocVectors(t *testing.T) []tocVector {
	t.Helper()
	data, err := os.ReadFile("../vectors/mlow/toc_vectors.json")
	require.NoError(t, err, "toc_vectors.json not found (run from the repo root)")
	var vectors []tocVector
	require.NoError(t, json.Unmarshal(data, &vectors))
	require.Len(t, vectors, 256, "the TOC table is exhaustive over the byte range")
	return vectors
}

func TestTocMatchesAll256(t *testing.T) {
	for _, rec := range loadTocVectors(t) {
		rec := rec
		t.Run("byte 0x"+toHex2(rec.B), func(t *testing.T) {
			toc := ParseMlowToc(rec.B)
			assert.Equal(t, rec.Std, toc.StdOpus, "std")
			assert.Equal(t, rec.SID, toc.SID, "sid")
			assert.Equal(t, rec.VAD, toc.VAD, "vad")
			assert.Equal(t, rec.SR, toc.SampleRate, "sr")
			assert.Equal(t, rec.Ms, toc.FrameMs, "ms")
			assert.Equal(t, rec.Voiced, toc.Voiced, "voiced")
			assert.Equal(t, rec.Active, toc.Active, "active")
			assert.Equal(t, rec.F2, toc.Flag2, "f2")
			assert.Equal(t, rec.F0, toc.Flag0, "f0")
		})
	}
}

func toHex2(b byte) string {
	const hexDigits = "0123456789abcdef"
	return string([]byte{hexDigits[b>>4], hexDigits[b&0xf]})
}
