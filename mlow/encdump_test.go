//go:build cgo

package mlow

import (
	"bufio"
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

func TestEncDumpCompare(t *testing.T) {
	hexPath := os.Getenv("MLOW_REF_HEX")
	if hexPath == "" {
		t.Skip("MLOW_REF_HEX not set")
	}
	f, err := os.Open(hexPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	raw, err := os.ReadFile("../vectors/mlow/synth_mic.raw")
	if err != nil {
		t.Fatal(err)
	}
	samples := make([]float32, len(raw)/2)
	for i := range samples {
		samples[i] = float32(int16(binary.LittleEndian.Uint16(raw[2*i:]))) / 32768.0
	}
	enc := NewMlowEncoder()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	pkt := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		off := pkt * 960
		if off+960 > len(samples) {
			break
		}
		got, err := enc.Encode(samples[off : off+960])
		if err != nil {
			t.Fatalf("pkt %d: %v", pkt, err)
		}
		var gotHex strings.Builder
		for _, b := range got {
			const d = "0123456789abcdef"
			gotHex.WriteByte(d[b>>4])
			gotHex.WriteByte(d[b&15])
		}
		if gotHex.String() != line {
			// find first differing byte
			n := len(got)
			if len(line)/2 < n {
				n = len(line) / 2
			}
			first := -1
			for i := 0; i < n; i++ {
				if got[i] != (hexVal(line[2*i])<<4 | hexVal(line[2*i+1])) {
					first = i
					break
				}
			}
			t.Fatalf("pkt %d diverges at byte %d (of got %d, ref %d):\n got %s\n ref  %s", pkt, first, len(got), len(line)/2, gotHex.String()[:min(120, gotHex.Len())], line[:min(120, len(line))])
		}
		pkt++
	}
	t.Logf("all %d packets byte-identical", pkt)
}

func hexVal(c byte) byte {
	switch {
	case c >= '0' && c <= '9':
		return c - '0'
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10
	default:
		return 0
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
