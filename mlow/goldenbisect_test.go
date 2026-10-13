//go:build cgo

package mlow

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"testing"
)

func TestGoldenBisect(t *testing.T) {
	refBytes, err := os.ReadFile("/tmp/golden-rust.hex")
	if err != nil {
		t.Skip("no reference")
	}
	sc := bufio.NewScanner(strings.NewReader(string(refBytes)))
	type rec struct {
		pkt int
		hex string
	}
	var refs []rec
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) != 2 {
			continue
		}
		n, _ := strconv.Atoi(f[0])
		refs = append(refs, rec{n, f[1]})
	}
	input := goldenSynthInput()
	enc := NewMlowEncoder()
	for _, r := range refs {
		off := r.pkt * goldenFrame
		got, err := enc.Encode(input[off : off+goldenFrame])
		if err != nil {
			t.Fatalf("pkt %d: %v", r.pkt, err)
		}
		var hs strings.Builder
		const d = "0123456789abcdef"
		for _, b := range got {
			hs.WriteByte(d[b>>4])
			hs.WriteByte(d[b&15])
		}
		if hs.String() != r.hex {
			t.Fatalf("FIRST DIVERGENCE pkt %d: got len %d ref len %d\n got %s\n ref  %s", r.pkt, len(got), len(r.hex)/2, hs.String(), r.hex)
		}
	}
	t.Logf("all %d packets identical", len(refs))
}
