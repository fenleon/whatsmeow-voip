package mlow

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type rcOp struct {
	Kind uint64 `json:"kind"`
	A    uint64 `json:"a"`
	B    uint64 `json:"b"`
}

type rcVectors struct {
	BytesHex    string     `json:"bytesHex"`
	Ops         []rcOp     `json:"ops"`
	CdfBytesHex string     `json:"cdfBytesHex"`
	CdfTables   [][]uint16 `json:"cdfTables"`
	CdfOps      []rcOp     `json:"cdfOps"`
}

func loadRCVectors(t *testing.T) *rcVectors {
	t.Helper()
	data, err := os.ReadFile("../vectors/mlow/rc_vectors.json")
	require.NoError(t, err)
	var v rcVectors
	require.NoError(t, json.Unmarshal(data, &v))
	return &v
}

// rcICDF is the fixed table the captured script encoded against (ftb=8).
var rcICDF = []byte{255, 200, 150, 90, 30, 0}

func TestRangeDecoderMatchesVectors(t *testing.T) {
	v := loadRCVectors(t)
	bytes, err := hex.DecodeString(v.BytesHex)
	require.NoError(t, err)
	d := NewRangeDecoder(bytes)
	for i, op := range v.Ops {
		i, op := i, op
		switch op.Kind {
		case 0:
			t.Run("icdf", func(t *testing.T) { assert.EqualValues(t, op.A, d.DecodeICDF(rcICDF, 8), "op %d", i) })
		case 1:
			t.Run("bits", func(t *testing.T) { assert.EqualValues(t, op.B, d.BitsN(uint32(op.A)), "op %d bits(%d)", i, op.A) })
		case 2:
			t.Run("bit_logp", func(t *testing.T) {
				assert.EqualValues(t, op.B, d.BitLogp(uint32(op.A)), "op %d bit_logp(%d)", i, op.A)
			})
		case 3:
			t.Run("uint", func(t *testing.T) {
				assert.EqualValues(t, op.B, d.DecodeUint(uint32(op.A)), "op %d uint(ft=%d)", i, op.A)
			})
		default:
			t.Fatalf("bad op kind %d", op.Kind)
		}
	}
	assert.EqualValues(t, 0, d.Err, "no decode error")
}

func TestRangeDecoderCDFMatchesVectors(t *testing.T) {
	v := loadRCVectors(t)
	bytes, err := hex.DecodeString(v.CdfBytesHex)
	require.NoError(t, err)
	d := NewRangeDecoder(bytes)
	for i, op := range v.CdfOps {
		ti := int(op.Kind)
		require.Less(t, ti, len(v.CdfTables))
		ti, op := ti, op
		t.Run("cdf op", func(t *testing.T) {
			assert.EqualValues(t, op.A, d.DecodeCDF(v.CdfTables[ti]), "cdf op %d table %d", i, ti)
		})
	}
	assert.EqualValues(t, 0, d.Err, "no decode error")
}

func TestRangeEncoderMatchesVectors(t *testing.T) {
	v := loadRCVectors(t)
	want, err := hex.DecodeString(v.BytesHex)
	require.NoError(t, err)
	e := NewRangeEncoder(len(want))
	for _, op := range v.Ops {
		switch op.Kind {
		case 0:
			e.EncodeICDF(int32(op.A), rcICDF, 8)
		case 1:
			e.BitsN(uint32(op.B), uint32(op.A))
		case 2:
			e.BitLogp(int32(op.B), uint32(op.A))
		case 3:
			e.EncodeUint(uint32(op.B), uint32(op.A))
		default:
			t.Fatalf("bad op kind %d", op.Kind)
		}
	}
	e.Done()
	assert.EqualValues(t, 0, e.Err(), "encoder error")
	assert.Equal(t, want, e.Bytes(), "encoder output differs from captured vectors")
}

func TestRangeEncoderCDFMatchesVectors(t *testing.T) {
	v := loadRCVectors(t)
	want, err := hex.DecodeString(v.CdfBytesHex)
	require.NoError(t, err)
	e := NewRangeEncoder(len(want))
	for _, op := range v.CdfOps {
		ti := int(op.Kind)
		require.Less(t, ti, len(v.CdfTables))
		e.EncodeCDF(int32(op.A), v.CdfTables[ti])
	}
	e.Done()
	assert.EqualValues(t, 0, e.Err(), "encoder error")
	assert.Equal(t, want, e.Bytes(), "cdf encoder output differs from captured vectors")
}

// rcOpScript is one op of the LCG-random round-trip script.
type rcOpScript func(e *RangeEncoder)

// buildRCScript builds one deterministic mixed-op script from an LCG seed,
// mirroring the Rust test (icdf / cdf zero+non-zero base / raw back-bits /
// range-stream raw symbol / 64-fine / uint small+large).
func buildRCScript(seed uint32) (ops []rcOpScript, checks []func(d *RangeDecoder) bool) {
	var s uint32 = seed
	next := func() uint32 {
		s = s*1664525 + 1013904223
		return s
	}
	cdf0 := []uint16{0, 3, 7, 12, 20}
	cdf1 := []uint16{5, 8, 14, 30}
	for i := 0; i < 256; i++ {
		r := next()
		switch r % 7 {
		case 0:
			sym := int32(r>>8) % 5
			ops = append(ops, func(e *RangeEncoder) { e.EncodeICDF(sym, rcICDF, 8) })
			checks = append(checks, func(d *RangeDecoder) bool { return d.DecodeICDF(rcICDF, 8) == sym })
		case 1:
			sym := int32(r>>8) % 4
			ops = append(ops, func(e *RangeEncoder) { e.EncodeCDF(sym, cdf0) })
			checks = append(checks, func(d *RangeDecoder) bool { return d.DecodeCDF(cdf0) == sym })
		case 2:
			sym := int32(r>>8) % 3
			ops = append(ops, func(e *RangeEncoder) { e.EncodeCDF(sym, cdf1) })
			checks = append(checks, func(d *RangeDecoder) bool { return d.DecodeCDF(cdf1) == sym })
		case 3:
			n := 1 + (r>>8)%16 // 1..=16 raw back-bits
			fl := next() & ((1 << n) - 1)
			ops = append(ops, func(e *RangeEncoder) { e.BitsN(fl, n) })
			checks = append(checks, func(d *RangeDecoder) bool { return d.BitsN(n) == fl })
		case 4:
			nbits := 1 + (r>>8)%6 // range-stream uniform symbol
			sym := next() & ((1 << nbits) - 1)
			ops = append(ops, func(e *RangeEncoder) { e.EncodeRawSymbol(sym, nbits) })
			checks = append(checks, func(d *RangeDecoder) bool { return d.DecodeRawSymbol(nbits) == sym })
		case 5:
			sym := int32(r>>8) % 64 // 64-symbol uniform: 0..=63
			ops = append(ops, func(e *RangeEncoder) { e.Encode64FineSym(sym) })
			checks = append(checks, func(d *RangeDecoder) bool { return d.Decode64FineSym() == sym })
		default:
			// Alternate small (ftb<=8) and large (ftb>8) uint branches.
			ft0 := uint32(200)
			if r&1 != 0 {
				ft0 = 2000
			}
			fl := next() % ft0
			ops = append(ops, func(e *RangeEncoder) { e.EncodeUint(fl, ft0) })
			checks = append(checks, func(d *RangeDecoder) bool { return d.DecodeUint(ft0) == fl })
		}
	}
	return ops, checks
}

func TestRangecoderRoundTripsRandomScripts(t *testing.T) {
	for _, seed := range []uint32{0x13579bdf, 0x2468ace0, 0xdeadbeef, 0x0badf00d} {
		seed := seed
		t.Run("seed", func(t *testing.T) {
			ops, checks := buildRCScript(seed)
			e := NewRangeEncoder(4096)
			for _, op := range ops {
				op(e)
			}
			e.Done()
			assert.EqualValues(t, 0, e.Err(), "seed %#x: encoder error", seed)
			// The encoder lays raw back-bits at the physical end of storage,
			// so the decoder must see the same storage size to mirror the
			// front/back read offsets. Truncating to ConsumedLen is only
			// valid when no back-bit op is used.
			bytes := e.Bytes()
			d := NewRangeDecoder(bytes)
			for i, check := range checks {
				if !check(d) {
					t.Fatalf("seed %#x: op %d decoded wrong", seed, i)
				}
			}
			assert.EqualValues(t, 0, d.Err, "seed %#x: decoder error", seed)
		})
	}
}
