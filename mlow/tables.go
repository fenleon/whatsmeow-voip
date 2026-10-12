// MLow runtime constant tables: each table ships as a zlib-compressed
// protobuf blob (vectors/mlow/{lsf,pitch,cc}_seed.bin, schema = the
// reference's tables.proto), inflated and decoded once at load. Ported from
// whatsapp-rust (MIT) wacore/src/voip/mlow/{smpl_tables_blob.rs,tables.proto};
// MPL-2.0 for this port.
//
// The blobs are decoded by field number; downstream stage vectors pin the
// expanded tables, so a decode error surfaces at the first consuming stage.
package mlow

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// maxTableBytes caps the inflated blob size (the reference's inflater cap).
const maxTableBytes = 4 << 20

// inflateTable zlib-inflates a table blob.
func inflateTable(compressed []byte) ([]byte, error) {
	r, err := zlib.NewReader(bytes.NewReader(compressed))
	if err != nil {
		return nil, fmt.Errorf("mlow table: zlib open: %w", err)
	}
	defer r.Close()
	// ponytail: no size-limited reader; blobs are in-repo trusted data.
	return io.ReadAll(r)
}

// LsfSeed is the irreducible LSF seed ROM (tables.proto LsfSeed). Flat
// row-major; the expanded LSF runtime tables are derived from this at load.
// u8/i8 arrays are raw bytes; f32 are packed repeated float; u16 are packed
// repeated uint32.
type LsfSeed struct {
	Rot8            []byte    // [2][16][16][16]
	RotCond8        []byte    // [2][2][16][16]
	St2AllQlvls8    []byte    // [9593]
	St2AllQlvlDcmfs []byte    // [9593]
	St2MinQi        []byte    // [2][2][17][16] i8
	St2MaxQi        []byte    // [2][2][17][16] i8
	Cb16            []uint16  // [2][16][16]
	Cinv16          []uint16  // [2][136]
	Cmf             []uint16  // [2][17]
	CmfCond         []uint16  // [2][18]
	LsfSel          []uint16  // [3][3]
	LsfExtra        []uint16  // [3]
	Mean            []float32 // [2][16]
	MinDist         []float32 // [2][17]
	RegCond         []float32 // [2]
	Qstep           []float32 // [2][2]
}

// PitchSeed is the irreducible pitch seed ROM (tables.proto PitchSeed).
type PitchSeed struct {
	BlocksegsBitstream  []byte // [676]
	Blocksegs2Idx       []byte // [217]
	BlocksegsIx         []byte // [187][2]
	FirstblockRange     []byte // [9][2]
	BlocksegIdxDcmf     []byte // [217]
	DeltaLagDcmfs       []byte // [3][319]
	BlockTransitionDcmf []byte // [9][9]
}

// CcSeed is the irreducible nrgres/gains + LTP gain seed ROM
// (tables.proto CcSeed). The runtime CDFs are derived from the DCMFs at load.
type CcSeed struct {
	NrgresGain4Dcmf    []byte  // [84]
	NrgresShape4Dcmf   []byte  // [98]
	FcbgOffsetDcmf     []byte  // [3][4][176]
	AcbgainsHrDcmf     []byte  // [17][16]
	FcbgainsVDcmf      []byte  // [34]
	FcbgainsVDeltaDcmf []byte  // [67]
	AcbgainsCbHrQ14    []int32 // [16][2] i16 (sint32, zigzag)
	GainReconBase      uint32
	GainRecon          []byte // gain-reconstruction int16 rodata
	NPulsesDcmfBgn     []byte // [17] pulse-count CDF seed (background noise)
	NPulsesDcmfUV      []byte // [33] pulse-count CDF seed (unvoiced)
	NPulsesDcmfV       []byte // [33] pulse-count CDF seed (voiced)
}

// --- minimal protobuf wire decoding for the table schemas ---

type pbField struct {
	wireType int
	varint   uint64
	fixed32  uint32
	bytes    []byte
}

func pbFields(data []byte) (map[int]pbField, error) {
	fields := make(map[int]pbField)
	i := 0
	for i < len(data) {
		tag, n := binary.Uvarint(data[i:])
		if n <= 0 {
			return nil, fmt.Errorf("mlow table: bad tag at %d", i)
		}
		i += n
		fieldNum := int(tag >> 3)
		wireType := int(tag & 7)
		f := pbField{wireType: wireType}
		switch wireType {
		case 0: // varint
			v, n := binary.Uvarint(data[i:])
			if n <= 0 {
				return nil, fmt.Errorf("mlow table: bad varint (field %d)", fieldNum)
			}
			i += n
			f.varint = v
		case 1: // fixed64
			if i+8 > len(data) {
				return nil, fmt.Errorf("mlow table: truncated fixed64 (field %d)", fieldNum)
			}
			f.varint = binary.LittleEndian.Uint64(data[i:])
			i += 8
		case 2: // length-delimited
			l, n := binary.Uvarint(data[i:])
			if n <= 0 || i+n+int(l) > len(data) {
				return nil, fmt.Errorf("mlow table: bad length (field %d)", fieldNum)
			}
			i += n
			f.bytes = data[i : i+int(l)]
			i += int(l)
		case 5: // fixed32
			if i+4 > len(data) {
				return nil, fmt.Errorf("mlow table: truncated fixed32 (field %d)", fieldNum)
			}
			f.fixed32 = binary.LittleEndian.Uint32(data[i:])
			i += 4
		default:
			return nil, fmt.Errorf("mlow table: unsupported wire type %d (field %d)", wireType, fieldNum)
		}
		fields[fieldNum] = f
	}
	return fields, nil
}

// pbBytes returns the raw length-delimited value of a field (absent = empty).
func pbBytes(fields map[int]pbField, num int) ([]byte, error) {
	f, ok := fields[num]
	if !ok {
		return nil, nil
	}
	if f.wireType != 2 {
		return nil, fmt.Errorf("mlow table: field %d is not length-delimited", num)
	}
	return f.bytes, nil
}

// pbUint32 returns the varint value of a field.
func pbUint32(fields map[int]pbField, num int) (uint32, error) {
	f, ok := fields[num]
	if !ok {
		return 0, nil
	}
	if f.wireType != 0 {
		return 0, fmt.Errorf("mlow table: field %d is not a varint", num)
	}
	return uint32(f.varint), nil
}

// pbPackedU16 decodes packed (or repeated single-value) u16s. The .proto
// stores u16 as repeated uint32.
func pbPackedU16(fields map[int]pbField, num int) ([]uint16, error) {
	f, ok := fields[num]
	if !ok {
		return nil, nil
	}
	var out []uint16
	appendVarint := func(data []byte) error {
		for i := 0; i < len(data); {
			v, n := binary.Uvarint(data[i:])
			if n <= 0 {
				return fmt.Errorf("mlow table: bad packed varint (field %d)", num)
			}
			i += n
			out = append(out, uint16(v))
		}
		return nil
	}
	switch f.wireType {
	case 2:
		if err := appendVarint(f.bytes); err != nil {
			return nil, err
		}
	case 0:
		out = append(out, uint16(f.varint))
	default:
		return nil, fmt.Errorf("mlow table: field %d unexpected wire type %d", num, f.wireType)
	}
	return out, nil
}

// pbPackedF32 decodes packed (or repeated single-value) float32s.
func pbPackedF32(fields map[int]pbField, num int) ([]float32, error) {
	f, ok := fields[num]
	if !ok {
		return nil, nil
	}
	var out []float32
	switch f.wireType {
	case 2:
		if len(f.bytes)%4 != 0 {
			return nil, fmt.Errorf("mlow table: field %d float length %d not a multiple of 4", num, len(f.bytes))
		}
		for i := 0; i < len(f.bytes); i += 4 {
			out = append(out, math.Float32frombits(binary.LittleEndian.Uint32(f.bytes[i:])))
		}
	case 5:
		out = append(out, math.Float32frombits(f.fixed32))
	default:
		return nil, fmt.Errorf("mlow table: field %d unexpected wire type %d", num, f.wireType)
	}
	return out, nil
}

// pbPackedSint32 decodes packed (or repeated single-value) zigzag sint32s.
func pbPackedSint32(fields map[int]pbField, num int) ([]int32, error) {
	f, ok := fields[num]
	if !ok {
		return nil, nil
	}
	var out []int32
	appendVarint := func(data []byte) error {
		for i := 0; i < len(data); {
			v, n := binary.Uvarint(data[i:])
			if n <= 0 {
				return fmt.Errorf("mlow table: bad packed varint (field %d)", num)
			}
			i += n
			out = append(out, int32(v>>1)^-int32(v&1))
		}
		return nil
	}
	switch f.wireType {
	case 2:
		if err := appendVarint(f.bytes); err != nil {
			return nil, err
		}
	case 0:
		out = append(out, int32(f.varint>>1)^-int32(f.varint&1))
	default:
		return nil, fmt.Errorf("mlow table: field %d unexpected wire type %d", num, f.wireType)
	}
	return out, nil
}

// LoadLsfSeed decodes the zlib+protobuf LSF seed blob.
func LoadLsfSeed(blob []byte) (*LsfSeed, error) {
	raw, err := inflateTable(blob)
	if err != nil {
		return nil, err
	}
	fields, err := pbFields(raw)
	if err != nil {
		return nil, err
	}
	s := &LsfSeed{}
	loaders := []struct {
		num  int
		load func() error
	}{
		{1, func() error { s.Rot8, err = pbBytes(fields, 1); return err }},
		{2, func() error { s.RotCond8, err = pbBytes(fields, 2); return err }},
		{3, func() error { s.St2AllQlvls8, err = pbBytes(fields, 3); return err }},
		{4, func() error { s.St2AllQlvlDcmfs, err = pbBytes(fields, 4); return err }},
		{5, func() error { s.St2MinQi, err = pbBytes(fields, 5); return err }},
		{6, func() error { s.St2MaxQi, err = pbBytes(fields, 6); return err }},
		{7, func() error { s.Cb16, err = pbPackedU16(fields, 7); return err }},
		{8, func() error { s.Cinv16, err = pbPackedU16(fields, 8); return err }},
		{9, func() error { s.Cmf, err = pbPackedU16(fields, 9); return err }},
		{10, func() error { s.CmfCond, err = pbPackedU16(fields, 10); return err }},
		{11, func() error { s.LsfSel, err = pbPackedU16(fields, 11); return err }},
		{12, func() error { s.LsfExtra, err = pbPackedU16(fields, 12); return err }},
		{13, func() error { s.Mean, err = pbPackedF32(fields, 13); return err }},
		{14, func() error { s.MinDist, err = pbPackedF32(fields, 14); return err }},
		{15, func() error { s.RegCond, err = pbPackedF32(fields, 15); return err }},
		{16, func() error { s.Qstep, err = pbPackedF32(fields, 16); return err }},
	}
	for _, l := range loaders {
		if err := l.load(); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// LoadPitchSeed decodes the zlib+protobuf pitch seed blob.
func LoadPitchSeed(blob []byte) (*PitchSeed, error) {
	raw, err := inflateTable(blob)
	if err != nil {
		return nil, err
	}
	fields, err := pbFields(raw)
	if err != nil {
		return nil, err
	}
	s := &PitchSeed{}
	byteFields := []struct {
		num int
		dst *[]byte
	}{{1, &s.BlocksegsBitstream}, {2, &s.Blocksegs2Idx}, {3, &s.BlocksegsIx},
		{4, &s.FirstblockRange}, {5, &s.BlocksegIdxDcmf}, {6, &s.DeltaLagDcmfs},
		{7, &s.BlockTransitionDcmf}}
	for _, bf := range byteFields {
		v, perr := pbBytes(fields, bf.num)
		if perr != nil {
			return nil, perr
		}
		*bf.dst = v
		if err != nil {
			return nil, err
		}
	}
	return s, nil
}

// LoadCcSeed decodes the zlib+protobuf cc seed blob.
func LoadCcSeed(blob []byte) (*CcSeed, error) {
	raw, err := inflateTable(blob)
	if err != nil {
		return nil, err
	}
	fields, err := pbFields(raw)
	if err != nil {
		return nil, err
	}
	s := &CcSeed{}
	byteFields := []struct {
		num int
		dst *[]byte
	}{{1, &s.NrgresGain4Dcmf}, {2, &s.NrgresShape4Dcmf}, {3, &s.FcbgOffsetDcmf},
		{4, &s.AcbgainsHrDcmf}, {5, &s.FcbgainsVDcmf}, {6, &s.FcbgainsVDeltaDcmf},
		{9, &s.GainRecon}, {10, &s.NPulsesDcmfBgn}, {11, &s.NPulsesDcmfUV},
		{12, &s.NPulsesDcmfV}}
	for _, bf := range byteFields {
		v, perr := pbBytes(fields, bf.num)
		if perr != nil {
			return nil, perr
		}
		*bf.dst = v
		if err != nil {
			return nil, err
		}
	}
	if s.AcbgainsCbHrQ14, err = pbPackedSint32(fields, 7); err != nil {
		return nil, err
	}
	if s.GainReconBase, err = pbUint32(fields, 8); err != nil {
		return nil, err
	}
	return s, nil
}
