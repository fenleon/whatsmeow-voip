// MLow heap window for the pitch lag/contour reads (the only consumer still
// addressing the WASM heap by absolute pointer), built from pitch_seed.bin at
// load. Ported from whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_mem.rs +
// smpl_pitch_seed.rs (contour-window half); MPL-2.0 for this port.
package mlow

import (
	"encoding/binary"
	"sync"
)

// Fixed WASM-build globals for the Group-D heap layout. The window is built
// at these absolute addresses so the pointer arithmetic lands unchanged.
const (
	gClk            = 0xb9f9a8
	gPitch          = 0xb9d378
	pcfgAddr        = gClk + 0x5704
	hdrContourMap   = 0xe7c10
	hdrLagCDF       = 0xbaa7b0
	hdrFracBase     = 0xbaa9be
	hdrDeltaCDF     = 0xbab13e
	deltaBoundsAddr = 0xe7ef0
	numContours     = 217
	numBlockTracks  = 187
)

// SmplMemRegion is one contiguous heap region at its absolute base address.
type SmplMemRegion struct {
	Base uint32
	Data []byte
}

// SmplMem is the pitch lag/contour heap window.
type SmplMem struct {
	Regions []SmplMemRegion
	GCC     uint32
	GNrg    uint32
	GPitch  uint32
	GClk    uint32
}

// contourWindowParts is the seed expansion the window layout consumes.
type contourWindowParts struct {
	records         [][2][]int // per contour: (lags, seglens)
	contourMap      []byte     // [217]
	firstblockRange [][2]int   // [9][2]
	lagCDF          []uint32   // 218
	fracCmfs        [][]uint32 // 3 × 320
	deltaCmfs       [][]uint32 // 9 × 10
}

// decodeBlocksegsSeed re-decodes the blocksegs bitstream into the 217
// per-contour (blocks, seglens) records.
func decodeBlocksegsSeed(seed *PitchSeed) [][2][]int {
	dec := NewRangeDecoder(seed.BlocksegsBitstream)
	records := make([][2][]int, 0, numContours)
	for i := 0; i < numContours; i++ {
		blocks, seglens := decodeBlockseg(dec)
		records = append(records, [2][]int{blocks, seglens})
	}
	return records
}

// decodeBlockseg decodes one blockseg: len = uniform(6)+1, then len pairs of
// (uniform(9), uniform(4)+1).
func decodeBlockseg(dec *RangeDecoder) (blocks []int, seglens []int) {
	const (
		nLen    = 6
		nBlock  = 9
		nSegLen = 4
	)
	length := int(ecDecodeUniform(dec, nLen)) + 1
	blocks = make([]int, 0, length)
	seglens = make([]int, 0, length)
	for i := 0; i < length; i++ {
		blocks = append(blocks, int(ecDecodeUniform(dec, nBlock)))
		seglens = append(seglens, int(ecDecodeUniform(dec, nSegLen))+1)
	}
	return blocks, seglens
}

// ecDecodeUniform decodes a uniform symbol in [0, n).
func ecDecodeUniform(dec *RangeDecoder, n uint32) uint32 {
	cmfLow := dec.Decode(n)
	dec.Update(cmfLow, cmfLow+1, n)
	return cmfLow
}

// dcmfToCmfU32 is the integer DCMF→CDF expansion (u32 variant used by the
// pitch tables).
func dcmfToCmfU32(dcmf []byte) []uint32 {
	dcmfLen := len(dcmf)
	cmf := make([]uint32, dcmfLen+1)
	var sum int64
	for n := 0; n < dcmfLen; n++ {
		tmp := int32(dcmf[n]) + 1
		tmp *= tmp
		if tmp > 65535 {
			tmp = 65535
		}
		cmf[n+1] = uint32(tmp)
		sum += int64(tmp)
	}
	cmf[0] = 0
	for n := 1; n < dcmfLen+1; n++ {
		prev := int64(cmf[n-1])
		add := (int64(cmf[n])*(32767-int64(dcmfLen)))/sum + 1
		cmf[n] = uint32(prev + add)
	}
	return cmf
}

// buildContourWindow expands the pitch seed into the contour-window tables.
func buildContourWindow(seed *PitchSeed) *contourWindowParts {
	records := decodeBlocksegsSeed(seed)
	firstblockRange := make([][2]int, 0, 9)
	for i := 0; i+1 < len(seed.FirstblockRange); i += 2 {
		firstblockRange = append(firstblockRange,
			[2]int{int(seed.FirstblockRange[i]), int(seed.FirstblockRange[i+1])})
	}
	var fracCmfs [][]uint32
	for i := 0; i+319 <= len(seed.DeltaLagDcmfs); i += 319 {
		fracCmfs = append(fracCmfs, dcmfToCmfU32(seed.DeltaLagDcmfs[i:i+319]))
	}
	var deltaCmfs [][]uint32
	for i := 0; i+9 <= len(seed.BlockTransitionDcmf); i += 9 {
		deltaCmfs = append(deltaCmfs, dcmfToCmfU32(seed.BlockTransitionDcmf[i:i+9]))
	}
	return &contourWindowParts{
		records:         records,
		contourMap:      seed.Blocksegs2Idx,
		firstblockRange: firstblockRange,
		lagCDF:          dcmfToCmfU32(seed.BlocksegIdxDcmf),
		fracCmfs:        fracCmfs,
		deltaCmfs:       deltaCmfs,
	}
}

func u16Bytes(v []uint32) []byte {
	out := make([]byte, 0, len(v)*2)
	for _, x := range v {
		out = binary.LittleEndian.AppendUint16(out, uint16(x))
	}
	return out
}

// BuildSmplMem lays the contour-window tables out at the fixed WASM
// addresses, reproducing the carved window byte-for-byte at every address the
// consumer reads.
func BuildSmplMem(seed *PitchSeed) *SmplMem {
	w := buildContourWindow(seed)

	var regions []SmplMemRegion
	push := func(base uint32, data []byte) {
		regions = append(regions, SmplMemRegion{Base: base, Data: data})
	}

	// region 0 @pcfg+0x1d38: 217 records (lags[8] | seglens[8] | seg_count,
	// 0x44 each), a 4-byte gap (=NUM_BLOCKTRACKS), then the 8-pointer header.
	r0 := make([]byte, 0, numContours*0x44+4+32)
	for _, rec := range w.records {
		lags, seglens := rec[0], rec[1]
		sc := len(lags)
		for i := 0; i < 8; i++ {
			var v int32
			if i < sc {
				v = int32(lags[i])
			}
			r0 = binary.LittleEndian.AppendUint32(r0, uint32(v))
		}
		for i := 0; i < 8; i++ {
			var v int32
			if i < sc {
				v = int32(seglens[i])
			}
			r0 = binary.LittleEndian.AppendUint32(r0, uint32(v))
		}
		r0 = binary.LittleEndian.AppendUint32(r0, uint32(int32(sc)))
	}
	r0 = binary.LittleEndian.AppendUint32(r0, numBlockTracks)
	for _, h := range []uint32{
		numContours, hdrContourMap, hdrLagCDF, hdrFracBase,
		0xe7d20, 0xe7ef0, 0xe8096, hdrDeltaCDF,
	} {
		r0 = binary.LittleEndian.AppendUint32(r0, h)
	}
	push(pcfgAddr+0x1d38, r0)

	// CDF tables (each its own region; the dead inter-table gaps the carve
	// included are unread).
	push(hdrLagCDF, u16Bytes(w.lagCDF))
	var frac []uint32
	for _, c := range w.fracCmfs {
		frac = append(frac, c...)
	}
	push(hdrFracBase, u16Bytes(frac))
	var delta []uint32
	for _, c := range w.deltaCmfs {
		delta = append(delta, c...)
	}
	push(hdrDeltaCDF, u16Bytes(delta))

	// contour_map[217].
	push(hdrContourMap, w.contourMap)

	// delta-lag bounds: firstblock_range u8 pairs, then the carve's 2
	// trailing pad bytes.
	bounds := make([]byte, 0, len(w.firstblockRange)*2+2)
	for _, p := range w.firstblockRange {
		bounds = append(bounds, byte(p[0]), byte(p[1]))
	}
	bounds = append(bounds, 0, 0)
	push(deltaBoundsAddr, bounds)

	return &SmplMem{
		Regions: regions,
		GPitch:  gPitch,
		GClk:    gClk,
	}
}

var (
	memOnce sync.Once
	memVal  *SmplMem
	memErr  error
)

// LoadSmplMem builds the pitch heap window once from the pitch seed blob.
func LoadSmplMem(blob []byte) (*SmplMem, error) {
	memOnce.Do(func() {
		seed, err := LoadPitchSeed(blob)
		if err != nil {
			memErr = err
			return
		}
		memVal = BuildSmplMem(seed)
	})
	return memVal, memErr
}

// regionFor returns the region containing [addr, addr+n) and the offset of
// addr within it, or nil.
func (m *SmplMem) regionFor(addr uint32, n int) (*SmplMemRegion, uint32) {
	for i := range m.Regions {
		r := &m.Regions[i]
		if addr >= r.Base && int(addr-r.Base)+n <= len(r.Data) {
			return r, addr - r.Base
		}
	}
	return nil, 0
}

// U8 reads a byte at the absolute WASM address (0 outside the window).
func (m *SmplMem) U8(addr uint32) byte {
	r, off := m.regionFor(addr, 1)
	if r == nil {
		return 0
	}
	return r.Data[off]
}

// U16 reads a little-endian u16.
func (m *SmplMem) U16(addr uint32) uint16 {
	r, off := m.regionFor(addr, 2)
	if r == nil {
		return 0
	}
	return binary.LittleEndian.Uint16(r.Data[off:])
}

// U32 reads a little-endian u32.
func (m *SmplMem) U32(addr uint32) uint32 {
	r, off := m.regionFor(addr, 4)
	if r == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(r.Data[off:])
}

// I32 reads a little-endian i32.
func (m *SmplMem) I32(addr uint32) int32 { return int32(m.U32(addr)) }

// CDFAt materializes the n-entry cumulative u16 CDF at addr. Entries outside
// the window read as 0, matching the WASM's out-of-region fallback.
func (m *SmplMem) CDFAt(addr uint32, n int) []uint16 {
	out := make([]uint16, n)
	for i := 0; i < n; i++ {
		out[i] = m.U16(addr + uint32(i)*2)
	}
	return out
}

// CDFBytes returns the raw [addr, addr+2n) byte slice when the whole CDF
// window sits inside one region (the common case), or nil at a region
// boundary / out of range, where the zero-fill semantics of CDFAt apply.
func (m *SmplMem) CDFBytes(addr uint32, n int) []byte {
	r, off := m.regionFor(addr, n*2)
	if r == nil {
		return nil
	}
	start := int(off)
	return r.Data[start : start+n*2]
}
