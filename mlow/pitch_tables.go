// Encoder-side pitch constant tables (blocksegs/blocktracks/CMFs), expanded
// once from the packed pitch seed ROM. Ported from whatsapp-rust (MIT)
// wacore/src/voip/mlow/smpl_pitch_enc.rs + smpl_pitch_seed.rs; MPL-2.0 for
// this port. Reuses the decoder's blocksegs range-decoder + DCMF expansion
// from smpl_mem.go.
package mlow

import (
	"math"
	"sync"
)

const (
	pitchNumBlocks    = 9 // ((MAXPITCH_MS - MINPITCH_MS) / PITCHBLOCK_MS)
	numBlocksegs      = 217
	numBlocktracks    = 187
	pitchNumSubframes = 8
)

// BlockSeg is one per-block lag-block layout.
type BlockSeg struct {
	Nblocks int
	Blocks  []int
	Seglens []int
}

// BlockTrack is one expanded track: the per-subframe lag block + mean/deltas.
type BlockTrack struct {
	Track       [pitchNumSubframes]int
	Meanblock   float32
	Trackdeltas float32
}

// PitchTables is the expanded encoder pitch table set.
type PitchTables struct {
	Blocksegs          []BlockSeg
	Blocktracks        []BlockTrack
	Blocksegs2idx      []int
	BlocksegIdxCmf     []uint32
	DeltaLagCmfs       [][]uint32
	BlocksegsIx        [][2]int
	FirstblockRange    [][2]int
	BlockTransitionCmf [][]uint32
}

// genBlocktracks expands each track's blockseg into the per-subframe track +
// mean/deltas.
func genBlocktracks(blocksegs []BlockSeg, blocksegsIx [][2]int) []BlockTrack {
	out := make([]BlockTrack, numBlocktracks)
	for trackIdx := 0; trackIdx < numBlocktracks; trackIdx++ {
		seg := &blocksegs[blocksegsIx[trackIdx][0]]
		var track [pitchNumSubframes]int
		segIdx := 0
		var meanblock, trackdeltas float32
		for b := 0; b < seg.Nblocks; b++ {
			for i := 0; i < seg.Seglens[b]; i++ {
				track[segIdx] = seg.Blocks[b]
				segIdx++
			}
			meanblock += float32(seg.Blocks[b] * seg.Seglens[b])
			if b != 0 {
				trackdeltas += float32(absInt(seg.Blocks[b-1] - seg.Blocks[b]))
			}
		}
		meanblock /= pitchNumSubframes
		out[trackIdx] = BlockTrack{Track: track, Meanblock: meanblock, Trackdeltas: trackdeltas}
	}
	return out
}

func absInt(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// buildPitchTables assembles the runtime tables from the expanded seed parts.
func buildPitchTables(seed *PitchSeed) *PitchTables {
	records := decodeBlocksegsSeed(seed)
	blocksegs := make([]BlockSeg, 0, numBlocksegs)
	for _, rec := range records {
		blocksegs = append(blocksegs, BlockSeg{
			Nblocks: len(rec[0]),
			Blocks:  rec[0],
			Seglens: rec[1],
		})
	}
	blocksegsIx := make([][2]int, 0, 187)
	for i := 0; i+1 < len(seed.BlocksegsIx); i += 2 {
		blocksegsIx = append(blocksegsIx,
			[2]int{int(seed.BlocksegsIx[i]), int(seed.BlocksegsIx[i+1])})
	}
	firstblockRange := make([][2]int, 0, 9)
	for i := 0; i+1 < len(seed.FirstblockRange); i += 2 {
		firstblockRange = append(firstblockRange,
			[2]int{int(seed.FirstblockRange[i]), int(seed.FirstblockRange[i+1])})
	}
	blocktracks := genBlocktracks(blocksegs, blocksegsIx)
	blocksegs2idx := make([]int, len(seed.Blocksegs2Idx))
	for i, v := range seed.Blocksegs2Idx {
		blocksegs2idx[i] = int(v)
	}
	blocksegIdxCmf := dcmfToCmfU32(seed.BlocksegIdxDcmf)
	var deltaLagCmfs [][]uint32
	for i := 0; i+319 <= len(seed.DeltaLagDcmfs); i += 319 {
		deltaLagCmfs = append(deltaLagCmfs, dcmfToCmfU32(seed.DeltaLagDcmfs[i:i+319]))
	}
	var blockTransitionCmf [][]uint32
	for i := 0; i+pitchNumBlocks <= len(seed.BlockTransitionDcmf); i += pitchNumBlocks {
		blockTransitionCmf = append(blockTransitionCmf,
			dcmfToCmfU32(seed.BlockTransitionDcmf[i:i+pitchNumBlocks]))
	}
	return &PitchTables{
		Blocksegs:          blocksegs,
		Blocktracks:        blocktracks,
		Blocksegs2idx:      blocksegs2idx,
		BlocksegIdxCmf:     blocksegIdxCmf,
		DeltaLagCmfs:       deltaLagCmfs,
		BlocksegsIx:        blocksegsIx,
		FirstblockRange:    firstblockRange,
		BlockTransitionCmf: blockTransitionCmf,
	}
}

var pitchTablesCache struct {
	sync.Once
	tables *PitchTables
}

// LoadPitchTables expands the embedded pitch seed once per process.
func LoadPitchTables() *PitchTables {
	pitchTablesCache.Do(func() {
		seed, err := LoadPitchSeed(pitchSeedBlob)
		if err != nil {
			panic("mlow: failed to load pitch seed: " + err.Error())
		}
		pitchTablesCache.tables = buildPitchTables(seed)
	})
	return pitchTablesCache.tables
}

// DebugChecksum is the FNV-1a digest over every field (order-deterministic),
// the seed-build tripwire (smpl_pitch_enc.rs:117).
func (t *PitchTables) DebugChecksum() uint64 {
	h := uint64(0xcbf29ce484222325)
	feed := func(x uint64) {
		h ^= x
		h *= 0x100000001b3
	}
	for i := range t.Blocksegs {
		s := &t.Blocksegs[i]
		feed(uint64(s.Nblocks))
		for _, b := range s.Blocks {
			feed(uint64(b))
		}
		for _, l := range s.Seglens {
			feed(uint64(l))
		}
	}
	for i := range t.Blocktracks {
		bt := &t.Blocktracks[i]
		for _, v := range bt.Track {
			feed(uint64(v))
		}
		feed(uint64(math.Float32bits(bt.Meanblock)))
		feed(uint64(math.Float32bits(bt.Trackdeltas)))
	}
	for _, v := range t.Blocksegs2idx {
		feed(uint64(v))
	}
	for _, v := range t.BlocksegIdxCmf {
		feed(uint64(v))
	}
	for _, row := range t.DeltaLagCmfs {
		for _, v := range row {
			feed(uint64(v))
		}
	}
	for _, p := range t.BlocksegsIx {
		feed(uint64(p[0]))
		feed(uint64(p[1]))
	}
	for _, p := range t.FirstblockRange {
		feed(uint64(p[0]))
		feed(uint64(p[1]))
	}
	for _, row := range t.BlockTransitionCmf {
		for _, v := range row {
			feed(uint64(v))
		}
	}
	return h
}
