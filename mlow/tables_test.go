package mlow

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func loadTableBlob(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile("../vectors/mlow/" + name)
	require.NoError(t, err, "%s not found (run from the repo root)", name)
	return data
}

// The shapes below are the tables.proto field shapes; decoding the committed
// blobs into exactly these shapes validates the wire decoder end to end.
// Downstream stage vectors (lsf_/pitch_/gains_*) pin the expanded values.
func TestLoadLsfSeedShapes(t *testing.T) {
	s, err := LoadLsfSeed(loadTableBlob(t, "lsf_seed.bin"))
	require.NoError(t, err)
	assert.Len(t, s.Rot8, 2*16*16*16)
	assert.Len(t, s.RotCond8, 2*2*16*16)
	assert.Len(t, s.St2AllQlvls8, 9593)
	assert.Len(t, s.St2AllQlvlDcmfs, 9593)
	assert.Len(t, s.St2MinQi, 2*2*17*16)
	assert.Len(t, s.St2MaxQi, 2*2*17*16)
	assert.Len(t, s.Cb16, 2*16*16)
	assert.Len(t, s.Cinv16, 2*136)
	assert.Len(t, s.Cmf, 2*17)
	assert.Len(t, s.CmfCond, 2*18)
	assert.Len(t, s.LsfSel, 3*3)
	assert.Len(t, s.LsfExtra, 3)
	assert.Len(t, s.Mean, 2*16)
	assert.Len(t, s.MinDist, 2*17)
	assert.Len(t, s.RegCond, 2)
	assert.Len(t, s.Qstep, 2*2)
}

func TestLoadPitchSeedShapes(t *testing.T) {
	s, err := LoadPitchSeed(loadTableBlob(t, "pitch_seed.bin"))
	require.NoError(t, err)
	assert.Len(t, s.BlocksegsBitstream, 676)
	assert.Len(t, s.Blocksegs2Idx, 217)
	assert.Len(t, s.BlocksegsIx, 187*2)
	assert.Len(t, s.FirstblockRange, 9*2)
	assert.Len(t, s.BlocksegIdxDcmf, 217)
	assert.Len(t, s.DeltaLagDcmfs, 3*319)
	assert.Len(t, s.BlockTransitionDcmf, 9*9)
}

func TestLoadCcSeedShapes(t *testing.T) {
	s, err := LoadCcSeed(loadTableBlob(t, "cc_seed.bin"))
	require.NoError(t, err)
	assert.Len(t, s.NrgresGain4Dcmf, 84)
	assert.Len(t, s.NrgresShape4Dcmf, 98)
	assert.Len(t, s.FcbgOffsetDcmf, 3*4*176)
	assert.Len(t, s.AcbgainsHrDcmf, 17*16)
	assert.Len(t, s.FcbgainsVDcmf, 34)
	assert.Len(t, s.FcbgainsVDeltaDcmf, 67)
	assert.Len(t, s.AcbgainsCbHrQ14, 16*2)
	assert.Equal(t, uint32(0xf35e0), s.GainReconBase, "gain_recon_base is a fixed address")
	assert.NotEmpty(t, s.GainRecon)
	assert.Len(t, s.NPulsesDcmfBgn, 17)
	assert.Len(t, s.NPulsesDcmfUV, 33)
	assert.Len(t, s.NPulsesDcmfV, 33)
}
