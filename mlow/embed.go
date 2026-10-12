// Package-level singleton access to the expanded codec tables, loaded from
// the embedded seed blobs (mlow/tables/, verbatim copies of
// vectors/mlow/*.bin — see that PROVENANCE.md).
package mlow

import (
	_ "embed"
	"sync"
)

//go:embed tables/lsf_seed.bin
var lsfSeedBlob []byte

//go:embed tables/pitch_seed.bin
var pitchSeedBlob []byte

//go:embed tables/cc_seed.bin
var ccSeedBlob []byte

var (
	lsfBuiltOnce  sync.Once
	lsfBuiltValue *LsfBuilt
	lsfBuiltErr   error
	ccOnce2       sync.Once
	ccValue2      *CcTables
	ccErr2        error
	memOnce2      sync.Once
	memValue2     *SmplMem
	memErr2       error
)

// LsfBuiltTables loads and expands the embedded LSF seed once.
func LsfBuiltTables() *LsfBuilt {
	lsfBuiltOnce.Do(func() {
		lsfBuiltValue, lsfBuiltErr = LoadLsfBuilt(lsfSeedBlob)
	})
	if lsfBuiltErr != nil {
		panic(lsfBuiltErr)
	}
	return lsfBuiltValue
}

// CcTablesTables loads and expands the embedded cc seed once.
func CcTablesTables() *CcTables {
	ccOnce2.Do(func() {
		ccValue2, ccErr2 = LoadCcTables(ccSeedBlob)
	})
	if ccErr2 != nil {
		panic(ccErr2)
	}
	return ccValue2
}

// SmplMemTables builds the embedded pitch heap window once.
func SmplMemTables() *SmplMem {
	memOnce2.Do(func() {
		memValue2, memErr2 = LoadSmplMem(pitchSeedBlob)
	})
	if memErr2 != nil {
		panic(memErr2)
	}
	return memValue2
}
