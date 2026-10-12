// MLow per-frame LSF parameter decode (the LSF block of the core decode)
// plus the cross-internal-frame predictor state. Ported from whatsapp-rust
// (MIT) wacore/src/voip/mlow/smpl_decode.rs; MPL-2.0 for this port.
//
// The smpl LSF coding is Meta-specific (NOT stock SILK CB1): a 2-way stage-1
// codebook selector, a stage-1 grid index, then 16 stage-2 residuals from
// g_lsf[stage1][config][grid][coeff]. The entropy primitive is the u16
// cumulative CDF decode, not the ICDF path.
package mlow

// SmplLsfState is the cross-internal-frame decoder state. The LSF block
// RESETS the pitch/LTP predictor fields to -1 whenever the stage-1 selector
// does not match the previous internal frame.
type SmplLsfState struct {
	PrevStage1  int32
	PrevMatch   bool
	HavePrev    bool
	PrevGainIdx int32
	PrevFiltIdx int32
	PrevLag     int32
	PrevFracLag int32
	// Encoder-only predictor mirror fields (unused by the decoder):
	PrevLagSamples float32
	PrevLagblk     int32
	PrevLagidx     int32
}

// AdvanceLsfState advances the LSF predictor mirror exactly as
// encode/decode_smpl_lsf does for an internal frame with the given stage-1
// selector: on a no-match (intf 0, or stage1 differs) reset the pitch/LTP
// predictor to -1, then record prev_stage1/prev_match.
func AdvanceLsfState(st *SmplLsfState, intf int, stage1 int32) {
	m := intf != 0 && stage1 == st.PrevStage1
	if !m {
		st.PrevGainIdx = -1
		st.PrevFiltIdx = -1
		st.PrevLag = -1
		st.PrevFracLag = -1
		st.PrevLagblk = -1
		st.PrevLagidx = -1
	}
	st.PrevStage1 = stage1
	st.PrevMatch = m
	st.HavePrev = true
}

// SmplLsfIndices is the decoded per-internal-frame LSF index set.
type SmplLsfIndices struct {
	Stage1 int32
	Grid   int32
	Stage2 [16]int32
	Extra  int32
}

// DecodeSmplLsf decodes the LSF block of one internal frame. config is the
// smpl config (0/1), intf the internal-frame index (0,1,2) within the 60 ms
// packet. Mutates st.
func DecodeSmplLsf(dec *RangeDecoder, t *SmplTables, st *SmplLsfState, config, intf int) SmplLsfIndices {
	var idx SmplLsfIndices

	// Read 1: stage-1 selector. The first internal frame uses the dedicated
	// row 0; later frames pick row 1/2 by the previous frame's stage-1 result.
	var sel int
	switch {
	case intf == 0:
		sel = 0
	case st.PrevStage1 != 0:
		sel = 2
	default:
		sel = 1
	}
	stage1 := dec.DecodeCDF(t.LsfSel[sel])
	idx.Stage1 = stage1

	// match := enter_match && stage1 == prev_stage1. enter_match is false for
	// the first internal frame and true afterwards (reset to 1 after each
	// synthesis). On no-match the pitch/LTP predictor resets to -1.
	enterMatch := intf != 0
	m := enterMatch && stage1 == st.PrevStage1
	if !m {
		st.PrevGainIdx = -1
		st.PrevFiltIdx = -1
		st.PrevLag = -1
		st.PrevFracLag = -1
	}
	st.PrevStage1 = stage1

	// Read 2: stage-1 grid. Inner select keys on the CURRENT stage1, outer
	// select on match.
	var gridCDF []uint16
	switch {
	case m:
		if stage1 != 0 {
			gridCDF = t.LsfGrid.Match1
		} else {
			gridCDF = t.LsfGrid.Match1Alt
		}
	case stage1 != 0:
		gridCDF = t.LsfGrid.Match0Alt
	default:
		gridCDF = t.LsfGrid.Match0
	}
	grid := dec.DecodeCDF(gridCDF)
	idx.Grid = grid
	st.PrevMatch = m
	st.HavePrev = true

	// Read 3: 16 stage-2 residuals, each coeff k from its own CDF
	// g_lsf[stage1][config][grid][k].
	st2 := t.LsfStage2[stage1][config][grid]
	for k := 0; k < 16; k++ {
		idx.Stage2[k] = dec.DecodeCDF(st2[k])
	}

	// "Extra" LSF read: a 3-symbol static CDF, always fires for our path
	// (p4=1, num_subfr>=2).
	idx.Extra = dec.DecodeCDF(t.LsfExtra)
	return idx
}
