// MLow pitch/LTP decode, run when the LSF stage-1 selector is 1 (voiced).
// Two parts: LTP gains (per-subframe gain index + optional 35-sym filter
// index, served from CcTables) and the lag block (primary lag absolute/delta,
// contour-map search, optional 64-sym fine read, per-segment fractional loop)
// which reads the contour window built from the pitch seed. Ported from
// whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_pitch.rs; MPL-2.0 for this
// port.
package mlow

// SmplPitchResult is the decoded LTP/pitch parameters for one internal frame.
type SmplPitchResult struct {
	GainIdx [4]int32
	FiltIdx [4]int32
	Lag     int32
	Contour int32
	// SampleLagQ6 is the per-segment reconstructed pitch lag in Q6
	// (1/64-sample).
	SampleLagQ6 [8]int32
	NumSeg      int32
	// IntLagQ6 is the per-subframe pitch lag in Q6 (the LTP lag table), kept
	// at 4 entries to match the vectors; the 8 per-40-block lags synthesis
	// consumes live in BlockLags.
	IntLagQ6 [4]int32
	// BlockLags is the per-40-sample-block laginds (8 per 20 ms frame).
	BlockLags [8]int32
	NumSubfr  int32
}

// decodeCDFMem is DecodeCDF on the n-entry CDF at heap address addr. Reads
// the region bytes in place when the window fits one region (the common
// case), falling back to the zero-filling CDFAt copy only at a region
// boundary, where the out-of-range entries must read 0.
func decodeCDFMem(dec *RangeDecoder, mem *SmplMem, addr uint32, n int) int32 {
	if bytes := mem.CDFBytes(addr, n); bytes != nil {
		return dec.DecodeCDFLE16(bytes)
	}
	return dec.DecodeCDF(mem.CDFAt(addr, n))
}

// DecodeSmplPitch decodes the pitch/LTP block of one internal frame.
// p2 = frame samples, p3 = num subframes, p6 = config (0/1); subfrCounts are
// the per-subframe pulse counts. Mutates st.
func DecodeSmplPitch(
	dec *RangeDecoder,
	mem *SmplMem,
	cc *CcTables,
	st *SmplLsfState,
	_ int32, // p2
	p3 int32,
	p6 int32,
	subfrCounts [4]int32,
) *SmplPitchResult {
	res := SmplPitchResult{FiltIdx: [4]int32{-1, -1, -1, -1}}

	// LTP gains loop: both selects key on p6 (active WB path is p6==0 HR, the
	// p6!=0 LR variant), both served from CcTables. The filter CDFs are shared
	// across p6.
	gainAccum := int32(0)
	for sf := 0; sf < 4 && sf < int(p3); sf++ {
		cnt := subfrCounts[sf]
		var gi int32
		if p6 != 0 {
			gi = dec.DecodeCDF(cc.AcbgainRowLr(st.PrevGainIdx))
		} else {
			gi = dec.DecodeCDF(cc.AcbgainRow(st.PrevGainIdx))
		}
		res.GainIdx[sf] = gi
		st.PrevGainIdx = gi

		var w0, w2 int32
		if p6 != 0 {
			w0, w2 = cc.AcbgainWeightsLr(gi)
		} else {
			w0, w2 = cc.AcbgainWeights(gi)
		}
		gainAccum += w0 + 2*w2

		if cnt > 0 {
			var fi int32
			if st.PrevFiltIdx == -1 {
				fi = dec.DecodeCDF(cc.FcbgainV())
			} else {
				fi = dec.DecodeCDF(cc.FcbgainVDelta(st.PrevFiltIdx))
			}
			res.FiltIdx[sf] = fi
			st.PrevFiltIdx = fi
		}
	}
	avgGain := gainAccum / p3 // drives the fractional-lag segment select

	// --- Lag block ---
	pcfg := mem.GClk + 0x5704
	numContoursI := int32(mem.U32(pcfg + 22240))
	lagCdf := mem.U32(pcfg + 22248)
	contourMap := mem.U32(pcfg + 22244)
	fracBase := mem.U32(pcfg + 22252)
	deltaCdf := mem.U32(pcfg + 22268)

	// primary lag:
	lag := int32(0)
	if st.PrevLag < 0 {
		lag = decodeCDFMem(dec, mem, lagCdf, int(maxI32(numContoursI+1, 0)))
	} else {
		di := decodeCDFMem(dec, mem, deltaCdf+uint32(st.PrevLag)*20, 10)
		lo := int32(mem.U8(deltaBoundsAddr + uint32(di)*2))
		hi := int32(mem.U8(deltaBoundsAddr + uint32(di)*2 + 1))
		rN := (hi - lo) + 2
		if rN < 2 {
			res.Lag = -1
			return &res // malformed delta interval
		}
		sym := decodeCDFMem(dec, mem, lagCdf+uint32(lo)*2, int(rN))
		lag = sym + lo
	}

	// contour-map search: find index where contour_map[i] == lag+1.
	target := lag + 1
	contour := int32(-1)
	for i := int32(0); i < numContours; i++ {
		if int32(mem.U8(contourMap+uint32(i))) == target {
			contour = i
			break
		}
	}
	res.Lag = lag
	res.Contour = contour
	if contour < 0 || contour >= numContoursI {
		return &res // out-of-range; stop consuming pitch bits
	}

	ctrBase := pcfg + uint32(contour)*0x44
	baseLag := mem.I32(ctrBase + 0x1d38) // contour base lag

	// (a) 64-symbol fine lag; read UNLESS prev_lag>=0 && -1 <= base_lag-prev_lag < 3.
	curLag2 := baseLag
	readFine := true
	if st.PrevLag >= 0 {
		delta := baseLag - st.PrevLag
		if delta >= -1 && delta < 3 {
			readFine = false
		}
	}
	subfrW := int32(0)
	if readFine {
		sym := dec.Decode64FineSym()
		curLag2 = (baseLag << 6) + sym
		st.PrevFracLag = curLag2
		st.PrevLag = baseLag
		segLen0 := mem.I32(ctrBase + 0x1d58)
		for i := int32(0); i < segLen0; i++ {
			if subfrW < 4 {
				res.IntLagQ6[subfrW] = curLag2
			}
			if subfrW < 8 {
				res.BlockLags[subfrW] = curLag2
			}
			subfrW++
		}
		if subfrW < 4 {
			res.IntLagQ6[subfrW] = curLag2 // trailing write, subfrW not incremented
		}
		if subfrW < 8 {
			res.BlockLags[subfrW] = curLag2
		}
	}

	// (b) fractional per-segment loop:
	cnt2 := mem.I32(ctrBase + 0x1d78)
	segSel := int32(0)
	if avgGain >= 10007 {
		if avgGain < 14085 {
			segSel = 1
		} else {
			segSel = 2
		}
	}
	fracSegBase := fracBase + uint32(segSel)*0x280
	l3 := st.PrevFracLag
	l2 := curLag2
	startSeg := int32(0)
	if readFine {
		startSeg = 1
	}
	res.NumSeg = cnt2
	for seg := startSeg; seg < cnt2; seg++ {
		segLag := mem.I32(ctrBase + 0x1d38 + uint32(seg)*4)
		nl2 := (l2 << 6) - l3 + ((segLag - l2) << 6)
		off := fracSegBase + uint32(nl2)*2 + 0xfe
		sym := decodeCDFMem(dec, mem, off, 65)
		l3 = sym + st.PrevFracLag + nl2
		if seg >= 0 && seg < 8 {
			res.SampleLagQ6[seg] = l3
		}
		segLen := mem.I32(ctrBase + 0x1d58 + uint32(seg)*4)
		for i := int32(0); i < segLen; i++ {
			if subfrW < 4 {
				res.IntLagQ6[subfrW] = l3
			}
			if subfrW < 8 {
				res.BlockLags[subfrW] = l3
			}
			subfrW++
		}
		l2 = segLag
		st.PrevFracLag = l3
		st.PrevLag = segLag
	}
	res.NumSubfr = subfrW
	return &res
}
