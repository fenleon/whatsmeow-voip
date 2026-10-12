// MLow gains + nrgres decode (the GAINS block), run for UNVOICED (stage-1
// selector == 0) internal frames; mutually exclusive with the pitch block.
// Ported from whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_gains.rs;
// MPL-2.0 for this port.
package mlow

// SmplGainResult is the decoded gain set for one internal frame.
type SmplGainResult struct {
	GainQ  [4]int32 // per-subframe quantized log-gain (Q-domain)
	NrgRes [4]int32 // per-subframe energy-residual symbol (only subframes with pulses are read)
}

// DecodeSmplGains decodes the gains+nrgres reads (the p3==4 path).
// subfrCounts are the per-subframe pulse counts.
func DecodeSmplGains(dec *RangeDecoder, cc *CcTables, p3 int32, subfrCounts [4]int32) *SmplGainResult {
	res := &SmplGainResult{}

	// main gain (n=85) + delta gain (n=99)
	gainMain := dec.DecodeCDF(cc.NrgresGain4())
	gainDelta := dec.DecodeCDF(cc.NrgresShape4())
	cfgSel := int32(2)

	// gain reconstruction. The index sf + p3*gainDelta is NOT bounded to the
	// visible 64-entry array (gainDelta up to 98); the WASM reads adjacent
	// rodata, which the seed carries verbatim.
	off6 := p3 * gainDelta
	base7 := gainMain*cc.NrgStep(cfgSel) - 0x154000
	for sf := 0; sf < 4 && sf < int(p3); sf++ {
		cbv := cc.GainRecon(p3 == 4, int32(sf)+off6)
		res.GainQ[sf] = base7 + (cbv << 4)
	}

	// nrgres: per-subframe bucketed CDF (n=92) with a gain-derived slice shift.
	for sf := 0; sf < 4 && sf < int(p3); sf++ {
		cnt := subfrCounts[sf]
		if cnt <= 0 {
			continue
		}
		bucket := (cnt & 0xffff) / 10
		if cnt >= 30 {
			bucket = 3
		}
		// g = clamp((gainQ[sf]+8192)>>14, floor -85); min_offset = -neg_part
		// (forward entry shift).
		g := (res.GainQ[sf] + 8192) >> 14
		if g < -85 {
			g = -85
		}
		negPart := (g >> 31) & g
		minOffset := -negPart
		res.NrgRes[sf] = dec.DecodeCDF(cc.FcbgOffset(int(cfgSel), int(bucket), int(minOffset)))
	}
	return res
}
