// MLow PVQ-style pulse decode (post-LSF blocks of the core decode). Ported
// from whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_pulse.rs; MPL-2.0 for
// this port. Wrapping semantics mirror the WASM's i32/u32 modular arithmetic.
package mlow

// smplPulseCountByte is the static gain-helper table (rodata 0xe8990):
// pulse-count byte index [config*3 + (p4+s1)].
var smplPulseCountByte = [8]byte{80, 160, 160, 16, 32, 32, 0, 0}

// mem8Static reads the static rodata byte at addr (0 outside the table).
func mem8Static(addr uint32) byte {
	if addr >= 0xe8990 && addr < 0xe8998 {
		return smplPulseCountByte[addr-0xe8990]
	}
	return 0
}

// SmplPulseResult is the decoded excitation for one internal (20 ms) frame.
type SmplPulseResult struct {
	Pulses []int32  // signed pulse magnitudes per sample position (len = p2)
	Subfr  [4]int32 // per-subframe pulse counts
}

// DecodeSmplPulses decodes the pulse blocks of one internal frame. p2 = frame
// samples (320), p3 = num subframes (4), p4 = regular flag (1), p6 = config
// (0/1), s1 = LSF stage-1 selector.
func DecodeSmplPulses(dec *RangeDecoder, cc *CcTables, p2, p3, p4, p6, s1 int32) *SmplPulseResult {
	res := &SmplPulseResult{
		Pulses: make([]int32, maxI32(p2, 0)),
	}

	idx := p4 + s1
	bByte := int32(mem8Static(uint32(p6*3+idx) + 0xe8990))
	frameLen4k := bByte * p2 / 320
	subfrLen16 := frameLen4k / p3
	posPerSubfr := p2 / p3

	// --- pulse COUNT ---
	var total int32
	if p6 != 0 {
		// WB low-rate: the pulse-count CDF for this voicing class.
		cdf := cc.NPulseCount(idx)
		total = dec.DecodeCDF(cdf)
	} else {
		// NB (config=0, our path): a TRIANGULAR prior over [0, frameLen4k].
		l := uint32(frameLen4k)
		triT := func(k uint32) uint32 {
			a := (k + 2) * (l + 1)
			b := (k - 1) * (k + 131070) >> 1
			return (a - b) & 0xffff
		}
		ft := triT(l)
		if ft == 0 {
			ft = 1
		}
		val := dec.Decode(ft)
		limit := uint32(frameLen4k) + 1
		var prevCum uint32
		var k uint32
		for {
			if k == limit {
				break
			}
			cum := triT(k)
			// found when prevCum <= val < cum (the cumulative-triangular
			// interval containing val).
			if prevCum <= val && val < cum {
				dec.Update(prevCum, cum, ft)
				break
			}
			prevCum = cum
			k++
		}
		total = int32(k)
	}

	// --- recursive binary SPLIT (p3==4 path) ---
	var split [8]int32
	if total != 0 {
		sum := maxI32(total-subfrLen16*2, 0)
		lo := maxI32(total-80, 0)
		if sum < lo {
			// min_split2 >= min_split assert path; treat as parse error
			// (zeroed subframes).
			return res
		}
		hiBound := total - lo
		if sum < hiBound {
			// window the split CDF at (sum - lo); n entries from the table base.
			n := int(hiBound-sum) + 2
			sum += dec.DecodeCDFWindow(cc.SplitCmf(total), int(sum-lo), n)
		}
		if sum > 0 {
			s0 := smplSplit3537(dec, cc, sum, subfrLen16)
			split[0] = s0
			split[1] = sum - s0
		}
		if sum < total {
			s2 := smplSplit3537(dec, cc, total-sum, subfrLen16)
			split[2] = s2
			split[3] = (total - sum) - s2
		}
		// A corrupt -1 from either half zeroes the whole split (and n_pulses).
		if split[0] == -1 || split[2] == -1 {
			split = [8]int32{}
		}
	}

	take := clampI32(p3, 0, 4)
	copy(res.Subfr[:take], split[:take])

	// --- MAGNITUDE block: per-subframe run-length pulse positions ---
	var posList, magList []int32
	pulseIdx := int32(-1)
	for subfr := int32(0); subfr < p3; subfr++ {
		cnt := split[subfr]
		if cnt <= 0 {
			continue
		}
		basePos := posPerSubfr * subfr
		runPos := basePos
		pos := posPerSubfr
		c := cnt
		for k := int32(0); k < cnt; k++ {
			if pos < 0 {
				break // defensive: malformed frame must not drive a huge CDF length
			}
			oct := (pos + 7) / 8
			bucket := cc.Runlen(oct)
			// window the c-pulses CDF by (max_samples - pos), reading pos+1
			// entries.
			start := int(bucket.MaxSamples() - pos)
			m := dec.DecodeCDFWindow(bucket.Cmf(c), start, int(pos+1))
			if m > 0 || k == 0 {
				pulseIdx++
				runPos += m
				posList = append(posList, runPos)
				magList = append(magList, 1)
				pos -= m
			} else if pulseIdx >= 0 {
				magList[pulseIdx]++
			}
			c--
		}
	}

	numPos := pulseIdx + 1

	// --- SIGN block: batched uniform sign reads (1 bit per position) ---
	if numPos > 0 {
		for p := int32(0); p <= pulseIdx; {
			nbits := numPos - p
			if nbits >= 15 {
				nbits = 15
			}
			if nbits <= 0 {
				break
			}
			sym := dec.DecodeRawSymbol(uint32(nbits))
			bitfield := sym << uint(16-nbits)
			end := p + nbits
			for q := p; q < end; q++ {
				sign := int32((bitfield>>14)&2) - 1 // +1 if MSB set else -1
				magList[q] *= sign
				bitfield <<= 1
			}
			p = end
		}
	}

	// scatter signed magnitudes into the pulse vector at their absolute
	// positions.
	for i := int32(0); i < numPos; i++ {
		pp := posList[i]
		if pp >= 0 && int(pp) < len(res.Pulses) {
			res.Pulses[pp] = magList[i]
		}
	}
	return res
}

// smplSplit3537 splits count pulses across a range, returning the count
// assigned to the first half (func 3537).
func smplSplit3537(dec *RangeDecoder, cc *CcTables, count, granularity int32) int32 {
	lo := minI32(count, granularity)
	minSplit := maxI32(count-granularity, 0)
	if lo < minSplit {
		return -1
	}
	if minSplit == lo {
		return minSplit
	}
	// window split_CMFs[count-1] at min_split, n entries.
	n := int(lo-minSplit) + 2
	return dec.DecodeCDFWindow(cc.SplitCmf(count), int(minSplit), n) + minSplit
}

func minI32(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}
