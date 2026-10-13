// SILK fixed-point primitives (zero-ULP integer lin2log/log2lin/sigm_Q15),
// shared by the MLow table builders and codecs. Ported from whatsapp-rust
// (MIT) wacore/src/voip/mlow/smpl_cc_tables.rs; MPL-2.0 for this port.
package mlow

import "math/bits"

const (
	silkInt32Max = 0x7FFF_FFFF
	silkInt16Max = 0x7FFF
)

// sat16 saturates to the int16 range.
func sat16(a int32) int32 {
	if a < -0x8000 {
		return -0x8000
	}
	if a > 0x7FFF {
		return 0x7FFF
	}
	return a
}

// smulwb: (a * (int16)b) >> 16, 64-bit intermediate.
func smulwb(a, b int32) int32 {
	return int32((int64(a) * int64(int16(b))) >> 16)
}

// smulww: (a * b) >> 16, 64-bit intermediate.
func smulww(a, b int32) int32 {
	return int32((int64(a) * int64(b)) >> 16)
}

// addPosSat32: silk_ADD_POS_SAT32, saturating add of two non-negative values.
func addPosSat32(a, b int32) int32 {
	if (uint32(a)+uint32(b))&0x80000000 != 0 {
		return silkInt32Max
	}
	return a + b
}

// rshiftRound: silk_RSHIFT_ROUND.
func rshiftRound(a int32, shift int32) int32 {
	if shift == 1 {
		return (a >> 1) + (a & 1)
	}
	return ((a >> uint(shift-1)) + 1) >> 1
}

// sqrtApprox: silk_SQRT_APPROX.
func sqrtApprox(x int32) int32 {
	if x <= 0 {
		return 0
	}
	lz, fracQ7 := clzFrac(x)
	var y int32
	if lz&1 != 0 {
		y = 32768
	} else {
		y = 46214
	}
	y >>= uint(lz >> 1)
	return smlawb(y, y, smulbb(213, fracQ7))
}

// smulbb: (int16)a * (int16)b (32-bit wrapping).
func smulbb(a, b int32) int32 {
	return int32(int16(a)) * int32(int16(b))
}

// smlabb: a + (int16)b * (int16)c.
func smlabb(a, b, c int32) int32 {
	return a + int32(int16(b))*int32(int16(c))
}

// smlawb: a + ((b * (int16)c) >> 16), 64-bit intermediate.
func smlawb(a, b, c int32) int32 {
	return int32(int64(a) + ((int64(b) * int64(int16(c))) >> 16))
}

// clzFrac: leading-zero count + Q7 fractional mantissa.
func clzFrac(inp int32) (lz int32, fracQ7 int32) {
	lz = int32(bits.LeadingZeros32(uint32(inp)))
	// Rotate RIGHT by (24-lz)&31 (Go's RotateLeft32 takes a negative amount
	// for right rotation).
	fracQ7 = int32(bits.RotateLeft32(uint32(inp), -int((24-lz)&31))) & 0x7f
	return lz, fracQ7
}

// lin2log approximates 128 * log2().
func lin2log(inLin int32) int32 {
	lz, fracQ7 := clzFrac(inLin)
	return smlawb(fracQ7, fracQ7*(128-fracQ7), 179) + ((31 - lz) << 7)
}

// log2lin approximates 2^() (inverse of lin2log).
func log2lin(inLogQ7 int32) int32 {
	if inLogQ7 < 0 {
		return 0
	}
	if inLogQ7 >= 3967 {
		return int32(^uint32(0) >> 1) // i32::MAX
	}
	out := int32(1) << uint(inLogQ7>>7)
	fracQ7 := inLogQ7 & 0x7f
	inner := smlawb(fracQ7, smulbb(fracQ7, 128-fracQ7), -174)
	if inLogQ7 < 2048 {
		out += (out * inner) >> 7
	} else {
		out += (out >> 7) * inner
	}
	return out
}

// sigmQ15 is the piecewise-linear sigmoid.
func sigmQ15(inQ5 int32) int32 {
	var slope = [6]int32{237, 153, 73, 30, 12, 7}
	var pos = [6]int32{16384, 23955, 28861, 31213, 32178, 32548}
	var neg = [6]int32{16384, 8812, 3906, 1554, 589, 219}
	if inQ5 < 0 {
		inQ5 = -inQ5
		if inQ5 >= 6*32 {
			return 0
		}
		ind := inQ5 >> 5
		return neg[ind] - smulbb(slope[ind], inQ5&0x1f)
	}
	if inQ5 >= 6*32 {
		return 32767
	}
	ind := inQ5 >> 5
	return pos[ind] + smulbb(slope[ind], inQ5&0x1f)
}
