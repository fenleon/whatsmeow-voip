// Opus/CELT range decoder and encoder (RFC 6716 §4.1), the entropy coder
// MLow's smpl_audio_codec reuses verbatim. Ported from whatsapp-rust (MIT)
// wacore/src/voip/mlow/rangecoder.rs; MPL-2.0 for this port.
//
// Range-coded symbols come from the front of the buffer, raw bits from the
// back. uint32 arithmetic wraps (Go's native uint32 semantics).
//
// The full primitive set is ported (raw bits / icdf / cdf / uint / fine) even
// where the decode path only exercises a subset, keeping the entropy coder
// faithful and reusable.
package mlow

import "math/bits"

const (
	ecSymBits    = 8
	ecCodeBits   = 32
	ecSymMax     = uint32(1<<ecSymBits - 1) // 255
	ecCodeTop    = uint32(1) << (ecCodeBits - 1)
	ecCodeBot    = ecCodeTop >> ecSymBits
	ecCodeExtra  = (ecCodeBits-2)%ecSymBits + 1 // 7
	ecWindowSize = 32
	ecUintBits   = 8
	ecCodeShift  = ecCodeBits - ecSymBits - 1 // 23
)

// ecIlog is floor(log2(x))+1 for x>0, 0 for x==0.
func ecIlog(x uint32) int32 {
	return int32(ecCodeBits - bits.LeadingZeros32(x))
}

// RangeDecoder is the RFC 6716 ec_dec. Err is a sticky decode error
// (degenerate/malformed table or exhausted bits); higher layers inspect it to
// fail loud instead of synthesizing from garbage.
type RangeDecoder struct {
	buf        []byte
	storage    uint32
	endOffs    uint32
	endWindow  uint32
	nendBits   int32
	nbitsTotal int32
	offs       uint32
	rng        uint32
	val        uint32
	ext        uint32
	rem        int32
	Err        int32
}

// NewRangeDecoder is RFC 6716 ec_dec_init.
func NewRangeDecoder(buf []byte) *RangeDecoder {
	d := &RangeDecoder{
		buf:        buf,
		storage:    uint32(len(buf)),
		nbitsTotal: int32(ecCodeBits+1) - int32(((ecCodeBits-ecCodeExtra)/ecSymBits)*ecSymBits),
		rng:        1 << ecCodeExtra,
	}
	d.rem = int32(d.readByte())
	d.val = d.rng - 1 - (uint32(d.rem) >> (ecSymBits - ecCodeExtra))
	d.normalize()
	return d
}

func (d *RangeDecoder) readByte() uint32 {
	if d.offs < d.storage {
		b := d.buf[d.offs]
		d.offs++
		return uint32(b)
	}
	return 0
}

func (d *RangeDecoder) readByteFromEnd() uint32 {
	if d.endOffs < d.storage {
		d.endOffs++
		return uint32(d.buf[d.storage-d.endOffs])
	}
	return 0
}

func (d *RangeDecoder) normalize() {
	for d.rng <= ecCodeBot {
		d.nbitsTotal += ecSymBits
		d.rng <<= ecSymBits
		sym0 := d.rem
		d.rem = int32(d.readByte())
		sym := (uint32(sym0)<<ecSymBits | uint32(d.rem)) >> (ecSymBits - ecCodeExtra)
		d.val = ((d.val << ecSymBits) + (ecSymMax & ^sym)) & (ecCodeTop - 1)
	}
}

// Decode returns the cumulative frequency in [0, ft) for the next symbol; the
// caller locates the symbol and calls Update.
func (d *RangeDecoder) Decode(ft uint32) uint32 {
	if ft == 0 {
		d.Err = 1
		d.ext = 1
		return 0
	}
	d.ext = d.rng / ft
	if d.ext == 0 {
		d.Err = 1
		d.ext = 1
		return 0
	}
	s := d.val / d.ext
	return ft - minU32(s+1, ft)
}

func (d *RangeDecoder) decodeBin(bitsN uint32) uint32 {
	d.ext = d.rng >> bitsN
	if d.ext == 0 {
		d.Err = 1
		d.ext = 1
		return 0
	}
	s := d.val / d.ext
	ft := uint32(1) << bitsN
	return ft - minU32(s+1, ft)
}

// DecodeRawSymbol decodes a uniform nbits-bit symbol directly off the range
// stream (sign coder).
func (d *RangeDecoder) DecodeRawSymbol(nbits uint32) uint32 {
	sym := d.decodeBin(nbits)
	d.Update(sym, sym+1, 1<<nbits)
	return sym
}

// Update advances past the symbol with cumulative range [fl,fh) out of ft.
func (d *RangeDecoder) Update(fl, fh, ft uint32) {
	s := d.ext * (ft - fh)
	d.val -= s
	if fl > 0 {
		d.rng = d.ext * (fh - fl)
	} else {
		d.rng -= s
	}
	d.normalize()
}

// BitLogp decodes one bit with P(0) = 1/2^logp (ec_dec_bit_logp).
func (d *RangeDecoder) BitLogp(logp uint32) int32 {
	r := d.rng
	dv := d.val
	s := r >> logp
	var ret int32
	if dv < s {
		ret = 1
		d.rng = s
	} else {
		d.val = dv - s
		d.rng = r - s
	}
	d.normalize()
	return ret
}

// DecodeICDF decodes a symbol against an inverse-CDF table (ec_dec_icdf);
// ftb = log2(ft).
func (d *RangeDecoder) DecodeICDF(icdf []byte, ftb uint32) int32 {
	if len(icdf) == 0 {
		d.Err = 1
		return 0
	}
	s0 := d.rng
	dv := d.val
	r := s0 >> ftb
	ret := int32(-1)
	var t uint32
	s := s0
	for {
		t = s
		ret++
		s = r * uint32(icdf[ret])
		if dv >= s || int(ret) >= len(icdf)-1 {
			break
		}
	}
	d.val = dv - s
	d.rng = t - s
	d.normalize()
	return ret
}

// DecodeCDF decodes a symbol against a u16 CUMULATIVE CDF table (the smpl
// primitive). The effective total is cdf[n-1]-cdf[0] (a non-zero base is
// subtracted out).
func (d *RangeDecoder) DecodeCDF(cdf []uint16) int32 {
	n := len(cdf)
	if n < 2 {
		d.Err = 1
		return 0
	}
	base := uint32(cdf[0])
	if uint32(cdf[n-1]) <= base {
		d.Err = 1
		return 0
	}
	ft := uint32(cdf[n-1]) - base
	fs := d.Decode(ft)
	target := base + fs
	k := 0
	for k < n-1 && uint32(cdf[k+1]) <= target {
		k++
	}
	d.Update(uint32(cdf[k])-base, uint32(cdf[k+1])-base, ft)
	return int32(k)
}

// DecodeCDFWindow is DecodeCDF against a length-n window of base starting at
// start, without materializing the window. Out-of-range positions read 0, so
// the math is identical to DecodeCDF over the zero-filled window.
func (d *RangeDecoder) DecodeCDFWindow(base []uint16, start, n int) int32 {
	if n < 2 {
		d.Err = 1
		return 0
	}
	at := func(i int) uint32 {
		if start+i < len(base) {
			return uint32(base[start+i])
		}
		return 0
	}
	cbase := at(0)
	last := at(n - 1)
	if last <= cbase {
		d.Err = 1
		return 0
	}
	ft := last - cbase
	fs := d.Decode(ft)
	target := cbase + fs
	k := 0
	for k < n-1 && at(k+1) <= target {
		k++
	}
	d.Update(at(k)-cbase, at(k+1)-cbase, ft)
	return int32(k)
}

// DecodeCDFLE16 is DecodeCDF reading the n-entry u16 CDF directly from a
// little-endian byte slice (len(bytes) == 2*n).
func (d *RangeDecoder) DecodeCDFLE16(bytes []byte) int32 {
	n := len(bytes) / 2
	at := func(i int) uint32 {
		return uint32(bytes[2*i]) | uint32(bytes[2*i+1])<<8
	}
	if n < 2 {
		d.Err = 1
		return 0
	}
	base := at(0)
	if at(n-1) <= base {
		d.Err = 1
		return 0
	}
	ft := at(n-1) - base
	fs := d.Decode(ft)
	target := base + fs
	k := 0
	for k < n-1 && at(k+1) <= target {
		k++
	}
	d.Update(at(k)-base, at(k+1)-base, ft)
	return int32(k)
}

// BitsN reads raw n bits from the BACK of the buffer (ec_dec_bits),
// LSB-first.
func (d *RangeDecoder) BitsN(n uint32) uint32 {
	window := d.endWindow
	available := d.nendBits
	for uint32(available) < n {
		window |= d.readByteFromEnd() << uint32(available)
		available += ecSymBits
		if uint32(available) > ecWindowSize-ecSymBits {
			break
		}
	}
	ret := window & ((1 << n) - 1)
	window >>= n
	available -= int32(n)
	d.endWindow = window
	d.nendBits = available
	d.nbitsTotal += int32(n)
	return ret
}

// DecodeUint decodes an integer uniformly distributed in [0, ft) for ft>1
// (ec_dec_uint).
func (d *RangeDecoder) DecodeUint(ft0 uint32) uint32 {
	ft := ft0 - 1
	ftb := ecIlog(ft)
	if ftb > ecUintBits {
		ftb -= ecUintBits
		t := (ft >> uint(ftb)) + 1
		s := d.Decode(t)
		d.Update(s, s+1, t)
		v := (s << uint(ftb)) | d.BitsN(uint32(ftb))
		if v <= ft {
			return v
		}
		d.Err = 1
		return ft
	}
	ft++
	s := d.Decode(ft)
	d.Update(s, s+1, ft)
	return s
}

// Decode64FineSym decodes the 64-symbol uniform fine-lag read:
// ext = rng>>6, sym = clamp(63 - val/ext, 0, 64), then Update(sym, sym+1, 64).
func (d *RangeDecoder) Decode64FineSym() int32 {
	d.ext = d.rng >> 6
	if d.ext == 0 {
		d.Err = 1
		d.ext = 1
		return 0
	}
	s := d.val / d.ext
	sym := clampI32(63-int32(s), 0, 64)
	d.Update(uint32(sym), uint32(sym)+1, 64)
	return sym
}

// Tell is the bits consumed so far, rounded up (ec_tell).
func (d *RangeDecoder) Tell() int32 {
	return d.nbitsTotal - ecIlog(d.rng)
}

func minU32(a, b uint32) uint32 {
	if a < b {
		return a
	}
	return b
}

func clampI32(v, lo, hi int32) int32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// RangeEncoder is the Opus/CELT ec_enc, the exact inverse of RangeDecoder,
// used by the mlow encoder. It writes range-coded symbols toward the front
// and raw bits toward the back; Done flushes and merges them, after which
// Bytes is the finished payload.
type RangeEncoder struct {
	buf        []byte
	storage    uint32
	endOffs    uint32
	endWindow  uint32
	nendBits   int32
	nbitsTotal int32
	offs       uint32
	rng        uint32
	val        uint32
	ext        uint32
	rem        int32
	err        int32
}

// NewRangeEncoder allocates a fixed-size output buffer.
func NewRangeEncoder(size int) *RangeEncoder {
	return &RangeEncoder{
		buf:        make([]byte, size),
		storage:    uint32(size),
		nbitsTotal: int32(ecCodeBits + 1),
		rng:        ecCodeTop,
		rem:        -1,
	}
}

// Reset starts a new stream while retaining the fixed output allocation.
func (e *RangeEncoder) Reset() {
	e.endOffs = 0
	e.endWindow = 0
	e.nendBits = 0
	e.nbitsTotal = int32(ecCodeBits + 1)
	e.offs = 0
	e.rng = ecCodeTop
	e.val = 0
	e.ext = 0
	e.rem = -1
	e.err = 0
}

// Err reports a sticky encode error (buffer exhausted or invalid input).
func (e *RangeEncoder) Err() int32 {
	return e.err
}

func (e *RangeEncoder) writeByte(b uint32) {
	if e.offs+e.endOffs < e.storage {
		e.buf[e.offs] = byte(b)
		e.offs++
	} else {
		e.err = -1
	}
}

func (e *RangeEncoder) writeByteAtEnd(b uint32) {
	if e.offs+e.endOffs < e.storage {
		e.endOffs++
		e.buf[e.storage-e.endOffs] = byte(b)
	} else {
		e.err = -1
	}
}

func (e *RangeEncoder) carryOut(c int32) {
	if uint32(c) != ecSymMax {
		carry := c >> ecSymBits
		if e.rem >= 0 {
			e.writeByte(uint32(e.rem + carry))
		}
		if e.ext > 0 {
			sym := uint32((int32(ecSymMax) + carry) & int32(ecSymMax))
			for {
				e.writeByte(sym)
				e.ext--
				if e.ext == 0 {
					break
				}
			}
		}
		e.rem = c & int32(ecSymMax)
	} else {
		e.ext++
	}
}

func (e *RangeEncoder) normalize() {
	for e.rng <= ecCodeBot {
		e.carryOut(int32(e.val >> ecCodeShift))
		e.val = (e.val << ecSymBits) & (ecCodeTop - 1)
		e.rng <<= ecSymBits
		e.nbitsTotal += ecSymBits
	}
}

// Encode encodes the symbol with cumulative range [fl,fh) out of ft.
func (e *RangeEncoder) Encode(fl, fh, ft uint32) {
	if ft == 0 {
		e.err = -1
		return
	}
	r := e.rng / ft
	if fl > 0 {
		e.val += e.rng - r*(ft-fl)
		e.rng = r * (fh - fl)
	} else {
		e.rng -= r * (ft - fh)
	}
	e.normalize()
}

// BitLogp encodes one bit with P(0) = 1/2^logp.
func (e *RangeEncoder) BitLogp(val int32, logp uint32) {
	r := e.rng
	l := e.val
	s := r >> logp
	r2 := r - s
	if val != 0 {
		e.val = l + r2
		e.rng = s
	} else {
		e.rng = r2
	}
	e.normalize()
}

// EncodeICDF encodes symbol s against an inverse-CDF table (ftb = log2(ft)).
func (e *RangeEncoder) EncodeICDF(s int32, icdf []byte, ftb uint32) {
	r := e.rng >> ftb
	if s > 0 {
		e.val += e.rng - r*uint32(icdf[s-1])
		e.rng = r * (uint32(icdf[s-1]) - uint32(icdf[s]))
	} else {
		e.rng -= r * uint32(icdf[s])
	}
	e.normalize()
}

// EncodeCDF is the inverse of DecodeCDF: encode symbol s against a u16
// cumulative CDF (ft = cdf[n-1]-cdf[0]).
func (e *RangeEncoder) EncodeCDF(s int32, cdf []uint16) {
	n := len(cdf)
	if n < 2 || s < 0 || int(s+1) >= n {
		e.err = -1
		return
	}
	base := uint32(cdf[0])
	if uint32(cdf[n-1]) <= base {
		e.err = -1
		return
	}
	ft := uint32(cdf[n-1]) - base
	e.Encode(uint32(cdf[s])-base, uint32(cdf[s+1])-base, ft)
}

// EncodeCDFWindow is EncodeCDF against a length-n window of base starting at
// start, without materializing the window. Out-of-range positions read 0.
func (e *RangeEncoder) EncodeCDFWindow(s int32, base []uint16, start, n int) {
	if n < 2 || s < 0 || int(s+1) >= n {
		e.err = -1
		return
	}
	at := func(i int) uint32 {
		if start+i < len(base) {
			return uint32(base[start+i])
		}
		return 0
	}
	cbase := at(0)
	last := at(n - 1)
	if last <= cbase {
		e.err = -1
		return
	}
	ft := last - cbase
	e.Encode(at(int(s))-cbase, at(int(s)+1)-cbase, ft)
}

// BitsN writes raw n bits toward the back of the buffer.
func (e *RangeEncoder) BitsN(fl, n uint32) {
	window := e.endWindow
	used := e.nendBits
	if used+int32(n) > ecWindowSize {
		for {
			e.writeByteAtEnd(window & ecSymMax)
			window >>= ecSymBits
			used -= ecSymBits
			if used < ecSymBits {
				break
			}
		}
	}
	window |= fl << uint32(used)
	used += int32(n)
	e.endWindow = window
	e.nendBits = used
	e.nbitsTotal += int32(n)
}

// EncodeUint is the inverse of DecodeUint for a value in [0, ft).
func (e *RangeEncoder) EncodeUint(fl, ft0 uint32) {
	ft := ft0 - 1
	ftb := ecIlog(ft)
	if ftb > ecUintBits {
		ftb -= ecUintBits
		t := (ft >> uint(ftb)) + 1
		e.Encode(fl>>uint(ftb), (fl>>uint(ftb))+1, t)
		e.BitsN(fl&((1<<uint(ftb))-1), uint32(ftb))
	} else {
		e.Encode(fl, fl+1, ft+1)
	}
}

// EncodeRawSymbol is the inverse of DecodeRawSymbol: encode a uniform
// nbits-bit symbol on the range stream.
func (e *RangeEncoder) EncodeRawSymbol(sym, nbits uint32) {
	e.Encode(sym, sym+1, 1<<nbits)
}

// Encode64FineSym is the inverse of Decode64FineSym.
func (e *RangeEncoder) Encode64FineSym(sym int32) {
	e.Encode(uint32(sym), uint32(sym)+1, 64)
}

// Done flushes the range coder and merges the back raw-bit stream. After
// this, Bytes is the payload.
func (e *RangeEncoder) Done() {
	l := int32(ecCodeBits) - ecIlog(e.rng)
	msk := (ecCodeTop - 1) >> uint(l)
	end := (e.val + msk) & ^msk
	if end|msk >= e.val+e.rng {
		l++
		msk >>= 1
		end = (e.val + msk) & ^msk
	}
	for l > 0 {
		e.carryOut(int32(end >> ecCodeShift))
		end = (end << ecSymBits) & (ecCodeTop - 1)
		l -= ecSymBits
	}
	if e.rem >= 0 || e.ext > 0 {
		e.carryOut(0)
	}
	window := e.endWindow
	used := e.nendBits
	for used >= ecSymBits {
		e.writeByteAtEnd(window & ecSymMax)
		window >>= ecSymBits
		used -= ecSymBits
	}
	if e.err == 0 {
		for i := e.offs; i < e.storage-e.endOffs; i++ {
			e.buf[i] = 0
		}
		if used > 0 {
			if e.endOffs >= e.storage-e.offs {
				e.err = -1
			} else {
				e.buf[e.storage-e.endOffs-1] |= byte(window)
			}
		}
	}
}

// Bytes returns the finished payload.
func (e *RangeEncoder) Bytes() []byte {
	return e.buf
}

// ConsumedLen is the meaningful body length = front (range) bytes + back
// (raw-bit) bytes; the gap between is zero-fill padding that Done wrote. The
// smpl encode path issues no back-bit ops, so endOffs stays 0 and this is
// just the front length; if a back-bit op is ever added, the front/back
// layout must be compacted (per the C ec_enc_shrink) before this stays
// correct.
func (e *RangeEncoder) ConsumedLen() int {
	return int(e.offs + e.endOffs)
}
