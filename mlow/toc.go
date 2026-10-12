// Package mlow is the Go port of the MLow codec (Meta's smpl_audio_codec,
// an Opus-forked split-band CELP), ported from the MIT-licensed
// whatsapp-rust Rust reference (wacore/src/voip/mlow, v0.7.0 tree) under
// this repo's MPL-2.0. It is verified byte-exact against
// vectors/mlow/ (see PROVENANCE.md for fixture provenance).
//
// One MLow frame = 3 chained 20 ms internal frames = 960 samples @ 16 kHz.
package mlow

// MlowToc is the decoded smpl TOC byte. StdOpus true means the remaining
// fields are unused and the frame is a standard Opus/CELT packet (decode
// with stock Opus, not the smpl path).
type MlowToc struct {
	StdOpus    bool
	SID        bool
	VAD        bool
	SampleRate int32
	FrameMs    int32
	Voiced     bool
	Active     bool
	Flag2      bool
	Flag0      bool
}

// standardOpusFrameMs is the frame duration (ms) of a standard Opus packet
// from its TOC config field b>>3 (RFC 6716 Table 2). 2.5 ms is rounded up;
// the smpl path only needs an output length for CNG frames.
func standardOpusFrameMs(b byte) int32 {
	config := b >> 3
	switch {
	case config < 12:
		// SILK NB/MB/WB
		return [4]int32{10, 20, 40, 60}[config&3]
	case config < 16:
		// Hybrid
		return [2]int32{10, 20}[(config-12)&1]
	default:
		switch config & 3 {
		case 0:
			return 3 // 2.5 ms rounded up
		case 1:
			return 5
		case 2:
			return 10
		default:
			return 20
		}
	}
}

// ParseMlowToc parses the smpl TOC byte: bit7=SID(DTX/CNG), bit6=VAD,
// bit5=internal rate (0→16k, 1→32k), bits4:3=frame size index into
// {10,20,60,120} ms, bit2=flag2, bit1=voiced-enable, bit0=flag0.
// (b & 0xC0) == 0xC0 is the in-profile standard Opus/CELT escape.
func ParseMlowToc(b byte) MlowToc {
	if b&0xC0 == 0xC0 {
		return MlowToc{
			StdOpus:    true,
			SampleRate: 16000,
			FrameMs:    standardOpusFrameMs(b),
		}
	}
	bit1 := b>>1&1 != 0
	vad := b>>6&1 != 0
	return MlowToc{
		SID:        b>>7 != 0,
		VAD:        vad,
		SampleRate: sampleRateFor(b),
		FrameMs:    [4]int32{10, 20, 60, 120}[(b>>3)&3],
		Voiced:     vad && bit1,
		Active:     vad || bit1,
		Flag2:      b>>2&1 != 0,
		Flag0:      b&1 != 0,
	}
}

func sampleRateFor(b byte) int32 {
	if b&0x20 != 0 {
		return 32000
	}
	return 16000
}
