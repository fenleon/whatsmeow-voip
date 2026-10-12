// MLow RED ("SplitRed") depacketization: the OUTERMOST wire layer of a
// WhatsApp MLow RTP audio payload. OPTIONAL: applied only when the call
// negotiated mlow_red_redundancy_level > 0. Ported from whatsapp-rust (MIT)
// wacore/src/voip/mlow/red.rs; MPL-2.0 for this port.
//
// On-wire (N = redundancy): [red_hdr[0..N] (2B each)][main_marker (1B)]
// [red_payloads][main_payload]. red_hdr[i]: byte0 = 0x80|(time_code&0x7f),
// byte1 = payload size; main_marker: high bit clear, low 7 bits = main time
// offset.
package mlow

import "errors"

// MlowFrame is one frame extracted from a SplitRed payload. Data borrows the
// input payload (no copy).
type MlowFrame struct {
	Data     []byte
	TimeCode uint8
	IsMain   bool
}

// Red depacketization errors.
var (
	ErrRedPktSizeZero    = errors.New("mlow red: empty packet")
	ErrRedHeaderTooShort = errors.New("mlow red: header too short")
	ErrRedRedundantShort = errors.New("mlow red: redundant block too short")
	ErrRedMainTooShort   = errors.New("mlow red: main too short")
)

// DepackSplitRed parses a SplitRed RED packet into its frames (redundant
// blocks in header order, then the main frame last). Only call when RED was
// negotiated: a bare MLow frame would be misparsed here (an active-voice TOC
// like 0x50 has its high bit clear and reads as a 0-redundancy main marker,
// not an error), so for a bare-frame stream feed the whole RTP payload as one
// MLow frame instead.
func DepackSplitRed(p []byte) ([]MlowFrame, error) {
	n := len(p)
	if n == 0 {
		return nil, ErrRedPktSizeZero
	}
	type redBlock struct {
		code uint8
		size uint8
	}
	var red []redBlock
	cur := 0
	rem := n
	for {
		if rem == 0 {
			return nil, ErrRedHeaderTooShort
		}
		b0 := p[cur]
		if b0 < 0x80 {
			// main marker (high bit clear) terminates the header run
			if rem <= 1 {
				return nil, ErrRedMainTooShort
			}
			break
		}
		if rem <= 2 {
			return nil, ErrRedRedundantShort
		}
		size := p[cur+1]
		if int(size)+2 >= rem {
			return nil, ErrRedRedundantShort
		}
		red = append(red, redBlock{code: b0 & 0x7f, size: size})
		cur += 2
		rem -= int(size) + 2
	}

	mainCode := p[cur] & 0x7f
	cur++

	frames := make([]MlowFrame, 0, len(red)+1)
	for _, r := range red {
		frames = append(frames, MlowFrame{
			Data:     p[cur : cur+int(r.size)],
			TimeCode: r.code,
		})
		cur += int(r.size)
	}
	mainSize := rem - 1 // total - header_size - sum(redundant sizes)
	frames = append(frames, MlowFrame{
		Data:     p[cur : cur+mainSize],
		TimeCode: mainCode,
		IsMain:   true,
	})
	return frames, nil
}
