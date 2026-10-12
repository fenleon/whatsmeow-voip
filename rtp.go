// Package wavoip — WhatsApp 1:1 call RTP framing.
//
// Ported from whatsapp-rust (MIT) wacore/src/voip/rtp.rs and warp.rs
// (v0.7.0 tree); MPL-2.0 for this port.
//
// WhatsApp's audio RTP headers are 16 B (speech) / 20 B (DTX, one extra
// 0xdebe extension word); video headers are 28 B with a 12 B four-extension
// block. The send-side audio sequencer starts seq at 1 and latches the RTP
// marker on the first speech frame after DTX/priming.
package wavoip

import (
	"encoding/binary"
	"errors"
)

// RTP payload types and clocks (wire values, quoted from the Rust source).
const (
	RTPPayloadTypeOpus          = 111
	RTPPayloadTypeWhatsAppAudio = 120 // MLOW or native Opus, per negotiation
	RTPPayloadTypeMLOW          = RTPPayloadTypeWhatsAppAudio
	RTPPayloadTypeMLOWRed       = 121
	RTPPayloadTypeH264          = 97
	RTPPayloadTypeAppData       = 119
	VideoClockRate              = 90000
	VideoTSStride15FPS          = VideoClockRate / 15
	VideoTSStride20FPS          = VideoClockRate / 20
)

const (
	whatsappRTPExtensionProfile = 0xdebe
	rtpFixedHeaderLen           = 12
	whatsappRTPHeaderSize       = 16
	whatsappRTPHeaderDTXSize    = 20
	whatsappVideoRTPHeaderSize  = 28
	whatsappRTPExtensionDTXWord = 0x30010000
	rtpVersion                  = 2
)

// Media frame info values for the video extension block. IDR = frame-present
// | keyframe (bit 3); delta = frame-present only.
//
// Wire delta vs upstream main @ 0dc38af0 (spec §8, PR #1463): the amended
// semantics are keyframe=bit 3 (0x08), delta=0x00; the v0.7.0 tree these
// vectors pin captured 0x09/0x01 (frame-present | WASM presence flag).
// Follow-up: switch emission to 0x08/0x00 alongside the spec, keeping the
// parser tolerant of both.
const (
	VideoMediaFrameInfoIDR   = 0x09
	VideoMediaFrameInfoDelta = 0x01
)

// Opus priming frames used by the short-tag ladder estimate.
var (
	// OpusPrimingFrame1 is the Android first priming frame (18 bytes).
	OpusPrimingFrame1 = [18]byte{0x12, 0x36, 0x26, 0x2b, 0x4a, 0xc8, 0x2b, 0x09, 0xc9, 0x1f, 0x34, 0xc2, 0xd6, 0x7a, 0x01, 0x73, 0x1b, 0x2e}
	// OpusPrimingFrame1WASM is the WASM/Web caller priming (24 bytes).
	OpusPrimingFrame1WASM = [24]byte{0x32, 0x36, 0x26, 0x2b, 0x4a, 0xcb, 0x1b, 0x5f, 0xba, 0x91, 0x68, 0x7e, 0xb8, 0x50, 0x93, 0x58, 0xe6, 0xd0, 0xa3, 0xa9, 0xd7, 0x1d, 0x81, 0x8c}
	// OpusPrimingFrame2 is the second priming frame (5 bytes).
	OpusPrimingFrame2 = [5]byte{0x90, 0xb8, 0x14, 0x14, 0xc4}
)

// warpAudioPiggybackExt is the DTX/piggyback extension word: on DTX packets it
// carries the WARP piggyback slot (the MI tag rides there from packet #3).
var warpAudioPiggybackExt = [4]byte{0x30, 0x01, 0x00, 0x00}

// warpPiggybackStartPacket is the first packet index that may piggyback.
const warpPiggybackStartPacket = 2

func audioPiggybackExtensionFor(packetIndex int, enabled bool, startPacket int) (uint32, bool) {
	if !enabled || packetIndex < startPacket {
		return 0, false
	}
	return binary.BigEndian.Uint32(warpAudioPiggybackExt[:]), true
}

// VideoRtpExtension is the four one-byte-header extensions emitted by
// WhatsApp's video sender (IDs 3/5/6/9 in the 0xdebe block).
type VideoRtpExtension struct {
	MediaFrameInfo    uint8
	InitialBandwidth  uint16
	ShortOffset       int16
	TransportSequence uint16
}

func (e VideoRtpExtension) encodeInto(out *[12]byte) {
	out[0] = 0x30
	out[1] = e.MediaFrameInfo
	out[2] = 0x51
	binary.BigEndian.PutUint16(out[3:5], e.InitialBandwidth)
	out[5] = 0x61
	binary.BigEndian.PutUint16(out[6:8], uint16(e.ShortOffset))
	out[8] = 0x91
	binary.BigEndian.PutUint16(out[9:11], e.TransportSequence)
	out[11] = 0
}

// IsOpusDTXPayload reports whether the payload is standard-Opus DTX /
// comfort noise or short warmup silence.
func IsOpusDTXPayload(payload []byte) bool {
	n := len(payload)
	switch {
	case n == 0:
		return false
	case n <= 2:
		// libopus defines packets up to two bytes as DTX; this also covers
		// MLow's one-byte SID token.
		return true
	case n <= 15:
		b0 := payload[0]
		if b0&0xf8 == 0x08 || b0 == 0x0a {
			return true
		}
		return b0&0xf0 == 0x30 && n <= 6
	default:
		return false
	}
}

// IsMlowDTXPayload reports the MLow SID shape (TOC bit 7 set).
func IsMlowDTXPayload(payload []byte) bool {
	return len(payload) > 0 && payload[0]&0xC0 == 0x80
}

// IsOpusMlowSpeechPayload reports an MLow speech frame
// (20 ms 0x48..0x4f or 60 ms 0x50..0x57).
func IsOpusMlowSpeechPayload(payload []byte) bool {
	if len(payload) < 18 {
		return false
	}
	b0 := payload[0]
	return b0&0xf8 == 0x48 || b0&0xf8 == 0x50
}

// IsOpusPrimingPayload reports one of the two known priming frames.
func IsOpusPrimingPayload(payload []byte) bool {
	return string(payload) == string(OpusPrimingFrame1[:]) ||
		string(payload) == string(OpusPrimingFrame2[:])
}

// IsWhatsAppOpusRTPLPayload reports whether payloadType is one of the audio
// payload types (Opus, WhatsApp audio/MLow, MLow-RED).
func IsWhatsAppOpusRTPPayloadType(payloadType uint8) bool {
	switch payloadType {
	case RTPPayloadTypeOpus, RTPPayloadTypeMLOW, RTPPayloadTypeMLOWRed:
		return true
	default:
		return false
	}
}

// EstimateSrtpRtpWireBytes estimates the on-wire SRTP size
// (header + payload + auth tag) for the DTX/priming/short-frame ladder.
func EstimateSrtpRtpWireBytes(opusPayload []byte) int {
	dtx := IsOpusDTXPayload(opusPayload)
	headerSize := whatsappRTPHeaderSize
	if dtx {
		headerSize = whatsappRTPHeaderDTXSize
	}
	shortTag := dtx || IsOpusPrimingPayload(opusPayload) || len(opusPayload) <= 18
	tagLen := 10
	if shortTag {
		tagLen = 4
	}
	return headerSize + len(opusPayload) + tagLen
}

// RtpHeader is the WhatsApp RTP header (fixed fields + optional extension).
type RtpHeader struct {
	Marker         bool
	PayloadType    uint8
	SequenceNumber uint16
	Timestamp      uint32
	SSRC           uint32
	// ExtensionWord, when set, emits one 0xdebe extension word (DTX/piggyback).
	ExtensionWord *uint32
	// VideoExtension, when set, emits the 12-byte video 0xdebe block.
	VideoExtension *VideoRtpExtension
}

// ByteSize is the full header length the encoder emits for this header.
func (h RtpHeader) ByteSize() int {
	switch {
	case h.VideoExtension != nil:
		return whatsappVideoRTPHeaderSize
	case h.ExtensionWord != nil:
		return whatsappRTPHeaderDTXSize
	default:
		return whatsappRTPHeaderSize
	}
}

// RTPHeaderByteLength returns the full on-wire RTP header length (fixed 12 +
// CSRC list + optional extension block), or 0 for malformed input.
func RTPHeaderByteLength(data []byte) int {
	if len(data) < rtpFixedHeaderLen || (data[0]>>6)&0x03 != rtpVersion {
		return 0
	}
	cc := int(data[0] & 0x0f)
	headerLen := rtpFixedHeaderLen + cc*4
	if len(data) < headerLen {
		return 0
	}
	if data[0]&0x10 == 0 {
		return headerLen
	}
	if len(data) < headerLen+4 {
		return 0
	}
	extWords := int(binary.BigEndian.Uint16(data[headerLen+2 : headerLen+4]))
	headerLen += 4 + extWords*4
	if len(data) < headerLen {
		return 0
	}
	return headerLen
}

// RTPExtensionProfileAndData returns the extension profile and its
// word-aligned payload. For a valid RTP packet without an extension it
// returns (0, nil, true); ok=false means malformed.
func RTPExtensionProfileAndData(data []byte) (profile uint16, payload []byte, ok bool) {
	if len(data) < rtpFixedHeaderLen || (data[0]>>6)&0x03 != rtpVersion {
		return 0, nil, false
	}
	extensionOffset := rtpFixedHeaderLen + int(data[0]&0x0f)*4
	if len(data) < extensionOffset {
		return 0, nil, false
	}
	if data[0]&0x10 == 0 {
		return 0, nil, true
	}
	if len(data) < extensionOffset+4 {
		return 0, nil, false
	}
	words := int(binary.BigEndian.Uint16(data[extensionOffset+2 : extensionOffset+4]))
	dataStart := extensionOffset + 4
	dataEnd := dataStart + words*4
	if dataEnd > len(data) || dataEnd < dataStart {
		return 0, nil, false
	}
	return binary.BigEndian.Uint16(data[extensionOffset : extensionOffset+2]),
		data[dataStart:dataEnd], true
}

// ParseWhatsAppVideoExtension decodes the exact WhatsApp video extension set.
// Other 0xdebe layouts, including the audio piggyback word, return nil.
func ParseWhatsAppVideoExtension(data []byte) *VideoRtpExtension {
	profile, extension, ok := RTPExtensionProfileAndData(data)
	if !ok || profile != whatsappRTPExtensionProfile || len(extension) != 12 ||
		extension[0] != 0x30 || extension[2] != 0x51 || extension[5] != 0x61 ||
		extension[8] != 0x91 || extension[11] != 0 {
		return nil
	}
	return &VideoRtpExtension{
		MediaFrameInfo:    extension[1],
		InitialBandwidth:  binary.BigEndian.Uint16(extension[3:5]),
		ShortOffset:       int16(binary.BigEndian.Uint16(extension[6:8])),
		TransportSequence: binary.BigEndian.Uint16(extension[9:11]),
	}
}

// IsRTPVersion2 reports a version-2 RTP fixed header.
func IsRTPVersion2(data []byte) bool {
	return len(data) >= rtpFixedHeaderLen && (data[0]>>6)&0x03 == rtpVersion
}

// ParseRTPHeader parses the fixed RTP header fields (the extension word is
// not decoded; videoExtension is, when present). ok=false for malformed data.
func ParseRTPHeader(data []byte) (header RtpHeader, ok bool) {
	if RTPHeaderByteLength(data) == 0 {
		return RtpHeader{}, false
	}
	header = RtpHeader{
		Marker:         data[1]>>7&1 == 1,
		PayloadType:    data[1] & 0x7f,
		SequenceNumber: binary.BigEndian.Uint16(data[2:4]),
		Timestamp:      binary.BigEndian.Uint32(data[4:8]),
		SSRC:           binary.BigEndian.Uint32(data[8:12]),
	}
	if ext := ParseWhatsAppVideoExtension(data); ext != nil {
		header.VideoExtension = ext
	}
	return header, true
}

// EncodeRTPHeader appends the encoded header to out.
func EncodeRTPHeaderInto(header RtpHeader, out []byte) []byte {
	size := header.ByteSize()
	var b [whatsappVideoRTPHeaderSize]byte
	b[0] = rtpVersion << 6
	if size > rtpFixedHeaderLen {
		b[0] |= 0x10 // X=1 (WhatsApp 0xdebe extension)
	}
	b[1] = boolBit(header.Marker)<<7 | header.PayloadType&0x7f
	binary.BigEndian.PutUint16(b[2:4], header.SequenceNumber)
	binary.BigEndian.PutUint32(b[4:8], header.Timestamp)
	binary.BigEndian.PutUint32(b[8:12], header.SSRC)
	if size >= 16 {
		binary.BigEndian.PutUint16(b[12:14], whatsappRTPExtensionProfile)
		// High byte stays zero: current extension blocks are at most three words.
		if header.VideoExtension != nil {
			b[15] = 3
		} else if header.ExtensionWord != nil {
			b[15] = 1
		}
	}
	if header.VideoExtension != nil {
		var encoded [12]byte
		header.VideoExtension.encodeInto(&encoded)
		copy(b[16:28], encoded[:])
	} else if size >= 20 && header.ExtensionWord != nil {
		binary.BigEndian.PutUint32(b[16:20], *header.ExtensionWord)
	}
	return append(out, b[:size]...)
}

// EncodeRTPHeader encodes the header to a fresh buffer.
func EncodeRTPHeader(header RtpHeader) []byte {
	return EncodeRTPHeaderInto(header, make([]byte, 0, header.ByteSize()))
}

func boolBit(b bool) byte {
	if b {
		return 1
	}
	return 0
}

// RtpStream is the send-side audio sequencer: seq starts at 1, timestamp
// advances by samplesPerPacket per packet, and the first speech frame after
// DTX/priming latches the marker (MLow profile).
type RtpStream struct {
	SSRC uint32

	seq                uint16
	timestamp          uint32
	lastSentTimestamp  uint32
	hasSentTimestamp   bool
	samplesPerPacket   uint32
	speechStarted      bool
	speechStartMarkers bool
	mlowProfile        bool
	audioPacketIndex   int
	warpPiggyback      bool
	payloadType        uint8
}

// NewRtpStream creates an audio sequencer for the given SSRC and framing.
func NewRtpStream(ssrc uint32, samplesPerPacket uint32, warpPiggyback bool) *RtpStream {
	return &RtpStream{
		SSRC:               ssrc,
		seq:                1,
		samplesPerPacket:   samplesPerPacket,
		speechStartMarkers: true,
		mlowProfile:        true,
		warpPiggyback:      warpPiggyback,
		payloadType:        RTPPayloadTypeMLOW,
	}
}

// SetPayloadType switches the payload type without resetting timing.
func (s *RtpStream) SetPayloadType(payloadType uint8) bool {
	if payloadType > 127 {
		return false
	}
	s.payloadType = payloadType
	return true
}

// SetMlowProfile toggles the MLow speech-start marker latching.
func (s *RtpStream) SetMlowProfile(enabled bool) {
	s.mlowProfile = enabled
	s.speechStartMarkers = enabled
}

// RTPTimestamp is the current send timestamp (last emitted), for an RTCP
// Sender Report.
func (s *RtpStream) RTPTimestamp() uint32 {
	if s.hasSentTimestamp {
		return s.lastSentTimestamp
	}
	return s.timestamp
}

func (s *RtpStream) resolveWarpExtension(dtx bool) (uint32, bool) {
	if dtx {
		return whatsappRTPExtensionDTXWord, true
	}
	if !s.warpPiggyback {
		return 0, false
	}
	idx := s.audioPacketIndex
	s.audioPacketIndex++
	return audioPiggybackExtensionFor(idx, true, warpPiggybackStartPacket)
}

func (s *RtpStream) emit(header RtpHeader) RtpHeader {
	s.lastSentTimestamp = header.Timestamp
	s.hasSentTimestamp = true
	s.seq++
	s.timestamp += s.samplesPerPacket
	return header
}

// NextPacket produces the header for the next audio packet.
func (s *RtpStream) NextPacket(payload []byte, marker bool) RtpHeader {
	var dtx bool
	if s.mlowProfile {
		dtx = IsMlowDTXPayload(payload)
	} else {
		dtx = IsOpusDTXPayload(payload)
	}
	priming := IsOpusPrimingPayload(payload)
	speech := !dtx && !priming
	useMarker := marker || (s.speechStartMarkers && speech && !s.speechStarted)
	switch {
	case dtx:
		s.speechStarted = false
	case speech:
		s.speechStarted = true
	}
	extWord, hasExt := s.resolveWarpExtension(dtx)
	var extPtr *uint32
	if hasExt {
		extPtr = &extWord
	}
	return s.emit(RtpHeader{
		Marker:         useMarker,
		PayloadType:    s.payloadType,
		SequenceNumber: s.seq,
		Timestamp:      s.timestamp,
		SSRC:           s.SSRC,
		ExtensionWord:  extPtr,
	})
}

// NextPreSpeechPacket produces a pre-speech ladder packet: it advances
// seq/timestamp without a marker or speech latch.
func (s *RtpStream) NextPreSpeechPacket() RtpHeader {
	extWord, hasExt := s.resolveWarpExtension(false)
	var extPtr *uint32
	if hasExt {
		extPtr = &extWord
	}
	return s.emit(RtpHeader{
		PayloadType:    s.payloadType,
		SequenceNumber: s.seq,
		Timestamp:      s.timestamp,
		SSRC:           s.SSRC,
		ExtensionWord:  extPtr,
	})
}

// VideoRtpStream is the send-side video sequencer: every packet of one access
// unit shares a timestamp that advances by tsStride only when the AU closes
// (marker packet).
type VideoRtpStream struct {
	SSRC uint32

	seq               uint16
	timestamp         uint32
	lastSentTimestamp uint32
	hasSentTimestamp  bool
	tsStride          uint32
	transportSequence uint16
}

// NewVideoRtpStream creates a video sequencer; tsStride must be non-zero.
func NewVideoRtpStream(ssrc uint32, tsStride uint32) (*VideoRtpStream, error) {
	if tsStride == 0 {
		return nil, errors.New("video rtp: tsStride must be non-zero")
	}
	return &VideoRtpStream{SSRC: ssrc, tsStride: tsStride}, nil
}

// RTPTimestamp is the current send timestamp (last emitted AU), for an RTCP
// Sender Report.
func (s *VideoRtpStream) RTPTimestamp() uint32 {
	if s.hasSentTimestamp {
		return s.lastSentTimestamp
	}
	return s.timestamp
}

// SetTimestampStride changes the cadence after the current AU, keeping
// sequence numbers and the RTP clock continuous across frame-rate changes.
func (s *VideoRtpStream) SetTimestampStride(tsStride uint32) bool {
	if tsStride == 0 {
		return false
	}
	s.tsStride = tsStride
	return true
}

// NextVideoPacket produces the header for the next packet of the current AU;
// lastInAU sets the marker and moves the timestamp to the next AU.
// mediaFrameInfo must be identical on every fragment of the AU.
func (s *VideoRtpStream) NextVideoPacket(lastInAU bool, mediaFrameInfo uint8) RtpHeader {
	header := RtpHeader{
		Marker:         lastInAU,
		PayloadType:    RTPPayloadTypeH264,
		SequenceNumber: s.seq,
		Timestamp:      s.timestamp,
		SSRC:           s.SSRC,
		VideoExtension: &VideoRtpExtension{
			MediaFrameInfo: mediaFrameInfo,
			// short_offset: zero is the truthful value when the API has no
			// capture-time origin to subtract from the RTP timestamp.
			TransportSequence: s.transportSequence,
		},
	}
	s.lastSentTimestamp = header.Timestamp
	s.hasSentTimestamp = true
	s.seq++
	s.transportSequence++
	if lastInAU {
		s.timestamp += s.tsStride
	}
	return header
}
