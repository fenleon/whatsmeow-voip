// STUN/WARP relay framing: RFC 5389 TLV encoder with WhatsApp's
// MESSAGE-INTEGRITY (HMAC-SHA1) and FINGERPRINT (CRC-32), the allocate
// builders (WASM + APK), the WhatsApp ping, and the subscription protobuf
// bodies.
//
// Ported from whatsapp-rust (MIT) wacore/src/voip/stun.rs (v0.7.0 tree);
// MPL-2.0 for this port. Transaction IDs are passed in so the builders stay
// pure and deterministically testable.
package wavoip

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/binary"
	"hash/crc32"
	"strconv"
)

const (
	stunMagic          = 0x2112a442
	stunFingerprintXOR = 0x5354554e
	stunXORPort        = 0x2112
)

var stunXORAddr = [4]byte{0x21, 0x12, 0xa4, 0x42}

const (
	stunAttrMessageIntegrity      = 0x0008
	stunAttrErrorCode             = 0x0009
	stunAttrWasmRelayEndpoint     = 0x0016
	stunAttrFingerprint           = 0x8028
	stunAttrRelayToken            = 0x4000
	stunAttrReceiverSubscriptions = 0x4021
	stunAttrStreamDescriptors     = 0x4024
	stunAttrSenderSubscriptionsV2 = 0x4025
	stunAttrParticipantCount      = 0x805a
)

// STUN message types.
const (
	MsgBindingRequest  = 0x0001
	MsgAllocateRequest = 0x0003
	MsgBindingSuccess  = 0x0101
	MsgBindingError    = 0x0111
	MsgAllocateSuccess = 0x0103
	MsgAllocateError   = 0x0113
	MsgWhatsAppPing    = 0x0801
	MsgWhatsAppPong    = 0x0802
)

func pad4(n int) int {
	return (4 - n%4) % 4
}

// stunAttr encodes a single STUN attribute (type, length, value, 4-byte
// alignment padding).
func stunAttr(attrType uint16, value []byte) []byte {
	out := make([]byte, 4, 4+len(value)+pad4(len(value)))
	binary.BigEndian.PutUint16(out[0:2], attrType)
	binary.BigEndian.PutUint16(out[2:4], uint16(len(value)))
	out = append(out, value...)
	return append(out, make([]byte, pad4(len(value)))...)
}

// stunPseudoHeader builds the 20-byte header used for MI/FINGERPRINT inputs.
func stunPseudoHeader(msgType uint16, msgLen uint16, transactionID [12]byte) [20]byte {
	var h [20]byte
	binary.BigEndian.PutUint16(h[0:2], msgType)
	binary.BigEndian.PutUint16(h[2:4], msgLen)
	binary.BigEndian.PutUint32(h[4:8], stunMagic)
	copy(h[8:20], transactionID[:])
	return h
}

// EncodeStunRequest encodes a STUN request per RFC 5389: header + attrs, then
// optional MESSAGE-INTEGRITY (over a pseudo-header whose length already counts
// the MI attr) and FINGERPRINT.
func EncodeStunRequest(msgType uint16, transactionID [12]byte, attrs []byte, integrityKey []byte, includeFingerprint bool) []byte {
	body := append([]byte(nil), attrs...)

	if integrityKey != nil {
		// attrs + MI attr (4 + 20)
		msgLen := uint16(len(body) + 24)
		header := stunPseudoHeader(msgType, msgLen, transactionID)
		mac := hmac.New(sha1.New, integrityKey)
		mac.Write(header[:])
		mac.Write(body)
		mi := mac.Sum(nil) // 20 bytes
		body = append(body, stunAttr(stunAttrMessageIntegrity, mi)...)
	}

	if includeFingerprint {
		// attrs + FINGERPRINT attr (4 + 4)
		msgLen := uint16(len(body) + 8)
		header := stunPseudoHeader(msgType, msgLen, transactionID)
		crcInput := append(header[:], body...)
		fp := crc32.ChecksumIEEE(crcInput) ^ stunFingerprintXOR
		var fpBytes [4]byte
		binary.BigEndian.PutUint32(fpBytes[:], fp)
		body = append(body, stunAttr(stunAttrFingerprint, fpBytes[:])...)
	}

	out := make([]byte, 20, 20+len(body))
	binary.BigEndian.PutUint16(out[0:2], msgType)
	binary.BigEndian.PutUint16(out[2:4], uint16(len(body)))
	binary.BigEndian.PutUint32(out[4:8], stunMagic)
	copy(out[8:20], transactionID[:])
	return append(out, body...)
}

// CreateNativeSenderSubscription is the native WA sender subscription:
// 1-byte count + big-endian SSRC (attr 0x4023).
func CreateNativeSenderSubscription(ssrc uint32) [5]byte {
	var buf [5]byte
	buf[0] = 1
	binary.BigEndian.PutUint32(buf[1:5], ssrc)
	return buf
}

// EncodeXorRelayEndpoint XOR-encodes an IPv4:port (6 bytes) for the WASM
// relay endpoint attribute. ok=false for a non-IPv4 address.
func EncodeXorRelayEndpoint(ipv4 string, port uint16) ([6]byte, bool) {
	var octets [4]byte
	n := 0
	start := 0
	for i := 0; i <= len(ipv4); i++ {
		if i != len(ipv4) && ipv4[i] != '.' {
			continue
		}
		if n == 4 {
			return [6]byte{}, false
		}
		v, err := strconv.ParseUint(ipv4[start:i], 10, 8)
		if err != nil {
			return [6]byte{}, false
		}
		octets[n] = byte(v)
		n++
		start = i + 1
	}
	if n != 4 {
		return [6]byte{}, false
	}
	var buf [6]byte
	binary.BigEndian.PutUint16(buf[0:2], port^stunXORPort)
	for i := 0; i < 4; i++ {
		buf[2+i] = octets[i] ^ stunXORAddr[i]
	}
	return buf, true
}

// createWasmRelayEndpointAttr is the WASM attr 0x0016 value: `00 01` followed
// by the 6-byte XOR relay endpoint.
func createWasmRelayEndpointAttr(endpointXOR [6]byte) [8]byte {
	var buf [8]byte
	binary.BigEndian.PutUint16(buf[0:2], 1)
	copy(buf[2:8], endpointXOR[:])
	return buf
}

// wasmStreamSlots are (streamIndex, subType, slotWord) in WASM wire order.
// Stream indices are audio/video0/video1; sub-types are media/FEC/NACK.
var wasmStreamSlots = [9][3]uint32{
	{0, 0, 0},
	{0, 1, 1},
	{0, 2, 4},
	{1, 0, 2},
	{1, 1, 3},
	{1, 2, 5},
	{2, 0, 7},
	{2, 1, 8},
	{2, 2, 6},
}

// WasmRelayStreamSlotWords returns the protocol slot order for the 9 relay
// streams (audio media/FEC/NACK, video0 ×3, video1 ×3).
func WasmRelayStreamSlotWords() [9]uint32 {
	var out [9]uint32
	for i, slot := range wasmStreamSlots {
		out[i] = slot[2]
	}
	return out
}

// BuildWasmStreamDescriptorsFromSSRCs builds WASM StreamDescriptors (0x4024)
// from an explicit collision-free local SSRC plan.
func BuildWasmStreamDescriptorsFromSSRCs(streamSSRCs [9]uint32, hbhFecSSRCs [2]uint32) []byte {
	var out []byte
	for i, slot := range wasmStreamSlots {
		streamIndex, subType := slot[0], slot[1]
		ssrc := streamSSRCs[i]
		if ssrc == 0 {
			continue
		}
		var d []byte
		if streamIndex != 0 {
			d = pbTag(d, 1, 0)
			d = pbVarint(d, uint64(streamIndex))
		}
		if subType != 0 {
			d = pbTag(d, 2, 0)
			d = pbVarint(d, uint64(subType))
		}
		d = pbTag(d, 3, 0)
		d = pbVarint(d, uint64(ssrc))
		out = pbLenDelim(out, 1, d)
	}
	for index, ssrc := range hbhFecSSRCs {
		if ssrc == 0 {
			continue
		}
		var descriptor []byte
		descriptor = pbTag(descriptor, 1, 0)
		descriptor = pbVarint(descriptor, uint64(index+3))
		descriptor = pbTag(descriptor, 2, 0)
		descriptor = pbVarint(descriptor, 3)
		descriptor = pbTag(descriptor, 3, 0)
		descriptor = pbVarint(descriptor, uint64(ssrc))
		out = pbLenDelim(out, 1, descriptor)
	}
	return out
}

// BuildWasmStreamDescriptors builds the call-specific WASM StreamDescriptors
// (0x4024): the 9-slot grid with each slot's SSRC derived from the call ID
// and participant ID.
func BuildWasmStreamDescriptors(callID, selfParticipantID string) ([]byte, error) {
	var ssrcs [9]uint32
	for i, slot := range WasmRelayStreamSlotWords() {
		ssrc, err := DeriveWasmParticipantSSRC(callID, selfParticipantID, slot)
		if err != nil {
			return nil, err
		}
		ssrcs[i] = ssrc
	}
	return BuildWasmStreamDescriptorsFromSSRCs(ssrcs, [2]uint32{}), nil
}

// wasmStreamDescriptorsTemplate is the captured WASM stream-descriptor
// protobuf (9-slot grid) pinned by kats.json's `stun.wasmAllocate`. The
// reference tree captured these SSRC values from a live WASM session before
// switching to per-call derivation (BuildWasmStreamDescriptors); the vector
// pins this byte-exact byte sequence, so the KAT verifier replays it.
var wasmStreamDescriptorsTemplate = []byte{
	0x0a, 0x06, 0x18, 0xca, 0xbc, 0x85, 0xae, 0x04, 0x0a, 0x08, 0x10, 0x01, 0x18, 0xa5, 0xac, 0xaf,
	0xae, 0x0a, 0x0a, 0x08, 0x10, 0x02, 0x18, 0xd6, 0xa4, 0xe6, 0xf9, 0x0f, 0x0a, 0x08, 0x08, 0x01,
	0x18, 0xf7, 0xdd, 0x9e, 0xb6, 0x0a, 0x0a, 0x0a, 0x08, 0x01, 0x10, 0x01, 0x18, 0xab, 0xcc, 0xb1,
	0xf3, 0x0d, 0x0a, 0x0a, 0x08, 0x01, 0x10, 0x02, 0x18, 0xda, 0xda, 0xef, 0x8a, 0x05, 0x0a, 0x08,
	0x08, 0x02, 0x18, 0xc5, 0xe9, 0xec, 0x8e, 0x0b, 0x0a, 0x0a, 0x08, 0x02, 0x10, 0x01, 0x18, 0xfd,
	0xc2, 0xb1, 0xb6, 0x0f, 0x0a, 0x0a, 0x08, 0x02, 0x10, 0x02, 0x18, 0xb0, 0x97, 0xf7, 0xb2, 0x09,
}

// BuildWasmStunAllocateRequest is the WASM/Web DataChannel Allocate:
// 0x4000 token + 0x4024 stream desc + 0x0016 endpoint + MI, no FINGERPRINT.
// Descriptors must carry this call's derived SSRCs rather than values
// captured from another call.
func BuildWasmStunAllocateRequest(transactionID [12]byte, relayToken []byte, endpointXOR [6]byte, integrityKey []byte, callID, selfParticipantID string) ([]byte, error) {
	descriptors, err := BuildWasmStreamDescriptors(callID, selfParticipantID)
	if err != nil {
		return nil, err
	}
	return buildWasmStunAllocateRequestWithDescriptors(transactionID, relayToken, endpointXOR, integrityKey, descriptors), nil
}

// buildWasmStunAllocateRequestWithDescriptors is the descriptor-agnostic
// allocate builder. The pinned kats.json wasmAllocate vector was captured
// with the WASM template descriptor set (see vectors/voip/README.md), so the
// KAT verifier passes the captured bytes here.
func buildWasmStunAllocateRequestWithDescriptors(transactionID [12]byte, relayToken []byte, endpointXOR [6]byte, integrityKey []byte, descriptors []byte) []byte {
	attrs := stunAttr(stunAttrRelayToken, relayToken)
	attrs = append(attrs, stunAttr(stunAttrStreamDescriptors, descriptors)...)
	endpointAttr := createWasmRelayEndpointAttr(endpointXOR)
	attrs = append(attrs, stunAttr(stunAttrWasmRelayEndpoint, endpointAttr[:])...)
	return EncodeStunRequest(MsgAllocateRequest, transactionID, attrs, integrityKey, false)
}

// BuildWhatsAppPing is the WhatsApp consent ping (type 0x0801, empty body).
func BuildWhatsAppPing(transactionID [12]byte) [20]byte {
	var out [20]byte
	binary.BigEndian.PutUint16(out[0:2], MsgWhatsAppPing)
	binary.BigEndian.PutUint32(out[4:8], stunMagic)
	copy(out[8:20], transactionID[:])
	return out
}

// StunAttribute is one parsed STUN attribute (value borrows the packet).
type StunAttribute struct {
	AttrType uint16
	Value    []byte
}

// IsStunPacket reports the STUN first-byte shape (top two bits zero).
func IsStunPacket(data []byte) bool {
	return len(data) >= 2 && data[0]&0xc0 == 0x00
}

// StunMessageType returns the message type of a STUN packet.
func StunMessageType(data []byte) (uint16, bool) {
	if len(data) < 2 {
		return 0, false
	}
	return uint16(data[0]&0x3f)<<8 | uint16(data[1]), true
}

// ParseStunAttributes parses the STUN attributes after the 20-byte header.
// A truncated final padding stops the walk instead of over-reading.
func ParseStunAttributes(data []byte) []StunAttribute {
	if !IsStunPacket(data) || len(data) < 20 {
		return nil
	}
	var attrs []StunAttribute
	off := 20
	for off < len(data) {
		if off+4 > len(data) {
			break
		}
		attrType := binary.BigEndian.Uint16(data[off : off+2])
		length := int(binary.BigEndian.Uint16(data[off+2 : off+4]))
		off += 4
		if off+length > len(data) {
			break
		}
		attrs = append(attrs, StunAttribute{AttrType: attrType, Value: data[off : off+length]})
		off += length + pad4(length)
	}
	return attrs
}

// --- Minimal protobuf wire encoding for the STUN subscription attrs ---

func pbVarint(out []byte, v uint64) []byte {
	for v >= 0x80 {
		out = append(out, byte(v)|0x80)
		v >>= 7
	}
	return append(out, byte(v))
}

func pbTag(out []byte, field, wire uint32) []byte {
	return pbVarint(out, uint64(field<<3|wire))
}

func pbZigzag(n int64) uint64 {
	return uint64((n << 1) ^ (n >> 63))
}

func pbLenDelim(out []byte, field uint32, bytes []byte) []byte {
	out = pbTag(out, field, 2)
	out = pbVarint(out, uint64(len(bytes)))
	return append(out, bytes...)
}

// CreateVoipSenderSubscriptions is `voip.SenderSubscriptions` (WASM, STUN
// attr 0x4000): one audio sender (ssrc as uint32).
func CreateVoipSenderSubscriptions(ssrc uint32) []byte {
	var sender []byte
	sender = pbTag(sender, 3, 0) // ssrc
	sender = pbVarint(sender, uint64(ssrc))
	sender = pbTag(sender, 5, 0) // stream_layer = AUDIO(0)
	sender = pbVarint(sender, 0)
	sender = pbTag(sender, 6, 0) // payload_type = MEDIA(0)
	sender = pbVarint(sender, 0)
	var out []byte
	return pbLenDelim(out, 1, sender) // senders[0]
}

// CreateApkSenderSubscriptions is `wa.voip.SenderSubscriptions` (APK, STUN
// attr 0x4025): one audio ssrc (sint64), optional pid.
func CreateApkSenderSubscriptions(ssrc uint32, pid *uint32) []byte {
	var ssrcLayers []byte
	ssrcLayers = pbTag(ssrcLayers, 1, 0) // ssrcs[0] (sint64, zigzag)
	ssrcLayers = pbVarint(ssrcLayers, pbZigzag(int64(ssrc)))
	if pid != nil {
		var p []byte
		p = pbTag(p, 1, 0) // pid (sint64)
		p = pbVarint(p, pbZigzag(int64(*pid)))
		p = pbLenDelim(p, 2, []byte("audio")) // layerId
		ssrcLayers = pbLenDelim(ssrcLayers, 2, p)
	}
	var ext []byte
	ext = pbLenDelim(ext, 1, ssrcLayers)
	var out []byte
	return pbLenDelim(out, 1, ext)
}

// CreateApkStreamDescriptors is `wa.voip.StreamDescriptors` (APK, STUN attr
// 0x4024): one audio/OPUS descriptor (ssrc sint64).
func CreateApkStreamDescriptors(ssrc uint32) []byte {
	var sd []byte
	sd = pbLenDelim(sd, 1, []byte("audio")) // stream_layer
	sd = pbLenDelim(sd, 2, []byte("OPUS"))  // payload_type
	sd = pbTag(sd, 3, 0)                    // ssrc (sint64)
	sd = pbVarint(sd, pbZigzag(int64(ssrc)))
	sd = pbTag(sd, 4, 0) // is_uplink_prefetch_enabled = false
	sd = pbVarint(sd, 0)
	var out []byte
	return pbLenDelim(out, 1, sd)
}

// BuildAndroidStunAllocateRequest is the APK Allocate: 0x4000 token + 0x4025
// sender subs + 0x4024 stream desc + MI.
func BuildAndroidStunAllocateRequest(transactionID [12]byte, relayToken []byte, ssrc uint32, pid *uint32, integrityKey []byte, includeFingerprint bool) []byte {
	attrs := stunAttr(stunAttrRelayToken, relayToken)
	attrs = append(attrs, stunAttr(stunAttrSenderSubscriptionsV2, CreateApkSenderSubscriptions(ssrc, pid))...)
	attrs = append(attrs, stunAttr(stunAttrStreamDescriptors, CreateApkStreamDescriptors(ssrc))...)
	return EncodeStunRequest(MsgAllocateRequest, transactionID, attrs, integrityKey, includeFingerprint)
}
