// RTCP builders: standard sender reports and WhatsApp's
// compact reports (PT 208/209), SDES chunks, and the WhatsApp CNAME shape.
//
// Ported from whatsapp-rust (MIT) wacore/src/voip/rtcp.rs (v0.7.0 tree);
// MPL-2.0 for this port. The SR's NTP timestamp is taken as a nowMs argument
// so the builders stay pure/no-clock.
package wavoip

import (
	"encoding/binary"
)

// RTCP payload types (wire values, quoted from the Rust source).
const (
	RTCPPtSR         = 200
	RTCPPtRR         = 201
	RTCPPtSDES       = 202
	RTCPPtRTPFB      = 205
	RTCPPtPSFB       = 206
	RTCPPtWACompact  = 208
	RTCPPtWACompact2 = 209
)

const (
	rtcpHeaderLen       = 8
	srtcpTrailerLen     = 14
	whatsappRTCPNameLen = 18
)

// ntpUnixOffsetSecs is the seconds between 1900-01-01 and 1970-01-01.
const ntpUnixOffsetSecs = 2_208_988_800

// IsRTCPPacket reports a version-2 RTCP packet (payload type in 192..=223).
func IsRTCPPacket(data []byte) bool {
	return len(data) >= rtcpHeaderLen && (data[0]>>6)&0x03 == 2 &&
		data[1] >= 192 && data[1] <= 223
}

// RTCPSenderStats are the sender-report counters.
type RTCPSenderStats struct {
	PacketsSent  uint32
	OctetsSent   uint32
	RTPTimestamp uint32
}

// RTCPReceptionReport is one RFC 3550 report block.
type RTCPReceptionReport struct {
	SSRC                       uint32
	FractionLost               uint8
	CumulativeLost             int32
	ExtendedHighestSequence    uint32
	Jitter                     uint32
	LastSenderReport           uint32
	DelaySinceLastSenderReport uint32
}

// BuildWhatsAppRTCPName builds the native CNAME: eleven random lowercase hex
// nibbles around a fixed "@pj…org" suffix, from 12 bytes of entropy.
func BuildWhatsAppRTCPName(entropy [12]byte) [whatsappRTCPNameLen]byte {
	const hexDigits = "0123456789abcdef"
	var randomHex [11]byte
	for nibble := range randomHex {
		b := entropy[6+nibble/2]
		var value byte
		if nibble&1 == 0 {
			value = b >> 4
		} else {
			value = b & 0x0f
		}
		randomHex[nibble] = hexDigits[value]
	}
	var name [whatsappRTCPNameLen]byte
	copy(name[:5], randomHex[:5])
	copy(name[5:8], "@pj")
	copy(name[8:14], randomHex[5:])
	copy(name[14:], ".org")
	return name
}

// BuildCompactRTCP208 builds the 12-byte compact RTCP (PT 208, RC=1);
// 26 bytes on the wire with the SRTCP trailer.
func BuildCompactRTCP208(localSSRC, remoteSSRC uint32) [12]byte {
	var buf [12]byte
	buf[0] = 0x81 // V=2, P=0, RC=1
	buf[1] = RTCPPtWACompact
	buf[3] = 2 // (2+1)*4 = 12 bytes
	binary.BigEndian.PutUint32(buf[4:8], localSSRC)
	binary.BigEndian.PutUint32(buf[8:12], remoteSSRC)
	return buf
}

// BuildCompactRTCP209 builds the 8-byte compact RTCP (PT 209, RC=1):
// the pre-speech ladder report.
func BuildCompactRTCP209(localSSRC uint32) [8]byte {
	var buf [8]byte
	buf[0] = 0x81
	buf[1] = RTCPPtWACompact2
	buf[3] = 1 // (1+1)*4 = 8 bytes
	binary.BigEndian.PutUint32(buf[4:8], localSSRC)
	return buf
}

// BuildSenderReport builds the 28-byte Sender Report (PT 200, RC=0).
// nowMs is the wall clock in milliseconds.
func BuildSenderReport(localSSRC uint32, stats RTCPSenderStats, nowMs uint64) [28]byte {
	var buf [28]byte
	buf[0] = 0x80 // V=2, RC=0
	buf[1] = RTCPPtSR
	buf[3] = 6 // (6+1)*4 = 28 bytes
	binary.BigEndian.PutUint32(buf[4:8], localSSRC)
	// NTP timestamp: seconds (upper 32) since 1900, fraction (lower 32). Both
	// fields truncate to uint32 (wrapping), matching the reference encoder.
	ntpSec := uint32(nowMs/1000 + ntpUnixOffsetSecs)
	ntpFrac := uint32(float64(nowMs%1000) / 1000.0 * 4_294_967_296.0)
	binary.BigEndian.PutUint32(buf[8:12], ntpSec)
	binary.BigEndian.PutUint32(buf[12:16], ntpFrac)
	binary.BigEndian.PutUint32(buf[16:20], stats.RTPTimestamp)
	binary.BigEndian.PutUint32(buf[20:24], stats.PacketsSent)
	binary.BigEndian.PutUint32(buf[24:28], stats.OctetsSent)
	return buf
}

func encodeReceptionReport(report RTCPReceptionReport, out []byte) []byte {
	out = binary.BigEndian.AppendUint32(out, report.SSRC)
	out = append(out, report.FractionLost)
	// cumulative_lost is a 24-bit signed field: only the low 3 bytes go out.
	var lost [4]byte
	binary.BigEndian.PutUint32(lost[:], uint32(report.CumulativeLost))
	out = append(out, lost[1:]...)
	out = binary.BigEndian.AppendUint32(out, report.ExtendedHighestSequence)
	out = binary.BigEndian.AppendUint32(out, report.Jitter)
	out = binary.BigEndian.AppendUint32(out, report.LastSenderReport)
	out = binary.BigEndian.AppendUint32(out, report.DelaySinceLastSenderReport)
	return out
}

// BuildSenderReportWithReports builds an SR carrying standard RFC 3550 report
// blocks (at most 31).
func BuildSenderReportWithReports(localSSRC uint32, stats RTCPSenderStats, nowMs uint64, reports []RTCPReceptionReport) []byte {
	if len(reports) > 31 {
		reports = reports[:31]
	}
	sr := BuildSenderReport(localSSRC, stats, nowMs)
	packet := append([]byte(nil), sr[:]...)
	packet[0] |= byte(len(reports))
	for _, report := range reports {
		packet = encodeReceptionReport(report, packet)
	}
	words := uint16(len(packet)/4 - 1)
	binary.BigEndian.PutUint16(packet[2:4], words)
	return packet
}

// BuildSourceDescription builds the native 32-byte SDES packet: one chunk,
// its 18-byte CNAME, END, and padding.
func BuildSourceDescription(localSSRC uint32, cname *[whatsappRTCPNameLen]byte) [32]byte {
	var packet [32]byte
	packet[0] = 0x81
	packet[1] = RTCPPtSDES
	binary.BigEndian.PutUint16(packet[2:4], 7)
	binary.BigEndian.PutUint32(packet[4:8], localSSRC)
	packet[8] = 1
	packet[9] = whatsappRTCPNameLen
	copy(packet[10:28], cname[:])
	return packet
}

// BuildWhatsAppSourceDescription sets the WhatsApp profile-extension bit
// (count nibble bit 4) on the SDES chunk when requested.
func BuildWhatsAppSourceDescription(localSSRC uint32, cname *[whatsappRTCPNameLen]byte, profileExtension bool) [32]byte {
	packet := BuildSourceDescription(localSSRC, cname)
	if profileExtension {
		packet[0] |= 0x10
	}
	return packet
}

// BuildEmptySourceDescription builds the empty 12-byte SDES chunk native
// RTCP sessions without a CNAME still send.
func BuildEmptySourceDescription(localSSRC uint32) [12]byte {
	var packet [12]byte
	packet[0] = 0x81
	packet[1] = RTCPPtSDES
	binary.BigEndian.PutUint16(packet[2:4], 2)
	binary.BigEndian.PutUint32(packet[4:8], localSSRC)
	return packet
}

// BuildSenderReportWithSDES builds WhatsApp's periodic sender report: a
// compound `SR + SDES` datagram.
func BuildSenderReportWithSDES(localSSRC uint32, stats RTCPSenderStats, nowMs uint64, cname *[whatsappRTCPNameLen]byte) []byte {
	return BuildSenderReportWithSDESAndReports(localSSRC, stats, nowMs, cname, nil)
}

// BuildSenderReportWithSDESAndReports builds the compound SR+SDES with
// standard RFC 3550 report blocks in the SR.
func BuildSenderReportWithSDESAndReports(localSSRC uint32, stats RTCPSenderStats, nowMs uint64, cname *[whatsappRTCPNameLen]byte, reports []RTCPReceptionReport) []byte {
	sdes := BuildSourceDescription(localSSRC, cname)
	senderReport := BuildSenderReportWithReports(localSSRC, stats, nowMs, reports)
	return append(senderReport, sdes[:]...)
}

// BuildWhatsAppSenderReportWithSDES builds the native 1:1 compound: the SR's
// report block is the RFC 3550 block followed by 24 bytes of WhatsApp
// transport/BWE statistics (signaled by count-nibble bit 4).
func BuildWhatsAppSenderReportWithSDES(localSSRC uint32, stats RTCPSenderStats, nowMs uint64, cname *[whatsappRTCPNameLen]byte, report *RTCPReceptionReport, profileExtension bool) []byte {
	sr := BuildSenderReport(localSSRC, stats, nowMs)
	senderReport := append([]byte(nil), sr[:]...)
	if profileExtension {
		senderReport[0] |= 0x10
	}
	if report != nil {
		senderReport[0] |= 1
		senderReport = encodeReceptionReport(*report, senderReport)
		senderReport = append(senderReport, make([]byte, 24)...)
		words := uint16(len(senderReport)/4 - 1)
		binary.BigEndian.PutUint16(senderReport[2:4], words)
	}
	sdes := BuildWhatsAppSourceDescription(localSSRC, cname, profileExtension)
	return append(senderReport, sdes[:]...)
}
