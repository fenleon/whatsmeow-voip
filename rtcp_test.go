package wavoip

import (
	"encoding/binary"
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompactReportsMatchKAT(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)

	compact208 := BuildCompactRTCP208(k.Inputs.SSRC, uint32(k.Rtcp.RemoteSsrc))
	compact209 := BuildCompactRTCP209(k.Inputs.SSRC)
	tests := []struct {
		name string
		got  []byte
		want string
	}{
		{"compact PT 208 (local + remote SSRC)", compact208[:], k.Rtcp.Compact208},
		{"compact PT 209 (pre-speech ladder)", compact209[:], k.Rtcp.Compact209},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, hex.EncodeToString(tc.got))
		})
	}
}

func TestSenderReportMatchesKAT(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	sr := BuildSenderReport(k.Inputs.SSRC, RTCPSenderStats{
		PacketsSent:  uint32(k.Rtcp.Stats.PacketsSent),
		OctetsSent:   uint32(k.Rtcp.Stats.OctetsSent),
		RTPTimestamp: uint32(k.Rtcp.Stats.RTPTimestamp),
	}, k.Rtcp.NowMs)
	assert.Equal(t, k.Rtcp.SenderReport, hex.EncodeToString(sr[:]))
}

func TestSenderReportNTPTimestamp(t *testing.T) {
	sr := BuildSenderReport(0x12345678, RTCPSenderStats{}, 1_700_000_000_250)
	ntpSec := binary.BigEndian.Uint32(sr[8:12])
	ntpFrac := binary.BigEndian.Uint32(sr[12:16])
	assert.Equal(t, uint32(2_208_988_800+1_700_000_000), ntpSec)
	assert.Equal(t, uint32(1<<30), ntpFrac, "250 ms = quarter of the NTP fraction")
}

func TestWhatsAppRTCPNameAndSDES(t *testing.T) {
	entropy := [12]byte{0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb}
	cname := BuildWhatsAppRTCPName(entropy)
	assert.Equal(t, "66778@pj899aab.org", string(cname[:]))

	sdes := BuildSourceDescription(0x12345678, &cname)
	assert.Equal(t, "81ca0007123456780112363637373840706a3839396161622e6f726700000000", hex.EncodeToString(sdes[:]))
	assert.True(t, IsRTCPPacket(sdes[:]))
}

func TestEmptySDESIsOnePaddedChunk(t *testing.T) {
	sdes := BuildEmptySourceDescription(0x12345678)
	assert.Equal(t, "81ca00021234567800000000", hex.EncodeToString(sdes[:]))
}

func TestSenderReportCompoundIncludesSDES(t *testing.T) {
	stats := RTCPSenderStats{PacketsSent: 5, OctetsSent: 600, RTPTimestamp: 1600}
	cname := BuildWhatsAppRTCPName([12]byte{0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb})
	compound := BuildSenderReportWithSDES(0x12345678, stats, 1_718_000_000_000, &cname)
	require.Len(t, compound, 28+32)
	assert.Equal(t, byte(RTCPPtSR), compound[1])
	assert.Equal(t, byte(RTCPPtSDES), compound[28+1])
	assert.True(t, IsRTCPPacket(compound))
}

func TestSenderReportEncodesReportBlocks(t *testing.T) {
	tests := []struct {
		name   string
		report RTCPReceptionReport
		block  []byte // the 24-byte report block bytes
	}{
		{
			"positive cumulative lost",
			RTCPReceptionReport{SSRC: 0x11112222, FractionLost: 0x40, CumulativeLost: 2, ExtendedHighestSequence: 0x00010002, Jitter: 123, LastSenderReport: 0x33445566, DelaySinceLastSenderReport: 0x778899aa},
			[]byte{0x11, 0x11, 0x22, 0x22, 0x40, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 123, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa},
		},
		{
			"negative cumulative lost wraps to 24-bit two's complement",
			RTCPReceptionReport{SSRC: 0x33334444, CumulativeLost: -1},
			[]byte{0x33, 0x33, 0x44, 0x44, 0, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			compound := BuildSenderReportWithSDESAndReports(0x55556666,
				RTCPSenderStats{PacketsSent: 9, OctetsSent: 4096, RTPTimestamp: 90000},
				1_700_000_000_000, &[whatsappRTCPNameLen]byte{'0'}, []RTCPReceptionReport{tc.report})
			require.Len(t, compound, 52+32)
			assert.Equal(t, []byte{0x81, RTCPPtSR, 0, 12}, compound[:4], "RC=1, (12+1)*4 = 52")
			assert.Equal(t, tc.block, compound[28:52])
			assert.Equal(t, []byte{0x81, RTCPPtSDES, 0, 7}, compound[52:56])
		})
	}
}

func TestWhatsAppSenderReportProfileExtension(t *testing.T) {
	report := RTCPReceptionReport{SSRC: 0x33334444, FractionLost: 5, CumulativeLost: 2, ExtendedHighestSequence: 91, Jitter: 7, LastSenderReport: 8, DelaySinceLastSenderReport: 9}
	cname := BuildWhatsAppRTCPName([12]byte{})
	compound := BuildWhatsAppSenderReportWithSDES(0x11112222,
		RTCPSenderStats{PacketsSent: 3, OctetsSent: 400, RTPTimestamp: 90000},
		1_700_000_000_000, &cname, &report, true)

	require.Len(t, compound, 108)
	assert.Equal(t, []byte{0x91, RTCPPtSR, 0, 18}, compound[:4], "profile bit 4 + RC=1")
	assert.Equal(t, make([]byte, 24), compound[52:76], "transport/BWE statistics are zeros when unavailable")
	assert.Equal(t, []byte{0x91, RTCPPtSDES, 0, 7}, compound[76:80])

	t.Run("without profile extension", func(t *testing.T) {
		compound := BuildWhatsAppSenderReportWithSDES(0x11112222,
			RTCPSenderStats{}, 1_700_000_000_000, &cname, nil, false)
		assert.Equal(t, byte(0x80), compound[0])
		assert.Equal(t, byte(0x81), compound[28])
	})
}

func TestIsRTCPPacket(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	sr := mustHex(t, k.Rtcp.SenderReport)

	tests := []struct {
		name  string
		data  []byte
		isRTC bool
	}{
		{"sender report", sr, true},
		{"rtp marker + h264 PT is outside the RTCP range", []byte{0x80, 0xe1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1}, false},
		{"rtp speech header", mustHex(t, k.Rtp.SpeechHeader16), false},
		{"truncated", sr[:7], false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.isRTC, IsRTCPPacket(tc.data))
		})
	}
}
