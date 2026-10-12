package wavoip

import (
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func mustHex(t *testing.T, s string) []byte {
	t.Helper()
	b, err := hex.DecodeString(s)
	require.NoError(t, err)
	return b
}

func TestEncodeRTPHeaderKAT(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	ssrc := k.Inputs.SSRC

	t.Run("speech header is 16 bytes", func(t *testing.T) {
		header := RtpHeader{Marker: true, PayloadType: 120, SequenceNumber: 1, Timestamp: 0, SSRC: ssrc}
		assert.Equal(t, k.Rtp.SpeechHeader16, hex.EncodeToString(EncodeRTPHeader(header)))
	})

	t.Run("dtx header carries the piggyback extension word", func(t *testing.T) {
		extWord := uint32(0x30010000)
		header := RtpHeader{PayloadType: 120, SequenceNumber: 2, Timestamp: 320, SSRC: ssrc, ExtensionWord: &extWord}
		assert.Equal(t, k.Rtp.DtxHeader20, hex.EncodeToString(EncodeRTPHeader(header)))
	})
}

func TestRTPHeaderRoundTrip(t *testing.T) {
	header := RtpHeader{Marker: true, PayloadType: 120, SequenceNumber: 0x1234, Timestamp: 0xdeadbeef, SSRC: 0x01020304}
	encoded := EncodeRTPHeader(header)
	assert.Equal(t, 16, RTPHeaderByteLength(encoded))
	parsed, ok := ParseRTPHeader(encoded)
	require.True(t, ok)
	assert.Equal(t, header, parsed)
}

func TestEstimateSrtpRtpWireBytes(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	payload := mustHex(t, k.Inputs.Payload)

	tests := []struct {
		name    string
		payload []byte
		want    uint64
	}{
		{"speech payload gets a 10-byte tag", payload, k.Rtp.EstimateSpeech12},
		{"dtx payload gets a 4-byte tag", []byte{0x10}, k.Rtp.EstimateDtx1},
		{"priming frame gets a 4-byte tag", OpusPrimingFrame2[:], k.Rtp.EstimatePriming2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, uint64(EstimateSrtpRtpWireBytes(tc.payload)))
		})
	}
}

func TestPayloadClassifiers(t *testing.T) {
	tests := []struct {
		name    string
		payload []byte
		opusDTX bool
		mlowDTX bool
		speech  bool
		priming bool
	}{
		{"opus dtx byte", []byte{0x10}, true, false, false, false},
		{"mlow sid token", []byte{0x90}, true, true, false, false},
		{"two-byte dtx", []byte{0xBB, 0x03}, true, true, false, false},
		{"empty payload", nil, false, false, false, false},
		{"mlow speech 20ms", make([]byte, 20), false, false, false, false},
		{"mlow speech too short", []byte{0x48, 0x00}, true, false, false, false},
		{"priming frame 1", OpusPrimingFrame1[:], false, false, false, true},
		{"priming frame 2", OpusPrimingFrame2[:], false, true, false, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.opusDTX, IsOpusDTXPayload(tc.payload))
			assert.Equal(t, tc.mlowDTX, IsMlowDTXPayload(tc.payload))
			assert.Equal(t, tc.speech, IsOpusMlowSpeechPayload(tc.payload))
			assert.Equal(t, tc.priming, IsOpusPrimingPayload(tc.payload))
		})
	}

	t.Run("audio payload types", func(t *testing.T) {
		for _, pt := range []uint8{RTPPayloadTypeOpus, RTPPayloadTypeMLOW, RTPPayloadTypeMLOWRed} {
			assert.True(t, IsWhatsAppOpusRTPPayloadType(pt), "payload type %d", pt)
		}
		assert.False(t, IsWhatsAppOpusRTPPayloadType(96))
	})
}

func TestRtpStreamSequenceAndMarker(t *testing.T) {
	s := NewRtpStream(0xabcd, 320, false)

	t.Run("priming and dtx before speech carry no marker", func(t *testing.T) {
		p := s.NextPacket(OpusPrimingFrame2[:], false)
		assert.Equal(t, uint16(1), p.SequenceNumber)
		assert.Equal(t, uint32(0), p.Timestamp)
		assert.False(t, p.Marker)
		// The priming frame starts with 0x90, which is also the MLow SID
		// shape, so it carries the DTX piggyback word.
		require.NotNil(t, p.ExtensionWord)
		assert.Equal(t, uint32(0x30010000), *p.ExtensionWord)

		d := s.NextPacket([]byte{0x90}, false)
		assert.Equal(t, uint16(2), d.SequenceNumber)
		assert.Equal(t, uint32(320), d.Timestamp)
		assert.False(t, d.Marker)
		require.NotNil(t, d.ExtensionWord)
		assert.Equal(t, uint32(0x30010000), *d.ExtensionWord)
		assert.Equal(t, uint32(320), s.RTPTimestamp())
	})

	t.Run("first speech frame latches the marker", func(t *testing.T) {
		sp := s.NextPacket(make([]byte, 40), false)
		assert.Equal(t, uint16(3), sp.SequenceNumber)
		assert.Equal(t, uint32(640), sp.Timestamp)
		assert.True(t, sp.Marker)
		assert.Equal(t, uint32(640), s.RTPTimestamp())
	})

	t.Run("subsequent speech has no marker", func(t *testing.T) {
		assert.False(t, s.NextPacket(make([]byte, 40), false).Marker)
	})

	t.Run("speech after dtx re-arms the marker", func(t *testing.T) {
		assert.False(t, s.NextPacket([]byte{0x90}, false).Marker)
		assert.True(t, s.NextPacket(make([]byte, 40), false).Marker)
	})
}

func TestRtpStreamPayloadTypeSwitchKeepsTiming(t *testing.T) {
	s := NewRtpStream(0x11223344, 2880, false)
	first := s.NextPacket(make([]byte, 20), false)
	assert.Equal(t, uint8(RTPPayloadTypeMLOW), first.PayloadType)
	assert.Equal(t, uint32(0), first.Timestamp)

	require.True(t, s.SetPayloadType(RTPPayloadTypeOpus))
	second := s.NextPacket(make([]byte, 20), false)
	assert.Equal(t, uint8(RTPPayloadTypeOpus), second.PayloadType)
	assert.Equal(t, uint16(2), second.SequenceNumber)
	assert.Equal(t, uint32(2880), second.Timestamp)
	assert.False(t, s.SetPayloadType(128))
}

func TestRtpStreamWarpPiggybackStartsAtPacketThree(t *testing.T) {
	s := NewRtpStream(1, 960, true)
	assert.Nil(t, s.NextPreSpeechPacket().ExtensionWord, "packet #0 too early")
	assert.Nil(t, s.NextPreSpeechPacket().ExtensionWord, "packet #1 too early")
	ext := s.NextPreSpeechPacket().ExtensionWord
	require.NotNil(t, ext, "packet #2 may piggyback")
	assert.Equal(t, uint32(0x30010000), *ext)
}

func TestNativeOpusCanDisableMlowSpeechMarkers(t *testing.T) {
	s := NewRtpStream(0xabcd, 960, false)
	s.SetMlowProfile(false)
	assert.False(t, s.NextPacket(make([]byte, 80), false).Marker)
	s.NextPacket([]byte{0x10}, false)
	assert.False(t, s.NextPacket(make([]byte, 80), false).Marker)
}

func TestShortMlowSpeechIsNotClassifiedAsDtx(t *testing.T) {
	s := NewRtpStream(0xabcd, 960, false)
	assert.True(t, s.NextPacket([]byte{0x48, 0x00}, false).Marker)
	next := s.NextPacket([]byte{0x48, 0x00}, false)
	assert.False(t, next.Marker)
	assert.Nil(t, next.ExtensionWord)
}

func TestVideoHeaderMatchesAndroidCapture(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)

	t.Run("idr capture", func(t *testing.T) {
		captured := mustHex(t, k.Rtp.AndroidVideoHeader28)
		header, ok := ParseRTPHeader(captured)
		require.True(t, ok)
		require.NotNil(t, header.VideoExtension)
		assert.Equal(t, RtpHeader{
			Marker: true, PayloadType: RTPPayloadTypeH264, SequenceNumber: 1,
			Timestamp: 114120, SSRC: 0x49c5fb8c,
			VideoExtension: &VideoRtpExtension{
				MediaFrameInfo: 0x09, InitialBandwidth: 0,
				ShortOffset: 29, TransportSequence: 0x0c3f,
			},
		}, header)
		assert.Equal(t, k.Rtp.AndroidVideoHeader28, hex.EncodeToString(EncodeRTPHeader(header)))
		_, extensionData, ok := RTPExtensionProfileAndData(captured)
		require.True(t, ok)
		assert.Equal(t, k.Rtp.AndroidVideoExtension12, hex.EncodeToString(extensionData))
	})

	t.Run("delta capture", func(t *testing.T) {
		captured := mustHex(t, k.Rtp.AndroidVideoDeltaHeader28)
		header, ok := ParseRTPHeader(captured)
		require.True(t, ok)
		require.NotNil(t, header.VideoExtension)
		assert.Equal(t, uint8(VideoMediaFrameInfoDelta), header.VideoExtension.MediaFrameInfo)
		assert.Equal(t, 36, int(header.VideoExtension.ShortOffset))
		assert.Equal(t, uint16(0x0a6a), header.VideoExtension.TransportSequence)
		assert.Equal(t, k.Rtp.AndroidVideoDeltaHeader28, hex.EncodeToString(EncodeRTPHeader(header)))
		assert.Equal(t, byte(VideoMediaFrameInfoDelta|0x08), byte(VideoMediaFrameInfoIDR))
	})
}

func TestVideoHeaderUsesWasmExtensionLayout(t *testing.T) {
	s, err := NewVideoRtpStream(0x11223344, VideoTSStride15FPS)
	require.NoError(t, err)
	header := s.NextVideoPacket(true, VideoMediaFrameInfoIDR)
	assert.Equal(t, []byte{
		0x90, 0xe1, // V=2, X=1, marker, PT=97
		0x00, 0x00, // sequence
		0x00, 0x00, 0x00, 0x00, // timestamp
		0x11, 0x22, 0x33, 0x44, // SSRC
		0xde, 0xbe, 0x00, 0x03, // WARP profile, three words
		0x30, 0x09, // ID 3: media_frame_info (keyframe | present)
		0x51, 0x00, 0x00, // ID 5: initial bandwidth
		0x61, 0x00, 0x00, // ID 6: short timestamp offset
		0x91, 0x00, 0x00, // ID 9: transport sequence
		0x00, // word alignment
	}, EncodeRTPHeader(header))
}

func TestVideoStreamMarkerAndTimestampPerAU(t *testing.T) {
	s, err := NewVideoRtpStream(0x11223344, VideoTSStride15FPS)
	require.NoError(t, err)

	p0 := s.NextVideoPacket(false, VideoMediaFrameInfoIDR)
	p1 := s.NextVideoPacket(false, VideoMediaFrameInfoIDR)
	p2 := s.NextVideoPacket(true, VideoMediaFrameInfoIDR)
	assert.Equal(t, []uint16{0, 1, 2}, []uint16{p0.SequenceNumber, p1.SequenceNumber, p2.SequenceNumber})
	assert.Equal(t, []uint32{0, 0, 0}, []uint32{p0.Timestamp, p1.Timestamp, p2.Timestamp})
	assert.Equal(t, []bool{false, false, true}, []bool{p0.Marker, p1.Marker, p2.Marker})
	assert.Equal(t, uint32(0), s.RTPTimestamp(), "SR maps to the emitted AU timestamp")
	for i, p := range []RtpHeader{p0, p1, p2} {
		assert.Equal(t, byte(VideoMediaFrameInfoIDR), p.VideoExtension.MediaFrameInfo)
		assert.Equal(t, uint16(i), p.VideoExtension.TransportSequence)
		assert.Equal(t, uint8(RTPPayloadTypeH264), p.PayloadType)
		assert.Nil(t, p.ExtensionWord)
		assert.Equal(t, 28, p.ByteSize())
	}

	p3 := s.NextVideoPacket(true, VideoMediaFrameInfoDelta)
	assert.Equal(t, uint32(VideoTSStride15FPS), p3.Timestamp)
	assert.Equal(t, uint16(3), p3.SequenceNumber)
	assert.Equal(t, uint32(VideoTSStride15FPS), s.RTPTimestamp())
	assert.Equal(t, byte(VideoMediaFrameInfoDelta), p3.VideoExtension.MediaFrameInfo)
	assert.Equal(t, uint16(3), p3.VideoExtension.TransportSequence)
}

func TestVideoStreamChangesCadenceWithoutReset(t *testing.T) {
	s, err := NewVideoRtpStream(0x11223344, VideoTSStride15FPS)
	require.NoError(t, err)
	first := s.NextVideoPacket(true, VideoMediaFrameInfoDelta)
	assert.Equal(t, uint32(0), first.Timestamp)

	require.True(t, s.SetTimestampStride(VideoTSStride20FPS))
	second := s.NextVideoPacket(true, VideoMediaFrameInfoDelta)
	third := s.NextVideoPacket(true, VideoMediaFrameInfoDelta)
	assert.Equal(t, uint32(VideoTSStride15FPS), second.Timestamp)
	assert.Equal(t, uint32(VideoTSStride15FPS+VideoTSStride20FPS), third.Timestamp)
	assert.False(t, s.SetTimestampStride(0))
	_, err = NewVideoRtpStream(1, 0)
	assert.Error(t, err)
}

func TestVideoStreamWrapsSeqAndTimestamp(t *testing.T) {
	s, err := NewVideoRtpStream(1, ^uint32(0))
	require.NoError(t, err)
	s.seq = 0xffff
	s.timestamp = 0xffffffff
	p := s.NextVideoPacket(true, VideoMediaFrameInfoIDR)
	assert.Equal(t, uint16(0xffff), p.SequenceNumber)
	p = s.NextVideoPacket(false, VideoMediaFrameInfoDelta)
	assert.Equal(t, uint16(0), p.SequenceNumber, "seq must wrap, not panic")
	assert.Equal(t, uint32(0xfffffffe), p.Timestamp, "timestamp must wrap, not panic")
}

func TestExtensionProfileAndData(t *testing.T) {
	packet := []byte{
		0x90, 97, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3,
		0xbe, 0xde, 0, 1, 0x31, 1, 2, 3, 0x65,
	}
	t.Run("one-byte profile", func(t *testing.T) {
		profile, data, ok := RTPExtensionProfileAndData(packet)
		require.True(t, ok)
		assert.Equal(t, uint16(0xbede), profile)
		assert.Equal(t, packet[16:20], data)
		assert.Equal(t, 20, RTPHeaderByteLength(packet))
	})
	t.Run("malformed word count", func(t *testing.T) {
		malformed := append([]byte(nil), packet...)
		malformed[15] = 2
		_, _, ok := RTPExtensionProfileAndData(malformed)
		assert.False(t, ok)
	})
	t.Run("short payload", func(t *testing.T) {
		_, _, ok := RTPExtensionProfileAndData(packet[:15])
		assert.False(t, ok)
	})
	t.Run("no extension", func(t *testing.T) {
		noExt := append([]byte(nil), packet[:12]...)
		noExt[0] = 0x80 // X=0
		_, data, ok := RTPExtensionProfileAndData(noExt)
		require.True(t, ok)
		assert.Nil(t, data)
	})
}
