package wavoip

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"strings"
)

// KATInputs are the shared inputs of vectors/voip/kats.json.
type KATInputs struct {
	CallKey       string `json:"callKey"`
	HbhKey        string `json:"hbhKey"`
	SelfLID       string `json:"selfLid"`
	PeerLID       string `json:"peerLid"`
	SSRC          uint32 `json:"ssrc"`
	Seq           uint16 `json:"seq"`
	Roc           uint32 `json:"roc"`
	Payload       string `json:"payload"`
	CallID        string `json:"callId"`
	SFrameCounter uint64 `json:"sframeCounter"`
	SamplePacket  string `json:"samplePacket"`
}

type katRtp struct {
	SpeechHeader16               string `json:"speechHeader16"`
	DtxHeader20                  string `json:"dtxHeader20"`
	AndroidVideoHeader28         string `json:"androidVideoHeader28"`
	AndroidVideoExtension12      string `json:"androidVideoExtension12"`
	AndroidVideoDeltaHeader28    string `json:"androidVideoDeltaHeader28"`
	AndroidVideoDeltaExtension12 string `json:"androidVideoDeltaExtension12"`
	EstimateSpeech12             uint64 `json:"estimateSpeech12"`
	EstimateDtx1                 uint64 `json:"estimateDtx1"`
	EstimatePriming2             uint64 `json:"estimatePriming2"`
}

type katRtcpStats struct {
	PacketsSent  uint64 `json:"packetsSent"`
	OctetsSent   uint64 `json:"octetsSent"`
	RTPTimestamp uint64 `json:"rtpTimestamp"`
}

type katRtcp struct {
	NowMs        uint64       `json:"nowMs"`
	RemoteSsrc   uint64       `json:"remoteSsrc"`
	Stats        katRtcpStats `json:"stats"`
	Compact208   string       `json:"compact208"`
	Compact209   string       `json:"compact209"`
	SenderReport string       `json:"senderReport"`
}

type katStun struct {
	Tx              string `json:"tx"`
	RelayToken      string `json:"relayToken"`
	MiKey           string `json:"miKey"`
	Crc32Abc        uint64 `json:"crc32_abc"`
	AttrToken       string `json:"attr_token"`
	XorEndpoint     string `json:"xorEndpoint"`
	NativeSenderSub string `json:"nativeSenderSub"`
	MinimalMi       string `json:"minimalMi"`
	WithFp          string `json:"withFp"`
	WasmAllocate    string `json:"wasmAllocate"`
	Ping            string `json:"ping"`
}

type katFile struct {
	Inputs     KATInputs         `json:"inputs"`
	SFrame     map[string]string `json:"sframe"`
	E2ESrtp    map[string]string `json:"e2e_srtp"`
	Wahkdf     map[string]string `json:"wahkdf"`
	HbhSrtp    map[string]string `json:"hbh_srtp"`
	VoipCrypto map[string]uint64 `json:"voip_crypto"`
	Rtp        katRtp            `json:"rtp"`
	Rtcp       katRtcp           `json:"rtcp"`
	Stun       katStun           `json:"stun"`
	StunProto  map[string]string `json:"stun_proto"`
}

// VerifyKats checks all sections of vectors/voip/kats.json — the media-crypto
// stack and the RTP/RTCP/STUN framing builders — against this implementation
// and returns every failure (nil = all pass).
func VerifyKats(data []byte) error {
	var k katFile
	if err := json.Unmarshal(data, &k); err != nil {
		return fmt.Errorf("parse kats: %w", err)
	}
	var failures []string
	check := func(name string, got []byte, wantHex string) {
		if !hexEqual(got, wantHex) {
			failures = append(failures, fmt.Sprintf("%s: got %s want %s", name, hex.EncodeToString(got), wantHex))
		}
	}
	checkStr := func(name, got, want string) {
		if got != want {
			failures = append(failures, fmt.Sprintf("%s: got %q want %q", name, got, want))
		}
	}

	callKey, err := hex.DecodeString(k.Inputs.CallKey)
	if err != nil {
		return fmt.Errorf("callKey: %w", err)
	}

	// wahkdf: HKDF(salt=0^32, ikm=callKey, info=peerLid, 46) -> master split.
	wahkdf, err := hkdfSHA256(make([]byte, 32), callKey[:32], []byte(k.Inputs.PeerLID), 46)
	if err != nil {
		return fmt.Errorf("wahkdf: %w", err)
	}
	check("wahkdf.okm46", wahkdf, k.Wahkdf["okm46"])
	check("wahkdf.masterKey", wahkdf[:16], k.Wahkdf["masterKey"])
	check("wahkdf.masterSalt", wahkdf[16:30], k.Wahkdf["masterSalt"])

	// e2e_srtp: derive with both LIDs; session keys, RTP IV, payload cipher, WARP tag.
	peer, err := DeriveE2ESrtpKeys(callKey, k.Inputs.PeerLID)
	if err != nil {
		return fmt.Errorf("e2e derive peer: %w", err)
	}
	self, err := DeriveE2ESrtpKeys(callKey, k.Inputs.SelfLID)
	if err != nil {
		return fmt.Errorf("e2e derive self: %w", err)
	}
	check("e2e_srtp.peer_cipherKey", peer.CipherKey[:], k.E2ESrtp["peer_cipherKey"])
	check("e2e_srtp.peer_salt", peer.Salt[:], k.E2ESrtp["peer_salt"])
	check("e2e_srtp.peer_authKey", peer.AuthKey[:], k.E2ESrtp["peer_authKey"])
	check("e2e_srtp.self_cipherKey", self.CipherKey[:], k.E2ESrtp["self_cipherKey"])
	check("e2e_srtp.self_salt", self.Salt[:], k.E2ESrtp["self_salt"])
	check("e2e_srtp.self_authKey", self.AuthKey[:], k.E2ESrtp["self_authKey"])

	iv := BuildE2ERtpIV(peer.Salt[:], k.Inputs.SSRC, k.Inputs.Roc, k.Inputs.Seq)
	check("e2e_srtp.rtpIv", iv[:], k.E2ESrtp["rtpIv"])

	payload, err := hex.DecodeString(k.Inputs.Payload)
	if err != nil {
		return fmt.Errorf("payload: %w", err)
	}
	ct, err := CryptE2EPayload(peer, k.Inputs.SSRC, k.Inputs.Seq, k.Inputs.Roc, payload)
	if err != nil {
		return fmt.Errorf("e2e crypt: %w", err)
	}
	check("e2e_srtp.cipher_out", ct, k.E2ESrtp["cipher_out"])
	pt, err := CryptE2EPayload(peer, k.Inputs.SSRC, k.Inputs.Seq, k.Inputs.Roc, ct)
	if err != nil {
		return fmt.Errorf("e2e crypt round-trip: %w", err)
	}
	if string(pt) != string(payload) {
		failures = append(failures, "e2e_srtp: payload round-trip mismatch")
	}

	samplePacket, err := hex.DecodeString(k.Inputs.SamplePacket)
	if err != nil {
		return fmt.Errorf("samplePacket: %w", err)
	}
	tag := ComputeWarpMITag(peer.AuthKey[:], samplePacket, k.Inputs.Roc, WarpMITagLen)
	check("e2e_srtp.warp_mi_tag4", tag, k.E2ESrtp["warp_mi_tag4"])

	// sframe: participant id, label, key, counter IV, header, encrypt.
	peerID := FormatParticipantID(k.Inputs.PeerLID)
	checkStr("sframe.participantPeerId", peerID, k.SFrame["participantPeerId"])
	checkStr("sframe.infoLabelPeer", KDFLabelE2ESframe+peerID, k.SFrame["infoLabelPeer"])
	peerKey, err := DeriveE2ESframeKey(callKey, peerID)
	if err != nil {
		return fmt.Errorf("sframe key: %w", err)
	}
	check("sframe.peerKey32", peerKey, k.SFrame["peerKey32"])
	selfKey, err := DeriveE2ESframeKey(callKey, FormatParticipantID(k.Inputs.SelfLID))
	if err != nil {
		return fmt.Errorf("sframe self key: %w", err)
	}
	check("sframe.selfKey32", selfKey, k.SFrame["selfKey32"])
	check("sframe.counterToIv_5", counterToIVBytes(5), k.SFrame["counterToIv_5"])
	check("sframe.header_5_0", BuildSframeHeader(5, 0), k.SFrame["header_5_0"])

	session, err := NewSframeSession(callKey, k.Inputs.SelfLID, k.Inputs.PeerLID)
	if err != nil {
		return fmt.Errorf("sframe session: %w", err)
	}
	session.TxCounter = k.Inputs.SFrameCounter
	frame, err := session.Encrypt(payload)
	if err != nil {
		return fmt.Errorf("sframe encrypt: %w", err)
	}
	check("sframe.encrypt_out", frame, k.SFrame["encrypt_out"])

	// hbh_srtp: two-stage HKDF derivations, keying split, both session expansions,
	// RTP ICM nonce + cipher, and the full SRTCP-protected sender report.
	hbh, err := hex.DecodeString(k.Inputs.HbhKey)
	if err != nil {
		return fmt.Errorf("hbhKey: %w", err)
	}
	uplinkSalt, err := hkdfSHA256(make([]byte, 32), hbh[:14], []byte(LabelUplinkSrtcpSalt), 32)
	if err != nil {
		return fmt.Errorf("hbh uplink salt: %w", err)
	}
	check("hbh_srtp.uplinkSalt32", uplinkSalt, k.HbhSrtp["uplinkSalt32"])
	uplinkKey, err := deriveHbhKeyWithLabels(hbh, LabelUplinkSrtcpSalt, LabelUplinkSrtcpKey)
	if err != nil {
		return fmt.Errorf("hbh uplink key: %w", err)
	}
	check("hbh_srtp.uplinkKey30", uplinkKey, k.HbhSrtp["uplinkKey30"])
	sharedSrtp, err := deriveHbhKeyWithLabels(hbh, LabelSharedSrtpSalt, LabelSharedSrtpKey)
	if err != nil {
		return fmt.Errorf("hbh shared srtp: %w", err)
	}
	check("hbh_srtp.sharedSrtpKey30", sharedSrtp, k.HbhSrtp["sharedSrtpKey30"])
	sharedSrtcp, err := deriveHbhKeyWithLabels(hbh, LabelSharedSrtcpSalt, LabelSharedSrtcpKey)
	if err != nil {
		return fmt.Errorf("hbh shared srtcp: %w", err)
	}
	check("hbh_srtp.sharedSrtcpKey30", sharedSrtcp, k.HbhSrtp["sharedSrtcpKey30"])
	downlinkKey, err := deriveHbhKeyWithLabels(hbh, LabelDownlinkSrtcpSalt, LabelDownlinkSrtcpKey)
	if err != nil {
		return fmt.Errorf("hbh downlink: %w", err)
	}
	check("hbh_srtp.downlinkKey30", downlinkKey, k.HbhSrtp["downlinkKey30"])

	keying, err := DeriveHbhSrtcpKeyingUplink(hbh)
	if err != nil {
		return fmt.Errorf("hbh keying: %w", err)
	}
	check("hbh_srtp.masterKey", keying.MasterKey[:], k.HbhSrtp["masterKey"])
	check("hbh_srtp.masterSalt", keying.MasterSalt[:], k.HbhSrtp["masterSalt"])

	rtpSession, err := ExpandLibsrtpSessionKeys(keying)
	if err != nil {
		return fmt.Errorf("hbh srtp expand: %w", err)
	}
	check("hbh_srtp.sessionKey", rtpSession.SessionKey[:], k.HbhSrtp["sessionKey"])
	check("hbh_srtp.sessionSalt", rtpSession.SessionSalt[:], k.HbhSrtp["sessionSalt"])
	check("hbh_srtp.authKey", rtpSession.AuthKey[:], k.HbhSrtp["authKey"])

	srtcpSession, err := ExpandLibsrtcpSessionKeys(keying)
	if err != nil {
		return fmt.Errorf("hbh srtcp expand: %w", err)
	}
	check("hbh_srtp.srtcpSessionKey", srtcpSession.SessionKey[:], k.HbhSrtp["srtcpSessionKey"])
	check("hbh_srtp.srtcpSessionSalt", srtcpSession.SessionSalt[:], k.HbhSrtp["srtcpSessionSalt"])
	check("hbh_srtp.srtcpAuthKey", srtcpSession.AuthKey[:], k.HbhSrtp["srtcpAuthKey"])

	packetIndex := uint64(k.Inputs.Roc)<<16 | uint64(k.Inputs.Seq)
	nonce := BuildRtpICMNonce(k.Inputs.SSRC, packetIndex)
	check("hbh_srtp.rtpIcmNonce", nonce[:], k.HbhSrtp["rtpIcmNonce"])
	hbhCT, err := CryptHbhRtpPayload(rtpSession, k.Inputs.SSRC, packetIndex, payload)
	if err != nil {
		return fmt.Errorf("hbh crypt: %w", err)
	}
	check("hbh_srtp.cipher_out", hbhCT, k.HbhSrtp["cipher_out"])

	// ssrc: HKDF(salt=slot_word LE32, ikm=callId, info=lid, 4) as LE u32.
	slot0, err := DeriveWasmParticipantSSRC(k.Inputs.CallID, k.Inputs.PeerLID, 0)
	if err != nil {
		return fmt.Errorf("ssrc slot0: %w", err)
	}
	slot1, err := DeriveWasmParticipantSSRC(k.Inputs.CallID, k.Inputs.PeerLID, 1)
	if err != nil {
		return fmt.Errorf("ssrc slot1: %w", err)
	}
	if slot0 != uint32(k.VoipCrypto["ssrc_slot0"]) {
		failures = append(failures, fmt.Sprintf("voip_crypto.ssrc_slot0: got %d want %d", slot0, k.VoipCrypto["ssrc_slot0"]))
	}
	if slot1 != uint32(k.VoipCrypto["ssrc_slot1"]) {
		failures = append(failures, fmt.Sprintf("voip_crypto.ssrc_slot1: got %d want %d", slot1, k.VoipCrypto["ssrc_slot1"]))
	}

	// rtp: WhatsApp speech/DTX/video headers + the SRTP wire-size ladder.
	tx, err := hex.DecodeString(k.Stun.Tx)
	if err != nil {
		return fmt.Errorf("stun.tx: %w", err)
	}
	var tx12 [12]byte
	copy(tx12[:], tx)

	speechHeader := EncodeRTPHeader(RtpHeader{
		Marker:         true,
		PayloadType:    RTPPayloadTypeMLOW,
		SequenceNumber: 1,
		Timestamp:      0,
		SSRC:           k.Inputs.SSRC,
	})
	check("rtp.speechHeader16", speechHeader, k.Rtp.SpeechHeader16)
	var dtxWord uint32 = whatsappRTPExtensionDTXWord
	dtxHeader := EncodeRTPHeader(RtpHeader{
		PayloadType:    RTPPayloadTypeMLOW,
		SequenceNumber: 2,
		Timestamp:      320,
		SSRC:           k.Inputs.SSRC,
		ExtensionWord:  &dtxWord,
	})
	check("rtp.dtxHeader20", dtxHeader, k.Rtp.DtxHeader20)
	for _, rec := range []struct {
		name      string
		captured  string
		extension string
	}{
		{"androidVideoHeader28", k.Rtp.AndroidVideoHeader28, k.Rtp.AndroidVideoExtension12},
		{"androidVideoDeltaHeader28", k.Rtp.AndroidVideoDeltaHeader28, k.Rtp.AndroidVideoDeltaExtension12},
	} {
		captured, err := hex.DecodeString(rec.captured)
		if err != nil {
			return fmt.Errorf("rtp %s: %w", rec.name, err)
		}
		header, ok := ParseRTPHeader(captured)
		if !ok || header.VideoExtension == nil {
			failures = append(failures, fmt.Sprintf("rtp.%s: parse failed", rec.name))
			continue
		}
		check("rtp."+rec.name, EncodeRTPHeader(header), rec.captured)
		ext, err := hex.DecodeString(rec.extension)
		if err != nil {
			return fmt.Errorf("rtp %s extension: %w", rec.name, err)
		}
		_, extensionData, ok := RTPExtensionProfileAndData(captured)
		if !ok || string(extensionData) != string(ext) {
			failures = append(failures, fmt.Sprintf("rtp.%s: extension block mismatch", rec.name))
		}
	}
	checkUint64 := func(name string, got uint64, want uint64) {
		if got != want {
			failures = append(failures, fmt.Sprintf("%s: got %d want %d", name, got, want))
		}
	}
	checkUint64("rtp.estimateSpeech12", uint64(EstimateSrtpRtpWireBytes(payload)), k.Rtp.EstimateSpeech12)
	checkUint64("rtp.estimateDtx1", uint64(EstimateSrtpRtpWireBytes([]byte{0x10})), k.Rtp.EstimateDtx1)
	checkUint64("rtp.estimatePriming2", uint64(EstimateSrtpRtpWireBytes(OpusPrimingFrame2[:])), k.Rtp.EstimatePriming2)

	// rtcp: compact PT 208/209 + sender report with NTP timestamp.
	compact208 := BuildCompactRTCP208(k.Inputs.SSRC, uint32(k.Rtcp.RemoteSsrc))
	check("rtcp.compact208", compact208[:], k.Rtcp.Compact208)
	compact209 := BuildCompactRTCP209(k.Inputs.SSRC)
	check("rtcp.compact209", compact209[:], k.Rtcp.Compact209)
	senderReport := BuildSenderReport(k.Inputs.SSRC, RTCPSenderStats{
		PacketsSent:  uint32(k.Rtcp.Stats.PacketsSent),
		OctetsSent:   uint32(k.Rtcp.Stats.OctetsSent),
		RTPTimestamp: uint32(k.Rtcp.Stats.RTPTimestamp),
	}, k.Rtcp.NowMs)
	check("rtcp.senderReport", senderReport[:], k.Rtcp.SenderReport)

	// stun: CRC-32, attribute TLV, XOR endpoint, native sender subscription,
	// MI + FINGERPRINT requests, the pinned WASM allocate, and the ping.
	relayToken, err := hex.DecodeString(k.Stun.RelayToken)
	if err != nil {
		return fmt.Errorf("stun.relayToken: %w", err)
	}
	miKey, err := hex.DecodeString(k.Stun.MiKey)
	if err != nil {
		return fmt.Errorf("stun.miKey: %w", err)
	}
	checkUint64("stun.crc32_abc", uint64(crc32.ChecksumIEEE([]byte("abc"))), k.Stun.Crc32Abc)
	tokenAttr := stunAttr(stunAttrRelayToken, relayToken)
	check("stun.attr_token", tokenAttr, k.Stun.AttrToken)
	endpointXOR, ok := EncodeXorRelayEndpoint("157.240.226.133", 3478)
	if !ok {
		return errors.New("stun: xor endpoint encode failed")
	}
	check("stun.xorEndpoint", endpointXOR[:], k.Stun.XorEndpoint)
	nativeSenderSub := CreateNativeSenderSubscription(k.Inputs.SSRC)
	check("stun.nativeSenderSub", nativeSenderSub[:], k.Stun.NativeSenderSub)
	check("stun.minimalMi", EncodeStunRequest(MsgAllocateRequest, tx12, tokenAttr, miKey, false), k.Stun.MinimalMi)
	check("stun.withFp", EncodeStunRequest(MsgAllocateRequest, tx12, tokenAttr, miKey, true), k.Stun.WithFp)
	// The pinned wasmAllocate carries the captured WASM template descriptor
	// set (the vector predates per-call SSRC derivation in the reference).
	check("stun.wasmAllocate",
		buildWasmStunAllocateRequestWithDescriptors(tx12, relayToken, endpointXOR, miKey, wasmStreamDescriptorsTemplate),
		k.Stun.WasmAllocate)
	ping := BuildWhatsAppPing(tx12)
	check("stun.ping", ping[:], k.Stun.Ping)

	// stun_proto: the protobuf subscription bodies.
	check("stun_proto.voip_sender_subscriptions", CreateVoipSenderSubscriptions(k.Inputs.SSRC), k.StunProto["voip_sender_subscriptions"])
	check("stun_proto.apk_sender_subscriptions_nopid", CreateApkSenderSubscriptions(k.Inputs.SSRC, nil), k.StunProto["apk_sender_subscriptions_nopid"])
	pid7 := uint32(7)
	check("stun_proto.apk_sender_subscriptions_pid", CreateApkSenderSubscriptions(k.Inputs.SSRC, &pid7), k.StunProto["apk_sender_subscriptions_pid"])
	check("stun_proto.apk_stream_descriptors", CreateApkStreamDescriptors(k.Inputs.SSRC), k.StunProto["apk_stream_descriptors"])

	// hbh_srtp.srtcpProtectedSrIndex1: SRTCP-protect the pinned sender report
	// with the uplink SRTCP session keys (libsrtp-pinned vector).
	srtcpKeys := E2ESrtpKeys{
		CipherKey: srtcpSession.SessionKey,
		Salt:      srtcpSession.SessionSalt,
		AuthKey:   srtcpSession.AuthKey,
	}
	plainSR, err := hex.DecodeString(k.Rtcp.SenderReport)
	if err != nil {
		return fmt.Errorf("rtcp.senderReport: %w", err)
	}
	protected, err := ProtectSRTCP(&srtcpKeys, 0x12345678, 1, plainSR)
	if err != nil {
		return fmt.Errorf("hbh srtcp protect: %w", err)
	}
	check("hbh_srtp.srtcpProtectedSrIndex1", protected, k.HbhSrtp["srtcpProtectedSrIndex1"])

	if len(failures) > 0 {
		return errors.New(strings.Join(failures, "\n"))
	}
	return nil
}

func counterToIVBytes(counter uint64) []byte {
	var iv [16]byte
	iv[8] = byte(counter)
	iv[9] = byte(counter >> 8)
	iv[10] = byte(counter >> 16)
	iv[11] = byte(counter >> 24)
	iv[12] = byte(counter >> 32)
	iv[13] = byte(counter >> 40)
	iv[14] = byte(counter >> 48)
	iv[15] = byte(counter >> 56)
	return iv[:]
}

func hexEqual(got []byte, wantHex string) bool {
	want, err := hex.DecodeString(wantHex)
	return err == nil && string(got) == string(want)
}
