package wavoip

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
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

type katFile struct {
	Inputs     KATInputs         `json:"inputs"`
	SFrame     map[string]string `json:"sframe"`
	E2ESrtp    map[string]string `json:"e2e_srtp"`
	Wahkdf     map[string]string `json:"wahkdf"`
	HbhSrtp    map[string]string `json:"hbh_srtp"`
	VoipCrypto map[string]uint64 `json:"voip_crypto"`
	// rtp/rtcp/stun sections pin the framing builders; they are verified by
	// the packetizer work, not by this crypto-only verifier.
}

// VerifyKats checks the crypto sections of vectors/voip/kats.json against this
// implementation and returns every failure (nil = all pass). The rtp/rtcp/stun
// sections are framing (packetizer) scope and intentionally skipped here.
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

	// The SRTCP-protected SR vector (srtcpProtectedSrIndex1) is verified by the
	// framing work — it needs the RTCP sender-report builder to produce its
	// plaintext input. Every crypto primitive it exercises is covered above.

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
