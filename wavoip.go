// Package wavoip implements the WhatsApp 1:1 call media-crypto stack:
// key derivation from the per-call callKey, E2E SRTP, hop-by-hop SRTP, and
// SFrame. Every primitive is pinned against the known-answer vectors in
// vectors/voip/kats.json (see cmd/verify-kats).
//
// Reference: docs/whatsapp-calls-protocol-spec.md §9; Rust reference:
// whatsapp-rust wacore/src/voip/{e2e_srtp,hbh_srtp,sframe,warp,ssrc}.rs.
package wavoip

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/hkdf"
	"crypto/hmac"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
)

const (
	// WarpMITagLen is the default WARP MESSAGE-INTEGRITY tag length (1..=20 legal).
	WarpMITagLen = 4
	// HbhKeyLen is the exact length of the signaling <hbh_key>: 14 B salt seed + 16 B key seed.
	HbhKeyLen = 30
	// SrtcpTagLen is the HMAC-SHA1-80 SRTCP auth tag length.
	SrtcpTagLen = 10
)

var (
	// ErrShortCallKey: a malformed peer callKey (>= 32 bytes for E2E SRTP, exactly 32 for SFrame).
	ErrShortCallKey = errors.New("callKey shorter than 32 bytes")
	// ErrBadHbhKey: the only valid hbh_key is exactly 30 bytes.
	ErrBadHbhKey = errors.New("hbh_key must be exactly 30 bytes")
)

// HKDF-SHA256 with explicit salt/ikm/info (the Rust reference's hkdf_sha256 helper).
func hkdfSHA256(salt, ikm, info []byte, length int) ([]byte, error) {
	return hkdf.Key(sha256.New, ikm, salt, string(info), length)
}

// E2ESrtpKeys are the expanded per-participant session keys.
type E2ESrtpKeys struct {
	CipherKey [16]byte
	Salt      [14]byte
	AuthKey   [20]byte
}

// AES-CM PRF (libsrtp KDF): IV = 14-byte master salt zero-padded to 16 with
// label XORed into byte 7, keystream via AES-128-CTR (full 128-bit counter).
func aesCMKDF(masterKey, masterSalt []byte, label byte, length int) ([]byte, error) {
	iv := make([]byte, 16)
	copy(iv, masterSalt)
	iv[7] ^= label
	block, err := aes.NewCipher(masterKey)
	if err != nil {
		return nil, fmt.Errorf("aes-cm kdf: %w", err)
	}
	out := make([]byte, length)
	cipher.NewCTR(block, iv).XORKeyStream(out, out)
	return out, nil
}

// DeriveE2ESrtpKeys derives the 1:1 SRTP session keys from the callKey and the
// SENDER's own participant LID (send keys from self LID, recv keys from peer
// LID; SFrame uses the opposite convention). HKDF-SHA256(salt=0^32,
// ikm=callKey[:32], info=lid, 46) -> 16 B master key + 14 B master salt, then
// the libsrtp AES-CM KDF with labels 0x00 cipher / 0x01 auth / 0x02 salt.
func DeriveE2ESrtpKeys(callKey []byte, participantLID string) (*E2ESrtpKeys, error) {
	return deriveE2EKeys(callKey, participantLID, 0x00, 0x01, 0x02)
}

// DeriveE2ESrtcpKeys is the SRTCP sibling: same master, RTCP labels 0x03/0x04/0x05.
func DeriveE2ESrtcpKeys(callKey []byte, participantLID string) (*E2ESrtpKeys, error) {
	return deriveE2EKeys(callKey, participantLID, 0x03, 0x04, 0x05)
}

func deriveE2EKeys(callKey []byte, participantLID string, encLabel, authLabel, saltLabel byte) (*E2ESrtpKeys, error) {
	if len(callKey) < 32 {
		return nil, ErrShortCallKey
	}
	master, err := hkdfSHA256(make([]byte, 32), callKey[:32], []byte(participantLID), 46)
	if err != nil {
		return nil, fmt.Errorf("derive e2e keys: %w", err)
	}
	masterKey, masterSalt := master[:16], master[16:30]
	var keys E2ESrtpKeys
	enc, err := aesCMKDF(masterKey, masterSalt, encLabel, 16)
	if err != nil {
		return nil, err
	}
	auth, err := aesCMKDF(masterKey, masterSalt, authLabel, 20)
	if err != nil {
		return nil, err
	}
	salt, err := aesCMKDF(masterKey, masterSalt, saltLabel, 14)
	if err != nil {
		return nil, err
	}
	copy(keys.CipherKey[:], enc)
	copy(keys.AuthKey[:], auth)
	copy(keys.Salt[:], salt)
	return &keys, nil
}

// BuildE2ERtpIV: salt right-aligned into 16 bytes, SSRC XORed at bytes 4-7,
// 48-bit packet index (roc<<16|seq) XORed at bytes 8-13.
func BuildE2ERtpIV(salt []byte, ssrc uint32, roc uint32, seq uint16) [16]byte {
	var iv [16]byte
	n := min(len(salt), 14)
	copy(iv[14-n:], salt[:n])
	iv[4] ^= byte(ssrc >> 24)
	iv[5] ^= byte(ssrc >> 16)
	iv[6] ^= byte(ssrc >> 8)
	iv[7] ^= byte(ssrc)
	packetIndex := uint64(roc)<<16 | uint64(seq)
	for i := 0; i < 6; i++ {
		iv[8+i] ^= byte(packetIndex >> (40 - 8*i))
	}
	return iv
}

// CryptE2EPayload is the symmetric AES-128-CTR of an RTP payload under the
// packet's IV (encrypt and decrypt are the same operation).
func CryptE2EPayload(keys *E2ESrtpKeys, ssrc uint32, seq uint16, roc uint32, payload []byte) ([]byte, error) {
	iv := BuildE2ERtpIV(keys.Salt[:], ssrc, roc, seq)
	block, err := aes.NewCipher(keys.CipherKey[:])
	if err != nil {
		return nil, fmt.Errorf("e2e payload: %w", err)
	}
	out := make([]byte, len(payload))
	cipher.NewCTR(block, iv[:]).XORKeyStream(out, payload)
	return out, nil
}

// ComputeWarpMITag: first tagLen bytes of HMAC-SHA1(authKey, packet || roc_be32),
// keyed by the per-participant SRTP auth key (not a separate callKey-derived key).
func ComputeWarpMITag(authKey, packet []byte, roc uint32, tagLen int) []byte {
	mac := hmac.New(sha1.New, authKey)
	mac.Write(packet)
	mac.Write(binary.BigEndian.AppendUint32(nil, roc))
	return mac.Sum(nil)[:tagLen]
}

// ProtectSRTCP (RFC 3711 §3.4): AES-CTR the body (first 8 bytes stay clear),
// append the E-flag+31-bit index word, then HMAC-SHA1-80 over the whole packet.
func ProtectSRTCP(keys *E2ESrtpKeys, senderSSRC uint32, index uint32, rtcp []byte) ([]byte, error) {
	const headerLen = 8
	split := min(len(rtcp), headerLen)
	iv := BuildE2ERtpIV(keys.Salt[:], senderSSRC, index>>16, uint16(index))
	block, err := aes.NewCipher(keys.CipherKey[:])
	if err != nil {
		return nil, fmt.Errorf("srtcp protect: %w", err)
	}
	body := make([]byte, len(rtcp)-split)
	cipher.NewCTR(block, iv[:]).XORKeyStream(body, rtcp[split:])
	out := make([]byte, 0, len(rtcp)+4+SrtcpTagLen)
	out = append(out, rtcp[:split]...)
	out = append(out, body...)
	out = binary.BigEndian.AppendUint32(out, 0x8000_0000|(index&0x7fff_ffff))
	mac := hmac.New(sha1.New, keys.AuthKey[:])
	mac.Write(out)
	return mac.Sum(out), nil
}

// HbhKeying is the 16 B master key + 14 B master salt split of a 30-byte crypto key.
type HbhKeying struct {
	MasterKey  [16]byte
	MasterSalt [14]byte
}

// LibsrtpSessionKeys are the expanded AES_CM_128_HMAC_SHA1_80 session keys.
type LibsrtpSessionKeys struct {
	SessionKey  [16]byte
	SessionSalt [14]byte
	AuthKey     [20]byte
}

// HBH wire labels (KAT-pinned).
const (
	LabelUplinkSrtcpSalt   = "uplink hbh srtcp salt"
	LabelUplinkSrtcpKey    = "uplink hbh srtcp key"
	LabelDownlinkSrtcpSalt = "downlink hbh srtcp salt"
	LabelDownlinkSrtcpKey  = "downlink hbh srtcp key"
	LabelSharedSrtcpSalt   = "hbh srtcp salt"
	LabelSharedSrtcpKey    = "hbh srtcp key"
	LabelSharedSrtpSalt    = "hbh srtp salt"
	LabelSharedSrtpKey     = "hbh srtp key"
)

// Two-stage SFU HKDF: salt = HKDF(null-salt, saltSeed, saltLabel, 32),
// cryptoKey = HKDF(salt, keySeed, keyLabel, 30).
func deriveHbhKeyWithLabels(hbhKey []byte, saltLabel, keyLabel string) ([]byte, error) {
	if len(hbhKey) != HbhKeyLen {
		return nil, ErrBadHbhKey
	}
	saltSeed, keySeed := hbhKey[:14], hbhKey[14:]
	salt, err := hkdfSHA256(make([]byte, 32), saltSeed, []byte(saltLabel), 32)
	if err != nil {
		return nil, fmt.Errorf("hbh salt: %w", err)
	}
	key, err := hkdfSHA256(salt, keySeed, []byte(keyLabel), HbhKeyLen)
	if err != nil {
		return nil, fmt.Errorf("hbh key: %w", err)
	}
	return key, nil
}

func hbhKeying(hbhKey []byte, saltLabel, keyLabel string) (*HbhKeying, error) {
	key, err := deriveHbhKeyWithLabels(hbhKey, saltLabel, keyLabel)
	if err != nil {
		return nil, err
	}
	var k HbhKeying
	copy(k.MasterKey[:], key[:16])
	copy(k.MasterSalt[:], key[16:30])
	return &k, nil
}

// DeriveHbhSrtpKeyingShared is the SRTP (media) keying, shared mode.
func DeriveHbhSrtpKeyingShared(hbhKey []byte) (*HbhKeying, error) {
	return hbhKeying(hbhKey, LabelSharedSrtpSalt, LabelSharedSrtpKey)
}

// DeriveHbhSrtcpKeyingUplink is the uplink SRTCP keying, directional mode.
func DeriveHbhSrtcpKeyingUplink(hbhKey []byte) (*HbhKeying, error) {
	return hbhKeying(hbhKey, LabelUplinkSrtcpSalt, LabelUplinkSrtcpKey)
}

// DeriveHbhSrtcpKeyingDownlink is the downlink SRTCP keying, directional mode.
func DeriveHbhSrtcpKeyingDownlink(hbhKey []byte) (*HbhKeying, error) {
	return hbhKeying(hbhKey, LabelDownlinkSrtcpSalt, LabelDownlinkSrtcpKey)
}

// DeriveHbhSrtcpKeyingShared is the shared-mode SRTCP keying.
func DeriveHbhSrtcpKeyingShared(hbhKey []byte) (*HbhKeying, error) {
	return hbhKeying(hbhKey, LabelSharedSrtcpSalt, LabelSharedSrtcpKey)
}

// hbhAESICM: libsrtp AES-ICM — counter = (salt zero-padded to 16) XOR iv,
// keystream = AES(counter), counter increments byte 15 with a single carry
// into byte 14 (NOT a full 128-bit counter).
func hbhAESICM(aesKey [16]byte, salt [14]byte, iv [16]byte, data []byte) ([]byte, error) {
	block, err := aes.NewCipher(aesKey[:])
	if err != nil {
		return nil, fmt.Errorf("aes-icm: %w", err)
	}
	counter := [16]byte{}
	copy(counter[:14], salt[:])
	for i := range counter {
		counter[i] ^= iv[i]
	}
	out := make([]byte, len(data))
	for pos := 0; pos < len(data); pos += 16 {
		ks := [16]byte{}
		block.Encrypt(ks[:], counter[:])
		n := min(16, len(data)-pos)
		for i := 0; i < n; i++ {
			out[pos+i] = data[pos+i] ^ ks[i]
		}
		counter[15]++
		if counter[15] == 0 {
			counter[14]++
		}
	}
	return out, nil
}

func (k *HbhKeying) combined30() [30]byte {
	var full [30]byte
	copy(full[:16], k.MasterKey[:])
	copy(full[16:], k.MasterSalt[:])
	return full
}

// ExpandLibsrtpSessionKeys expands the session keys (labels 0x00 enc, 0x01
// auth, 0x02 salt) through the libsrtp KDF: IV all-zero except byte 7 = label,
// keystream via AES-ICM keyed by the 30-byte (key||salt) material.
func ExpandLibsrtpSessionKeys(k *HbhKeying) (*LibsrtpSessionKeys, error) {
	return expandSessionKeys(k, 0x00, 0x01, 0x02)
}

// ExpandLibsrtcpSessionKeys expands the RTCP-labelled session keys (0x03/0x04/0x05).
func ExpandLibsrtcpSessionKeys(k *HbhKeying) (*LibsrtpSessionKeys, error) {
	return expandSessionKeys(k, 0x03, 0x04, 0x05)
}

func expandSessionKeys(k *HbhKeying, encLabel, authLabel, saltLabel byte) (*LibsrtpSessionKeys, error) {
	full := k.combined30()
	derive := func(label byte, length int) ([20]byte, error) {
		var iv [16]byte
		iv[7] = label
		var zero [20]byte
		out, err := hbhAESICM([16]byte(full[:16]), [14]byte(full[16:30]), iv, make([]byte, length))
		if err != nil {
			return zero, err
		}
		var res [20]byte
		copy(res[:], out)
		return res, nil
	}
	enc, err := derive(encLabel, 16)
	if err != nil {
		return nil, err
	}
	auth, err := derive(authLabel, 20)
	if err != nil {
		return nil, err
	}
	salt, err := derive(saltLabel, 14)
	if err != nil {
		return nil, err
	}
	return &LibsrtpSessionKeys{
		SessionKey:  [16]byte(enc[:16]),
		AuthKey:     auth,
		SessionSalt: [14]byte(salt[:14]),
	}, nil
}

// BuildRtpICMNonce: zero nonce, SSRC big-endian at bytes 4-8,
// (packetIndex << 16) as a big-endian u64 at bytes 8-16.
func BuildRtpICMNonce(ssrc uint32, packetIndex uint64) [16]byte {
	var iv [16]byte
	binary.BigEndian.PutUint32(iv[4:8], ssrc)
	binary.BigEndian.PutUint64(iv[8:16], packetIndex<<16)
	return iv
}

// CryptHbhRtpPayload encrypts/decrypts an RTP payload under the expanded HBH
// session key with the libsrtp ICM (2-byte carry counter).
func CryptHbhRtpPayload(session *LibsrtpSessionKeys, ssrc uint32, packetIndex uint64, payload []byte) ([]byte, error) {
	nonce := BuildRtpICMNonce(ssrc, packetIndex)
	return hbhAESICM(session.SessionKey, session.SessionSalt, nonce, payload)
}

// DeriveWasmParticipantSSRC: HKDF-SHA256(salt=slotWord LE32, ikm=callID,
// info=lid, 4), read back as a little-endian uint32.
func DeriveWasmParticipantSSRC(callID, lid string, slotWord uint32) (uint32, error) {
	okm, err := hkdfSHA256(binary.LittleEndian.AppendUint32(nil, slotWord), []byte(callID), []byte(lid), 4)
	if err != nil {
		return 0, fmt.Errorf("ssrc: %w", err)
	}
	return binary.LittleEndian.Uint32(okm), nil
}
