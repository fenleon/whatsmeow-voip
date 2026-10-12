// SFrame end-to-end media encryption (AES-128-GCM) over the relay.
//
// Three byte-level traps, all KAT-pinned (see cmd/verify-kats):
//  1. GCM nonce = 8 zero bytes || counter as u64 LITTLE-endian (16-byte nonce,
//     GHASH-derived J0 — Go's crypto/cipher.NewGCMWithNonceSize handles it).
//  2. The SFrame header is appended AFTER the ciphertext+tag and is NOT AAD.
//  3. Wire layout is [ciphertext || 16-byte tag || varint header].
package wavoip

import (
	"crypto/aes"
	"crypto/cipher"
	"errors"
	"fmt"
)

// KDFLabelE2ESframe is the SFrame HKDF info prefix: "e2e sframe key" + participant id.
const KDFLabelE2ESframe = "e2e sframe key"

const (
	sframeKeyLen  = 16
	sframeTagLen  = 16
	sframeCallKey = 32
)

var errSframeCallKey = errors.New("sframe: callKey must be exactly 32 bytes")

// FormatParticipantID qualifies a participant JID for both E2E-SRTP and SFrame
// HKDF info: strip the resource, keep an existing ":N@lid" device suffix, give
// a bare "@lid" an implicit ":0", pass everything else through.
func FormatParticipantID(jid string) string {
	bare := jid
	for i := 0; i < len(bare); i++ {
		if bare[i] == '/' {
			bare = bare[:i]
			break
		}
	}
	at := -1
	for i := len(bare) - 1; i >= 0; i-- {
		if bare[i] == '@' {
			at = i
			break
		}
	}
	if at <= 0 {
		return bare
	}
	user, domain := bare[:at], bare[at+1:]
	if domain == "lid" && !containsColon(user) {
		return user + ":0@" + domain
	}
	return bare
}

func containsColon(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == ':' {
			return true
		}
	}
	return false
}

// DeriveE2ESframeKey derives the per-participant SFrame key:
// HKDF-SHA256(salt=callKey[0:16], ikm=callKey[16:32], info=label+participantID, 32).
func DeriveE2ESframeKey(callKey []byte, participantID string) ([]byte, error) {
	if len(callKey) != sframeCallKey {
		return nil, errSframeCallKey
	}
	return hkdfSHA256(callKey[:16], callKey[16:32],
		[]byte(KDFLabelE2ESframe+participantID), 32)
}

// CounterToIV is the non-standard 16-byte GCM nonce:
// 8 zero bytes || counter as u64 LITTLE-endian.
func CounterToIV(counter uint64) [16]byte {
	var iv [16]byte
	iv[8] = byte(counter)
	iv[9] = byte(counter >> 8)
	iv[10] = byte(counter >> 16)
	iv[11] = byte(counter >> 24)
	iv[12] = byte(counter >> 32)
	iv[13] = byte(counter >> 40)
	iv[14] = byte(counter >> 48)
	iv[15] = byte(counter >> 56)
	return iv
}

func appendVarint(out []byte, v uint64) []byte {
	for v > 0x7f {
		out = append(out, byte(v&0x7f)|0x80)
		v >>= 7
	}
	return append(out, byte(v))
}

// BuildSframeHeader: varint(counter) + varint(keyID) + total-length byte.
func BuildSframeHeader(counter, keyID uint64) []byte {
	header := appendVarint(appendVarint(nil, counter), keyID)
	return append(header, byte(len(header)+1))
}

// SframeSession holds the per-direction keys (encrypt for peer, decrypt for self).
type SframeSession struct {
	encryptKey [sframeKeyLen]byte
	decryptKey [sframeKeyLen]byte
	// TxCounter is the outbound frame counter (callers set/advance it).
	TxCounter uint64
}

// NewSframeSession builds a session from the callKey and self/peer JIDs.
// Send keys derive from the PEER's participant id, recv keys from SELF's
// (the opposite of E2E-SRTP's orientation).
func NewSframeSession(callKey []byte, selfJID, peerJID string) (*SframeSession, error) {
	selfID := FormatParticipantID(selfJID)
	peerID := FormatParticipantID(peerJID)
	sendKey, err := DeriveE2ESframeKey(callKey, peerID)
	if err != nil {
		return nil, err
	}
	recvKey, err := DeriveE2ESframeKey(callKey, selfID)
	if err != nil {
		return nil, err
	}
	s := &SframeSession{}
	copy(s.encryptKey[:], sendKey)
	copy(s.decryptKey[:], recvKey)
	return s, nil
}

func sframeAEAD(key [16]byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key[:])
	if err != nil {
		return nil, fmt.Errorf("sframe: %w", err)
	}
	return cipher.NewGCMWithNonceSize(block, 16)
}

// Encrypt one frame: [ciphertext || tag || varint-header] under the current
// TxCounter (which it consumes and increments).
func (s *SframeSession) Encrypt(plaintext []byte) ([]byte, error) {
	counter := s.TxCounter
	s.TxCounter++
	aead, err := sframeAEAD(s.encryptKey)
	if err != nil {
		return nil, err
	}
	iv := CounterToIV(counter)
	out := aead.Seal(nil, iv[:], plaintext, nil)
	return append(out, BuildSframeHeader(counter, 0)...), nil
}

// (SFrame recv outcome is expressed by Decrypt's ([]byte, bool) return — no
// result type needed.)

// Decrypt authenticates and opens one frame. ok=false means "not SFrame; the
// raw frame is the payload" — the peer ships plain Opus inside E2E-SRTP
// without SFrame-wrapping, so any failure (too short, header mismatch, bad
// GCM tag) returns the caller to the raw bytes rather than forging a plaintext.
func (s *SframeSession) Decrypt(frame []byte) ([]byte, bool) {
	if len(frame) < sframeTagLen+3 {
		return frame, false
	}
	headerLen := int(frame[len(frame)-1])
	if headerLen < 3 || headerLen > len(frame) {
		return frame, false
	}
	header := frame[len(frame)-headerLen:]
	ciphertext := frame[:len(frame)-headerLen]
	if len(ciphertext) < sframeTagLen+1 {
		return frame, false
	}
	counter, _, ok := parseSframeHeader(header)
	if !ok {
		return frame, false
	}
	aead, err := sframeAEAD(s.decryptKey)
	if err != nil {
		return frame, false
	}
	iv := CounterToIV(counter)
	plain, err := aead.Open(nil, iv[:], ciphertext, nil)
	if err != nil {
		return frame, false
	}
	return plain, true
}

func parseSframeHeader(header []byte) (counter, keyID uint64, ok bool) {
	if len(header) < 2 {
		return 0, 0, false
	}
	totalLen := int(header[len(header)-1])
	if totalLen != len(header) {
		return 0, 0, false
	}
	body := header[:len(header)-1]
	counter, n, ok := decodeVarint(body, 0)
	if !ok {
		return 0, 0, false
	}
	keyID, _, ok = decodeVarint(body, n)
	return counter, keyID, ok
}

func decodeVarint(data []byte, offset int) (uint64, int, bool) {
	var value uint64
	shift := uint(0)
	for i := offset; i < len(data); i++ {
		b := data[i]
		value |= uint64(b&0x7f) << shift
		if b&0x80 == 0 {
			return value, i + 1, true
		}
		shift += 7
		if shift > 63 {
			return 0, 0, false
		}
	}
	return 0, 0, false
}
