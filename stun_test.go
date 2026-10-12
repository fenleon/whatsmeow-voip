package wavoip

import (
	"encoding/hex"
	"hash/crc32"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func katTx12(t *testing.T, k *katFile) [12]byte {
	t.Helper()
	tx := mustHex(t, k.Stun.Tx)
	var tx12 [12]byte
	copy(tx12[:], tx)
	return tx12
}

func TestCrc32IsIEEE(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	assert.Equal(t, k.Stun.Crc32Abc, uint64(crc32.ChecksumIEEE([]byte("abc"))))
	assert.Equal(t, uint64(0x352441c2), uint64(crc32.ChecksumIEEE([]byte("abc"))))
}

func TestStunAttrAndEndpointMatchKAT(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	token := mustHex(t, k.Stun.RelayToken)

	t.Run("relay token attribute TLV", func(t *testing.T) {
		assert.Equal(t, k.Stun.AttrToken, hex.EncodeToString(stunAttr(stunAttrRelayToken, token)))
	})

	t.Run("xor relay endpoint", func(t *testing.T) {
		ep, ok := EncodeXorRelayEndpoint("157.240.226.133", 3478)
		require.True(t, ok)
		assert.Equal(t, k.Stun.XorEndpoint, hex.EncodeToString(ep[:]))
	})

	t.Run("native sender subscription", func(t *testing.T) {
		sub := CreateNativeSenderSubscription(k.Inputs.SSRC)
		assert.Equal(t, k.Stun.NativeSenderSub, hex.EncodeToString(sub[:]))
	})

	t.Run("malformed endpoints are rejected", func(t *testing.T) {
		for _, bad := range []string{"1.2.3", "1.2.3.4.5", "a.b.c.d", ""} {
			_, ok := EncodeXorRelayEndpoint(bad, 3478)
			assert.False(t, ok, "endpoint %q must be rejected", bad)
		}
	})
}

func TestEncodeStunRequestMiAndFingerprintMatchKAT(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	tx := katTx12(t, k)
	token := mustHex(t, k.Stun.RelayToken)
	miKey := mustHex(t, k.Stun.MiKey)
	attrs := stunAttr(stunAttrRelayToken, token)

	t.Run("message integrity only", func(t *testing.T) {
		assert.Equal(t, k.Stun.MinimalMi, hex.EncodeToString(EncodeStunRequest(MsgAllocateRequest, tx, attrs, miKey, false)))
	})

	t.Run("message integrity + fingerprint", func(t *testing.T) {
		assert.Equal(t, k.Stun.WithFp, hex.EncodeToString(EncodeStunRequest(MsgAllocateRequest, tx, attrs, miKey, true)))
	})
}

func TestWasmAllocateAndPingMatchKAT(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	tx := katTx12(t, k)
	token := mustHex(t, k.Stun.RelayToken)
	miKey := mustHex(t, k.Stun.MiKey)
	ep, ok := EncodeXorRelayEndpoint("157.240.226.133", 3478)
	require.True(t, ok)

	t.Run("allocate with the pinned descriptor template", func(t *testing.T) {
		alloc := buildWasmStunAllocateRequestWithDescriptors(tx, token, ep, miKey, wasmStreamDescriptorsTemplate)
		assert.Equal(t, k.Stun.WasmAllocate, hex.EncodeToString(alloc))
	})

	t.Run("dynamic descriptors carry the call's derived SSRCs", func(t *testing.T) {
		alloc, err := BuildWasmStunAllocateRequest(tx, token, ep, miKey, "CALL-ID-0001", "12345:0@lid")
		require.NoError(t, err)
		attrs := ParseStunAttributes(alloc)
		require.NotEmpty(t, attrs)
		assert.Equal(t, uint16(stunAttrRelayToken), attrs[0].AttrType)
		var descriptors []byte
		var mi *StunAttribute
		for i := range attrs {
			switch attrs[i].AttrType {
			case stunAttrRelayToken:
			case stunAttrStreamDescriptors:
				descriptors = attrs[i].Value
			case stunAttrMessageIntegrity:
				mi = &attrs[i]
			case stunAttrWasmRelayEndpoint:
			default:
				t.Fatalf("unexpected attribute %#x", attrs[i].AttrType)
			}
		}
		require.NotEmpty(t, descriptors)
		expected, err := BuildWasmStreamDescriptors("CALL-ID-0001", "12345:0@lid")
		require.NoError(t, err)
		assert.Equal(t, expected, descriptors)
		require.NotNil(t, mi)
		assert.Len(t, mi.Value, 20)
	})

	t.Run("ping", func(t *testing.T) {
		ping := BuildWhatsAppPing(tx)
		assert.Equal(t, k.Stun.Ping, hex.EncodeToString(ping[:]))
	})
}

func TestProtobufSubscriptionBodiesMatchKAT(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	pid := uint32(7)

	tests := []struct {
		name string
		got  []byte
		want string
	}{
		{"voip sender subscriptions", CreateVoipSenderSubscriptions(k.Inputs.SSRC), k.StunProto["voip_sender_subscriptions"]},
		{"apk sender subscriptions without pid", CreateApkSenderSubscriptions(k.Inputs.SSRC, nil), k.StunProto["apk_sender_subscriptions_nopid"]},
		{"apk sender subscriptions with pid", CreateApkSenderSubscriptions(k.Inputs.SSRC, &pid), k.StunProto["apk_sender_subscriptions_pid"]},
		{"apk stream descriptors", CreateApkStreamDescriptors(k.Inputs.SSRC), k.StunProto["apk_stream_descriptors"]},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, hex.EncodeToString(tc.got))
		})
	}
}

func TestAndroidAllocateCarriesThreeAttrs(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	tx := katTx12(t, k)
	token := mustHex(t, k.Stun.RelayToken)
	miKey := mustHex(t, k.Stun.MiKey)
	packet := BuildAndroidStunAllocateRequest(tx, token, k.Inputs.SSRC, nil, miKey, false)

	attrs := ParseStunAttributes(packet)
	require.Len(t, attrs, 4)
	assert.Equal(t, uint16(stunAttrRelayToken), attrs[0].AttrType)
	assert.Equal(t, uint16(stunAttrSenderSubscriptionsV2), attrs[1].AttrType)
	assert.Equal(t, uint16(stunAttrStreamDescriptors), attrs[2].AttrType)
	assert.Equal(t, uint16(stunAttrMessageIntegrity), attrs[3].AttrType)
	assert.Equal(t, CreateApkStreamDescriptors(k.Inputs.SSRC), attrs[2].Value)
}

func TestParseStunAttributes(t *testing.T) {
	data := loadKats(t)
	k := loadKatFile(t, data)
	minimal := mustHex(t, k.Stun.MinimalMi)

	t.Run("minimal allocate carries token + integrity", func(t *testing.T) {
		require.True(t, IsStunPacket(minimal))
		msgType, ok := StunMessageType(minimal)
		require.True(t, ok)
		assert.Equal(t, uint16(MsgAllocateRequest), msgType)
		attrs := ParseStunAttributes(minimal)
		require.Len(t, attrs, 2)
		assert.Equal(t, uint16(stunAttrRelayToken), attrs[0].AttrType)
		assert.Equal(t, mustHex(t, k.Stun.RelayToken), attrs[0].Value)
		assert.Equal(t, uint16(stunAttrMessageIntegrity), attrs[1].AttrType)
		assert.Len(t, attrs[1].Value, 20)
	})

	t.Run("truncated final padding does not over-read", func(t *testing.T) {
		p := make([]byte, 20)
		p[0] = 0x00
		p = append(p, 0x40, 0x00, 0x00, 0x01, 0xAB)
		attrs := ParseStunAttributes(p)
		require.Len(t, attrs, 1)
		assert.Equal(t, []byte{0xAB}, attrs[0].Value)
	})

	t.Run("non-stun and short packets", func(t *testing.T) {
		assert.False(t, IsStunPacket([]byte{0xc0, 0x00}))
		assert.Nil(t, ParseStunAttributes([]byte{0x00, 0x01}))
	})
}
