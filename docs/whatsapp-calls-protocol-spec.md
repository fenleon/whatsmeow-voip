# WhatsApp 1:1 Call Protocol Specification

**Status:** draft — reverse-engineered, unofficial. **Date:** 2026-10-12.
**Primary source:** [whatsapp-rust](https://github.com/oxidezap/whatsapp-rust) v0.7 `voip` (MIT) and its `wacore` crates, cross-validated on device against official WhatsApp clients (Light Phone III tool, 2026-09/10). Identifiers are quoted verbatim from source. Provenance tags: **[S]** = derived from source code, **[V]** = validated live against official clients.

> **Disclaimer.** This documents an undocumented, proprietary protocol for
> interoperability research. Unofficial clients violate WhatsApp's Terms of
> Service; accounts used with them risk suspension. Use a secondary number.
> Nothing here is endorsed by or affiliated with Meta/WhatsApp.

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [Signaling transport](#2-signaling-transport)
3. [Call lifecycle](#3-call-lifecycle)
4. [Offer contents & capabilities](#4-offer-contents--capabilities)
5. [Relay & media transport](#5-relay--media-transport)
6. [Audio framing](#6-audio-framing)
7. [The MLow codec](#7-the-mlow-codec)
8. [RTP/RTCP details](#8-rtprtcp-details)
9. [Media cryptography](#9-media-cryptography)
10. [Video](#10-video)
11. [Call log sync](#11-call-log-sync)
12. [Test vectors](#12-test-vectors)
13. [Open questions](#13-open-questions)
14. [Sources](#14-sources)

---

## 1. Architecture overview

A WhatsApp 1:1 call has three planes:

```
           signaling (same WebSocket as messaging)
  caller ──────────────────────────────────────────────── callee
     │          <call> stanzas: offer/accept/terminate …     │
     │                                                        │
     └────────────── media ───────────────────────────────────┘
            via one WhatsApp relay (TURN-like), no P2P:
            STUN allocate → DTLS/SCTP DataChannel → RTP/SRTP
```

- **Signaling** rides the ordinary binary WebSocket protocol (Noise-encrypted
  transport) as `call` stanzas. Call signaling nodes themselves are *plaintext
  application-layer nodes* — visible to the server — except the per-call
  **callKey**, which is Signal-encrypted end-to-end. **[S]**
- **Media** always flows through a WhatsApp-provided relay; this
  implementation contains no direct-P2P path (a `p2p-cand-round` signaling
  attribute exists but is unused). **[S]**
- **Media keys** derive end-to-end from the callKey, opaque to the server;
  the relay adds an optional hop-by-hop SRTP layer. **[S]**

## 2. Signaling transport

**[S]** Call signaling is ordinary WhatsApp binary-protocol stanzas with root
tag `call`, carried on the same WebSocket as messaging (Noise transport
encryption only — *not* wrapped in a Signal-protocol message).

Wrapper:

```xml
<call to=peer-jid id=wrapper-id>  <!-- receive side adds from, t, notify,
                                     platform, version, recipient, offline -->
  <offer|accept|reject|terminate|transport|video|relaylatency|heartbeat .../>
</call>
```

- Every action carries `call-id` and `call-creator` attributes.
- Group calls address the wrapper `to={call_id}@call`; 1:1 calls address the
  peer JID.
- The caller's relay data is returned by the server in an
  `<ack class="call" type="offer">` reply correlated by the offer wrapper's
  stanza `id`. The callee acknowledges the ring with
  `<receipt><offer call-id call-creator/></receipt>`.
- Typed in-call acks use `<ack class="call" type="video">` etc.
- **Stanza child order is load-bearing** on `<offer>`: the server rejects a
  wrongly ordered offer (nack 439).
- Wrapper `id` is mandatory on `offer` and `accept`; an idless accept is
  silently dropped by the server.

Action tags: `offer`, `offer_notice`, `preaccept`, `accept`, `reject`,
`terminate`, `transport`, `relaylatency`, `video`, `group_update`,
`enc_rekey`, `waiting_room_update`, `user_action`, `screen_share`.

## 3. Call lifecycle

**[S]** Local state machine: `Idle → Calling → Ringing → Connecting → Active
→ Ended` (a `WaitingRoom` phase exists for group calls).

| Step | Sender | Node | Notes |
|---|---|---|---|
| place call | caller | `<offer call-id call-creator>` + children (§4) | relay comes back in the server's `<ack type=offer>` |
| start ringing | callee | `<preaccept>` then audio/video `<encopt keygen="2">`, `<capability>` | preaccept acks the ring |
| answer | callee | `<accept>` (wrapper `id` required) | may carry a `<te>`; media connect follows |
| ICE exchange | both | `<transport p2p-cand-round …>` with `<te priority>` + `<net medium="2" protocol="0">` | `transport-message-type`: 1 = relay candidate, 3 = peer ICE, 9 = reply/keepalive |
| latency probes | both | `<relaylatency><te latency relay_name/></relaylatency>` | wire latency value = `0x02000000 + rtt_ms` |
| keepalive | both | `<heartbeat call-id call-creator/>` to `{call_id}@call` | 1 s cadence during active call |
| hang up | either | `<terminate reason=…>` | see below |
| decline | callee | `<reject call-id call-creator count="0">` | `reason="busy"` = that device can't take it; peer keeps ringing. No `reason` = explicit decline |

**Terminate reasons:** `accepted_elsewhere`, `rejected_elsewhere`, `timeout`,
`group_call_ended`, or absent. Receive mapping: `timeout` / `group_call_ended`
/ absent → **missed**; `accepted_elsewhere` / `rejected_elsewhere` → the call
was answered/declined on another of *your* devices (the caller dismisses
siblings — one terminate per device JID, no fan-out attribute).

**Edge cases.** Offers replayed from offline queue (wrapper attr `offline`)
are never rung — they surface as missed calls. `call-id` format:
`"00" + hex(15 random bytes)` (32 lowercase hex chars). The caller generates a
random 32-byte **callKey** per call and Signal-encrypts it per callee device
inside the offer's `<enc type="pkmsg"|"msg" v="2" count="0">` — it is the root
of all media keying (§9) and is opaque to the server.

## 4. Offer contents & capabilities

**[S]** `<offer call-id call-creator [caller_pn]>` children, **in this order**
(order is server-enforced):

1. `<privacy>` — the caller's tc token bytes
2. `<audio enc="opus" rate="8000|16000">` — signaling rates from
   `DEFAULT_AUDIO_RATES = ["8000", "16000"]`
3. optional `<video enc="h264" dec="h264" orientation="0"
   screen_width="1920" screen_height="1080" device_orientation="0">`
4. `<net medium="3">`
5. `<capability ver="1">` — 7-byte capability blob (below)
6. callKey `<enc>` (bare = single device) or
   `<destination><to jid=…><enc/></to></destination>` (multi-device)
7. `<encopt keygen="2">`
8. `<device-identity>` — the caller's Signal identity blob

**Capability blobs** (7 bytes; they gate codec selection — there is no SDP
anywhere in the protocol):

| blob | meaning |
|---|---|
| `01 05 f7 09 e0 bb 13` | audio offer, MLow |
| `01 05 f7 09 e0 fa 13` | video offer (`0xfa` at byte 5 = video) |
| preaccept audio variant ends `0x07` | |
| `…bb` → `…7f` variants | standard Opus instead of MLow (clearing bit 0x80 of byte 5 = capability bit 31) |

The peer's blob byte 5 high bit selects `use_mlow_codec_v1` vs standard Opus.
`<accept>` additionally carries `voip_settings` (advertises
`rtcp_interval_ms=1500`; `standard_opus_voip_settings` for a 48 kHz Opus
clock) and `audio_rates`. No SDP is exchanged at any point; SDP-like
`ice-ufrag`/`ice-pwd` are synthesized locally from relay material (§5).

**Fingerprinting note [V]:** offers carrying a captured-WA-Web-shaped
`caller_pn` are silently dropped by the server (no ack) — the fork sends
`caller_pn: None`. Client fingerprinting gates some signaling shapes.

## 5. Relay & media transport

**[S]** Media path is **relay-only**. The relay block arrives either inside
the incoming `<call>` subtree (callee) or in the caller's `<ack type=offer>`:

- `<key>` — base64, 16 bytes; its ASCII form is the STUN MESSAGE-INTEGRITY
  key for allocation
- `<hbh_key>` — 30 bytes = 14 B salt seed + 16 B key seed, for hop-by-hop SRTP
  (§9.3); the accept ack can carry a `<relay>` patch merging it
- `<warp_mi_tag_len>` — default 4 (§9.2)
- `<token id=N>` / `<auth_token id=N>` — indexed relay/auth token arrays
  (index cap 64)
- `<te2>` — relay endpoints; content = raw address bytes (6 B = IPv4+port
  big-endian, 18 B = IPv6+port); attrs `relay_id relay_name token_id
  auth_token_id is_fna protocol c2r_rtt`
- Relay node attrs: `uuid self_pid peer_pid`. Default relay port **3478**;
  web clients use **3480** — relays answering only on 3478 complete the
  handshake but never forward the peer's stream. `is_fna=1, auth_token_id=0`
  relays are inbound-only.

**Transport stack:** the client performs a STUN allocate against one relay
endpoint, then runs all media (STUN, RTP, RTCP) as binary messages inside a
**pre-negotiated WebRTC DataChannel**: SCTP (port 5000) over DTLS (client
role, self-signed cert, server cert *not* verified — SRTP keys come from
callKey, not DTLS) over UDP. DataChannel label `"pre-negotiated"`, id 0.
Synthetic SDP: `ice-ufrag` = base64(auth token), `ice-pwd` = base64(relay
key). STUN uses RFC 5389 framing (magic `0x2112a442`) with WhatsApp
extensions: message types ping `0x0801`/pong `0x0802` (1 s keepalive),
`RELAY_TOKEN 0x4000`, `STREAM_DESCRIPTORS 0x4024`,
`RECEIVER_SUBSCRIPTIONS 0x4021`, `PARTICIPANT_COUNT 0x805a`,
`WASM_RELAY_ENDPOINT 0x0016` (XOR-obfuscated endpoint), and a 9-slot stream
descriptor grid `[0,1,4,2,3,5,7,8,6]` = audio/video/video1 × media/FEC/NACK.
Audio SSRC = `HKDF-SHA256(salt=slot_word_LE32, ikm=call_id, info=lid, 4)` as
LE u32. Allocate deadline 10 s; failure is terminal
(`RelayAllocateFailed`/`RelayAllocateTimedOut`). **No TURN channels or
bandwidth allocation is used.**

## 6. Audio framing

**[S]** The native operating point (capability-gated MLow profile):

| field | value |
|---|---|
| channels / sample rate | mono, 16 000 Hz |
| frame | 960 samples = 60 ms |
| RTP clock rate | 16 000, timestamp step 960 |
| payload type | 120 (`WHATSAPP_PAYLOAD_TYPE_MLOW`) |

`AudioSource` carries **raw PCM: 960-sample mono i16 @ 16 kHz** — buffers of
any other length are dropped, no RTP is sent. Muted mic sends the one-byte DTX
token `0x90`. Other negotiated audio formats exist: standard Opus
(RFC 7587 PT 111, 48 kHz clock, ts step 2880), Opus-in-MLow-profile
(PT 120, 16 kHz clock), and the MLow *escape* (§7.3). Sequencer: audio `seq`
starts at 1, timestamp at 0, advancing 960/frame; the first speech frame after
DTX/priming latches the RTP **marker** (MLow profile). RTP header sizes:
audio speech 16 B, DTX 20 B, video 28 B; extension profile `0xdebe`; DTX
packets carry extension word `0x30010000` (the WARP piggyback, §9.2).

## 7. The MLow codec

### 7.1 What it is

**[S]** MLow is Meta's proprietary `smpl_audio_codec`: an **Opus-forked
split-band CELP codec**, the default for WhatsApp 1:1 calls. Decoder guard:
exactly 16 kHz wideband, `low_rate=0`, 60 ms — frames at other internal rates
are dropped (never mis-decoded; a range-coder desync would otherwise corrupt
the stream). One MLow frame = 3 chained 20 ms internal frames = 960 samples.
Encoder emits active `config=0` (`0x50`) frames; each internal frame is either
voiced (LTP pitch block) or unvoiced (gains block), never both.

### 7.2 Packet format

**[S]** Payload = **TOC byte + range-coded body**:

- TOC bit 7 = SID (DTX/CNG), bit 6 = VAD, bit 5 = internal rate (0 → 16k,
  1 → 32k), bits 4:3 = frame size index into {10, 20, 60, 120} ms.
- Speech frames: 20 ms TOC `0x48..0x4f`, 60 ms `0x50..0x57` (≥18 B payload).
- DTX/SID: `(b & 0xC0) == 0x80`; canonical SID token `0x90`.
- `(b & 0xC0) == 0xC0` → in-profile **standard Opus/CELT escape** (§7.3).
- Symbol order per internal frame: LSF → pulses → pitch (voiced) / gains
  (unvoiced). Optional **SplitRed** redundancy wrapper exists (PT 121) when
  `mlow_red_redundancy_level > 0`: `[red_hdr[i] 2B each][main_marker 1B]
  [red payloads][main payload]`.

### 7.3 The Opus escape

**[S]** CELT-only Opus packets ride inside the MLow RTP profile transcode-free
via a **TOC rewrite**: stock TOC `0xC0|config…` forms are rewritten to the
MLow TOC grammar (code-1/code-2 shorthand gets an explicit descriptor byte);
libopus DTX (≤2 B packets) and CELT hangover silence map to SID `0x90`. The
inverse restores the RFC TOC. Encoder settings for the escape: libopus
`Application::LowDelay`, wideband max (CELT-only), 24 kbps, complexity 5,
DTX on. Stock Opus *mis-decodes* raw MLow — both directions of the escape
exist precisely because of that.

### 7.4 Reference implementation & vectors

The MIT Rust implementation lives at `wacore/src/voip/mlow/` (~18k LOC,
30+ modules: analysis, `smpl_lpc`/`lsf_quant`, `smpl_pitch`, `smpl_pulse`,
`smpl_gains`, `smpl_celp`/`synth`, harmonic postfilters, VAD, range coder,
tables from `tables.proto`). Golden vectors and fixture provenance:
`wacore/src/voip/testdata/` + `mlow/testdata/PROVENANCE.md` (encoder/decoder
stage dumps, e2e cross-checks, all 256 TOC bytes, pinned checksums
`0x8782cb8cd7b5f262` decode / `0xb1793f1072d15a7d` bitstream). Export of
language-neutral vectors to this repo: **planned** (§12).

## 8. RTP/RTCP details

**[S]** Payload types: 111 Opus (RFC 7587), **120 WhatsApp audio/MLow**,
121 MLow-RED, **97 H.264**, 119 app-data. RTCP PTs: SR 200, RR 201, SDES 202,
RTPFB 205, PSFB 206, plus WhatsApp-compact **PT 208** (12 B: RC=1, local +
remote SSRC) and **PT 209** (8 B, pre-speech ladder). RTCP cadence 1500 ms;
periodic compound `SR + SDES`; SDES CNAME = 11 random hex nibbles around a
fixed `@pj…org` host. Reception reports extend the RFC 3550 24-byte block with
**24 extra bytes** of transport/bandwidth stats, signaled by bit 4 of the
count nibble.

Keyframe feedback: peer requests are PSFB **PLI (FMT 1)** matching the video
SSRC, or **FIR (FMT 4)** (target in the media field or 8-byte FCI rows).
Video RTP: clock 90 000, timestamp stride `90_000/fps` (6000 @ 15 fps, 4500
@ 20 fps); all packets of one access unit share a timestamp, which advances
only when the AU closes (marker). Video header extension block (12 B,
profile `0xdebe`): ID 3 `media_frame_info` (1 B; IDR = `0x09`, delta = `0x01`,
identical across fragments of an AU), ID 5 `initial_bandwidth` (2 B),
ID 6 `short_offset` (2 B), ID 9 `transport_sequence` (2 B).

## 9. Media cryptography

All media crypto derives from the per-call **callKey** (32 B, §3) — the
server never sees it.

### 9.1 E2E SRTP (primary path)

**[S]** Key schedule: `HKDF-SHA256(salt = 0^32, ikm = callKey, info =
participant_lid, 46)` → 30-byte master (16 B key + 14 B salt). The `info` is
the **sender's own** device-qualified participant LID; send keys use self
LID, receive keys use peer LID (SFrame uses the opposite orientation).
Session keys expand via the libsrtp **AES-CM KDF** (IV = 14 B master salt,
label XORed at byte 7): labels `0x00` cipher key (16 B), `0x01` auth key
(20 B, HMAC-SHA1), `0x02` salt (14 B). Payload: **AES-128-CTR** with IV =
salt right-aligned, SSRC XORed at bytes 4–7, 48-bit packet index
`(ROC << 16) | seq` XORed at bytes 8–13. Auth tag: **10 B** normally, **4 B**
on DTX/priming/short frames. SRTCP: labels `0x03/0x04/0x05`, RFC 3711 §3.4
layout. ROC tracking follows RFC 3711 (send-side counter; receive-side guess
index committed only after the MI tag verifies).

### 9.2 WARP MESSAGE-INTEGRITY tag

**[S]** Each packet carries a 4-byte tag (`warp_mi_tag_len`): first 4 bytes of
`HMAC-SHA1(auth_key, packet || roc_be32)`, keyed by the per-participant SRTP
auth key. On DTX/priming packets the tag piggybacks inside the RTP extension
word `0x30010000`, sent from packet #3 onward.

### 9.3 Hop-by-hop SRTP (client ↔ relay)

**[S]** Protects the client↔relay leg, keyed from the signaling
`<hbh_key>` (30 B = 14 B salt seed + 16 B key seed) via two-stage
HKDF-SHA256 with wire labels `"hbh srtp salt"/"key"`, `"hbh srtcp salt"/"key"`,
and directional `"uplink|downlink hbh srtcp salt"/"key"` variants (mode
selected by a WhatsApp transport rollout flag). Suite:
`AES_CM_128_HMAC_SHA1_80`, libsrtp AES-ICM with the 2-byte carry counter.

### 9.4 SFrame

**[S]** E2E frame encryption, **AES-128-GCM** with a non-standard 16-byte
nonce = 8 zero bytes ‖ counter as u64 **little-endian**. Key =
`HKDF-SHA256(salt = callKey[0..16], ikm = callKey[16..32], info =
"e2e sframe key" + participant_id, 32)`, per-participant per-direction. Wire
layout: `[ciphertext ‖ 16 B tag ‖ varint header]`; header = varint(counter) +
varint(key_id) + length byte, and is **not** GCM AAD. In 1:1 calls SFrame is
**receive-side only** in practice — official Android peers ship plain audio
inside E2E-SRTP, and decryption fails closed to plaintext passthrough. Group
calls use per-epoch group keys instead.

## 10. Video

### 10.1 Signaling — the upgrade dance

**[S]** Mid-call video uses `<video call-id call-creator state=N [dec=…]
[voip_settings="video"] [device_orientation=0..3]>`. States:

| N | meaning |
|---|---|
| 0 | disabled |
| 1 | enabled (camera on) |
| 2 | paused |
| 3 / 4 / 5 | upgrade request / accept / reject (legacy) |
| 6 | stopped (camera off, stays in video mode) |
| 7 / 8 / 9 | reject-by-timeout / cancel / cancel-by-timeout |
| 10 | unknown peer |
| 11 | **UpgradeRequestV2** (current; sends `voip_settings="video"`) |
| 20 | error |

Current flow: initiator sends `state=11 dec="H264"`, arms a 5 s timeout that
sends `state=9` on expiry; acceptor replies `state=4 dec="H264,AV1"` then
`state=1 dec="H264"`. Camera off = `state=6`; camera back on = standalone
`state=1 dec="H264" device_orientation=N`. `dec` lists codecs the *sender*
can decode. A declined upgrade is, in practice, mapped by clients to
`state=6` (the explicit `state=5` path is unreferenced in the Rust client).
Offers/accepts carry `<video enc="h264" dec="h264" …>` (§4).

### 10.2 Media

**[S][V]** Codec: H.264 (signaled `avc1.42E01F` — Constrained Baseline);
validated operating point: MediaCodec `video/avc`, Constrained Baseline
L3.1, no B-frames, one slice/frame; tiers 640×480@15fps ~600 kbps (default)
and 1280×720@20fps ~1.8 Mbps (WiFi). Packetization (RFC 6184): single-NAL
(≤800 B) or FU-A (798 B fragments); inbound single-NAL / STAP-A / FU-A; an AU
completes on marker or timestamp change (4 MiB reassembly cap). **AUD NALs
are omitted** because WhatsApp's decoder requires SPS/PPS to lead every IDR —
senders prepend CSD to each IDR, which the receiver packs into one STAP-A.
Decoder: hold AUs until the first keyframe; after relay reconnect, delta AUs
are dropped until a fresh keyframe. Orientation is the signaling
`device_orientation` value **masked `& 0x03`** (official clients send
unmasked values, e.g. wire 4 = 270°), applied as `rotationZ = 90° × value`.
Peer PLI/FIR → keyframe request surfaces on the sender's video-control
channel; local outbound overflow requests an IDR from the local encoder.

## 11. Call log sync

**[S]** Call history arrives via encrypted app-state sync: module
`WAWebCallLogSync`, collection `Regular` (version 1, account scope), action
`callLog`, index `["call_log", callId]`. The `CallLogRecord` protobuf carries:
`callResult` (CONNECTED, REJECTED, CANCELLED, ACCEPTEDELSEWHERE, MISSED,
INVALID, UNAVAILABLE, UPCOMING, FAILED, ABANDONED, ONGOING), `isDndMode`,
`silenceReason` (NONE/SCHEDULED/PRIVACY/LIGHTWEIGHT), `duration`, `startTime`,
`isIncoming`, `isVideo`, `isCallLink`, `callLinkToken`, `scheduledCallId`,
`callId`, `callCreatorJid`, `groupJid`, `participants[] {userJid, callResult}`,
`callType` (REGULAR / SCHEDULED_CALL / VOICE_CHAT).

## 12. Test vectors

Status: **planned** for this repo. The Rust reference currently pins:

- `wacore/src/voip/testdata/kats.json` — SRTP/HBH/SFrame known-answer tests
- `mlow/testdata/` — codec stage dumps (`fe_dump.json`, `lsf_quant_io.json`,
  `pitchio_ground_truth.json`, `sigmode_ground_truth.json`,
  `vad_ground_truth.json`, `gennoise_*`), decoder cross-checks
  (`lsf_vectors.json`, `pitch_vectors.json`, `pulse_vectors.json`,
  `gains_vectors.json`, `rc_vectors.json`, `toc_vectors.json` — all 256 TOC
  bytes), reference decode (`ref_usesmpl_expected.raw`, `e2e_vectors.json`),
  `inbound_capture_frames.json`, `synth_mic.raw` input
  (s16le/16 kHz/mono/110×960), with provenance in
  `mlow/testdata/PROVENANCE.md`
- Round-trip checksums: decode FNV-1a `0x8782cb8cd7b5f262`, bitstream
  `0xb1793f1072d15a7d`

Export format and language-neutral packaging TBD (first consumer: a Go port
for whatsmeow).

## 13. Open questions

- Official callee-side accept timeout (not in source; observed behavior TBD).
- `offer_notice` server fan-out policy beyond its attributes.
- Whether official clients ever originate PLI/FIR (the Rust client only
  parses them) and any direct-P2P ICE path (attribute exists, unused).
- Exact semantics/timing of `enc_rekey` and `user_action`.
- Group-call media (SFrame epoch keying, waiting rooms) — out of scope for
  this spec's first revision.

## 14. Sources

- [whatsapp-rust](https://github.com/oxidezap/whatsapp-rust) v0.7 + `wacore`
  (MIT) — primary; identifiers quoted verbatim. Key files:
  `wacore/src/stanza/call.rs`, `wacore/src/types/call.rs`,
  `wacore/src/voip/*` (relay_parse, rtp, rtcp, e2e_srtp, hbh_srtp, sframe,
  warp, stun, engine, h264, mlow/), `src/voip/facade.rs`,
  `src/voip/transport.rs`, `src/handlers/call.rs`, `src/features/call_log.rs`,
  `waproto/src/whatsapp.proto`.
- On-device validation against official clients (2026-09/10): capability
  gating / `caller_pn` fingerprinting, CSD-per-IDR, orientation masking,
  relay port 3480 requirement, video tiers.
- [whatspec](https://github.com/oxidezap/whatspec) — the ecosystem's
  language-neutral protocol IR (extracted from the WA Web bundle; supplies
  the `waproto` definitions). **whatspec covers iq/proto/appstate/enums/
  tokens/notif/wam — it has no voip domain.** This spec documents that
  missing surface; its vectors aim to follow whatspec's packaging
  conventions (versioned JSON + schema + manifest with provenance).
- Sibling work: [open-madrid](https://github.com/fenleon/open-madrid) (the
  iMessage equivalent) — same methodology: public spec + vectors, MIT.
