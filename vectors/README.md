# Test vectors

WhatsApp 1:1 call test vectors, exported **verbatim** from
[whatsapp-rust](https://github.com/oxidezap/whatsapp-rust) (MIT) — the only
open-source client with a working call media path. `manifest.json` carries the
sha256 of every file; verify after cloning:

```sh
sha256sum -c <(jq -r '.files[] | "\(.sha256)  vectors/\(.path)"' vectors/manifest.json)
```

Everything here is **synthetic** — no production captures, no real PII
(keys in `kats.json` are sequential dummies, JIDs are `111…/222…@lid`,
audio is generated). Full per-fixture provenance and regeneration recipes:
[`mlow/PROVENANCE.md`](mlow/PROVENANCE.md).

## Domains

### `voip/kats.json` — media crypto + framing KATs

One self-contained known-answer record covering §9 of the spec:

- `wahkdf` — callKey → 46-byte HKDF-SHA256 output → master key/salt split
- `e2e_srtp` — per-participant session keys, AES-128-CTR RTP IV construction,
  ciphertext, and the 4-byte WARP MESSAGE-INTEGRITY tag
- `hbh_srtp` — two-stage HKDF from the 30-byte signaling `hbh_key`
  (`"hbh srtp salt"/"key"` labels), session-key expansion, ICM nonce, and a
  full SRTCP-protected packet
- `sframe` — key derivation (`"e2e sframe key" + participant-id`), the
  little-endian counter IV, varint header, ciphertext+tag
- `rtp` — WhatsApp audio speech (16 B) / DTX (20 B) / video (28 B) header
  shapes incl. the `0xdebe` extension blocks and `media_frame_info` bytes
- `rtcp` — compact PT 208/209, sender report, reception-report extension
- `stun` / `stun_proto` — ALLOCATE with WhatsApp attributes
  (`RELAY_TOKEN 0x4000`, stream descriptors, XOR endpoint), ping `0x0801`,
  MESSAGE-INTEGRITY + FINGERPRINT, and the protobuf-fragment subscription
  bodies

A Go (or any) implementation reproducing this record byte-for-byte has the
entire crypto stack pinned.

### `mlow/` — the MLow codec verification pack

Two kinds of fixtures (full recipes in `PROVENANCE.md`):

1. **Bit-exact wire decode** — `inbound_capture_frames.json` (external
   encoder's actual wire frames, TOC `0x10`/`0x12`/`0x50`) + per-stage
   decoder cross-checks (`lsf_`, `pitch_`, `pulse_`, `gains_`, `rc_`,
   `toc_vectors.json` — the full 256-entry TOC table) +
   `ref_usesmpl_expected.raw` / `e2e_vectors.json` (reference PCM from
   libopus built with `useSmpl`). These are what a port must match to claim
   interoperability.
2. **Encoder/decoder stage dumps** — `fe_dump`, `lsf_quant_io`,
   `pitchio_ground_truth`, `sigmode_ground_truth`, `vad_ground_truth`,
   `gennoise_*`, postfilter `.raw` dumps: pin each internal stage against the
   `smpl` C reference, so a port that desyncs can bisect to the exact stage.

`*_seed.bin` are the codec's **runtime tables** (zlib+protobuf), not test
vectors — a port needs them as data. Pinned round-trip checksums (FNV-1a):
decode `0x8782cb8cd7b5f262`, bitstream `0xb1793f1072d15a7d`.

## Regenerating from the Rust reference

From the whatsapp-rust repo root (deterministic; see `mlow/README.md` and
`PROVENANCE.md`):

```sh
VOIP_GOLDEN=1   cargo test -p wacore --features voip   # codec goldens
MLOW_GEN_SYNTH=1 cargo test -p wacore --features voip regen_synth_mic_raw
VOIP_GEN_TABLES=1 cargo test -p wacore --features voip gen_runtime_tables
```

## License

Exported from whatsapp-rust under the MIT license; the vectors keep that
license. The rest of this repo is MPL-2.0.
