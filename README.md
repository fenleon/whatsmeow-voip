# whatsmeow-voip

**VoIP (calls) support for the WhatsApp Go ecosystem — [whatsmeow](https://github.com/tulir/whatsmeow) / mautrix.**

WhatsApp's 1:1 call media path has no open-source Go implementation. This repo
changes that, in stages, with the goal of landing the work upstream:

1. **Spec** — [`docs/whatsapp-calls-protocol-spec.md`](docs/whatsapp-calls-protocol-spec.md):
   the WhatsApp 1:1 call protocol (signaling, relay transport, MLow codec,
   RTP/RTCP, media crypto, video), reverse-engineered from
   [whatsapp-rust](https://github.com/oxidezap/whatsapp-rust) (MIT, the only
   open-source client with a working call media path) and validated on device
   against official clients.
2. **Test vectors** — golden vectors (MLow codec stages, SRTP key derivation,
   packet framing) exported from the Rust reference so the Go port can be
   verified byte-exact. *Status: planned.*
3. **Go implementation** — `whatsmeow-voip`: call signaling, then media
   (Pion for DTLS/SCTP/SRTP transport, MLow codec ported against the vectors).
   **Started:** the media-crypto stack (`wavoip` package — E2E SRTP, WARP MI
   tag, HBH SRTP, SFrame, SSRC derivation) is implemented and verified
   byte-exact against `vectors/voip/kats.json`:

   ```sh
   go run ./cmd/verify-kats   # verify-kats: all known-answer vectors pass
   go test ./...
   ```

   Contribution target: whatsmeow (signaling/media hooks) and mautrix-whatsapp
   (client surface).

## Upstream process

Contributions follow the [mautrix contributing guidelines](https://docs.mau.fi/bridges/general/contributing.html)
and the Beeper Go guidelines:

- Non-trivial plans are discussed in the whatsmeow/mautrix Matrix rooms
  **before** PRs open.
- PR descriptions, issues, and code comments are written by the (human)
  author; AI assistance is fine for code, but the author understands and
  stands behind every line.
- `pre-commit` + golangci-lint on every commit; latest stable Go.

## License

MPL-2.0, matching whatsmeow, so the code can move upstream without
relicensing.

## Related

- [whatsapp-rust](https://github.com/oxidezap/whatsapp-rust) — the MIT Rust
  reference implementation (`wacore/src/voip/` carries the KAT vectors this
  repo exports).
- Unofficial, community effort; unofficial clients violate WhatsApp's ToS.
