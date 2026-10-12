# MLow (smpl_audio_codec) Rust → Go port map

Reference: `calls/rust/whatsapp-rust/wacore/src/voip/mlow/` (~18.0k LOC, 30 modules).
Paths below are relative to that dir (`mlow/…:line`). Vectors: `whatsmeow-voip/vectors/mlow/`
(see its `PROVENANCE.md` — every fixture maps to a named Rust test).

## 1. Module map

Side: D = decoder-only, E = encoder-only, S = shared, T = test-only.

| module | LOC | role | key public items | state deps (mlow) | pinned by | side |
|---|---|---|---|---|---|---|
| `mod.rs` | 45 | crate root; re-exports `MlowDecoder`/`MlowEncoder` | — | all | — | S |
| `toc.rs` | 130 | smpl TOC byte parse | `parse_mlow_toc`, `MlowToc`, `standard_opus_frame_ms` | none | `toc_vectors.json` (exhaustive 256) | D |
| `red.rs` | 146 | SplitRED depacketization (outermost wire layer) | `depack_split_red`, `MlowFrame`, `RedError` | none | unit tests only | D |
| `rangecoder.rs` | 854 | Opus ec_dec/ec_enc (RFC 6716 §4.1), u32 wrapping | `RangeDecoder` (`err`, `decode_cdf`, `decode_cdf_le16`, `decode`, raw bits), `RangeEncoder` (`encode`, `encode_cdf`, `encode_cdf_window`, `encode_raw_symbol`, `done`) | none | `rc_vectors.json` | S |
| `smpl_decode.rs` | 189 | LSF block decode + runtime CDF container + cross-frame LSF/pitch predictor | `load_smpl_tables`, `SmplTables`, `SmplLsfState`, `smpl_advance_lsf_state`, `decode_smpl_lsf`, `SmplLsfIndices` | rangecoder, smpl_lsf_seed | `lsf_vectors.json` | S (state mirrored by encoder) |
| `smpl_lsf_seed.rs` | 758 | LSF seed ROM → expanded CDF tables (`SmplTables`) + synthesis tables (`LsfBuilt`) | `lsf_built()`, `LsfSeed::build()`, `seed_from_json` | smpl_tables_blob | `lsf_seed.bin` (runtime) | S |
| `smpl_lsf_quant.rs` | 639 | encoder LSF VQ (stage-1 shortlist + RD beam, per-coeff stage-2 clamps) | `load_lsf_cb`, `lsf_quant`, `lsf_quant_cond`, `lsf_weights_laroia` | — (verbatim tables) | `lsf_quant_io.json` | E |
| `smpl_synth.rs` | 550 | NLSF reconstruct (decoder), NLSF2A, LPC synth, frac-LTP (encoder AbyS), decoder state struct | `smpl_reconstruct_nlsf`, `smpl_nlsf2a`, `SmplSynthTables`, `SmplDecoderState`, `synth_internal_frame`, `smpl_gain_lin` | smpl_lsf_seed, smpl_celpdec, smpl_decode, smpl_harm_postfilter | via e2e | S |
| `smpl_pitch.rs` | 271 | pitch/LTP block decode (gains from CcTables, lags from heap window) | `decode_smpl_pitch`, `SmplPitchResult`, `decode_cdf_mem` | rangecoder, smpl_mem, smpl_cc_tables, smpl_decode | `pitch_vectors.json` | D |
| `smpl_pitch_enc.rs` | 1180 | SILK-style pitch estimator + wire lag encode | `smpl_pitch`, `smpl_encode_lags_wire`, `smpl_lags_predictor_after`, `load_pitch_tables`, `PitchEstState`, `PitchResult` | smpl_pitch_seed, smpl_signal_mode, smpl_tables_blob | `pitchio_ground_truth.json` | E |
| `smpl_pitch_seed.rs` | 235 | pitch seed ROM → `PitchTables` + contour-window parts | `PitchSeed::build()`, `build_contour_window()`, `seed_from_json` | smpl_tables_blob | `pitch_seed.bin` (runtime) | S |
| `smpl_mem.rs` | 297 | WASM heap window (absolute addresses) for contour/lag reads | `load_smpl_mem`, `SmplMem` (`u8/i16/u32/i32`, `cdf_at`, `cdf_bytes`) | smpl_pitch_seed, smpl_tables_blob | `pitch_seed.bin` (runtime) | S (real consumer = decode) |
| `smpl_cc_tables.rs` | 629 | gains/pulse CDFs from `cc_seed.bin` (dcmf→cmf, split/runlen builders, fixed-pt helpers) | `load_cc_tables`, `CcTables`, `RunlenCmfs`, `seed_from_json` | smpl_tables_blob | `cc_seed.bin` (runtime) | S |
| `smpl_pulse.rs` | 274 | PVQ-style pulse decode (count/split/magnitude/sign) | `decode_smpl_pulses`, `SmplPulseResult`, `mem8_static`, `smpl_split_3537` | rangecoder, smpl_cc_tables | `pulse_vectors.json` | D |
| `smpl_gains.rs` | 104 | unvoiced gains+nrgres decode | `decode_smpl_gains`, `SmplGainResult` | rangecoder, smpl_cc_tables | `gains_vectors.json` | D |
| `smpl_nrgres.rs` | 133 | encoder residual-energy scalar+shape quantizer | `quant_nrg_res_4`, `NrgResQuant` (98×4 shape CB verbatim) | — | `gennoise_params_dump.json` (via `param_decode_match.rs`) | E |
| `smpl_celp.rs` | 2386 | CELP excitation encoder: FCB greedy + delayed-decision beam search, ACB, gain quant | `CelpEncoder`, `CelpSubframeOut`, `cb_acbgains_hr_q14/lr_q14`, `smpl_distribute_fcb_surv` | smpl_celpdec tables | indirect (e2e/golden) | E |
| `smpl_celpdec.rs` | 658 | decoder CELP synthesis (per-subframe: interp → ACB → gennoise → LPC → HP postfilter) | `CelpDecState::synth_frame`, `CelpDecParams`, `celp_decode`, `lpc_interpol`, `acb_dequant` | smpl_gennoise, smpl_harmcomb, smpl_lsf_seed | `exc_pre_lags.json`, e2e ref | D |
| `smpl_gennoise.rs` | 741 | shaped residual-noise generator + nrgres decode helper | `NoiseGenerator`, `smpl_celp_gen_noise`, `smpl_decode_resnrg`, `smpl_get_normalized_bitrate` | — | `gennoise_vectors.json`, `gennoise_params_dump.json` | D (bitrate helper also used per-packet) |
| `smpl_vad.rs` | 544 | SILK VAD, fully fixed-point | `SmplVadState::new/process_packet`, `VadPacketResult` | — | `vad_ground_truth.json` | E |
| `smpl_signal_mode.rs` | 294 | VUV voicing classifier + spectral harmonicity | `VuvMode`, `smpl_get_signal_mode`, `build_f2w`, `harm_strength_at` | smpl_lpc (`SMPL_F_LEN`) | `sigmode_ground_truth.json` | E |
| `smpl_lpc.rs` | 747 | LPC front-end: window, FFT-autocorr → RC → A, BWE, A2NLSF, LPC interpolation | `smpl_window_lpc20`, `smpl_lpc_analyze_with_f2`, `smpl_a2nlsf_16`, `smpl_lpc_interpol(_idx)`, `new_lpc_fft_scratch`, `SMPL_F_LEN` | smpl_perc (FFT), silk_lsf_cos_tab | `fe_dump.json`, `lsf_quant_io.json` | E |
| `smpl_perc.rs` | 1217 | perceptual weighting model + bitrate controller + portable mixed-radix FFT | `FftScratch`, `rfft_forward_ordered_sc`, `smpl_perc_model`, `smpl_perc_ac2a`, `PercModelState`, `BitrateController::control` | — | none direct | E |
| `smpl_harmcomb.rs` | 463 | HP pitch-harmonic postfilter (per 20 ms frame) + encoder input-HP helpers | `HpPostfilterState`, `smpl_hp_postfilter`, `smpl_get_hp_coefs`, `smpl_filt_arma2` | — | `hp_postfilter_vectors.raw` | D (+ E input HP) |
| `smpl_harm_postfilter.rs` | 377 | final per-packet harmonic comb, 48-sample group delay | `HarmPostfilterState`, `smpl_harm_postfilter` | — | `harm_postfilter_vectors.raw` | D |
| `analysis.rs` | 1281 | encoder top level: analysis-by-synthesis, voicing, rate control, TOC=0x50 | `SmplEncoderState`, `smpl_analyze_frame_st` (+ internal candidates) | nearly every E module | indirect (e2e/golden) | E |
| `encode.rs` | 793 | entropy encoder (exact inverse of the decode field order) | `MlowEncoder`, `MlowError`, `encode_smpl_frame_into` | analysis, params, rangecoder, smpl_decode/cc_tables/mem, smpl_pitch_enc | `golden.rs` bitstream checksum | E |
| `decoder.rs` | 609 | decoder top level: RED → TOC → 3×20 ms → postfilters | `MlowDecoder` | everything D | `inbound_capture_frames.json` + `ref_usesmpl_expected.raw` | D |
| `params.rs` | 63 | encoder param structs (LSF/pulse/pitch/gain per internal frame) | `SmplFrameParams`, `SmplInternalParams`, … | — | — | S |
| `silk_lsf_cos_tab.rs` | 22 | SILK Q12 cos table (129 entries) for A2NLSF | const array | — | via `a2nlsf_matches_c` | E |
| `smpl_tables_blob.rs` | 119 | zlib+protobuf blob loader (§2) | `load_blob_buffa`, `inflate`, `tables` protos | — | — | S |
| `golden.rs` | 155 | no-fixture regression tripwire (§4) | test | encoder+decoder | its own committed checksums | T |
| `quality_metrics.rs` | 216 | decode-quality metrics vs `ref_usesmpl_expected.raw` | test | decoder | `ref_usesmpl_expected.raw` | T |
| `quality_tests.rs` | 814 | synthetic corpus (`synth_mic.raw`) + energy-envelope tests + capture tripwire | test | decoder | `synth_mic.raw`, `e2e_vectors.json`, `inbound_capture_frames.json` | T |
| `param_decode_match.rs` | 88 | per-subframe unvoiced gain params vs C `dec_param` dump | test | decoder vs encoder tables | `gennoise_params_dump.json` | T |

## 2. Runtime tables

Loader: `smpl_tables_blob.rs:36-48` — each `testdata/<x>.bin` is a **zlib stream** (inflated with a
4 MiB cap, `smpl_tables_blob.rs:33`) wrapping a **proto3** message from `mlow/tables.proto`
(schema `package mlow.tables`); decoded by field number. Generator: `gen_runtime_tables`
(`smpl_tables_blob.rs:90-118`), zlib level 9 (`:29`), from gitignored JSON dumps.

- `lsf_seed.bin` → `LsfSeed` (`tables.proto:25-43`): rot_8 `[2][16][16][16]`, rot_cond_8
  `[2][2][16][16]`, stage-2 qlvls/dcmfs `[9593]`, min/max qi, cb_16/cinv_16/cmf/cmf_cond u16,
  lsf_sel `[3][3]`, lsf_extra `[3]`, mean/min_dist/reg_cond/qstep f32. Consumed by
  `smpl_lsf_seed.rs:565-571` (`lsf_built()` → `SmplTables` CDFs used by `smpl_decode.rs` and
  `SmplSynthTables` used by `smpl_synth.rs`; grid16 tables **derived at load**, `tables.proto:42`).
- `pitch_seed.bin` → `PitchSeed` (`tables.proto:47-55`): packed blocksegs bitstream `[676]`,
  blocksegs2idx `[217]`, blocksegs_ix `[187][2]`, firstblock_range `[9][2]`, blockseg_idx_dcmf
  `[217]`, delta_lag_dcmfs `[3][319]`, block_transition_dcmf `[9][9]`. Two consumers:
  `smpl_pitch_enc.rs:82-88` (encoder `PitchTables`, blocktracks decoded from the bitstream) and
  `smpl_mem.rs:72-75` (`build_smpl_mem` → contour window carved at the fixed WASM heap addresses
  `smpl_mem.rs:57-68`, `g_pitch=0xb9d378`, 217 contours) which the **decoder** `smpl_pitch.rs` reads.
- `cc_seed.bin` → `CcSeed` (`tables.proto:60-73`): nrgres gain/shape dcmfs, fcbg_offset
  `[3][4][176]`, acbgains_hr `[17][16]`, fcbgains_v (+delta), `acbgains_cb_hr_q14` i16 pairs,
  `gain_recon` int16 rodata at `gain_recon_base = 0xf35e0`, n_pulses dcmfs (bgn/uv/v). Consumed by
  `smpl_cc_tables.rs:303-308` (`load_cc_tables`; CDFs derived via `dcmf_to_cmf` at load).
- `tables.desc` = compiled FileDescriptorSet of `tables.proto`; `tables.desc.sha256` pins both files
  (`proto 9ebaa32a…`, `desc fcbce2424…`).

Go note: the CDFs are u16 little-endian cumulative; a Go port can keep the same load path
(compress/zlib + a minimal proto decoder for these fixed schemas, or precompute the expanded tables
at build time — but byte-identity of the .bin must be verified against the committed blobs).

## 3. Decoder pipeline

Entry `MlowDecoder::decode` (`decoder.rs:77-96`), per RTP payload:

1. **RED strip** (only if negotiated): `depack_split_red` (`red.rs:33`); the *last* frame decodes.
2. **TOC parse** `parse_mlow_toc` (`toc.rs:42-86`); bit layout in `toc.rs:1-7`.
   - `std_opus` (`b&0xC0==0xC0`) → `sr/1000*ms` samples of silence (`decoder.rs:103-110`).
   - `sid || !active` → 60 ms silence, **range coder never opened** (`decoder.rs:116-119`).
   - Operating-point guard (`decoder.rs:127-147`): drop (silence, no coder touch) if rate≠16k,
     `flag2` (low_rate=1), or `frame_ms≠60`; otherwise the 3×20 ms geometry would desync the coder.
3. **Active decode** `decode_active_frame` (`decoder.rs:151-273`): `config = (b>>2)&1`,
   `RangeDecoder::new(&frame[1..])`, then per internal frame `f in 0..3`:
   1. `decode_smpl_lsf` (`smpl_decode.rs:84-155`): stage-1 selector (row keyed on intf/prev) → grid
      (match-dependent CDF) → 16 stage-2 symbols (per-coeff CDF) → `extra`. Mutates `SmplLsfState`.
   2. `decode_smpl_pulses` (`smpl_pulse.rs:28`): pulse count (NB **triangular prior**, wrapping u32
      math, `smpl_pulse.rs:55-66`), recursive binary splits, magnitude run-lengths, raw sign bits →
      `pulses[320]` + `subfr[4]`.
   3. `stage1==1` (voiced): `decode_smpl_pitch` (`smpl_pitch.rs:45`) — LTP gain idx (HR CDF row keyed
      on `prev_gain_idx`, `encode.rs:371` mirrors), filt idx (delta CDF), primary lag abs/delta,
      optional 64-sym fine, per-segment fractional loop reading the heap contour window →
      `block_lags[8]`. Decoder maps `lag = laginds*0.5 + 32` (`decoder.rs:200-203`).
     `stage1==0` (unvoiced): `decode_smpl_gains` (`smpl_gains.rs:18-58`) — gain_main/delta →
      `gain_q` via `gain_recon`; per-subframe `nrg_res` from bucketed CDF.
   4. `smpl_reconstruct_nlsf` (`smpl_synth.rs:107-165`); `state.prev_nlsf` updated (`decoder.rs:240`).
   5. `CelpDecState::synth_frame` (`smpl_celpdec.rs:368-482`): per-subframe LPC interpolation
      (`SMPL_LSF_INTERPOL_4`, `smpl_celpdec.rs:37`, over `lsf_prev`), excitation = pulses ×
      `fcb_gains[voiced][fcbg_idx]`, `celp_decode` ACB (two 40-block lags per subframe),
      `smpl_celp_gen_noise` (`smpl_gennoise.rs:415`) using `smpl_decode_resnrg`
      (`smpl_gennoise.rs:335`), `filt_ar16` LPC synthesis, then per-frame
      `smpl_hp_postfilter` (`smpl_celpdec.rs:474`, lag = energy-weighted mean of the 8 lags).
4. **Per-packet** `smpl_harm_postfilter` (`decoder.rs:247`, `smpl_harm_postfilter.rs:239`) over all
   960 samples with the 24 lags + `avg_norm_br/3`; clamp to ±1.

State that must persist across the 110-frame stream (`SmplDecoderState`, `smpl_synth.rs:543-550`):
`lstate: SmplLsfState` (prev stage-1/match, prev gain/filt/lag predictors, reset to −1 on no-match,
`smpl_decode.rs:59-72`); `prev_nlsf` (16 f32); `celp: CelpDecState` (`smpl_celpdec.rs:333-345`:
`NoiseGenerator` — env/smoothing/ARMA states/`rand_seed`/`prev_voiced`/`since_unvoiced`,
`smpl_gennoise.rs:346-356`; `acb_state` 728-sample ring; `lpc_synth_mem[16]`; `lsf_prev[16]`;
`prev_nrgres`; `hp: HpPostfilterState`, `smpl_harmcomb.rs:35-47`); `harm: HarmPostfilterState`
(`smpl_harm_postfilter.rs:91-97`, incl. 320-sample `state_comb` history). `reset()` restores all.

## 4. Encoder pipeline

Entry `MlowEncoder::encode_into` (`encode.rs:73-88`): sanitize (NaN→0, clamp ±1) →
`smpl_analyze_frame_st` (`analysis.rs:173-388`) → `encode_smpl_frame_into` (`encode.rs:91-135`).

Analysis per 60 ms packet (`analysis.rs:173-388`):
1. SILK VAD on int16 PCM, per-internal-frame probabilities + `coded_as_active_voice`
   (`analysis.rs:190-201`, `smpl_vad.rs:400`).
2. Input HP ARMA2 @35 Hz (`analysis.rs:206-218`, `smpl_harmcomb.rs:195-201`); int16-scaled `x[576]`
   with 16-sample lead history (`analysis.rs:221-228`).
3. Per internal frame `smpl_analyze_internal` (`analysis.rs:1042-1121`):
   - perceptual autocorrelation via the perc model, **exactly once per frame**
     (`analysis.rs:1058`, `smpl_perc.rs:734/784`);
   - roll `ltp_buf`, run pitch estimator `smpl_pitch` (`analysis.rs:1069`,
     `smpl_pitch_enc.rs:701` — HP filter, 2× downsample, coarse block-track survivors,
     full-rate refinement, rate/prev-lag/harmonicity biases);
   - voicing decision `smpl_get_signal_mode` + `VuvMode` hysteresis
     (`analysis.rs:1080-1089`, `smpl_signal_mode.rs:167`); unvoiced resets the lag predictors;
   - front-end: `smpl_window_lpc20` → `smpl_lpc_analyze_with_f2` (512-pt FFT autocorr →
     `brute_dct` → ac2rc → rc2a → BWE ×0.9999^i) → `smpl_a2nlsf_16` (fixed-point SILK A2NLSF) →
     `lsf_quant`/`lsf_quant_cond` (`analysis.rs:404-437`);
   - voiced: CELP encode (`CelpEncoder`, `smpl_celp.rs:740-776`: ACB + delayed-decision FCB beam
     search + gain quant) and `quant_nrg_res_4` (`smpl_nrgres.rs:61`); unvoiced: AbyS candidate
     (`analysis.rs:495`); bitrate controller sets per-subframe pulse budget
     (`smpl_perc.rs:856-868`);
   - commit reconstructed NLSF (`analysis.rs:1112`), replay predictor state
     (`smpl_replay_pitch_state`, `analysis.rs:1126-1144`).
4. Persisted encoder state `SmplEncoderState` (`analysis.rs:51-94`): input history `hist`,
   HP state, `CelpEncoder`, perc model, `BitrateController`, `lpc_hist`, `prev_lsfq`,
   `prev_voiced`, `SmplVadState`, `VuvMode`, `hp_pitch_hist`, `ltp_buf`, `PitchEstState`, FFT scratch.

Bitstream emit (`encode.rs:91-135`): `enc.reset()` → per internal frame
`encode_smpl_lsf` → `encode_smpl_pulses` → (`encode_smpl_pitch` | `encode_smpl_gains`)
mirroring the decode field order exactly, then `enc.done()`; output = `[toc][range-coded body]`.

What pins what:
- **Bitstream**: `golden.rs` pins the encoder output bytes (fnv1a over concatenated packets,
  `golden.rs:92`) and decoded PCM bytes (`golden.rs:89`) on a synthetic corpus — no external
  fixture. This is the only encoder bitstream oracle (the external capture is NOT Rust-reproducible,
  `PROVENANCE.md:77-79`), so the Go encoder can at best match this class of check.
- **Per-stage dumps**: `fe_dump.json` (`smpl_lpc.rs:655`), `lsf_quant_io.json`
  (`smpl_lsf_quant.rs:596`), `pitchio_ground_truth.json` (`smpl_pitch_enc.rs:1097`),
  `sigmode_ground_truth.json` (`smpl_signal_mode.rs:232`), `vad_ground_truth.json`
  (`smpl_vad.rs:508`), `gennoise_vectors.json` (`smpl_gennoise.rs:654`),
  `gennoise_params_dump.json` (`param_decode_match.rs:15`).

## 5. Suggested Go port order (decoder-first)

Rust arithmetic per stage, read from code (no guessing):

| # | stage | gate vector | arithmetic in Rust |
|---|---|---|---|
| 1 | `rangecoder.rs` ec_dec/ec_enc | `rc_vectors.json` | pure u32, `wrapping_*` semantics (`rangecoder.rs:3`) — Go: uint32 with natural wrap |
| 2 | `toc.rs` parse | `toc_vectors.json` (256 exhaustive) | integer bit ops |
| 3 | table loader (zlib+proto, §2) | e2e decode later; `PitchTables::debug_checksum` (`smpl_pitch_enc.rs:117`) | — |
| 4 | `smpl_decode.rs` LSF block | `lsf_vectors.json` | integer symbols, u16 CDFs |
| 5 | `smpl_cc_tables.rs` + `smpl_pulse.rs` | `pulse_vectors.json` | u32 wrapping (triangular prior), i32; fixed-pt helpers (`smulbb`, `smpl_cc_tables.rs:14-83`) |
| 6 | `smpl_gains.rs` | `gains_vectors.json` | i32 only (Q14 shifts, `smpl_gains.rs:50`) |
| 7 | `smpl_pitch_seed.rs` + `smpl_mem.rs` + `smpl_pitch.rs` | `pitch_vectors.json` | integer; heap addressed by absolute u32 addresses — keep the region-slice addressing |
| 8 | `smpl_lsf_seed.rs` expansion + `smpl_synth.rs` reconstruct/NLSF2A | `lsf_quant_io.json` / e2e | f32 with f64 in NLSF2A polys (`smpl_synth.rs:224-257`) |
| 9 | `smpl_gennoise.rs` | `gennoise_vectors.json` + `gennoise_params_dump.json` | f32; Q14 int input (`smpl_decode_resnrg`, `smpl_gennoise.rs:335`) |
| 10 | `smpl_celpdec.rs` synthesis | `exc_pre_lags.json` (`smpl_celpdec.rs:501`) | **all f32** — excitation, ACB, `filt_ar16` (`smpl_celpdec.rs:307-315`, plain f32 accumulation), noise add. (f64 accumulation exists only in the encoder-AbyS `smpl_synth.rs:261-278` LPC path, not the decoder) |
| 11 | `smpl_harmcomb.rs` HP postfilter | `hp_postfilter_vectors.raw` | f32 |
| 12 | `smpl_harm_postfilter.rs` | `harm_postfilter_vectors.raw` | f32; **tolerance, not bit-exact** (`smpl_harm_postfilter.rs:5-7`: C built with `-ffast-math -mavx`; matches to i16 quantization) |
| 13 | `decoder.rs` e2e | `inbound_capture_frames.json` → `ref_usesmpl_expected.raw` (corr>0.95, `decoder.rs:394-407`); energy envelopes via `e2e_vectors.json` | — |
| 14 | `smpl_vad.rs` | `vad_ground_truth.json` | **fully fixed-point i32** (silk ops `smpl_vad.rs:14-122`); port exactly, no floats |
| 15 | `smpl_lpc.rs` front-end | `fe_dump.json`, `a2nlsf_matches_c` (`smpl_lpc.rs:628`) | f32 window/FFT + f64 DCT rows (`smpl_lpc.rs:93-183`); A2NLSF fixed Q16 i32 (`smpl_lpc.rs:597-`) |
| 16 | `smpl_lsf_quant.rs` | `lsf_quant_io.json` | f32 |
| 17 | `smpl_pitch_enc.rs` | `pitchio_ground_truth.json` | f32 (unrolled filters for stable accumulation, `smpl_pitch_enc.rs:213-251`) |
| 18 | `smpl_signal_mode.rs` | `sigmode_ground_truth.json` | f32 |
| 19 | `smpl_celp.rs` + `smpl_nrgres.rs` + `analysis.rs` + `encode.rs` | `golden.rs` round-trip; `param_decode_match.rs` | f32 CELP; f64 autocorr in analysis (`analysis.rs:848`); f32 nrgres dB math (`smpl_nrgres.rs:61-70`) |

FFT/SIMD/unsafe audit:
- **No `unsafe` anywhere** in mlow/ (grepped). No SIMD intrinsics; no external FFT crate.
- The C reference used PFFFT; the Rust uses a **portable mixed-radix complex FFT** in
  `smpl_perc.rs:315-599` (PERCW_NFFT = 576 = 2⁶·3², radix-2/3 + naive base; ordered real packing)
  shared by the LPC front-end (512-pt, `smpl_lpc.rs:249-271`). Consequence: front-end `A`/NLSF match
  the C to *tight float tolerance*, not bit-for-bit (`smpl_lpc.rs:11-13`) — same caveat applies to a
  Go FFT; the committed float-tolerance fixtures absorb this.
- Pointer tricks to port carefully:
  - heap-window addressing by absolute address with region-boundary zero-fill fallback
    (`smpl_mem.rs:155-197`, `smpl_pitch.rs:37-42`);
  - in-place overlapping FIR (`smpl_synth.rs:307-318`: reads see prior writes — Go must not
    introduce a temp buffer);
  - `wrapping` u32 modular arithmetic in the pulse count/splits (`smpl_pulse.rs:55-66`,
    `encode.rs:210-226`) and the range coder;
  - unrolled 5-sample AR1 accumulation in the pitch HP filter (`smpl_pitch_enc.rs:224-251`) — the
    float *summation order* is normative; keep it.

## 6. Multi-frame TOC details

Bit layout (`toc.rs:1-7`, parse `toc.rs:42-86`): bit7=SID(DTX/CNG), bit6=VAD, bit5=rate
(0→16k, 1→32k), bits4:3→frame ms index into {10,20,60,120}, bit2=low_rate (flag2), bit1=voiced
enable, bit0=flag0. `active = vad || bit1`, `voiced = vad && bit1`. `(b&0xC0)==0xC0` is the
standard-Opus/CELT escape, not a global discriminator (`toc.rs:1-3`).

Families seen on the wire (`quality_tests.rs:794-813` tripwire):
- `0x50` = 0b0101_0000: active, VAD, 16k, 60 ms, low_rate=0, config 0 — the normal speech frame;
  full body: LSF → pulses → (pitch | gains) per internal frame.
- `0x12` = 0b0001_0010: active (bit1 set, VAD clear), 16k, 60 ms, config 0 ("config-2" in the
  capture) — same body geometry; the decoder decodes it identically (`vad||bit1` ⇒ active,
  `toc.rs:70`).
- `0x10` = 0b0001_0000: **inactive** (VAD clear, bit1 clear), 10 ms nominal ("config-1") — the body
  omits *every* symbol: the decoder silences it without opening the range coder
  (`decoder.rs:116-119`), so no body geometry can desync the coder. Same for any SID frame.

Encoder TOC emission in the vendored code: **constant `0x50`**. `smpl_analyze_frame_st` hardcodes
`toc: 0x50, config: 0` (`analysis.rs:383-387`) and `encode_smpl_frame_into` prepends `fp.toc`
(`encode.rs:132`); the R9 test asserts every emitted frame has TOC `0x50` (`encode.rs:619-625`,
"the encoder is config-0 / no-DTX; codec routing comes from negotiation, not this byte"). The
encoder has **no DTX/SID path** — silence is encoded as a low-energy active frame
(`golden.rs:22-23`). So the vendored v0.7.0 code emits activity-independent `0x50` and does NOT
reproduce the upstream main behavior (activity-dependent `0x50`/`0x12`/`0x10`); the Go port should
do the same unless the WhatsApp peer requires the upstream variant (the external capture with
`0x10`/`0x12` came from the C reference encoder, `PROVENANCE.md:75-79`).

Operating-point reality of the decoder (mirror it in Go): 16 kHz / low_rate=0 / 60 ms only;
everything else active is dropped to silence without touching the coder (`decoder.rs:127-147`).
