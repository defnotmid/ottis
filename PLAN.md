# ottis plan

A native Odin port of Kyutai's Pocket TTS, specific to that one model, with a
permissive phonemizer beside it. CPU first on Linux x86-64 (AVX-512) and macOS
arm64 (NEON); Slang compute later through SDL_GPU (SPIR-V + MSL), the way
../paf, ../oasic and ../onimite build their shaders. First consumer: onimite
(voice tracks, word-level captions, lip-sync).

Research date 2026-10-12, against upstream `kyutai-labs/pocket-tts` main
@ b94db17.

## What we port

Default checkpoint `english` (= `english_2026-09_2`), one file
`languages/english/model.safetensors`, 219 MB, BF16, 109.5 M params, weights
CC-BY-4.0 (credit Kyutai), code MIT. `kyutai/pocket-tts` is gated (encoder
for voice cloning); `kyutai/pocket-tts-without-voice-cloning` is public, has
the encoder zeroed, and ships the tokenizers and precomputed voices.

| stage | shape | params |
| --- | --- | --- |
| text | normalize, capitalize, terminal punctuation, split into <= 50-token chunks on sentence then `,;:` boundaries; SentencePiece Unigram, vocab 4000, byte fallback; LUT 4001x1024 | 4.1 M |
| backbone `flow_lm` | 6 layers, d 1024, 16 heads x 64, FFN 4096 GELU-tanh, pre-LayerNorm eps 1e-5, no biases, fused in_proj, interleaved-pair RoPE (period 1e4), causal, KV cache = [bos_before_voice, voice] [text] [audio frames] | 75.5 M |
| head | adaLN MLP (6 res blocks, d 512), LSD 1 step: `x = z + v(c, s=0, t=1, z)`, `z ~ N(0, 0.3)`; EOS logit > -4 after 6 frames, then a few tail frames | 9.8 M |
| Mimi decoder | latent 32 @ 12.5 Hz -> denorm -> 1x1 32->512 -> depthwise convT k32 s16 -> 2-layer transformer (d 512, 8 heads, window 250, LayerScale) -> SEANet (convT strides 6,5,4, res blocks, ELU) -> 24 kHz | 10.3 M |
| Mimi encoder | voice cloning only: SEANet down 4,5,6 -> transformer -> conv k32 s16 -> 32 -> speaker_proj 32->1024 | 9.8 M |

Exact quirks to copy: the head's "RMSNorm" is unbiased mean-subtracted
variance (N-1); GELU is tanh (upstream fixed erf -> tanh in Sept 2026);
streaming convs carry a left buffer, convTs carry an overlap-add partial with
bias removed; the first latent input is NaN replaced by `bos_emb`. A voice is
just the backbone KV cache after the voice prefix; shipped voices are those
caches in safetensors, tied to the checkpoint.

Cost: ~4.3 GMAC per second of audio (decode is GEMV-bound: the backbone
streams ~300 MB of f32 weights a frame, 150 MB as bf16). Upstream reports
~6x real time on 2 M4 cores.

Randomness enters in exactly one place, the head's [32] Gaussian, one draw
per frame plus one per prompt pass (discarded). ottis owns a seeded RNG
(stream per scene/utterance), so a seed pins the audio bit for bit on one
build and platform.

Not ported: the `*_24l` variants (6x larger), the drifting head, flow
matching with Euler steps, training. Other languages (fr, de, es, it, pt, nl)
are the same graph with another tokenizer and checkpoint; they come for free
once English matches.

## Code layout (proposed)

Model-specific code, no tensor library or graph. Every weight is a named,
typed field checked once at load; every kernel is one of the model's steps
at sizes fixed at load.

```
ottis/            cli: text + voice + seed -> wav + words json
pocket/           model structs, safetensors load (bf16 -> f32 or kept bf16), backbone, head, generate loop
codec/            Mimi decoder (streaming), encoder (cloning)
text/             normalize, chunking, SentencePiece Unigram (tokenizer.json)
align/            word timings from backbone attention
phon/             phonemizer: phone set, lexicon, OOV model, durations, visemes
audio/            wav io, polyphase resampler
shaders/          later: *.slang -> spv/msl, build.sh as in paf/onimite
```

Kernel inventory (each later becomes one `cs_main`): attention step with KV
append and interleaved RoPE; MLP (linear, GELU-tanh, linear); LayerNorm
variants; adaLN res block; streaming conv1d, convT1d (normal and depthwise);
ELU; windowed attention for Mimi (ring buffer of 250).

## Oracle (never committed)

Python run through `uv`, outside the repo (`.oracle/`, gitignored) or in a
scratch dir: upstream `pocket-tts` with the noise taken from a file instead
of torch's RNG, dumping raw f32 per stage for a fixed text set and voice:
token ids, LUT rows, per-layer backbone outputs, `c`, head output, eos
logits, latents, Mimi intermediate and final audio. Odin tests read those
files from a configured path and skip when they are missing; the repo keeps
only the regeneration note and output hashes.

The timestamp fork (`dpm63/pocket-tts-timestamped`, MIT) is the oracle for
word timings; `g2p_en` for the OOV model.

## Milestone 0: speaking through onnxruntime

**Status (2026-10-13): working on Linux x64 and macOS arm64.** `ottis say`
produces speech that Whisper small.en transcribes word for word, with word
timings from forced alignment, deterministic per seed across runs (threads
fixed at 4: at 8 or more ONNX Runtime gives other samples).

What landed differs from the sketch below in three ways:

- Self-contained: the binary downloads and SHA-256-verifies the pinned ORT
  1.30.0 release for its platform (libcurl on Linux/macOS, WinHTTP on
  Windows), extracts the library (tar.gz / zip in Odin) and loads it with
  `core:dynlib`; nothing is linked or installed.
- Word timings come from forced alignment (`align/`): wav2vec2-base-960h
  (ONNX, Apache-2.0) and a CTC Viterbi pass over each chunk's transcript,
  20 ms frames, wildcards for words it cannot spell. It replaced a first
  version that read two backbone attention heads (pocket-tts-timestamped's
  method, through a graph patch): at pause onsets the aligner lands on
  the voice where attention was often well off.
- Text handling follows upstream `text_chunking.py`, not the ONNX wrapper's
  simplified copy; the loop adds upstream's 6-frame EOS minimum and 5 ms
  decoder fade-in.

Measured on the 9900X3D, `-o:speed`, 4 intra-op threads per graph:

| Text | Audio | Synthesis | Real time | Backbone | Flow | Decode |
| --- | --- | --- | --- | --- | --- | --- |
| 1 chunk, 46 tokens | 8.64 s | 1.66-1.81 s | 4.8-5.2x | 1.27-1.41 s | 0.11 s | 0.25 s |
| 4 chunks, 124 tokens | 21.36 s | 4.60 s | 4.65x | 3.55 s | 0.29 s | 0.63 s |

Word starts land on the voice's onsets where a word has one (onimite's
`hit` cue check passes for "lamp" and "dark"); numbers and symbols are
wildcards spanning what was said, marked `"estimated"`. Alignment costs
about 2% of real time.

Verified: tokenizer ids equal SentencePiece's (`odin test spm`); fetch,
extract and patch on Linux x64; zip (Windows) and macOS/arm64 tar
extraction on the real release archives; `odin check` for macOS arm64 and
Linux arm64, and a Windows x64 object build. Run on macOS arm64 (M4,
2026-10-13): fetch, `say` and `fetch -verify` work, both test suites pass,
and output is deterministic across runs. Not yet run: Windows and Linux
arm64 binaries.

Next for this lane:

- Process time is ~6x wall time (ORT threads spin): try disabling spinning,
  fewer threads for the small graphs, the int8 graphs, and decoding on a
  second thread while the backbone runs (upstream does), then measure.
- onimite: a `say()` binding that synthesizes at load (content-addressed
  cache, one engine per process on onimite's pool) and registers the clip,
  its captions and word cues.
- Voice cloning from a WAV through `mimi_encoder.onnx` (needs a WAV reader
  and resampler).
- Run on Windows and Linux arm64.

The original sketch: the quickest route to audio, after `../dangersnail`'s pattern: copy its
API-27 adapter (`extras/onnx_runtime`, ~1300 lines: runtime/session owners,
fixed caller-owned tensors, names/types/shapes validated at setup,
sequential single-thread sessions) and a checksum-pinned fetch script like
`tools/model_assets/person_follow/fetch.sh`, assets under an ignored
`.data/`.

- Model: `KevinAHM/pocket-tts-onnx`, bundle `english_2026-04` (CC-BY-4.0):
  `text_conditioner`, `flow_lm_main` (18 KV states, cache [2,1,1000,16,64],
  so at most 1000 positions per chunk), `flow_lm_flow`, `mimi_decoder` (56
  conv/attention states), `mimi_encoder` (cloning), `tokenizer.model`,
  `bos_before_voice.npy`, and `bundle.json` describing every state's shape,
  dtype and fill. fp32 and int8 variants. Also fr, de, it, pt, es (+ 24l).
- Runtime: ORT 1.29 is installed here (`/usr/lib/libonnxruntime.so`, the
  ROCm build); MIT, permissive deps; macOS ships universal binaries.
- Ours in Odin: SentencePiece reader, chunking, the generate loop (state
  plumbing from `bundle.json`, the LSD step `x = z + flow(c, 0, 1, z)` with
  our seeded noise, EOS rule), voice prompt from a WAV through the encoder,
  WAV writer.
- Lost against the native port: word timings from attention (the graphs do
  not output attention or queries), so words are timed by the phonemizer's
  duration estimate inside each chunk; the current `english` checkpoint
  (2026-09_2, tanh GELU fix) until someone re-exports; one dependency
  (~20 MB library per platform); no path to Slang.
- Afterwards it is the native port's oracle for the 2026-04 checkpoint, the
  way dangersnail's CPU ORT lane is its semantic oracle; the native port then
  replaces it, one path in the end.

## Milestones

Each one lands with a parity test against the oracle.

1. **Load and tokenize.** safetensors reader; tokenizer.json Unigram Viterbi.
   Gate: token ids equal upstream over a text corpus (punctuation, numbers,
   unicode, byte fallback).
2. **Mimi decoder.** Oracle latents in, audio out, streamed frame by frame.
   Gate: max abs error per stage within an f32 tolerance; whole-utterance and
   streaming give the same samples.
3. **Backbone and head.** Voice cache + text prefix + generate with injected
   noise. Gate: per-frame `c`, eos logits and latents within tolerance;
   frame counts identical.
4. **End to end.** Chunking, EOS tail, fade-in, ottis RNG, WAV out. Gate:
   same seed -> identical hash across runs and thread counts; listening pass.
5. **Word timings.** Softmax of q.k over the text keys for the fork's heads
   (English: L3H8 + L2H0), monotonic word tracking. Gate: word starts/ends
   match the fork; output is the Whisper-style JSON onimite's
   `captions.odin` already reads.
6. **Voice cloning.** WAV reader, resampler, `end_on_pause` trim, Mimi
   encoder, voice cache save/load. Gate: cache equals upstream's for the same
   clip.
7. **CPU performance.** Phase timings (load, tokenize, prompt, per-frame
   backbone/head/decoder), wall vs. per-thread busy, peak RSS, real-time
   factor, time to first audio. Candidates: bf16 weights with f32
   accumulate, one worker path splitting GEMV rows, Mimi pipelined against
   the backbone. macOS: same `#simd` kernels on NEON; Accelerate only if a
   profile shows it wins, and then as the one path on that platform.
8. **Slang.** One shader per kernel, SDL_GPU compute, KV cache and conv state
   resident on the GPU; the CPU path is the parity oracle, as onimite's
   software backend is for its renderer.

## Phonemizer

Pocket TTS reads text, so phonemes never enter the model and cannot change
how it pronounces a word. Their uses are elsewhere:

- **Pronunciation control** goes through *respelling*: the user lexicon maps
  a word to text Pocket says right (`Kyutai -> Kyoo-tie`), applied before
  tokenizing. Each entry also carries phones, for the uses below.
- **Lip-sync:** word times come from attention (milestone 5); phones split
  each word's span, giving phone and viseme timings.
- **Later phoneme backends** (Kokoro) would need misaki-compatible output.

Stack, all permissive, all Odin:

1. **Phone set:** misaki's English set (49 phones, what Kokoro was trained
   on), with ARPAbet and IPA conversion, and a phone -> viseme table.
2. **Normalizer:** numbers, ordinals, currency, dates, times, abbreviations,
   rules after Flite's `usenglish` (BSD). Its words are mapped back to the
   source words so attention timings still apply ("1984" spans four spoken
   words).
3. **Lexicon:** user lexicon, then CMUdict (BSD-2-style; attribution in
   THIRD_PARTY_NOTICES), compiled by an Odin tool into a sorted, flat binary
   table embedded with `#load`. Port misaki's -s / -ed / -ing rules.
4. **OOV model:** g2p_en's GRU (Apache-2.0, ~165 k params, 0.66 MB):
   matvec, sigmoid, tanh, argmax. Measure it against Flite's CART
   letter-to-sound (BSD) on held-out CMUdict words (phone and word error
   rate) and keep the better one. misaki's BART fallback (Apache, 752 k)
   only if a Kokoro backend is wanted.
5. **Phone durations:** per-phone priors (vowel/consonant, stress, word
   final), scaled to fill the attention-derived word span.

Avoid: espeak-ng, Piper, Sequitur (GPL); ipa-dict en_UK (GPL); num2words
(LGPL); MMS aligner weights (non-commercial). Wiktionary/WikiPron data is
CC-BY-SA: only as a separate data file, if ever.

Multilingual: Pocket's other languages need no G2P to speak; timings come
from attention the same way. Lip-sync for them can start with a
character-class duration split and get per-language lexicons later.

## Open questions

- Which checkpoint to pin: `english_2026-04` matches the ONNX bundle, so
  the ORT lane is the oracle with no Python; `english` (2026-09_2) is
  upstream's current default and needs the Python oracle.
- Ship the gated cloning weights, or start without cloning (public repo,
  shipped voices only)?
- Is a Kokoro backend ever wanted? It decides between the GRU and porting
  misaki's BART.
- Keep weights in bf16 at runtime (half the memory traffic) or widen to f32
  at load: decided by milestone 7's measurements.

## Sources

- Upstream: https://github.com/kyutai-labs/pocket-tts (config/english.yaml,
  models/tts_model.py, flow_lm.py, mimi.py, modules/*), weights
  https://huggingface.co/kyutai/pocket-tts-without-voice-cloning
- CALM paper: https://arxiv.org/abs/2509.06926
- GELU fix: https://github.com/kyutai-labs/pocket-tts/commit/2dd548a811
- Timestamps fork: https://github.com/dpm63/pocket-tts-timestamped
- Native references: https://github.com/gradium-ai/xn-ptts (Rust),
  https://github.com/babybirdprd/pocket-tts (Candle),
  https://github.com/VolgaGerm/PocketTTS.cpp
- misaki: https://github.com/hexgrad/misaki (EN_PHONES.md, en.py)
- g2p_en: https://github.com/Kyubyong/g2p
- CMUdict: https://github.com/cmusphinx/cmudict
- Flite: https://github.com/festvox/flite
