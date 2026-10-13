# ottis

<img src="art/logo.png" width="96" alt="Ottis, ottis's mascot: a pixel-art sea otter holding a microphone">

Text to speech with word timings, in Odin. It runs Kyutai's
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (100M parameters, CPU)
through ONNX Runtime and writes a WAV plus a Whisper-style JSON of when each
word is spoken, which onimite's `captions{}` reads as is.

```sh
odin build . -o:speed -out:build/ottis
build/ottis say "The keeper climbed the spiral stairs." -o line.wav -words line.json
build/ottis say -f script.txt -voice marius -seed 3 -o scene.wav -words scene.json
build/ottis voices
```

The binary needs nothing installed. On first use it downloads, into
`$OTTIS_DATA` or the user data directory (`-data DIR` to override):

- ONNX Runtime 1.30.0's library for the platform (Linux x64/arm64, macOS
  arm64, Windows x64),
- the `english_2026-04` ONNX bundle (about 440 MB) and the chosen voice,
- the word aligner, wav2vec2-base-960h as ONNX (378 MB),

each checked against a pinned SHA-256 (`assets/pins.odin`). `ottis fetch`
does this up front; `ottis fetch -verify` re-hashes everything. Downloads use
the system libcurl on Linux and macOS and WinHTTP on Windows.

**Word timings** come from forced alignment: each chunk's text is aligned
to its generated audio by a CTC speech recognizer (20 ms frames), so cues
land where the word is actually spoken. Words it cannot spell, such as numbers
("$3.50" said as "three fifty"), share the span between their neighbours
and are marked `"estimated": true`.

**Determinism**: the only randomness is the flow head's noise, drawn from
ottis's own seeded generator; the same text, voice and `-seed` give the same
samples on a given CPU (`say` prints their SHA-256). ONNX Runtime's thread
count changes results, so ottis fixes it (`pocket.THREADS`).

## As a library

```odin
import "deps:ottis/assets"   // with -collection:deps=.. from a sibling project
import "deps:ottis/pocket"

root, _ := assets.data_dir()
library, err := assets.fetch(root, "alba")          // runtime, bundle, aligner, voice
engine: pocket.Engine
pocket.engine_init(&engine, root, library)           // once per process
voice, _ := pocket.voice_load(&engine, root, "alba") // once per voice
result: pocket.Result
pocket.synthesize(&engine, &voice, "Hello there.", pocket.DEFAULT_OPTIONS, &result)
// result.audio: mono f32 at result.sample_rate (24 kHz)
// result.words: { text, index, start, end, estimated } for every source word
// result.chunks: generated chunks with their spans and word ranges
```

After `engine_init` the engine is read-only: `synthesize` can run on several
threads at once with their own `Result`s, and gives the same samples as when
run alone. Each call uses `pocket.THREADS` (4) runtime threads per graph.
`pocket.VERSION` changes whenever the output for the same inputs does: a
store of synthesized lines keys on it, the pins in `assets`, the text, the
voice and the options.

Layout: `assets/` (pins, download, archive extraction), `ort/` (ONNX
Runtime C API, loaded at run time), `spm/` (SentencePiece Unigram
tokenizer), `pocket/` (chunking, generation), `align/` (resampling, CTC
forced alignment), `wav/`,
`main.odin` (CLI). `PLAN.md` holds the plan, including the native port.
`tools/fetch_third_party.sh` clones the upstream Pocket TTS sources, at the
pinned commit the port follows, into `third_party/` (gitignored, not needed
to build).

ottis is MIT-licensed (`LICENSE`). Licences of what is downloaded, and of
behaviour ported from elsewhere: `THIRD_PARTY_NOTICES.md`. Audio made with
ottis should credit *Pocket TTS by Kyutai (CC-BY-4.0)*.

The mascot, Ottis, is drawn by `tools/mascot` (`odin run tools/mascot --
art`): `art/logo.svg` keeps its eyes, mouths (closed, "ah", "oo", "ee") and
sound waves as layers a script can switch by id, so Ottis can lip-sync to
ottis's own word cues.
