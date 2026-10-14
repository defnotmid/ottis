# Third-party notices

ottis's source is its own. It downloads the following at run time (`ottis
fetch`, or the first `ottis say`), each pinned by SHA-256 in
`assets/pins.odin`; none of it is in this repository or the binary.

| What | From | Licence |
| --- | --- | --- |
| ONNX Runtime 1.30.0 (CPU) | github.com/microsoft/onnxruntime releases | MIT; its `LICENSE` and `ThirdPartyNotices.txt` are extracted next to the library |
| Pocket TTS `english_2026-04` weights | Kyutai, huggingface.co/kyutai/pocket-tts | CC-BY-4.0 |
| ONNX export of those weights | huggingface.co/KevinAHM/pocket-tts-onnx | CC-BY-4.0 (fetched as `english_2026-04/LICENSE`) |
| Precomputed voices | huggingface.co/kyutai/pocket-tts-without-voice-cloning | CC-BY-4.0; each voice's source recording carries its own licence, listed at huggingface.co/kyutai/tts-voices (`cosette` comes from Expresso and `jean` from EARS: non-commercial) |
| wav2vec2-base-960h (word aligner) | Facebook AI, exported to ONNX by huggingface.co/onnx-community/wav2vec2-base-960h-ONNX | Apache-2.0 |

Attribution for audio made with ottis: *Pocket TTS by Kyutai (CC-BY-4.0).*

Kyutai's acceptable-use terms for Pocket TTS prohibit, among other things,
voice impersonation or cloning without explicit and lawful consent, and
presenting generated audio as genuine recordings of real people or events.

## Behaviour ported from other projects

- **Pocket TTS** (github.com/kyutai-labs/pocket-tts, MIT, Kyutai): text
  preparation and chunking, the generation loop's end-of-speech rule and
  lengths, and the decoder fade-in (`pocket/text.odin`, `pocket/tts.odin`).
  Its licence, from kyutai-labs/pocket-tts at b94db17:
  `LICENSES/pocket-tts-MIT.txt`.
- **torchaudio's CTC forced alignment** (BSD-2-Clause): the Viterbi
  formulation `align/align.odin` follows (reimplemented).
- **SentencePiece** (github.com/google/sentencepiece, Apache-2.0): the Unigram
  Viterbi encoding `spm/spm.odin` reproduces (reimplemented, not copied).
- **ONNX Runtime's C API** (github.com/microsoft/onnxruntime, MIT): the
  function table slots in `ort/ort.odin` follow `OrtApi`'s field order in
  `onnxruntime_c_api.h`.
