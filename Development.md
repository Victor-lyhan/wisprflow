# Dev Log
Development journal of a voice to text + LLM to imitate Wisprflow workflow

## 8/2/2026 — Phase 0

Scoped the project: local speech-to-text for a dental operatory, modular enough to embed in other projects, open-weight models only. Dev on M4 Max, **production on Windows** with no guaranteed GPU — which ruled out MLX (Apple-only) and CUDA-bound tooling (NeMo, vLLM) as production paths.

### Research that changed the design

- **No public dental ASR dataset exists.** Recent dental and orthodontic ASR studies each built private corpora. Closest public analog is PriMock57 (mock primary-care consultations, audio + manual transcripts). A bootstrap step is unavoidable.
- **The orthodontic ASR study is the strongest available evidence.** 11 systems; all significantly worse on clinical vocabulary than general speech (*P* < 0.001) **except** the LLM-corrected configuration, which also won overall (3.5% domain WER; Dragon Professional Anywhere 33.9% WER). Noise substantially increased errors. Clinically significant errors in every system, 2%–66%. → ASR + constrained LLM correction, domain WER as the headline metric, both transcripts retained with a diff, human verification assumed.
- **Parakeet-TDT-0.6b-v3 has community ONNX exports** (CC-BY-4.0) that run without NeMo — the thing that makes one codebase work on Mac dev and Windows CPU production.
- **pyannote 3.1 is MIT**, community-1 is CC-BY-4.0. Both self-hostable commercially.
- **CDT and SNODENT are ADA copyright** and require a paid license to redistribute, though practices may use CDT freely in their own records. → never bundled; runtime hook instead.
- **Whisper's `initial_prompt` reads only its last 224 tokens** and gets hallucination-prone when stuffed → hot-words are a small nudge; the lexicon belongs in the correction stage.

### Central tension

Transcribing during recording conflicts with diarization, which needs the whole recording. Resolved with **two tiers**: a live tier (~2–4 s, provisional labels) and a final authoritative tier at end of recording. The final tier *is* the offline pipeline, so batch work isn't throwaway. Batch is modeled as a degenerate case of streaming — one code path.

### Built

Contracts and stage protocols, entry-point backend registry, YAML/env config, audio ingestion (file, **growing-WAV for live recording**, push queue, PyAV resampling), faster-whisper backend, CLI, five sinks, evaluation harness (WER/CER, domain WER, tooth accuracy, corrector insertion rate), dental seed lexicon (278 terms), 175 tests. Lint and format clean.

### Measured

- Growing-WAV reader: 44.1 kHz stereo → 16 kHz mono, **first audio at 0.21 s** while the file was still being written (0.84 s total). The live-transcription premise holds.
- Smoke set (10 synthetic dental utterances, tiny.en): **WER 9.57%, DWER 18.52%**, tooth accuracy 8/8, RTF 0.05.
- **Domain WER is ~2× overall WER** — independently reproducing the published finding on our own harness.

### Three bugs worth remembering

1. **Number words adjacent to punctuation didn't normalize.** `"nineteen."` didn't match the lookup, so tooth extraction missed every sentence-final tooth reference — and tooth accuracy scored over a short reference list, i.e. failed *upward*.
2. **`"one hundred thousand"` ≠ `"100,000"`.** Anaesthetic concentrations are dictated both ways. Fixing it dropped one sample's WER from 20% to 6.98% — most of the apparent error was a normalization artifact. Compounding is gated on an explicit multiplier word so `"three two three"` still stays three probing depths rather than collapsing to 323.
3. **Bad TTS voices produced a 42% WER that measured nothing.** macOS legacy formant voices (Fred, Kathy) rendered "carpules" as "car appeals" — audio a human can barely parse. Modern voices gave 0–11% on the same scripts. The generator now refuses them by name.

The lesson common to all three: an unvalidated metric produced a number that looked like a model result and wasn't. Worth re-checking normalization before believing any accuracy change.

### Real errors seen (clean synthetic audio, tiny.en)

- `buccal` → `buckle`
- `mesial` → `mutual`
- `irreversible pulpitis` → `a reversible pulpitis` — **inverts the diagnosis.** Exactly the clinically-significant-error class the study flagged.

### Next

Streaming policy (LocalAgreement-2), Silero VAD, pyannote diarization, LLM corrector with diff output and insertion-rate guard, Parakeet ONNX backend, Windows packaging + PHI hardening.

Blocking dependency: the real gold set needs people reading the scripts in an operatory. Nothing measures real-world accuracy until that exists — the current numbers describe the harness, not the clinic.

## 7/16/2026
