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

## 8/2/2026 (later) — Phases 1-3

### Correction works, and it is the whole ballgame

Domain WER 15.38% -> 3.08% (80% relative), overall WER 9.57% -> 5.32%, tooth
accuracy held at 100%, RTF 0.20. Qwen3-8B via Ollama on loopback.

The shape of the result matters as much as the size. Verbatim: DWER (15.38%)
*above* WER (9.57%). Corrected: DWER (3.08%) *below* WER (5.32%). Clinical
vocabulary stopped being the weak point -- the same inversion the orthodontic
study reported for its LLM-corrected system.

### Candidate selection took three tries

Feeding the model plausible alternatives is what makes the prompt small and the
corrections accurate. Getting there:

1. Phrases were indexed whole, so "irreversible" -- which exists in the lexicon
   only inside "irreversible pulpitis" -- was unreachable from a misrecognized
   "reversible". Now component words are indexed too.
2. Global score ranking let one token monopolize every slot: "tooth" matches
   "root" at 0.67 and buried "mutual" -> "mesial" at 0.50. Now round-robin, so
   each suspicious word contributes its best candidate first.
3. Chairside vocabulary ("open", "wider") matched clinical terms and filled the
   list with noise. Stopworded.

"mutual" -> "mesial" is still not reachable by string similarity -- "mouth" and
"gutta" both score *higher*, and phonetic matching does not help either
(Metaphone MTL vs MSL). That correction has to come from the model's contextual
knowledge that "mesial occlusal distal" is canonical. Worth knowing where the
technique's limit is.

### The error class correction cannot touch

"reversible pulpitis" for "irreversible pulpitis" is correctly spelled, in the
lexicon, and names the opposite treatment decision. Similarity search finds
nothing because nothing is wrong with the string. Worse, the "skip known-good
words" optimization actively blinded the system to it.

Handled with a curated confusion list and, deliberately, **flagging rather than
correcting**. When asked, the model declined to flip the diagnosis -- correctly.
Inferring a diagnosis inversion from context is a clinical judgement, and an LLM
silently making it is worse than the original error. Flags point; humans decide.

### Streaming

First partial 1.30s, first confirmed 4.24s, finalization 1.2s after the
recording ended (15.9s audio written in real time).

The two tiers justified themselves immediately: live output read
"mesialocleucle distal" where the final full-context pass produced "mesial
occlusal, distal". Live is fast and rough; final is authoritative.

### Also

- Tooth notation across Universal/FDI/Palmer, verified against anatomy. The
  collision is real: FDI 18 is Universal 1, Universal 18 is FDI 37.
- mypy strict clean across 35 files; py.typed added.
- Renamed the "null" backend to "passthrough" -- `backend: null` in YAML parses
  as None, not the string.
- A bug found only because the status question prompted actually running mypy:
  `av.AVError` does not exist in PyAV 12+, so every corrupt-audio handler raised
  AttributeError instead of AudioError. Neither existing error test reached
  PyAV. Documenting a command is not the same as running it.

### Next

Silero VAD, Parakeet ONNX (the fast Windows CPU path), lexicon build from
MeSH/RxNorm/UMLS. Diarization is written but unverified -- pyannote is HF-gated
and needs a login before it loads at all.

Still blocking on real data: every number above comes from synthetic TTS. No
handpiece noise, no masks, no crosstalk, no disfluency.

## 7/16/2026
