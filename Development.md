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

## 8/2/2026 (later still) — Phase 4, all backends built

### Systematic comparison, 10 synthetic samples, 68 domain terms

| engine | WER | DWER | terms | tooth | RTF | ins |
|---|---|---|---|---|---|---|
| whisper-tiny | 8.51% | 11.76% | 60/68 | 100% | 0.047 | - |
| whisper-tiny+llm | 3.72% | 0.00% | 68/68 | 100% | 0.240 | 4.0% |
| whisper-small | 8.51% | 14.71% | 58/68 | 100% | 0.319 | - |
| whisper-small+llm | 5.32% | 4.41% | 65/68 | 100% | 0.319 | 4.1% |
| parakeet-v3 | 7.98% | 8.82% | 62/68 | 100% | 0.076 | - |
| **parakeet-v3+llm** | 5.85% | **2.94%** | 66/68 | 100% | **0.161** | 2.6% |

Four things fall out of this:

1. **Correction beats model size, decisively.** tiny+llm (DWER 0.00%) beats small
   alone (14.71%). Spending compute on the correction stage buys far more
   clinical accuracy than spending it on a bigger recognizer.
2. **whisper-small scored *worse* than whisper-tiny on domain terms** (14.71% vs
   11.76%) while tying on overall WER. On 68 terms that is 2 terms of difference
   and could easily be noise -- but it is a reminder that general-purpose model
   size does not predict clinical vocabulary accuracy.
3. **Parakeet is the best uncorrected engine and the best production choice.**
   8.82% DWER against whisper-small's 14.71%, at a quarter the RTF. With
   correction: 2.94% DWER at RTF 0.161 -- the pick for a Windows CPU box.
4. **Insertion rate 2.6-4.1%.** The corrector does add some content. Parakeet+llm
   is lowest. Non-zero means the guard is doing real work and warrants watching.

Treat tiny+llm's 0.00% as the small-sample artifact it probably is, not a claim.

### Built

Parakeet ONNX (production engine), Silero VAD (gates the streaming decode),
lexicon build from MeSH + RxNorm (286 -> 787 terms), benchmark script, Windows
CI matrix, cross-platform TTS.

### Bugs found by running rather than reading

- **CoreML auto-selection was 45x slower than CPU.** It supports 1012 of 2115
  nodes, so the graph fragments into ~291 partitions and transfer overhead
  dominates. RTF 3.69 vs 0.082. Removed from auto-detect.
- **Whitespace tokens are word boundaries.** Dropping them produced "number3" and
  "tooth number19" -- which then defeated tooth extraction entirely.
- **`path` means opposite things in onnx-asr and faster-whisper.** One is a
  pre-staged weights directory, the other a download cache.

### Still outstanding

Diarization remains unverified anywhere -- pyannote is HF-gated. Windows is
untested until CI runs. And every number above is synthetic speech.

## 8/2/2026 (evening) — Windows verified

CI green on all seven jobs: windows/macos/ubuntu x py3.11/3.12, plus a
Windows-only job running real faster-whisper and PyAV wheels end to end.

**Verified on Windows**: the full suite, mypy strict, lint, and real model
inference. Specifically resolved: `GrowingWavSource` handles Windows file-sharing
(the highest-risk item -- it polls a WAV while another handle writes it), and
PyAV wheels decode without a separate ffmpeg install.

**Still unverified anywhere**: GPU paths (CI runners have none), microphone
capture, diarization (HF-gated), and real clinic audio.

### Three bugs CI caught that were structurally invisible locally

1. **`.gitignore` was eating source code.** Unanchored `data/` and `audio/`
   patterns -- written to keep patient recordings out -- also matched
   `src/flowscribe/audio/` and `src/flowscribe/dental/data/`. The audio package
   and the entire lexicon were never committed. All 270 local tests passed
   because an editable install imports from the working tree, so they were
   testing files no user would ever receive. Ruff and the formatter had also been
   silently skipping the package for the same reason.
2. **A drifted venv hid 11 mypy errors.** A clean `uv sync` resolves numpy 1.26,
   whose stubs require type arguments on `ndarray`; the local venv had drifted to
   2.4. Annotating properly then exposed a genuinely wrong annotation:
   `to_float32_mono` and `PCMResampler.push` were typed as taking float32 but
   accept int16/int32 and convert.
3. **`from tests.conftest import ...`** resolved only because sys.path happened
   to contain the repo root.

Plus one in the workflow itself: PowerShell is the default shell on Windows
runners and does not expand `dist/*.whl`, so uv received the glob verbatim.

The common thread is that a local development environment cannot test what it
imports around. Editable installs, a warm venv, and an accidentally-correct
sys.path each hid a real defect. Nothing short of a clean checkout on another
machine would have found them.

## 7/16/2026
