# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`flowscribe` — a modular, fully-local speech-to-text pipeline for dental clinical audio. Audio in, speaker-labeled transcript out. No paid APIs, no PHI leaving the machine.

Development happens on macOS (Apple Silicon); **production targets Windows**, GPU not guaranteed. That constraint drives most backend choices: MLX is Apple-only and can never be the production path, and CUDA-only tooling (NeMo, vLLM) is off the table. Portable engines only — CTranslate2, ONNX Runtime, llama.cpp.

## Commands

```bash
uv venv --python 3.11
uv pip install -e ".[dev,whisper,eval]"      # extras are per-backend; see pyproject

pytest                                        # full suite
pytest -m "not slow"                          # skips tests that load real models
pytest tests/test_metrics.py::TestDomainWordErrorRate -v
ruff check src/ tests/ && ruff format src/ tests/
mypy src/

flowscribe backends                           # list registered backends per stage
flowscribe transcribe visit.wav --format text
flowscribe transcribe live.wav --live         # WAV still being written
flowscribe fetch-models --asr large-v3        # provisioning; the only online step
flowscribe eval data/smoke/manifest.jsonl --model tiny.en --language en --allow-network

python scripts/make_smoke_dataset.py --out data/smoke   # macOS only, synthetic
```

`--allow-network` is required for anything that downloads. `offline_only` defaults to **true** because this handles patient audio; inference must never reach the network.

## Architecture

```
audio → preprocess → VAD → ASR → streaming policy → diarize → LLM correct → sink
```

Every stage is a `Protocol` in `protocols.py`, resolved by name through `registry.py` via `importlib.metadata` entry points. Backends declare themselves in `pyproject.toml`; **external projects register their own the same way**, which is the mechanism behind "pluggable into other projects". Resolution is lazy — listing backends never imports them, so uninstalled extras cost nothing.

Key files:
- `contracts.py` — all data types. Hot-path types (`AudioChunk`, `VadSegment`) are slotted dataclasses; serialized types (`Utterance`, `Transcript`, events) are pydantic. Do not convert one family to the other.
- `pipeline.py` — stage wiring and execution. Currently implements the final tier only.
- `normalize.py` — top-level, **not** under `evaluation/`. Both the dental layer and the eval harness depend on it; nesting it caused a circular import.

### Two-tier output

"Start transcribing while recording" conflicts with diarization, which needs the whole recording to cluster speakers. Resolved by two tiers:

- **Live** — provisional, ~2–4 s latency, online speaker labels that may be revised.
- **Final** — authoritative, full-context ASR plus offline diarization, produced when recording ends.

`Transcript.tier` records which. The final tier *is* the offline pipeline, so batch work is not throwaway. **Batch is a degenerate case of streaming**: a finished-file `AudioSource` just yields its chunks and sets `is_last`. There is no separate batch path — don't add one.

### Verbatim and corrected are both retained

`TranscriptionResult` carries `verbatim`, `corrected`, and `edits`. The corrector returns a diff, and every change is auditable. Published dental ASR evaluation found clinically significant errors in **every** system tested (2%–66%), so an LLM-rewritten clinical transcript without a diff is not verifiable. Keeping both is also what makes correction *measurable* rather than assumed.

## Working on this codebase

**Domain WER, not WER, is the metric.** Overall WER is dominated by ordinary words and hides clinical-vocabulary failure — the published finding, and reproduced here: on the smoke set, WER 9.57% against DWER 18.52%. When changing anything that could affect accuracy, run `flowscribe eval` and compare DWER.

**Normalization is part of the metric.** `normalize.py` decides what counts as an error. Two rules there are load-bearing and easy to break:
- Number words fold to digits, but compounding only happens with an explicit multiplier ("hundred"/"thousand"). `"three two three"` must stay three probing depths, never `323`. There are regression tests for this.
- Number words adjacent to punctuation (`"nineteen."`) must still map, or tooth extraction silently misses sentence-final references.

**Tooth extraction is deliberately conservative.** Only explicitly marked forms (`tooth 14`, `#14`, `tooth number fourteen`). A bare number in dental speech is usually a probing depth or a carpule count. Over-extraction inflates the tooth metric in the flattering direction.

**Never bundle CDT or SNODENT.** Both are ADA copyright requiring a paid commercial license to redistribute, even though practices may use CDT freely in their own records. `dental/data/seed_lexicon.txt` is generic terminology only; practices supply codes at runtime via `CorrectionConfig.user_codes_path`. There is a test asserting no CDT codes are present.

**The null diarizer leaves `speaker` as `None`, not `SPEAKER_00`.** A fabricated single label is indistinguishable downstream from a genuine single-speaker result, and would misattribute the patient's words to the dentist.

**Adding an ASR backend**: implement the `ASREngine` protocol, register an entry point under `flowscribe.asr`, then subclass `ASREngineContract` in `tests/test_contracts_and_sinks.py`. That suite is what makes interchangeability a checked property — the real faster-whisper engine and the test fake pass identically.

**Synthetic audio measures the harness, not accuracy.** `scripts/make_smoke_dataset.py` uses macOS `say`. It has no handpiece noise, no masks, no crosstalk, no disfluency. Never quote its numbers as accuracy figures. It also refuses legacy formant-synthesis voices (Fred, Kathy, Zarvox…) — those produced a 42% WER that measured the *synthesizer*, not the recognizer.

### Correction is guarded, flagging is not correction

`correct/guard.py` is safety-critical: it stands between LLM output and a patient record. Three checks reject a proposal — edit ratio, numeric token count, tooth reference count. Values may change (correcting a misheard tooth number is the point); counts may not.

Separately, `dental/review.py` **flags** clinically confusable terms without changing them. This handles what correction structurally cannot: a misrecognition landing on *another valid clinical term* ("reversible pulpitis" for "irreversible pulpitis"). Nothing looks wrong, so similarity search can't find it — and an LLM inverting a diagnosis on inference is worse than the original error. Flags point; humans decide. Never make flagging auto-correct.

### Tooth notation is never inferred

`dental/teeth.py` converts between Universal, FDI, and Palmer explicitly. "Tooth 18" is the upper-right third molar in FDI and the lower-left second molar in Universal — opposite corners of the mouth. Guessing notation from context would silently record the wrong tooth.

## Status

Working end to end: contracts, registry, config, audio ingestion (file / growing-WAV / queue), faster-whisper ASR, **streaming with LocalAgreement-2**, **LLM correction via Ollama**, review flagging, tooth notation, CLI, sinks, eval harness. 270 tests, mypy strict clean.

Measured: domain WER 15.38% → 3.08% with correction; first partial at 1.30 s and first confirmed text at 4.24 s on a growing file.

Implemented but **unverified**: `diarize/pyannote_diarizer.py` — pyannote models are HF-gated, so it needs `huggingface-cli login` plus accepting the model conditions before it can run at all.

Not built: Silero VAD backend, Parakeet ONNX backend (the fast Windows CPU path), lexicon build from MeSH/RxNorm/UMLS, FastAPI service, Windows testing. See `Development.md`.
