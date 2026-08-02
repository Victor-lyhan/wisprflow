# flowscribe

Modular, fully-local speech-to-text for dental clinical audio. Audio in, speaker-labeled transcript out — no paid APIs, no patient data leaving the machine.

Built to be embedded: install the core, add only the engines you need, and swap any stage without touching the rest.

## Why this exists

General-purpose ASR handles dental dictation badly, and it fails in a way that ordinary metrics hide. A 2026 evaluation of 11 ASR systems on orthodontic clinical records found every system was significantly worse on clinical vocabulary than on general speech (*P* < 0.001) — with one exception, the configuration that added an LLM correction pass, which was also the most accurate overall. The same study found clinically significant errors in **every** system tested, ranging 2%–66%.

That shapes three decisions here:

1. **ASR plus a constrained LLM correction stage**, not ASR alone.
2. **Domain WER is the headline metric**, not overall WER.
3. **Verbatim and corrected transcripts are both kept, with a diff** — so corrections are auditable and human verification stays possible.

Reproduced on this repo's own smoke set: overall WER 9.57%, domain WER 18.52% — the clinical vocabulary fails at roughly twice the rate the headline number suggests.

## Install

```bash
uv venv --python 3.11
uv pip install -e ".[whisper,eval]"
```

Extras are per-backend: `whisper`, `parakeet`, `vad`, `diarize`, `llm`, `mlx`, `denoise`, `eval`, `dev`. The core install is light — engines are optional and load lazily.

## Use

```bash
flowscribe fetch-models --asr large-v3      # provisioning; the only step that needs network
flowscribe transcribe visit.wav --format text
flowscribe transcribe live.wav --live       # a WAV that is still being recorded
flowscribe eval data/gold/manifest.jsonl    # WER, domain WER, tooth accuracy
```

```python
from flowscribe import Config, transcribe_file

result = transcribe_file("visit.wav", Config())
print(result.verbatim.text)          # raw ASR
print(result.best.text)              # corrected, when correction is enabled
for edit in result.edits:            # every change, auditable
    print(f"{edit.kind}: {edit.before!r} -> {edit.after!r}")
```

## Design

```
audio → preprocess → VAD → ASR → streaming policy → diarize → LLM correct → sink
```

Each stage is a `Protocol` with backends resolved by name through entry points. Registering your own needs no fork:

```toml
[project.entry-points."flowscribe.asr"]
my-engine = "my_package.engines:MyEngine"
```

A shared contract test suite verifies backends are genuinely interchangeable — the bundled faster-whisper engine and the test fake pass the same assertions.

**Two-tier output.** Transcribing during recording conflicts with diarization, which needs the whole recording to cluster speakers. So there are two tiers: a *live* tier (~2–4 s latency, provisional speaker labels) and a *final* tier produced when recording ends (full-context ASR, offline diarization) that is authoritative. Batch is modeled as a degenerate case of streaming, so there is only one code path.

**Runs on Windows.** Development is on Apple Silicon, production on Windows with no guaranteed GPU. Every default is portable — CTranslate2 and ONNX Runtime for ASR, PyAV for decoding (no external ffmpeg install), llama.cpp for the corrector. ONNX execution providers are auto-detected: CUDA → DirectML → CPU.

## Handling patient audio

`offline_only` defaults to **true**. Models are fetched in an explicit provisioning step and pinned; at inference time the model libraries run offline, telemetry is disabled, and a non-loopback correction endpoint is rejected outright. A test runs the full pipeline with every outbound socket refused, so the guarantee is enforced rather than documented.

This is a drafting aid that requires clinician review, not unattended documentation — see the error rates above.

## Licensing note

CDT procedure codes and SNODENT are American Dental Association copyright and require a paid commercial license to redistribute. A practice may use CDT freely in its own records, but this package never bundles either. The included lexicon is generic clinical terminology from freely redistributable sources; practices supply their own code list at runtime via `CorrectionConfig.user_codes_path`.

## Status

Phase 0 complete — contracts, registry, config, audio ingestion, faster-whisper backend, CLI, sinks, evaluation harness, 175 tests. Streaming policy, VAD, diarization, and the LLM corrector are next. See `Development.md`.
