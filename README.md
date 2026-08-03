# flowscribe

Local speech-to-text for dental clinical audio. Speak, watch text appear, get a corrected transcript. Nothing leaves the machine and nothing costs money.

```bash
uv pip install -e ".[whisper,mic,ui]"
flowscribe ui --allow-network        # http://127.0.0.1:8000
```

---

## Why it exists

General-purpose speech recognition handles dental dictation badly, and it fails in a way ordinary metrics hide. A 2026 evaluation of 11 ASR systems on orthodontic records found every system was significantly worse on clinical vocabulary than on general speech (*P* < 0.001) — with one exception: the configuration that added an LLM correction pass, which was also the most accurate overall. The same study found clinically significant errors in **every** system tested, ranging 2%–66%.

Reproduced here, then fixed:

| | overall WER | **domain WER** |
|---|---|---|
| verbatim | 9.57% | **15.38%** |
| corrected | 5.32% | **3.08%** |

Clinical vocabulary fails at roughly twice the headline rate. With correction, domain WER drops ~80% relative and falls *below* general WER — the same inversion the study reported.

That drives three decisions: **domain WER is the metric**, **correction is a separate constrained stage**, and **both transcripts are kept with a diff** so corrections stay auditable.

---

## How it works

```
microphone ─┐
audio file ─┼─→ resample 16k mono ─→ VAD ─→ ASR ─→ LocalAgreement-2 ─→ live text
growing WAV─┘                                            │
                                                         └─→ on stop ─→ full-context ASR
                                                                     ─→ LLM correction
                                                                     ─→ review flags
                                                                     ─→ final transcript
```

### Two tiers, and why

A recognizer revises its output as more context arrives. Emitting every hypothesis directly produces text that visibly rewrites itself — and can briefly display a wrong drug name or tooth number before correcting.

- **Live tier** — text is held back until **two successive decodes agree** on it (LocalAgreement-2). First partial ~1.3 s, first confirmed ~4.2 s. Explicitly provisional.
- **Final tier** — on stop, the complete audio is re-transcribed with full context, then corrected. This is the record.

The difference is real. On the same recording, live produced `"mesialocleucle distal"` where the final pass produced `"mesial occlusal, distal"`.

The audio buffer scrolls past confirmed text, so decode cost stays bounded over a long appointment instead of growing with it.

### The dental layer

**787-term lexicon** built from MeSH (public domain) and RxNorm (open API), merged with a hand-written seed list. Regenerate with `python scripts/build_lexicon.py`.

**Correction is constrained, not free rewriting.** The model gets one utterance plus a short list of candidate terms chosen by similarity — given `"buckle"` it sees `"buccal"`. Three guards reject a proposal before it can reach a record:

| guard | rejects |
|---|---|
| edit ratio | rewriting >25% of an utterance — that's paraphrasing, not term repair |
| numeric token count | inventing or dropping a number; values may change, the count may not |
| tooth reference count | adding a tooth that was never dictated |

Every accepted change is recorded as an `Edit` with before/after, so the diff is reviewable.

**Review flags handle what correction cannot.** Some misrecognitions land on *another valid clinical term*: `"reversible pulpitis"` for `"irreversible pulpitis"` is correctly spelled, in the lexicon, and names the opposite treatment. Nothing about the string looks wrong, so similarity search can't find it — and an LLM inverting a diagnosis on inference is worse than the original error. These are **flagged for a human, never auto-corrected**.

Flags are high-severity only by default. Directional pairs like mesial/distal are genuinely confusable but appear in nearly every note; flagging them every time buries the rare flag that matters.

**Tooth notation** converts across Universal, FDI and Palmer explicitly, never inferred — "tooth 18" is the upper-right third molar in FDI and the lower-left second molar in Universal.

**Number handling** folds words to digits, but only compounds with an explicit multiplier: `"one hundred thousand"` = `100000`, while `"three two three"` stays three probing depths rather than collapsing to `323`.

---

## Using it

### Browser UI

```bash
flowscribe ui --allow-network            # first run downloads models
flowscribe ui -c configs/demo.yaml       # with LLM correction
```

Pick an input device (built-in mic, AirPods, USB interface), press start, speak. Live text on the left; the final transcript lands in an **editable textbox** with **Download .txt**, and the raw JSON sits beside it with **Download .json**.

### Terminal

```bash
flowscribe listen --list-devices
flowscribe listen --device 2 -o visit.json
flowscribe transcribe visit.wav --format text
flowscribe transcribe live.wav --live          # a WAV still being written
flowscribe eval data/smoke/manifest.jsonl      # WER, domain WER, tooth accuracy
```

`listen` emits one JSON object per line:

```json
{"type":"partial","provisional":true,"start":0.0,"end":2.1,"text":"tooth number three has a"}
{"type":"final","provisional":false,"id":"u00000","start":0.0,"end":4.2,"text":"..."}
{"type":"complete","text":"...","verbatim":{...},"corrected":{...},"edits":[],"flags":[]}
```

---

## Embedding in another project

The core is a library; the CLI and UI are thin wrappers over it.

```python
from flowscribe import Config, transcribe_file

result = transcribe_file("visit.wav", Config())
result.verbatim.text      # raw ASR
result.best.text          # corrected when correction is enabled
result.edits              # every change, with before/after
result.flags              # spans a human should verify
```

Live, with your own audio source:

```python
from flowscribe import Config
from flowscribe.pipeline import Pipeline
from flowscribe.audio import MicrophoneSource   # or QueueAudioSource for a socket feed

with Pipeline(Config()) as pipeline, MicrophoneSource() as mic:
    for event in pipeline.stream(mic):
        ...   # PartialUtterance | FinalUtterance | TranscriptComplete
```

### Swapping a stage

Every stage is a `Protocol` resolved by name through entry points. Registering your own needs no fork:

```toml
[project.entry-points."flowscribe.asr"]
my-engine = "my_package.engines:MyEngine"
```

| stage | protocol | bundled backends |
|---|---|---|
| source | `AudioSource` | file, growing WAV, microphone, push queue |
| VAD | `VAD` | `silero` |
| ASR | `ASREngine` | `parakeet-onnx`, `faster-whisper` |
| policy | `StreamingPolicy` | LocalAgreement-N |
| correction | `Corrector` | `ollama`, `passthrough` |
| sink | `Sink` | json, jsonl, text, srt, vtt |

A shared contract suite (`tests/test_contracts_and_sinks.py`) verifies backends are genuinely interchangeable — the real faster-whisper engine and the test fake pass identical assertions. Subclass `ASREngineContract` and your backend is held to the same bar.

Install only what you use: `whisper`, `parakeet`, `vad`, `mic`, `llm`, `ui`, `eval`, `dev`. The core is light and engines load lazily, so an uninstalled extra costs nothing.

---

## Migrating to another computer

Verified, not assumed — a wheel was built, installed into a clean environment with no source tree, and used to transcribe a file:

```bash
uv build --wheel                                   # 88 KB
uv pip install "flowscribe-0.1.0-py3-none-any.whl[whisper,mic,ui]"
flowscribe fetch-models --asr large-v3             # one-time, ~2.4 GB cached
flowscribe transcribe visit.wav --format text      # runs offline from here
```

Two things move: the **wheel** (88 KB) and the **model cache** (~2.4 GB in `~/.cache/huggingface`). Copy the cache directly to skip re-downloading, or let `fetch-models` pull it once.

**Runs on Windows.** Development is macOS, production Windows with no guaranteed GPU, so every default is portable — CTranslate2 and ONNX Runtime for recognition, PyAV for decoding (no separate ffmpeg install), llama.cpp via Ollama for correction. CI runs the full suite on Windows, macOS and Linux across Python 3.11 and 3.12, plus a Windows job that installs the real model wheels and transcribes a file end to end. See `docs/windows.md`.

ONNX execution providers are auto-detected: CUDA → DirectML (any Windows GPU) → CPU.

---

## Cost and privacy

**No paid services.** The only network addresses anywhere in the source are `127.0.0.1` and `localhost`. No API keys, no accounts, no metered calls. Every model is open-weight:

| model | role | licence |
|---|---|---|
| Parakeet-TDT-0.6b-v3 | recognition | CC-BY-4.0 |
| Whisper (faster-whisper) | recognition | MIT |
| Silero VAD | speech detection | MIT |
| Qwen3 | correction (optional) | Apache-2.0 |

**`offline_only` defaults to true.** Models are fetched in an explicit provisioning step; at inference the model libraries run offline, telemetry is disabled, and a non-loopback correction endpoint is rejected outright. A test runs the whole pipeline with every outbound socket refused, so the guarantee is enforced rather than merely documented.

**CDT and SNODENT are excluded.** Both are American Dental Association copyright and require a paid licence to redistribute. A practice may use CDT freely in its own records, so supply your own list via `correction.user_codes_path`.

---

## Performance

Measured on an M4 Max, CPU only:

| configuration | domain WER | RTF |
|---|---|---|
| parakeet-v3 | 8.82% | 0.076 |
| **parakeet-v3 + correction** | **2.94%** | 0.161 |
| whisper-small | 14.71% | 0.319 |
| whisper-small + correction | 4.41% | 0.319 |

Correction beats model size decisively: whisper-**tiny** with correction reaches 0.00% domain WER on this set, where whisper-**small** alone sits at 14.71%.

Reproduce with `python scripts/benchmark.py data/smoke/manifest.jsonl --correct`.

---

## What these numbers do and don't mean

**Every figure above comes from synthetic speech** — text-to-speech with no handpiece whine, no suction, no surgical masks, no crosstalk, no disfluency. Published work finds background noise substantially increases error rates, so **real operatory audio will score worse**. Treat the tables as a ranking of configurations, not a forecast of clinical accuracy.

Acoustic capture already shows this: the same audio played through speakers and re-captured by a microphone turned `"carpules"` into `"car peels"` and `"mesial"` into `"measly"` — before any operatory noise is involved.

Getting real numbers needs a gold set: the scripts in `scripts/make_smoke_dataset.py` read aloud by several people in an actual operatory. Because the script is the reference, that yields exact ground truth at near-zero labelling cost.

**This is a drafting aid that requires clinician review, not unattended documentation.**

---

## Development

```bash
uv venv --python 3.11
uv pip install -e ".[dev,whisper,parakeet,vad,mic,llm,ui,eval]"
pytest                       # 293 tests
pytest -m "not slow"         # skips model downloads
ruff check src/ tests/ && ruff format src/ tests/
mypy src/                    # strict
```

To reproduce a CI failure locally — a warm venv drifts from a clean resolve and hides real bugs:

```bash
git clone --branch <branch> . /tmp/citest && cd /tmp/citest
uv sync --python 3.11 --extra dev --extra whisper --extra eval --extra llm
uv run mypy src/ && uv run pytest -m "not slow" -q
```

See `CLAUDE.md` for architecture notes and `Development.md` for the decision log.
