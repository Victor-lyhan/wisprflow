# Testing on Windows

Windows is the production target, but development happens on macOS. Nothing here
has been run on Windows yet — this document is the plan for closing that gap, and
it flags where breakage is most likely rather than claiming it will work.

There are two ways to get Windows results. Use both: CI catches regressions on
every push, a real machine catches things CI cannot (GPU, microphone, a clinic's
actual audio hardware).

---

## 1. CI — no Windows machine required

`.github/workflows/ci.yml` runs the suite on `windows-latest` alongside macOS and
Linux, on every push. Push the branch and read the results:

```bash
git push -u origin phase-0-pipeline-skeleton
gh run watch          # or the Actions tab
```

Two jobs:

| job | what it proves |
|---|---|
| `test` (3 OSes × 2 Pythons) | Lint, mypy, and 270 tests pass on Windows |
| `models` (Windows only) | CTranslate2 and PyAV **wheels actually resolve and run** on Windows, and a real file transcribes end to end |

The second job matters more than it looks. Most Windows breakage in a Python
audio stack is not logic — it is a wheel that does not exist for the platform, or
a native library that needs a Visual C++ runtime.

## 2. On a real Windows machine

```powershell
winget install --id=astral-sh.uv -e
git clone https://github.com/Victor-lyhan/wisprflow.git
cd wisprflow

uv venv --python 3.11
uv sync --extra dev --extra whisper --extra eval --extra llm

uv run pytest -q                          # full suite, including model-backed tests
uv run mypy src/
uv run flowscribe backends

# Regenerate the smoke set locally -- this uses Windows SAPI, no install needed
uv run python scripts/make_smoke_dataset.py --out data/smoke
uv run flowscribe eval data/smoke/manifest.jsonl --model tiny.en --language en --allow-network
```

Expect **different numbers than macOS**: SAPI voices are not the macOS voices, so
the WER is not comparable across platforms. Compare Windows-to-Windows over time,
not Windows-to-macOS.

### With a GPU

```powershell
uv pip install "onnxruntime-gpu"      # NVIDIA / CUDA
uv pip install "onnxruntime-directml" # any GPU: NVIDIA, AMD, Intel
```

`device: auto` picks CUDA when torch reports it, otherwise CPU. DirectML applies
to the Parakeet ONNX backend, which **is not implemented yet** — until then, GPU
acceleration on Windows only reaches faster-whisper via CUDA.

### LLM correction

```powershell
winget install Ollama.Ollama
ollama pull qwen3:8b
uv run flowscribe transcribe visit.wav -c configs/ollama.yaml --correct --show-edits
```

On a CPU-only box use `qwen3:4b` instead — the 8B model will run but correction
will dominate wall-clock time.

---

## Where this is most likely to break

Listed in rough order of risk. Each is a specific thing to check, not a general
worry.

**1. `GrowingWavSource` file-sharing semantics.** The highest-risk item by some
margin. It polls a WAV while another handle is still writing it. Windows file
locking is stricter than POSIX: CPython's `open()` permits shared reads by
default, so this *should* work, but the entire live-transcription path depends on
it and it has never been exercised there. `tests/test_audio.py::TestGrowingWavSource`
covers it — watch that class specifically.

**2. Hugging Face cache symlinks.** The HF cache uses symlinks, which on Windows
need Developer Mode or an elevated shell. Without it, files are copied instead —
functional but larger on disk, with a warning per download. Either enable
Developer Mode or set:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

**3. Long paths.** Model cache directories nest deeply and can exceed the legacy
260-character `MAX_PATH`. If downloads fail with path errors:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1
```

**4. PyAV wheels.** Chosen specifically so Windows needs no separate ffmpeg
install and no PATH configuration — the codecs ship inside the wheel. If the
`models` CI job fails at import, that assumption is wrong and is worth knowing
early.

**5. `pyannote` / torch.** The largest install by far and the most likely to be
awkward. Also still unverified anywhere, since the models are HF-gated (see
below).

**6. Line endings.** `.gitattributes` pins `.txt` and `.jsonl` to LF, so the seed
lexicon, confusion sets, and dataset manifests parse identically on both
platforms. A lexicon test failing on Windows and nowhere else would still point
here first.

---

## Not yet possible to test anywhere

- **Diarization.** `pyannote/speaker-diarization-3.1` is gated on Hugging Face.
  It needs a one-time `huggingface-cli login` plus accepting the model conditions
  on its model page before the pipeline will load at all. Until then the backend
  is written but unrun, on every platform.
- **Real accuracy.** Every number in this repo comes from synthetic speech. No
  handpiece whine, no suction, no masks, no crosstalk, no disfluency.
