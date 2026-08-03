# Testing on Windows

Windows is the production target, development happens on macOS. CI now runs the
suite on `windows-latest` on every push, so this is a record of what has actually
been verified there rather than a list of hopes.

**Verified on Windows** (GitHub Actions, `windows-latest`): the full test suite,
mypy strict, lint, and — in the `models` job — real faster-whisper and PyAV
wheels installing and transcribing a file end to end.

**Not verified anywhere**: GPU paths, microphone capture, real clinic audio
hardware, and diarization (HF-gated). A CI runner has no GPU and no sound card.

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

**1. ~~`GrowingWavSource` file-sharing semantics.~~ RESOLVED.** This was the
highest-risk item: it polls a WAV while another handle writes it, and Windows
file locking is stricter than POSIX. `tests/test_audio.py::TestGrowingWavSource`
passes on `windows-latest`, so CPython's default shared-read behaviour does hold
and the live-transcription path works there.

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

**4. ~~PyAV wheels.~~ RESOLVED.** Chosen so Windows needs no separate ffmpeg
install and no PATH configuration. The `models` job decodes and transcribes a
real file on Windows, so the assumption holds.

**5. `pyannote` / torch.** The largest install by far and the most likely to be
awkward. Also still unverified anywhere, since the models are HF-gated (see
below).

**6. Shell portability in CI.** PowerShell is the default shell on Windows
runners and does not expand globs into arguments the way bash does —
`--with dist/*.whl` reached uv verbatim and failed. Any workflow step relying on
shell expansion needs an explicit `shell: bash`.

**7. Line endings.** `.gitattributes` pins `.txt` and `.jsonl` to LF, so the seed
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
- **GPU acceleration.** CI runners have no GPU, so the CUDA and DirectML provider
  paths in `parakeet_onnx.py` are unexercised. They are selected by
  `_resolve_providers` and will only be proven on real hardware.

## What CI caught that local development could not

Worth recording, because each was invisible on the development machine:

1. **`.gitignore` was excluding source.** Unanchored `data/` and `audio/`
   patterns also matched `src/flowscribe/audio/` and
   `src/flowscribe/dental/data/`, so the audio package and the entire lexicon
   were never committed. Editable installs import from the working tree, so all
   270 local tests passed against files no user would receive.
2. **A drifted venv hid 11 mypy errors.** A clean `uv sync` resolves numpy 1.26
   whose stubs require `ndarray` type arguments; the local venv had drifted to
   2.4. Fixing the annotations then exposed a genuinely wrong one.
3. **`from tests.conftest import ...`** worked only because `sys.path` happened
   to contain the repository root.
