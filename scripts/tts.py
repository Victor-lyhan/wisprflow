"""Cross-platform speech synthesis for generating test audio.

Exists so the smoke dataset can be built on the production platform, not only on
the development machine. A Windows operator who cannot regenerate the test audio
cannot check whether a change broke anything.

Backends, in order of preference per platform:

* macOS   -- ``say``, restricted to modern concatenative voices.
* Windows -- SAPI via PowerShell (``System.Speech.Synthesis``).
* Linux   -- ``espeak-ng`` if present.

None of these produce audio resembling a dental operatory. They validate the
harness; they do not measure accuracy.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["synthesize", "available_voices", "backend_name", "UNUSABLE_VOICES"]

# macOS ships legacy formant-synthesis voices whose output is barely intelligible
# to a human, let alone an ASR model. Including them once produced a 42% WER that
# measured the synthesizer rather than the recognizer -- "carpules" came back as
# "car appeals".
UNUSABLE_VOICES = frozenset(
    {
        "Albert",
        "Bad",
        "Bahh",
        "Bells",
        "Boing",
        "Bubbles",
        "Cellos",
        "Deranged",
        "Fred",
        "Good",
        "Hysterical",
        "Jester",
        "Junior",
        "Kathy",
        "Organ",
        "Superstar",
        "Ralph",
        "Trinoids",
        "Whisper",
        "Wobble",
        "Zarvox",
    }
)

_MACOS_VOICES = ["Samantha", "Daniel", "Karen", "Moira"]
# SAPI ships two voices on a default Windows install. Names vary by locale, so
# an empty string means "whatever the system default is" rather than failing.
_WINDOWS_VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
_LINUX_VOICES = ["en-us", "en-gb"]


def backend_name() -> str | None:
    """Which synthesis backend is usable here, if any."""
    system = platform.system()
    if system == "Darwin" and shutil.which("say"):
        return "say"
    if system == "Windows" and shutil.which("powershell"):
        return "sapi"
    if shutil.which("espeak-ng"):
        return "espeak-ng"
    return None


def available_voices() -> list[str]:
    """Voice names to cycle through on this platform."""
    return {
        "say": _MACOS_VOICES,
        "sapi": _WINDOWS_VOICES,
        "espeak-ng": _LINUX_VOICES,
    }.get(backend_name() or "", [])


def _say(text: str, out: Path, voice: str, rate: int) -> None:
    if voice in UNUSABLE_VOICES:
        raise ValueError(f"{voice!r} is a legacy formant voice; its output is unusable.")
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", str(out), "--data-format=LEI16@22050", text],
        check=True,
        capture_output=True,
    )


def _sapi(text: str, out: Path, voice: str, rate: int) -> None:
    # SAPI rate is -10..10 rather than words per minute; 180 wpm is roughly 0.
    sapi_rate = max(-10, min(10, round((rate - 180) / 15)))
    # Single-quote escaping for PowerShell string literals.
    safe_text = text.replace("'", "''")
    safe_voice = voice.replace("'", "''")
    script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{ $synth.SelectVoice('{safe_voice}') }} catch {{ }}
$synth.Rate = {sapi_rate}
$synth.SetOutputToWaveFile('{out}')
$synth.Speak('{safe_text}')
$synth.Dispose()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
    )


def _espeak(text: str, out: Path, voice: str, rate: int) -> None:
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", str(rate), "-w", str(out), text],
        check=True,
        capture_output=True,
    )


def synthesize(text: str, out: Path, *, voice: str | None = None, rate: int = 180) -> Path:
    """Render ``text`` to a WAV file. Raises if no backend is available."""
    backend = backend_name()
    if backend is None:
        raise RuntimeError(
            "No speech synthesizer found. macOS needs 'say', Windows needs "
            "PowerShell, Linux needs espeak-ng (apt install espeak-ng)."
        )

    voices = available_voices()
    chosen = voice or (voices[0] if voices else "")
    out.parent.mkdir(parents=True, exist_ok=True)

    {"say": _say, "sapi": _sapi, "espeak-ng": _espeak}[backend](text, out, chosen, rate)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render text to a WAV file.")
    parser.add_argument("--text", default="Tooth number three has a mesial occlusal composite.")
    parser.add_argument("--out", type=Path, default=Path("out.wav"))
    parser.add_argument("--voice", default=None)
    args = parser.parse_args()

    if backend_name() is None:
        print("error: no speech synthesizer available on this platform", file=sys.stderr)
        raise SystemExit(1)

    print(f"backend: {backend_name()}")
    print(f"wrote:   {synthesize(args.text, args.out, voice=args.voice)}")
