#!/usr/bin/env python3
"""Generate a synthetic smoke dataset for exercising the evaluation harness.

Cross-platform: macOS (say), Windows (SAPI), Linux (espeak-ng). Deliberately so
-- production is Windows, and an operator who cannot regenerate the test audio
cannot check whether a change broke anything.

WHAT THIS IS NOT
================
This is not the dental gold set, and numbers from it must never be quoted as
accuracy figures. macOS ``say`` produces clean, perfectly articulated,
single-speaker audio with no handpiece whine, no suction, no surgical mask
attenuation, no crosstalk, and no spontaneous-speech disfluency. Published dental
ASR work found background noise substantially increases error rates, so real
operatory audio will score far worse than this.

WHAT IT IS FOR
==============
Verifying that the harness itself works end to end -- manifest loading, scoring,
domain WER, tooth extraction, aggregation -- without waiting on human recording
sessions. Because the script is the reference, ground truth is exact.

The real gold set replaces this: the same scripts read by several people in an
operatory, which is what makes the reference exact there too, at near-zero
labelling cost.

Usage:
    python scripts/make_smoke_dataset.py --out data/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tts import UNUSABLE_VOICES, available_voices, backend_name, synthesize  # noqa: E402

# Scenarios chosen to stress what general ASR gets wrong in dentistry: surface
# abbreviations, tooth numbers in both spoken and digit form, charted number
# runs, anaesthetic concentrations, and drug names.
SCRIPTS: list[tuple[str, str]] = [
    (
        "restorative-01",
        "Tooth number three has a mesial occlusal distal composite restoration. "
        "The margins are intact and occlusion checks within normal limits.",
    ),
    (
        "perio-01",
        "Probing depths on the buccal of tooth number fourteen are three, two, three. "
        "Lingual readings are four, three, four with bleeding on probing.",
    ),
    (
        "anesthesia-01",
        "Administered two carpules of lidocaine two percent with epinephrine "
        "one to one hundred thousand via inferior alveolar nerve block.",
    ),
    (
        "endo-01",
        "Tooth number nineteen presents with irreversible pulpitis and periapical "
        "radiolucency. Recommend root canal therapy followed by a crown.",
    ),
    (
        "extraction-01",
        "Extracted tooth number seventeen. The patient tolerated the procedure well. "
        "Hemostasis achieved and post operative instructions given.",
    ),
    (
        "hygiene-01",
        "Completed prophylaxis with scaling and fluoride varnish. Moderate calculus "
        "on the mandibular anterior lingual surfaces. Generalized gingivitis noted.",
    ),
    (
        "prosth-01",
        "Crown preparation on tooth number thirty. Impression taken with polyvinyl "
        "siloxane. Temporary cemented and occlusion verified.",
    ),
    (
        "perio-02",
        "Scaling and root planing completed on the upper right quadrant. Furcation "
        "involvement noted on the buccal of tooth number two.",
    ),
    (
        "exam-01",
        "Caries detected on the distal of tooth number twelve and the occlusal of "
        "tooth number thirty one. Bitewing radiographs confirm interproximal lesions.",
    ),
    (
        "ortho-01",
        "Bonded brackets on the maxillary arch. Initial archwire placed with elastics. "
        "Overjet measures five millimeters with moderate crowding.",
    ),
]


def generate(out_dir: Path, voices: list[str], rate: int) -> list[dict]:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for sample_id, text in SCRIPTS:
        # Cycle voices so the set is not overfitted to one timbre. Not a
        # substitute for real speakers.
        voice = voices[len(records) % len(voices)] if voices else None
        wav = audio_dir / f"{sample_id}.wav"

        try:
            synthesize(text, wav, voice=voice, rate=rate)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  ! {sample_id}: {exc}", file=sys.stderr)
            continue

        records.append(
            {
                "id": sample_id,
                "audio": f"audio/{wav.name}",
                "reference": text,
                "language": "en",
                "speakers": 1,
                "metadata": {"voice": voice, "synthetic": True, "backend": backend_name()},
            }
        )
        print(f"  + {sample_id:<16} [{voice or 'default'}]")

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/smoke"))
    parser.add_argument("--rate", type=int, default=180, help="Speaking rate, words per minute.")
    args = parser.parse_args()

    if backend_name() is None:
        print(
            "error: no speech synthesizer available. macOS needs 'say', Windows "
            "needs PowerShell, Linux needs espeak-ng.",
            file=sys.stderr,
        )
        return 1

    voices = available_voices()
    banned = UNUSABLE_VOICES & set(voices)
    if banned:
        print(
            f"error: legacy formant-synthesis voice(s) selected: {', '.join(sorted(banned))}. "
            "Their output is unintelligible and produces meaningless error rates.",
            file=sys.stderr,
        )
        return 1

    print(f"Generating {len(SCRIPTS)} samples into {args.out} via {backend_name()} ...")
    records = generate(args.out, voices, args.rate)
    if not records:
        print("error: no samples generated", file=sys.stderr)
        return 1

    manifest = args.out / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {manifest} ({len(records)} samples)")
    print("\nSYNTHETIC AUDIO -- validates the harness, not real-world accuracy.")
    print(f"\n  flowscribe eval {manifest} --model small.en --language en --allow-network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
