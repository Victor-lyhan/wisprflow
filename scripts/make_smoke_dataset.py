#!/usr/bin/env python3
"""Generate a synthetic smoke dataset for exercising the evaluation harness.

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
import shutil
import subprocess
import sys
from pathlib import Path

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

# Distinct voices stand in for speaker variation, and the accent spread
# (US/GB/AU/IE) is deliberate given multilingual deployment.
#
# Only modern concatenative voices. macOS also ships legacy formant-synthesis
# voices -- Fred, Kathy, Albert, Zarvox and friends -- whose output is barely
# intelligible to a human, let alone an ASR model. Using them produced a 42% WER
# that measured the synthesizer, not the recognizer: "carpules" came back as
# "car appeals". A generator that quietly includes them yields numbers that look
# like model failure and are nothing of the sort.
VOICES = ["Samantha", "Daniel", "Karen", "Moira"]

# Guard against reintroducing the above by name.
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


def generate(out_dir: Path, voices: list[str], rate: int) -> list[dict]:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for sample_id, text in SCRIPTS:
        voice = voices[len(records) % len(voices)]
        wav = audio_dir / f"{sample_id}.wav"

        result = subprocess.run(
            [
                "say",
                "-v",
                voice,
                "-r",
                str(rate),
                "-o",
                str(wav),
                "--data-format=LEI16@22050",
                text,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ! {sample_id}: {result.stderr.strip()}", file=sys.stderr)
            continue

        records.append(
            {
                "id": sample_id,
                "audio": f"audio/{wav.name}",
                "reference": text,
                "language": "en",
                "speakers": 1,
                "metadata": {"voice": voice, "synthetic": True},
            }
        )
        print(f"  + {sample_id:<16} [{voice}]")

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/smoke"))
    parser.add_argument("--rate", type=int, default=180, help="Speaking rate, words per minute.")
    args = parser.parse_args()

    if shutil.which("say") is None:
        print(
            "error: 'say' not found. This generator is macOS-only; on Windows use "
            "real recordings or another TTS.",
            file=sys.stderr,
        )
        return 1

    banned = UNUSABLE_VOICES & set(VOICES)
    if banned:
        print(
            f"error: legacy formant-synthesis voice(s) selected: {', '.join(sorted(banned))}. "
            "Their output is unintelligible and produces meaningless error rates.",
            file=sys.stderr,
        )
        return 1

    print(f"Generating {len(SCRIPTS)} samples into {args.out} ...")
    records = generate(args.out, VOICES, args.rate)
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
