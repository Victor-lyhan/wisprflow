#!/usr/bin/env python3
"""Compare ASR backends, with and without correction, over a dataset.

Produces the table that decides which engine ships. Every configuration is
scored on identical audio against an identical reference, so the numbers are
comparable to each other -- which is the only comparison that means anything
here.

Two cautions on reading the output:

* **Domain WER is computed over the loaded lexicon.** Regenerating the lexicon
  changes the denominator, so DWER is comparable only within one lexicon version.
  The version in use is printed with the results.
* **Synthetic audio measures the harness, not the clinic.** These figures come
  from TTS with no handpiece noise, no masks and no crosstalk. Treat them as a
  ranking of configurations, never as expected accuracy.

Usage:
    python scripts/benchmark.py data/smoke/manifest.jsonl
    python scripts/benchmark.py data/smoke/manifest.jsonl --correct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dentascribe.config import ASRConfig, Config, CorrectionConfig  # noqa: E402
from dentascribe.dental.lexicon import BUILT_PATH, SEED_PATH, load_seed_lexicon  # noqa: E402
from dentascribe.evaluation import evaluate, load_manifest  # noqa: E402

ENGINES: list[tuple[str, str, str]] = [
    ("whisper-tiny", "faster-whisper", "tiny.en"),
    ("whisper-small", "faster-whisper", "small.en"),
    ("parakeet-v3", "parakeet-onnx", "nemo-parakeet-tdt-0.6b-v3"),
]


def run(name: str, backend: str, model: str, manifest: Path, correct: bool) -> dict | None:
    config = Config(
        offline_only=False,
        final=ASRConfig(backend=backend, model=model, language="en"),
        correction=CorrectionConfig(
            enabled=correct, backend="ollama" if correct else "passthrough"
        ),
    )

    started = time.monotonic()
    try:
        result = evaluate(load_manifest(manifest), config)
    except Exception as exc:  # noqa: BLE001 - one engine failing must not stop the sweep
        print(f"  ! {name}: {exc}", file=sys.stderr)
        return None

    card = result.corrected if correct and result.corrected else result.verbatim
    return {
        "name": name + ("+llm" if correct else ""),
        "wer": card.wer,
        "dwer": card.domain_wer,
        "terms": f"{card.domain_hits}/{card.domain_terms}",
        "tooth": card.tooth_accuracy,
        "rtf": result.real_time_factor,
        "insertions": result.mean_insertion_rate if correct else 0.0,
        "wall": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--correct", action="store_true", help="Also run each engine with the LLM.")
    parser.add_argument("--only", default=None, help="Run one engine by name.")
    args = parser.parse_args()

    lexicon = load_seed_lexicon()
    source = "generated (MeSH+RxNorm)" if BUILT_PATH.exists() else "seed only"
    print(f"lexicon: {len(lexicon)} terms, {source}")
    print(f"dataset: {args.manifest}\n")

    engines = [e for e in ENGINES if args.only is None or e[0] == args.only]
    rows: list[dict] = []
    for name, backend, model in engines:
        print(f"running {name} ...", file=sys.stderr)
        row = run(name, backend, model, args.manifest, correct=False)
        if row:
            rows.append(row)
        if args.correct:
            print(f"running {name}+llm ...", file=sys.stderr)
            row = run(name, backend, model, args.manifest, correct=True)
            if row:
                rows.append(row)

    if not rows:
        print("no results", file=sys.stderr)
        return 1

    header = f"{'engine':<18}{'WER':>9}{'DWER':>9}{'terms':>10}{'tooth':>9}{'RTF':>8}{'ins':>8}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<18}{row['wer']:>8.2%}{row['dwer']:>9.2%}"
            f"{row['terms']:>10}{row['tooth']:>9.2%}{row['rtf']:>8.3f}"
            f"{row['insertions']:>8.1%}"
        )

    best = min(rows, key=lambda r: r["dwer"])
    print(f"\nlowest domain WER: {best['name']} at {best['dwer']:.2%}")
    print("\nSynthetic audio -- a ranking of configurations, not expected clinical accuracy.")
    if BUILT_PATH.exists():
        print(
            f"DWER computed over {BUILT_PATH.name}; not comparable to runs using {SEED_PATH.name}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
