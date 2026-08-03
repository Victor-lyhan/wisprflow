#!/usr/bin/env python3
"""Build the dental lexicon from freely redistributable sources.

Replaces the hand-written seed list with terminology pulled from public
vocabularies. The seed list stays in the repo as the offline fallback and as the
floor: whatever this produces is merged with it, never substituted for it.

SOURCES AND WHY THESE ONES
==========================
* **MeSH** (NLM) -- public domain, no account. The dental subtree (category E06
  plus the stomatognathic anatomy branch A03/A14) covers anatomy, conditions and
  procedures.
* **RxNorm** (NLM) -- open REST API, no key. Supplies drug ingredient names, which
  matters because a misheard anaesthetic or antibiotic is among the highest-cost
  errors in a dental record.

DELIBERATELY EXCLUDED
=====================
* **CDT** and **SNODENT** are American Dental Association copyright and require a
  paid commercial licence to redistribute. A practice may use CDT freely in its
  own records, so a licensed practice supplies its own file at runtime via
  ``CorrectionConfig.user_codes_path``. Nothing here fetches or embeds them.
* **SNOMED CT** is reachable through UMLS but its licence is per-user and its
  redistribution terms are not compatible with shipping a derived file. Adding it
  would mean every user needs their own UMLS account, so it is left out.

This script reaches the network by design and is a build-time tool. It is never
invoked at inference time -- that path stays offline.

Usage:
    python scripts/build_lexicon.py --out src/flowscribe/dental/data/lexicon.txt
    python scripts/build_lexicon.py --dry-run          # report counts, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = REPO_ROOT / "src" / "flowscribe" / "dental" / "data" / "seed_lexicon.txt"

MESH_TREES = {
    "E06": "dentistry procedures",
    "A03.556.500": "mouth and oral anatomy",
    "A14": "stomatognathic system",
    "C07": "stomatognathic diseases",
}

# Ingredient names relevant to dental practice. RxNorm is queried per name rather
# than bulk-downloaded: the full release is large and most of it is irrelevant
# here, and the API needs no key.
DENTAL_DRUGS = [
    "lidocaine",
    "articaine",
    "mepivacaine",
    "prilocaine",
    "bupivacaine",
    "epinephrine",
    "levonordefrin",
    "benzocaine",
    "chlorhexidine",
    "amoxicillin",
    "clindamycin",
    "penicillin",
    "metronidazole",
    "azithromycin",
    "ibuprofen",
    "acetaminophen",
    "nitrous oxide",
    "midazolam",
    "triazolam",
    "fluoride",
    "sodium fluoride",
    "hydrogen peroxide",
    "eugenol",
]

# Terms accepted from any source must look like clinical vocabulary rather than
# free text. MeSH entries include long descriptive phrases that would bloat the
# lexicon and never appear in speech.
_ACCEPTABLE = re.compile(r"^[a-z][a-z \-']{2,40}$")
_MAX_WORDS = 3

# Terms that arrive from the sources but are not dental vocabulary: RxNorm
# returns co-formulated and unrelated systemic drugs, and MeSH contributes bare
# adjectives. Left in, they inflate the domain-WER denominator with words whose
# recognition says nothing about clinical accuracy.
_REJECT = frozenset(
    {
        "artificial",
        "anesthesia",
        "acid",
        "acetic acid",
        "ascorbic acid",
        "adapalene",
        "allantoin",
        "alpha-tocopherol acetate",
        "water",
        "alcohol",
        "oil",
        "gel",
        "paste",
        "solution",
        "powder",
        "cream",
        "spray",
        "tablet",
    }
)


def _http_get(url: str, timeout: float = 30.0) -> str | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "flowscribe-lexicon-build"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  ! {url.split('?')[0]}: {exc}", file=sys.stderr)
        return None


def _acceptable(term: str) -> bool:
    term = term.strip().lower()
    if not _ACCEPTABLE.match(term):
        return False
    if len(term.split()) > _MAX_WORDS or term in _REJECT:
        return False
    # MeSH uses inverted headings ("Tooth, Deciduous"); those are handled by the
    # caller splitting on the comma, so anything still containing one is prose.
    return "," not in term


def fetch_mesh() -> set[str]:
    """Descriptor names from the dental MeSH subtrees."""
    terms: set[str] = set()
    for tree, label in MESH_TREES.items():
        url = (
            "https://id.nlm.nih.gov/mesh/sparql?format=JSON&limit=5000&query="
            + urllib.parse.quote(
                "PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#> "
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
                "SELECT ?label WHERE { "
                "?d meshv:treeNumber ?t . "
                f'FILTER(STRSTARTS(STR(?t), "http://id.nlm.nih.gov/mesh/{tree}")) '
                "?d rdfs:label ?label . }"
            )
        )
        payload = _http_get(url)
        if payload is None:
            continue

        try:
            rows = json.loads(payload)["results"]["bindings"]
        except (ValueError, KeyError):
            print(f"  ! {tree}: unexpected response shape", file=sys.stderr)
            continue

        found = 0
        for row in rows:
            raw = row.get("label", {}).get("value", "")
            # "Tooth, Deciduous" -> both "tooth" and "deciduous" as candidates.
            for part in raw.split(","):
                candidate = part.strip().lower()
                if _acceptable(candidate):
                    terms.add(candidate)
                    found += 1
        print(f"  + MeSH {tree:<12} {found:>5} terms  ({label})")

    return terms


def fetch_rxnorm() -> set[str]:
    """Ingredient names and synonyms for dental pharmacology."""
    terms: set[str] = set()
    for drug in DENTAL_DRUGS:
        url = "https://rxnav.nlm.nih.gov/REST/drugs.json?name=" + urllib.parse.quote(drug)
        payload = _http_get(url)
        if payload is None:
            continue
        try:
            groups = json.loads(payload).get("drugGroup", {}).get("conceptGroup", []) or []
        except ValueError:
            continue

        terms.add(drug)
        for group in groups:
            for concept in group.get("conceptProperties", []) or []:
                name = concept.get("name", "").split("[")[0].strip().lower()
                # Strip dose forms: "lidocaine 20 MG/ML" is not spoken vocabulary.
                name = re.sub(r"\d.*$", "", name).strip()
                if _acceptable(name):
                    terms.add(name)

    print(f"  + RxNorm      {len(terms):>5} terms")
    return terms


def load_seed() -> set[str]:
    terms = set()
    for line in SEED.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip().lower()
        if text:
            terms.add(text)
    return terms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SEED.parent / "lexicon.txt")
    parser.add_argument("--dry-run", action="store_true", help="Report counts, write nothing.")
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--skip-rxnorm", action="store_true")
    args = parser.parse_args()

    print("Building dental lexicon from public sources ...")
    print("(CDT and SNODENT are ADA copyright and are never fetched or embedded.)\n")

    seed = load_seed()
    print(f"  = seed        {len(seed):>5} terms  (bundled, always included)")

    collected = set(seed)
    if not args.skip_mesh:
        collected |= fetch_mesh()
    if not args.skip_rxnorm:
        collected |= fetch_rxnorm()

    added = collected - seed
    print(f"\n  total {len(collected)} terms ({len(added)} beyond the seed list)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        for term in sorted(added)[:25]:
            print(f"    {term}")
        return 0

    if len(collected) <= len(seed):
        print(
            "\nerror: no terms were fetched, so the result would only duplicate the "
            "seed list. Refusing to overwrite -- check network access.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Dental lexicon, generated by scripts/build_lexicon.py\n"
        "#\n"
        "# Sources: MeSH (public domain) and RxNorm (open API), merged with the\n"
        "# hand-written seed list. No CDT and no SNODENT -- both are ADA copyright\n"
        "# and require a commercial licence to redistribute. A licensed practice\n"
        "# supplies its own codes via CorrectionConfig.user_codes_path.\n"
        "#\n"
        "# Regenerate rather than hand-editing; edits belong in seed_lexicon.txt.\n\n"
    )
    args.out.write_text(header + "\n".join(sorted(collected)) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out} ({len(collected)} terms)")
    print("\nPoint the corrector at it:  correction.lexicon_path: " + str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
