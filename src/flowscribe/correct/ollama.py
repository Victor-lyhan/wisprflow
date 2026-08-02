"""LLM correction via a local Ollama server.

Ollama rather than an in-process runtime because it installs cleanly on both
macOS and Windows with GPU acceleration handled for you, which matters when
production is a Windows box of unknown specification. It listens on loopback, so
transcripts never leave the machine and the ``offline_only`` check passes.

Correction is per utterance. Batching the whole transcript would be cheaper, but
per-utterance keeps each edit attributable to a specific span, bounds the damage
of a bad generation to one utterance, and lets the guard reject that utterance
alone rather than discarding the run.

Requires: ``pip install 'flowscribe[llm]'`` and a running ``ollama serve``.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts import Edit, Transcript, Utterance
from ..dental.lexicon import Confusions, Lexicon, load_confusions, load_seed_lexicon
from ..errors import MissingDependency
from .guard import check, classify
from .prompt import SYSTEM_PROMPT, build_user_prompt, parse_response

__all__ = ["OllamaCorrector"]

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"


class OllamaCorrector:
    """Constrained terminology correction against a local Ollama model."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3:8b",
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        max_edit_ratio: float = 0.25,
        lexicon_path: str | Path | None = None,
        user_codes_path: str | Path | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        context_window: int = 1,
        **options: object,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise MissingDependency(
                "httpx is required for the Ollama corrector. Install with: "
                "pip install 'flowscribe[llm]'"
            ) from exc

        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.max_edit_ratio = max_edit_ratio
        self.temperature = temperature
        self.context_window = context_window
        self._options = options
        self._client = httpx.Client(timeout=timeout)

        self.lexicon = Lexicon.from_file(lexicon_path) if lexicon_path else load_seed_lexicon()
        if user_codes_path:
            # Practice-supplied procedure codes. CDT and SNODENT are ADA
            # copyright and cannot be bundled, so they enter only here.
            self.lexicon = self.lexicon.extend(
                Path(user_codes_path).read_text(encoding="utf-8").splitlines()
            )

        self.confusions: Confusions = load_confusions()

        # Populated per run so callers can see what the guard threw out.
        self.rejected: list[tuple[str, str]] = []

    # -- transport ---------------------------------------------------------- #

    def available(self) -> bool:
        """Whether the Ollama server is reachable."""
        try:
            return self._client.get(f"{self.endpoint}/api/tags").status_code == 200
        except Exception:
            return False

    def _generate(self, prompt: str) -> str | None:
        import httpx

        try:
            response = self._client.post(
                f"{self.endpoint}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {"temperature": self.temperature},
                },
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            return str(content) if content else None
        except (httpx.HTTPError, ValueError, KeyError):
            # A correction failure must never fail the transcription. The
            # verbatim transcript is the deliverable; correction is an
            # enhancement on top of it.
            return None

    # -- Corrector protocol -------------------------------------------------- #

    def correct(self, transcript: Transcript) -> tuple[Transcript, list[Edit]]:
        self.rejected = []
        utterances = transcript.utterances
        corrected: list[Utterance] = []
        edits: list[Edit] = []

        for index, utterance in enumerate(utterances):
            text = utterance.text.strip()
            if not text:
                corrected.append(utterance)
                continue

            window = self.context_window
            before = " ".join(u.text for u in utterances[max(0, index - window) : index])
            after = " ".join(u.text for u in utterances[index + 1 : index + 1 + window])

            raw = self._generate(
                build_user_prompt(
                    text,
                    self.lexicon,
                    confusions=self.confusions,
                    context_before=before,
                    context_after=after,
                )
            )
            proposal = parse_response(raw) if raw is not None else None

            if proposal is None or proposal == text:
                corrected.append(utterance)
                continue

            verdict = check(text, proposal, max_edit_ratio=self.max_edit_ratio)
            if not verdict:
                self.rejected.append((utterance.id, verdict.reason))
                corrected.append(utterance)
                continue

            edits.append(
                Edit(
                    utterance_id=utterance.id,
                    before=text,
                    after=proposal,
                    kind=classify(text, proposal, self.lexicon),  # type: ignore[arg-type]
                    reason=f"{self.name}:{self.model}",
                )
            )
            # Word timings belong to the verbatim audio alignment and no longer
            # describe the corrected string, so they are dropped rather than
            # left to imply a precision that is gone.
            corrected.append(utterance.model_copy(update={"text": proposal, "words": []}))

        return transcript.model_copy(update={"utterances": corrected}), edits

    def close(self) -> None:
        self._client.close()
