"""Packaging integrity.

These exist because an editable install hides packaging bugs completely. Local
development ran against ``src/`` directly and everything passed, while the built
wheel was missing the entire audio package and the whole lexicon. Every platform
in CI failed with ``ModuleNotFoundError: No module named 'dentascribe.audio'``.

The cause was ``.gitignore``: unanchored ``data/`` and ``audio/`` patterns,
written to keep patient recordings out of the repository, also matched
``src/dentascribe/audio/`` and ``src/dentascribe/dental/data/``. The files were
never committed, so they could not appear in a wheel built from a clean checkout.

Nothing in the ordinary test suite could catch that, because the ordinary test
suite imports from the working tree.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "dentascribe"

# Every subpackage that must reach an installed wheel. Listed explicitly rather
# than discovered, so that deleting one is a test failure rather than a silently
# shorter list.
REQUIRED_SUBPACKAGES = [
    "asr",
    "audio",
    "correct",
    "dental",
    "evaluation",
    "sinks",
    "streaming",
    "vad",
]

REQUIRED_DATA = [
    "dental/data/seed_lexicon.txt",
    "dental/data/confusions.txt",
]


class TestSourceTree:
    """Cheap checks that run everywhere."""

    @pytest.mark.parametrize("package", REQUIRED_SUBPACKAGES)
    def test_subpackage_is_importable(self, package: str) -> None:
        __import__(f"dentascribe.{package}")

    @pytest.mark.parametrize("relative", REQUIRED_DATA)
    def test_data_file_exists(self, relative: str) -> None:
        assert (SRC / relative).is_file(), f"missing bundled data file: {relative}"

    @pytest.mark.parametrize("package", REQUIRED_SUBPACKAGES)
    def test_subpackage_is_tracked_by_git(self, package: str) -> None:
        """A file git ignores cannot reach a wheel built from a clean checkout.

        This is the check that would have caught the original bug, and it costs
        nothing to run.
        """
        result = subprocess.run(
            ["git", "ls-files", f"src/dentascribe/{package}/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("not a git checkout")
        assert result.stdout.strip(), f"src/dentascribe/{package}/ is untracked or ignored"

    @pytest.mark.parametrize("relative", REQUIRED_DATA)
    def test_data_file_is_tracked_by_git(self, relative: str) -> None:
        result = subprocess.run(
            ["git", "ls-files", f"src/dentascribe/{relative}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("not a git checkout")
        assert result.stdout.strip(), f"src/dentascribe/{relative} is untracked or ignored"


@pytest.mark.slow
class TestBuiltWheel:
    """Builds an actual wheel and inspects it.

    Slow because it shells out to the build backend, but it is the only check
    that tests what a user receives rather than what the working tree contains.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
        out = tmp_path_factory.mktemp("wheel")
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(REPO_ROOT)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"could not build wheel: {result.stderr[-400:]}")
        wheels = list(out.glob("*.whl"))
        if not wheels:
            pytest.skip("build produced no wheel")
        return wheels[0]

    @pytest.mark.parametrize("package", REQUIRED_SUBPACKAGES)
    def test_wheel_contains_subpackage(self, wheel: Path, package: str) -> None:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        assert any(name.startswith(f"dentascribe/{package}/") for name in names), (
            f"wheel is missing dentascribe/{package}/"
        )

    @pytest.mark.parametrize("relative", REQUIRED_DATA)
    def test_wheel_contains_data_file(self, wheel: Path, relative: str) -> None:
        """The lexicon is data, not code, and is the thing most likely to be
        dropped by a build backend that only collects .py files."""
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        assert f"dentascribe/{relative}" in names, f"wheel is missing dentascribe/{relative}"
