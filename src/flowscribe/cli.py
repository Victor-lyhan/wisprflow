"""Command-line interface.

flowscribe transcribe visit.wav --format text
flowscribe transcribe live.wav --live          # still being recorded
flowscribe fetch-models --asr large-v3         # provisioning; the only online step
flowscribe backends
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from pathlib import Path
from typing import Annotated, Any

import typer

from .config import Config
from .errors import FlowscribeError
from .registry import ASR, CORRECTOR, SINK

app = typer.Typer(
    name="flowscribe",
    help="Local speech-to-text for dental clinical audio.",
    no_args_is_help=True,
    add_completion=False,
)


def _fail(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def transcribe(
    audio: Annotated[Path, typer.Argument(help="Audio file to transcribe.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write here instead of stdout.")
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="json, jsonl, text, srt, or vtt.")
    ] = "text",
    model: Annotated[str | None, typer.Option(help="ASR model name.")] = None,
    language: Annotated[
        str | None, typer.Option(help="Language code. Omit to auto-detect.")
    ] = None,
    live: Annotated[
        bool, typer.Option("--live", help="Read a WAV that is still being recorded.")
    ] = False,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML config file.")
    ] = None,
    correct: Annotated[
        bool | None,
        typer.Option("--correct/--no-correct", help="Override the config file setting."),
    ] = None,
    allow_network: Annotated[
        bool,
        typer.Option("--allow-network", help="Permit downloads. Off by default to protect PHI."),
    ] = False,
    show_edits: Annotated[
        bool, typer.Option("--show-edits", help="Print every correction made.")
    ] = False,
) -> None:
    """Transcribe an audio file."""
    from .pipeline import Pipeline

    try:
        config = Config.from_yaml(config_file) if config_file else Config()
    except FlowscribeError as exc:
        _fail(str(exc))

    if model:
        config.final.model = model
    if language:
        config.final.language = language
    if allow_network:
        config.offline_only = False
    if correct is not None:
        config.correction.enabled = correct

    # Correction needs a real backend; the default is a no-op, so silently
    # producing unchanged output would look like the feature ran and found
    # nothing to do.
    if correct and config.correction.backend in ("null", "passthrough"):
        _fail(
            "--correct needs a corrector backend. Set correction.backend in a config "
            f"file (available: {', '.join(CORRECTOR.names())})."
        )

    try:
        sink = SINK.create(fmt)
    except FlowscribeError:
        _fail(f"Unknown format {fmt!r}. Available: {', '.join(SINK.names())}")

    from .audio.preprocess import open_source

    try:
        source = open_source(audio, live=live, chunk_seconds=config.chunk_seconds)
        with Pipeline(config) as pipeline:
            result = pipeline.transcribe(source)
            stats = pipeline.stats
    except FlowscribeError as exc:
        _fail(str(exc))
    except KeyboardInterrupt:
        typer.secho("interrupted", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from None

    rendered = sink.emit(result.best)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.secho(f"wrote {output}", fg=typer.colors.GREEN, err=True)
    else:
        sys.stdout.write(rendered + "\n")

    if show_edits and result.edits:
        typer.secho(f"\n{len(result.edits)} correction(s):", fg=typer.colors.CYAN, err=True)
        for edit in result.edits:
            typer.echo(f"  [{edit.kind}] {edit.before!r} -> {edit.after!r}", err=True)

    # Diagnostics go to stderr so piping stdout to a file stays clean.
    typer.secho(
        f"\n{stats.audio_seconds:.1f}s audio in {stats.total_seconds:.1f}s "
        f"(RTF {stats.real_time_factor:.2f})",
        fg=typer.colors.BRIGHT_BLACK,
        err=True,
    )


@app.command()
def listen(
    seconds: Annotated[
        float | None,
        typer.Option(help="Stop after this many seconds. Omit to run until Ctrl-C."),
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="Input device index or name. Omit for the system default.")
    ] = None,
    model: Annotated[str | None, typer.Option(help="ASR model for the live tier.")] = None,
    language: Annotated[str | None, typer.Option(help="Language code.")] = None,
    config_file: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    correct: Annotated[
        bool | None,
        typer.Option("--correct/--no-correct", help="Override the config file setting."),
    ] = None,
    allow_network: Annotated[bool, typer.Option("--allow-network")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the final transcript here.")
    ] = None,
    list_devices: Annotated[
        bool, typer.Option("--list-devices", help="Show capture devices and exit.")
    ] = False,
) -> None:
    """Transcribe live from a microphone, emitting JSON events.

    One JSON object per line on stdout: `partial` while text is still unstable,
    `final` once the streaming policy has confirmed it, then a single `complete`
    record carrying the authoritative transcript.

    Stop with Ctrl-C. The recording is finalized rather than discarded.
    """
    from .audio.microphone import MicrophoneSource, default_input_device, list_input_devices
    from .live import event_to_dict
    from .pipeline import Pipeline

    if list_devices:
        try:
            for entry in list_input_devices():
                typer.echo(json.dumps(entry))
        except FlowscribeError as exc:
            _fail(str(exc))
        return

    try:
        config = Config.from_yaml(config_file) if config_file else Config()
    except FlowscribeError as exc:
        _fail(str(exc))

    if model:
        config.live.model = model
        config.final.model = model
    if language:
        config.live.language = language
        config.final.language = language
    if allow_network:
        config.offline_only = False
    if correct is not None:
        config.correction.enabled = correct

    if correct and config.correction.backend in ("null", "passthrough"):
        _fail("--correct needs a corrector backend; set correction.backend in a config file.")

    selected: int | str | None = device
    if device is not None and device.isdigit():
        selected = int(device)

    try:
        info = default_input_device() if device is None else {"name": str(device)}
        source = MicrophoneSource(device=selected, chunk_seconds=config.chunk_seconds)
    except FlowscribeError as exc:
        _fail(str(exc))

    typer.secho(f"listening on {info['name']}  (Ctrl-C to stop)", fg=typer.colors.CYAN, err=True)

    # Ctrl-C closes the capture device and lets the stream end on its own, so the
    # final tier still runs. Killing the generator outright would throw away a
    # whole appointment because someone stopped the recording.
    stopping = threading.Event()

    def handle_interrupt(signum: int, frame: object) -> None:
        if not stopping.is_set():
            stopping.set()
            typer.secho("\nfinalizing ...", fg=typer.colors.YELLOW, err=True)
            source.stop()

    signal.signal(signal.SIGINT, handle_interrupt)

    deadline = threading.Timer(seconds, source.stop) if seconds else None
    if deadline:
        deadline.daemon = True
        deadline.start()

    final_record: dict[str, Any] | None = None
    try:
        with Pipeline(config) as pipeline, source:
            for event in pipeline.stream(source):
                record = event_to_dict(event)
                if record["type"] == "complete":
                    final_record = record
                sys.stdout.write(json.dumps(record) + "\n")
                sys.stdout.flush()
    except FlowscribeError as exc:
        _fail(str(exc))
    finally:
        if deadline:
            deadline.cancel()

    if source.overflows or source.dropped_blocks:
        typer.secho(
            f"warning: {source.overflows} input overflow(s), "
            f"{source.dropped_blocks} dropped block(s) -- audio was lost",
            fg=typer.colors.YELLOW,
            err=True,
        )

    if output and final_record:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(final_record, indent=2), encoding="utf-8")
        typer.secho(f"wrote {output}", fg=typer.colors.GREEN, err=True)


@app.command()
def ui(
    host: Annotated[str, typer.Option(help="Bind address. Loopback by default.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 8000,
    model: Annotated[str | None, typer.Option(help="ASR model.")] = None,
    language: Annotated[str | None, typer.Option(help="Language code.")] = None,
    config_file: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    correct: Annotated[
        bool | None,
        typer.Option("--correct/--no-correct", help="Override the config file setting."),
    ] = None,
    allow_network: Annotated[bool, typer.Option("--allow-network")] = False,
) -> None:
    """Serve the browser demo UI.

    Binds to loopback by default: the page streams live clinical audio, and
    exposing that on a LAN interface should be a deliberate act, not a default.
    """
    try:
        from .server import serve
    except ImportError:
        _fail("The demo UI needs extra packages. Install with: pip install 'flowscribe[ui]'")

    try:
        config = Config.from_yaml(config_file) if config_file else Config()
    except FlowscribeError as exc:
        _fail(str(exc))

    if model:
        config.live.model = model
        config.final.model = model
    if language:
        config.live.language = language
        config.final.language = language
    if allow_network:
        config.offline_only = False
    if correct is not None:
        config.correction.enabled = correct

    typer.secho(f"demo UI on http://{host}:{port}", fg=typer.colors.CYAN)
    serve(config, host=host, port=port)


@app.command("fetch-models")
def fetch_models(
    asr: Annotated[str, typer.Option(help="ASR model to download.")] = "large-v3",
    model_dir: Annotated[Path | None, typer.Option(help="Where to store models.")] = None,
) -> None:
    """Download models ahead of time.

    Provisioning is deliberately separate from inference. Once models are cached,
    the pipeline runs with networking disabled, so no patient audio is ever in
    memory during a network call.
    """
    config = Config(offline_only=False)
    target = model_dir or config.model_dir
    target.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Downloading ASR model {asr!r} to {target} ...")
    try:
        from faster_whisper import WhisperModel

        WhisperModel(asr, device="cpu", compute_type="int8", download_root=str(target))
    except ImportError:
        _fail("faster-whisper is not installed. Install with: pip install 'flowscribe[whisper]'")
    except Exception as exc:  # noqa: BLE001 - surface whatever the downloader failed with
        _fail(f"Download failed: {exc}")

    typer.secho(f"ready: {asr}", fg=typer.colors.GREEN)
    typer.secho(
        "Inference can now run offline. Set FLOWSCRIBE_MODEL_DIR "
        f"={target} if this is not the default location.",
        fg=typer.colors.BRIGHT_BLACK,
    )


@app.command("eval")
def evaluate_cmd(
    manifest: Annotated[Path, typer.Argument(help="JSONL dataset manifest.")],
    model: Annotated[str | None, typer.Option(help="ASR model name.")] = None,
    language: Annotated[str | None, typer.Option(help="Language code.")] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML config file.")
    ] = None,
    correct: Annotated[
        bool | None,
        typer.Option("--correct/--no-correct", help="Override the config file setting."),
    ] = None,
    lexicon_file: Annotated[
        Path | None, typer.Option("--lexicon", help="Term list for domain WER.")
    ] = None,
    allow_network: Annotated[bool, typer.Option("--allow-network")] = False,
    per_sample: Annotated[bool, typer.Option(help="Print a line per sample.")] = False,
) -> None:
    """Score the pipeline against a dataset.

    Reports domain WER alongside overall WER. Domain WER is the number to watch:
    overall WER is dominated by ordinary words and can look healthy while
    clinical vocabulary is failing.
    """
    try:
        from .dental.lexicon import Lexicon, load_seed_lexicon
        from .evaluation import evaluate, load_manifest
    except FlowscribeError as exc:
        _fail(str(exc))

    try:
        config = Config.from_yaml(config_file) if config_file else Config()
        samples = load_manifest(manifest)
    except FlowscribeError as exc:
        _fail(str(exc))

    if model:
        config.final.model = model
    if language:
        config.final.language = language
    if allow_network:
        config.offline_only = False
    if correct is not None:
        config.correction.enabled = correct

    lexicon = Lexicon.from_file(lexicon_file) if lexicon_file else load_seed_lexicon()
    typer.secho(
        f"{len(samples)} sample(s), {len(lexicon)} lexicon terms\n",
        fg=typer.colors.BRIGHT_BLACK,
        err=True,
    )

    try:
        result = evaluate(samples, config, lexicon=lexicon, progress=per_sample)
    except FlowscribeError as exc:
        _fail(str(exc))

    if per_sample:
        typer.echo("")
        for s in result.samples:
            typer.echo(f"  {s.sample_id:<24} {s.verbatim.summary()}")
        typer.echo("")

    typer.echo(result.report())


@app.command()
def backends() -> None:
    """List registered backends for each stage."""
    for label, registry in (
        ("asr", ASR),
        ("corrector", CORRECTOR),
        ("sink", SINK),
    ):
        names = registry.names()
        typer.secho(f"{label:>10}: ", nl=False, fg=typer.colors.CYAN)
        typer.echo(", ".join(names) if names else "(none)")


@app.command()
def version() -> None:
    """Print the version."""
    from . import __version__

    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
