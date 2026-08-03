"""Demo UI server.

Serves a single page that streams live transcription over a WebSocket. The
browser captures nothing itself -- audio comes from the *server's* default input
device, so this demonstrates the same path a clinic workstation would use rather
than a browser-only toy.

Binds to loopback by default. The stream carries clinical audio and, once
transcribed, clinical text; putting that on a LAN interface should be an explicit
choice.

Requires: ``pip install 'flowscribe[ui,mic]'``
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .config import Config
from .live import event_to_dict

# Imported at module scope, not inside create_app, and that placement is
# load-bearing. This module uses `from __future__ import annotations`, so every
# annotation is a string that FastAPI resolves against the *module* globals when
# it builds a route. With `WebSocket` imported as a function local, the handler's
# `socket: WebSocket` annotation was unresolvable, FastAPI fell back to treating
# the parameter as a query field, and every handshake was rejected with a bare
# HTTP 403 and no traceback.
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    FASTAPI_AVAILABLE = False

__all__ = ["create_app", "serve"]

PAGE = Path(__file__).parent / "static" / "index.html"


def create_app(config: Config) -> Any:
    """Build the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "The demo UI needs fastapi and uvicorn. Install with: pip install 'flowscribe[ui]'"
        )

    app = FastAPI(title="flowscribe demo")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.get("/devices")
    async def devices() -> dict[str, Any]:
        from .audio.microphone import default_input_device, list_input_devices

        try:
            return {"default": default_input_device(), "devices": list_input_devices()}
        except Exception as exc:  # noqa: BLE001 - reported to the page, not fatal
            return {"error": str(exc), "devices": []}

    @app.websocket("/stream")
    async def stream(socket: WebSocket) -> None:
        from .audio.microphone import MicrophoneSource
        from .pipeline import Pipeline

        await socket.accept()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        source = MicrophoneSource(chunk_seconds=config.chunk_seconds)

        def run_pipeline() -> None:
            """Drive the blocking pipeline on a worker thread.

            Transcription is CPU-bound and synchronous. Running it on the event
            loop would stall every other connection and the socket itself, so it
            gets a thread and pushes results back through a queue.
            """
            try:
                with Pipeline(config) as pipeline, source:
                    for event in pipeline.stream(source):
                        loop.call_soon_threadsafe(queue.put_nowait, event_to_dict(event))
            except Exception as exc:  # noqa: BLE001 - surface to the page
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        worker = loop.run_in_executor(None, run_pipeline)

        async def watch_for_stop() -> None:
            """A client message means stop; a disconnect means the same."""
            try:
                while True:
                    await socket.receive_text()
                    source.stop()
                    return
            except (WebSocketDisconnect, RuntimeError):
                source.stop()

        watcher = asyncio.create_task(watch_for_stop())

        try:
            while True:
                record = await queue.get()
                if record is None:
                    break
                await socket.send_text(json.dumps(record))
        except WebSocketDisconnect:
            source.stop()
        finally:
            source.stop()
            watcher.cancel()
            await worker
            try:
                await socket.close()
            except RuntimeError:
                pass

    return app


def serve(config: Config, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the demo server."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The demo UI needs uvicorn. Install with: pip install 'flowscribe[ui]'"
        ) from exc

    uvicorn.run(create_app(config), host=host, port=port, log_level="warning")
