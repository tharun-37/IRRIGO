"""
WebSocket fan-out for live telemetry.

One process, many dashboards. Frames are small and dropped rather than queued if
a client cannot keep up: on a monitoring surface a slow consumer is better served
by the newest frame than by a growing backlog of stale ones. A valve command is
never sent over this channel, so nothing safety-relevant can be lost by dropping
a telemetry frame.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class TelemetryHub:
    """Tracks connected dashboards and broadcasts to all of them."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Remember the serving loop, so worker threads can publish to it.

        Ingest is synchronous because SQLite writes block, so it runs in a thread
        pool where there is no running event loop. Without this the hub cannot be
        reached from ingest at all, and every live frame is silently dropped
        while the API still reports success.
        """
        self._loop = loop

    def submit(self, coro: Any) -> bool:
        """Hand a coroutine to the serving loop from any thread."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            running.create_task(coro)
            return True
        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
            return True
        return False

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def broadcast(self, frame_type: str, payload: Any, **extra: Any) -> None:
        """Send one frame to every client, dropping any that cannot keep up."""
        if not self._clients:
            return
        message: dict[str, Any] = {"type": frame_type, "payload": payload, **extra}
        async with self._lock:
            targets = list(self._clients)

        dead: list[WebSocket] = []
        for client in targets:
            try:
                await client.send_json(message)
            except Exception:
                dead.append(client)
        if dead:
            async with self._lock:
                for client in dead:
                    self._clients.discard(client)

    async def ping(self) -> None:
        """Keep intermediaries from dropping an idle connection."""
        await self.broadcast("ping", {"at": asyncio.get_event_loop().time()})


hub = TelemetryHub()


def safe_broadcast(frame_type: str, payload: Any, **extra: Any) -> None:
    """
    Publish one frame without letting a failure reach the caller.

    Deliberately synchronous and non-blocking, so it can be called from the
    thread pool that serves the sync ingest endpoints, from async endpoints, and
    from background threads. A dashboard that fails to update must never fail a
    valve command, so every failure is swallowed here.
    """
    coro = hub.broadcast(frame_type, payload, **extra)
    if not hub.submit(coro):
        coro.close()
