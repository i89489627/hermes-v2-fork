"""hermes-v2 MCP stdio client (plan P21.T3, issue #156143).

Upstream bug: ``tools/call`` arguments were intermittently dropped on the
stdio hop; the server executed the tool with schema defaults and the caller
saw no error. The v2 client closes every gap that allowed silent drops:

1. **Strict framing** — all I/O goes through :mod:`hermes.mcp.protocol`
   (newline-delimited JSON-RPC 2.0, one locked write per frame, hard errors
   on malformed reads).
2. **Caller-side validation** — when a schema is supplied, arguments are
   validated with :func:`hermes.mcp.argument_validator.validate_arguments`
   BEFORE the request is written; violations raise :class:`McpArgumentError`.
3. **Server echo verification** — servers built on the hermes contract answer
   ``tools/call``-shaped requests (``params`` containing an ``arguments``
   key) with a ``result`` that echoes the exact received arguments under
   ``arguments_received``. Any mismatch — or a missing echo — raises
   :class:`McpEchoMismatchError`, so a dropped-arguments call can never
   complete silently again.
4. **Full call logging** — every request is logged with its complete
   arguments on logger ``hermes.mcp.client``.

Concurrency: responses are correlated by JSON-RPC id — a single reader thread
dispatches to per-request queues, writes are serialized under a lock — so one
client is safely shareable across threads (used by the chaos suite).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import queue
import subprocess
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from hermes.mcp.argument_validator import validate_arguments
from hermes.mcp.protocol import FramingError, read_message, write_message

__all__ = [
    "McpArgumentError",
    "McpClientError",
    "McpEchoMismatchError",
    "McpRemoteError",
    "McpStdioClient",
    "McpTimeoutError",
]

_DEFAULT_TIMEOUT: Final[float] = 30.0
_EOF_ERROR_CODE: Final[int] = -32000

_LOGGER = logging.getLogger("hermes.mcp.client")


class McpClientError(Exception):
    """Base class for MCP client failures."""


class McpArgumentError(McpClientError):
    """Request arguments failed the caller-side schema validation."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("argument validation failed: " + "; ".join(errors))
        self.errors: Final[list[str]] = list(errors)


class McpTimeoutError(McpClientError):
    """The server did not answer within the request timeout."""


class McpRemoteError(McpClientError):
    """The server answered with a JSON-RPC error object."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"MCP server error {code}: {message}")
        self.code = code
        self.message = message


class McpEchoMismatchError(McpClientError):
    """The server's ``arguments_received`` echo was missing or did not match
    the arguments that were sent — the issue-#156143 drop signature."""


class McpStdioClient:
    """JSON-RPC client for a newline-delimited stdio MCP server subprocess.

    Spawns ``server_command`` with binary stdio pipes. ``request`` validates
    (optional schema), logs the full arguments, writes one framed request and
    blocks for the correlated response. ``close`` (idempotent) shuts the
    subprocess down; usable as a context manager.
    """

    def __init__(self, server_command: list[str], cwd: str | Path | None = None) -> None:
        self._pending: dict[int, queue.SimpleQueue[dict[str, Any]]] = {}
        self._next_id = itertools.count(1)
        self._lock = threading.Lock()
        self._closed = False
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._process: subprocess.Popen[bytes] = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd is not None else None,
        )
        self._reader = threading.Thread(
            target=self._read_loop, name="hermes-mcp-reader", daemon=True
        )
        self._stderr_drain = threading.Thread(
            target=self._drain_stderr, name="hermes-mcp-stderr", daemon=True
        )
        self._reader.start()
        self._stderr_drain.start()

    # ── public API ────────────────────────────────────────────────────────

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        schema: Mapping[str, Any] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Any:
        """Send one JSON-RPC request and return the ``result`` value.

        Order of defenses (issue #156143): schema-validate ``params``'s
        ``arguments`` BEFORE sending; log the full arguments; write one
        locked frame; verify the server's ``arguments_received`` echo when
        the request carried arguments. Raises :class:`McpArgumentError` on
        validation failure, :class:`McpEchoMismatchError` on echo mismatch,
        :class:`McpRemoteError` on a JSON-RPC error response and
        :class:`McpTimeoutError` when no response arrives in time.
        """
        sent_params: dict[str, Any] = dict(params) if params is not None else {}
        if schema is not None:
            arguments = sent_params.get("arguments")
            errors = validate_arguments(
                schema, arguments if isinstance(arguments, Mapping) else {}
            )
            if errors:
                raise McpArgumentError(errors)

        with self._lock:
            if self._closed:
                raise McpClientError("MCP client is closed")
            request_id = next(self._next_id)
            message: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": sent_params,
            }
            waiter: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
            self._pending[request_id] = waiter  # registered BEFORE the write
            self._log_request(request_id, method, sent_params)
            self._write_locked(message)

        response = self._await_response(waiter, request_id, method, timeout)

        error = response.get("error")
        if error is not None:
            if isinstance(error, Mapping):
                raise McpRemoteError(
                    int(error.get("code", _EOF_ERROR_CODE)),
                    str(error.get("message", "unknown error")),
                )
            raise McpRemoteError(_EOF_ERROR_CODE, str(error))
        if "result" not in response:
            raise McpClientError(f"malformed response for id={request_id}: {response!r}")
        result = response["result"]

        if "arguments" in sent_params:
            self._verify_echo(result, sent_params["arguments"])
        return result

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        schema: Mapping[str, Any] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Any:
        """Convenience wrapper: ``request("tools/call", ...)`` for one tool."""
        return self.request(
            "tools/call",
            {"name": tool_name, "arguments": dict(arguments)},
            schema=schema,
            timeout=timeout,
        )

    def close(self) -> None:
        """Terminate the server and fail any in-flight requests. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._process.stdin is not None:
            with contextlib.suppress(OSError):
                self._process.stdin.close()
        with contextlib.suppress(OSError):
            self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        self._reader.join(timeout=5)
        self._stderr_drain.join(timeout=2)
        for stream in (self._process.stdout, self._process.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    def __enter__(self) -> McpStdioClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── internals ─────────────────────────────────────────────────────────

    def _write_locked(self, message: dict[str, Any]) -> None:
        """Write one frame; caller holds ``self._lock`` (serializes writes)."""
        stdin = self._process.stdin
        assert stdin is not None, "client spawned with stdin=PIPE"
        try:
            write_message(stdin, message)
        except OSError as exc:
            request_id = message.get("id")
            if isinstance(request_id, int):
                waiter = self._pending.pop(request_id, None)
                if waiter is not None:
                    waiter.put(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": _EOF_ERROR_CODE,
                                "message": "failed to write request to MCP server",
                            },
                        }
                    )
            raise McpClientError(f"failed to write request: {exc}") from exc

    def _log_request(
        self, request_id: int, method: str, params: Mapping[str, Any]
    ) -> None:
        """Full-argument request log — the audit trail for dropped arguments."""
        _LOGGER.info(
            "mcp request id=%s method=%s params=%s",
            request_id,
            method,
            json.dumps(dict(params), ensure_ascii=False, default=str),
        )

    def _await_response(
        self,
        waiter: queue.SimpleQueue[dict[str, Any]],
        request_id: int,
        method: str,
        timeout: float,
    ) -> dict[str, Any]:
        """Block on the id-correlated queue; deregister the entry on timeout."""
        try:
            return waiter.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                self._pending.pop(request_id, None)
            raise McpTimeoutError(
                f"no response for id={request_id} method={method!r} within {timeout}s"
            ) from None

    def _verify_echo(self, result: Any, sent_arguments: Any) -> None:
        """Require ``result.arguments_received == sent_arguments`` (#156143)."""
        if not isinstance(result, Mapping) or "arguments_received" not in result:
            raise McpEchoMismatchError(
                "MCP server did not echo received arguments "
                "(missing 'arguments_received' in result)"
            )
        echoed = result["arguments_received"]
        if echoed != sent_arguments:
            raise McpEchoMismatchError(
                "server echoed different arguments: "
                f"sent={sent_arguments!r} received={echoed!r}"
            )

    def _read_loop(self) -> None:
        """Reader thread: frame reads, dispatch by id, fail pendants on EOF."""
        stdout = self._process.stdout
        assert stdout is not None, "client spawned with stdout=PIPE"
        while True:
            try:
                message = read_message(stdout)
            except FramingError as exc:
                if exc.eof:
                    break
                _LOGGER.warning("discarding malformed frame from MCP server: %s", exc)
                continue
            self._dispatch_response(message)
        self._fail_all_pending("MCP server closed the stream")

    def _dispatch_response(self, message: dict[str, Any]) -> None:
        """Route one server message to its pending queue by JSON-RPC id."""
        request_id = message.get("id")
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            _LOGGER.warning("ignoring server message without numeric id: %r", message)
            return
        with self._lock:
            waiter = self._pending.pop(request_id, None)
        if waiter is None:
            _LOGGER.warning("no pending request for id=%s", request_id)
            return
        _LOGGER.debug("mcp response id=%s", request_id)
        waiter.put(message)

    def _fail_all_pending(self, reason: str) -> None:
        """Answer every queued waiter with an error so no caller hangs."""
        with self._lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for request_id, waiter in pending:
            waiter.put(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": _EOF_ERROR_CODE, "message": reason},
                }
            )

    def _drain_stderr(self) -> None:
        """Keep the server's stderr pipe drained (deadlock prevention)."""
        stderr = self._process.stderr
        if stderr is None:
            return
        for raw in iter(stderr.readline, b""):
            self._stderr_tail.append(raw.decode("utf-8", errors="replace").rstrip())
