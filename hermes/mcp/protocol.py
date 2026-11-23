"""hermes-v2 MCP stdio framing (plan P21.T3, issue #156143).

Upstream bug: MCP stdio ``tools/call`` arguments were intermittently dropped
— the server executed the tool with defaults while the caller saw no error.
One contributing cause was sloppy stdio framing: multi-write payloads that
could interleave under concurrency, hand-rolled parsing that skipped or
truncated frames, and no verification that the server received what was
sent.

This module pins down the strict wire layer the v2 client and test servers
share:

- **Framing**: newline-delimited JSON-RPC 2.0 over *binary* streams.
  :func:`write_message` emits ``json.dumps(compact) + "\\n"`` as a SINGLE
  write under a module-level lock, so concurrent writers can never interleave
  partial lines (an argument-dropping vector). :func:`read_message` reads
  exactly one line and raises :class:`FramingError` on EOF or malformed
  content instead of silently returning junk.
- **Server loop**: :func:`serve_stdio` dispatches each request to a handler
  and wraps its return value into a JSON-RPC response (or a ``-32603`` error
  response if the handler raises, so a buggy tool call never wedges the
  framing loop). Requests without an ``id`` are notifications: the handler
  runs but no response is written.

The echo contract used against the argument-dropping bug lives in
:mod:`hermes.mcp.client`: servers answer ``tools/call`` with an
``arguments_received`` field containing the EXACT arguments received.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Callable, Mapping
from typing import IO, Any, Final

__all__ = [
    "FramingError",
    "RequestHandler",
    "read_message",
    "serve_stdio",
    "write_message",
]

_LOGGER = logging.getLogger("hermes.mcp.server")

#: Serializes every :func:`write_message` call process-wide so a frame is
#: always ONE uninterruptible write; interleaved partial lines are exactly
#: how arguments got dropped upstream (issue #156143).
_WRITE_LOCK: Final[threading.Lock] = threading.Lock()

_JSONRPC_PARSE_ERROR: Final[int] = -32700
_JSONRPC_INTERNAL_ERROR: Final[int] = -32603

#: Handler contract: receive the raw request mapping, return the RESULT
#: payload (``serve_stdio`` wraps it with ``jsonrpc``/``id``), or ``None`` for
#: a notification (no response written).
RequestHandler = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]


class FramingError(Exception):
    """A stream violated the newline-delimited JSON framing contract.

    ``eof`` marks the benign end-of-stream case (the peer closed the pipe)
    so callers can distinguish "server went away" from "malformed frame".
    """

    def __init__(self, message: str, *, eof: bool = False) -> None:
        super().__init__(message)
        self.eof = eof


def write_message(stream: IO[bytes], message: Mapping[str, Any]) -> None:
    """Write one message: compact JSON + ``"\\n"``, single write under a lock.

    The whole frame is encoded first and flushed in one ``write`` call so a
    concurrent writer on another thread can never interleave bytes into the
    middle of it (issue #156143).
    """
    line = json.dumps(dict(message), separators=(",", ":"), ensure_ascii=False)
    payload = (line + "\n").encode("utf-8")
    with _WRITE_LOCK:
        stream.write(payload)
        stream.flush()


def read_message(stream: IO[bytes]) -> dict[str, Any]:
    """Read one newline-terminated JSON object from ``stream``.

    Raises :class:`FramingError` (``eof=True``) on a closed stream and
    :class:`FramingError` (``eof=False``) on malformed frames — empty lines,
    undecodable bytes, non-JSON payloads or non-object JSON. Never returns
    a silent partial message.
    """
    line = stream.readline()
    if not line:
        raise FramingError("unexpected end of stream", eof=True)
    stripped = line.strip()
    if not stripped:
        raise FramingError("empty frame")
    try:
        message = json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FramingError(f"malformed JSON frame: {exc}") from exc
    if not isinstance(message, dict):
        raise FramingError(f"frame is not a JSON object: {type(message).__name__}")
    return message


def serve_stdio(
    handler: RequestHandler,
    *,
    stdin: IO[bytes] | None = None,
    stdout: IO[bytes] | None = None,
) -> None:
    """Minimal newline-delimited JSON-RPC server loop over binary streams.

    Reads requests from ``stdin`` (default ``sys.stdin.buffer``) until EOF.
    Each request is dispatched to ``handler``; the returned mapping becomes
    the ``result`` of a JSON-RPC response written to ``stdout`` (default
    ``sys.stdout.buffer``). A handler exception becomes a ``-32603`` error
    response with the request's id, keeping the loop alive. A malformed
    frame gets a ``-32700`` parse error and the loop continues; EOF ends it.

    Used by the hermes test MCP servers (subprocess bootstraps that import
    this module) and suitable as the skeleton for real servers.
    """
    input_stream = stdin if stdin is not None else sys.stdin.buffer
    output_stream = stdout if stdout is not None else sys.stdout.buffer
    _LOGGER.debug("serve_stdio loop starting")
    while True:
        try:
            request = read_message(input_stream)
        except FramingError as exc:
            if exc.eof:
                _LOGGER.debug("serve_stdio: stdin closed, exiting")
                return
            _LOGGER.warning("serve_stdio: %s", exc)
            write_message(
                output_stream,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": _JSONRPC_PARSE_ERROR, "message": str(exc)},
                },
            )
            continue
        response = _dispatch(handler, request)
        if response is not None:
            write_message(output_stream, response)


def _dispatch(
    handler: RequestHandler, request: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Run ``handler`` and build the response mapping (``None`` = no reply).

    Notifications (no ``id`` key) never produce a response; handler failures
    on requests become JSON-RPC internal-error responses so the caller always
    gets an answer instead of hanging.
    """
    is_notification = "id" not in request
    request_id = request.get("id")
    try:
        result = handler(request)
    except Exception as exc:  # handler bugs must not wedge the framing loop
        _LOGGER.exception("handler failed for request id=%r", request_id)
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": _JSONRPC_INTERNAL_ERROR,
                "message": f"{type(exc).__name__}: {exc}",
            },
        }
    if is_notification or result is None:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}
