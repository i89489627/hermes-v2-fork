"""Writer IPC protocol: newline-delimited JSON over a TCP loopback socket (plan P6.T2).

Subagents and the main agent send memory write operations to the
:class:`~hermes.concurrency.writer.WriterProcess` over this wire contract.

**Platform note (deliberate deviation from the plan text).** The plan
specifies a Unix-domain socket for the writer queue; Unix sockets do not
exist on Windows (``AF_UNIX`` is unavailable on Windows Python), so the
transport is deliberately TCP on the loopback interface — host
``127.0.0.1`` by default with ``port=0`` meaning "OS-assigned at
:meth:`WriterProcess.start` time". Loopback TCP is the portable equivalent:
same-process and same-machine clients connect, remote hosts cannot.

**Framing** follows the :mod:`hermes.mcp.protocol` pattern — newline-delimited
JSON over *binary* streams, one frame emitted as a SINGLE write + flush so
concurrent writers can never interleave partial lines, and
:class:`ProtocolError` raised on EOF or malformed frames instead of silently
returning junk — but the payload is simpler than JSON-RPC:

    WriteRequest  = {"request_id": int | str, "client_id": str, "op": {...}}
    WriteResponse = {"request_id": int | str | null, "ok": bool,
                     "result": any | null, "error": str | null}

``request_id`` is client-assigned and echoed verbatim in the response, so a
client matches responses to requests and verifies per-connection write
order. ``client_id`` names the caller (main agent, subagent-1, ...) for
logs and the crash-recovery replay seam (plan P6.T3). An error response
carries a human-readable ``error`` and never kills the connection.

**Op shapes** (the ``op`` object; applied by the writer through
:class:`~hermes.memory.store.MemoryStore`):

- ``{"kind": "add", "text": str, "type": EntryType?, "context": str?}`` —
  ``type`` defaults to ``"Feedback"`` (one of ``User``/``Feedback``/
  ``Project``/``Reference``), ``context`` defaults to ``"global"``. The
  response result carries ``entry_id``/``type``/``context``/``trust_score``.
- ``{"kind": "update", "entry_id": str, "text": str?, "trust_score": num?}``
  — both ``text`` and ``trust_score`` are optional; an unknown
  ``entry_id`` answers ``ok=False`` (``KeyError``).
- ``{"kind": "remove", "entry_id": str}`` — the result carries
  ``removed`` (``False`` when the id is unknown; removing an unknown id is
  not an error).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import IO, Any, Final, TypeGuard

__all__ = [
    "ProtocolError",
    "WriteRequest",
    "WriteResponse",
    "decode_request",
    "decode_response",
    "encode_request",
    "encode_response",
    "read_request",
    "read_response",
    "write_request",
    "write_response",
]

#: Serializes every frame write process-wide so a frame is always ONE
#: uninterruptible write + flush — the issue-#156143 lesson from
#: :mod:`hermes.mcp.protocol` applied to the writer transport.
_WRITE_LOCK: Final[threading.Lock] = threading.Lock()


class ProtocolError(Exception):
    """A stream violated the writer wire contract.

    ``eof`` marks the benign end-of-stream case (the peer closed the
    connection) so the server can distinguish "client went away" from
    "malformed frame" (the line-oriented framing stays usable after a
    malformed frame — exactly one line was consumed).
    """

    def __init__(self, message: str, *, eof: bool = False) -> None:
        super().__init__(message)
        self.eof = eof


def _valid_request_id(value: Any) -> TypeGuard[int | str]:
    """True for a client-assigned id: a JSON int (not bool) or a string."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, str))


@dataclass(frozen=True)
class WriteRequest:
    """One queued memory write operation (a single wire frame's payload)."""

    #: Client-assigned id echoed verbatim in the response.
    request_id: int | str
    #: Caller identity (e.g. ``"main"``, ``"subagent-1"``).
    client_id: str
    #: The operation to apply — see the module docstring for the shapes.
    op: dict[str, Any]

    def to_frame(self) -> dict[str, Any]:
        """The JSON object this request is encoded as."""
        return {"request_id": self.request_id, "client_id": self.client_id, "op": self.op}

    @classmethod
    def from_frame(cls, frame: dict[str, Any]) -> WriteRequest:
        """Validate a decoded JSON object into a :class:`WriteRequest`.

        Raises :class:`ProtocolError` on missing/mistyped fields.
        """
        request_id = frame.get("request_id")
        if not _valid_request_id(request_id):
            raise ProtocolError(
                f"request_id must be an int or string, got {request_id!r}"
            )
        client_id = frame.get("client_id")
        if not isinstance(client_id, str):
            raise ProtocolError(f"client_id must be a string, got {client_id!r}")
        op = frame.get("op")
        if not isinstance(op, dict):
            raise ProtocolError(f"op must be a JSON object, got {type(op).__name__}")
        return cls(request_id=request_id, client_id=client_id, op=op)


@dataclass(frozen=True)
class WriteResponse:
    """The writer's answer to one :class:`WriteRequest`.

    ``ok=True`` means the op was applied through the store (``result``
    carries the applied outcome, e.g. the new ``entry_id``). ``ok=False``
    carries ``error`` and leaves the connection usable — a failing op
    never wedges the writer or the client.
    """

    #: Echo of the request's id (``None`` when the frame was too malformed
    #: to carry one, e.g. undecodable JSON).
    request_id: int | str | None
    #: True when the op was applied through the store.
    ok: bool
    #: Applied outcome on success (e.g. ``{"entry_id": ...}``), else None.
    result: Any | None = None
    #: Human-readable failure reason on ``ok=False``, else None.
    error: str | None = None

    def to_frame(self) -> dict[str, Any]:
        """The JSON object this response is encoded as."""
        return {
            "request_id": self.request_id,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_frame(cls, frame: dict[str, Any]) -> WriteResponse:
        """Validate a decoded JSON object into a :class:`WriteResponse`.

        Raises :class:`ProtocolError` on missing/mistyped fields.
        """
        request_id = frame.get("request_id")
        if request_id is not None and not _valid_request_id(request_id):
            raise ProtocolError(
                f"request_id must be an int, string or null, got {request_id!r}"
            )
        ok = frame.get("ok")
        if not isinstance(ok, bool):
            raise ProtocolError(f"ok must be a bool, got {ok!r}")
        result = frame.get("result")
        error = frame.get("error")
        if error is not None and not isinstance(error, str):
            raise ProtocolError(f"error must be a string, got {error!r}")
        return cls(request_id=request_id, ok=ok, result=result, error=error)


def encode_request(request: WriteRequest) -> bytes:
    """One complete wire frame for ``request`` (compact JSON + ``"\\n"``)."""
    return _encode_frame(request.to_frame())


def encode_response(response: WriteResponse) -> bytes:
    """One complete wire frame for ``response`` (compact JSON + ``"\\n"``)."""
    return _encode_frame(response.to_frame())


def _encode_frame(frame: dict[str, Any]) -> bytes:
    """Compact-JSON encoding of one frame with the newline terminator."""
    line = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
    return (line + "\n").encode("utf-8")


def decode_request(frame: dict[str, Any]) -> WriteRequest:
    """Alias of :meth:`WriteRequest.from_frame` (encode/decode helper pair)."""
    return WriteRequest.from_frame(frame)


def decode_response(frame: dict[str, Any]) -> WriteResponse:
    """Alias of :meth:`WriteResponse.from_frame` (encode/decode helper pair)."""
    return WriteResponse.from_frame(frame)


def _read_frame(stream: IO[bytes]) -> dict[str, Any]:
    """Read one newline-terminated JSON object from ``stream``.

    Raises :class:`ProtocolError` (``eof=True``) on a closed stream and
    :class:`ProtocolError` (``eof=False``) on malformed frames — empty
    lines, undecodable bytes, non-JSON payloads or non-object JSON.
    """
    line = stream.readline()
    if not line:
        raise ProtocolError("unexpected end of stream", eof=True)
    stripped = line.strip()
    if not stripped:
        raise ProtocolError("empty frame")
    try:
        frame = json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"malformed JSON frame: {exc}") from exc
    if not isinstance(frame, dict):
        raise ProtocolError(f"frame is not a JSON object: {type(frame).__name__}")
    return frame


def _write_frame(stream: IO[bytes], payload: bytes) -> None:
    """Emit one frame as a single write + flush under the process-wide lock."""
    with _WRITE_LOCK:
        stream.write(payload)
        stream.flush()


def read_request(stream: IO[bytes]) -> WriteRequest:
    """Read and validate one :class:`WriteRequest` frame from ``stream``."""
    return decode_request(_read_frame(stream))


def read_response(stream: IO[bytes]) -> WriteResponse:
    """Read and validate one :class:`WriteResponse` frame from ``stream``."""
    return decode_response(_read_frame(stream))


def write_request(stream: IO[bytes], request: WriteRequest) -> None:
    """Write one :class:`WriteRequest` frame to ``stream`` (single write)."""
    _write_frame(stream, encode_request(request))


def write_response(stream: IO[bytes], response: WriteResponse) -> None:
    """Write one :class:`WriteResponse` frame to ``stream`` (single write)."""
    _write_frame(stream, encode_response(response))
