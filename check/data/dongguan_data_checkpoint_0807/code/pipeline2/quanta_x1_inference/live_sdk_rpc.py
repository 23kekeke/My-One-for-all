"""Length-prefixed JSON-RPC over TCP for live SDK daemon <-> live_runner."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse


def parse_host_port(url: str, *, default_port: int) -> tuple[str, int]:
    if "://" in url:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or default_port
        return host, int(port)
    if ":" in url:
        host, port_text = url.rsplit(":", 1)
        return host, int(port_text)
    return url, default_port


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_message(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def decode_message(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, 4)
    length = struct.unpack(">I", header)[0]
    if length > 64 * 1024 * 1024:
        raise ValueError(f"RPC message too large: {length} bytes")
    body = recv_exact(sock, length)
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("RPC payload must be a JSON object")
    return value


class JsonRpcClient:
    """One persistent TCP connection; serializes requests with a lock."""

    def __init__(self, url: str, *, default_port: int = 15100, timeout_sec: float = 120.0):
        self.url = url
        self.default_port = default_port
        self.timeout_sec = timeout_sec
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._request_id = 0

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def _ensure_connected(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        host, port = parse_host_port(self.url, default_port=self.default_port)
        sock = socket.create_connection((host, port), timeout=self.timeout_sec)
        sock.settimeout(self.timeout_sec)
        self._sock = sock
        return sock

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            request = {"id": req_id, "method": method, "params": params or {}}
            last_error: Exception | None = None
            for attempt in (1, 2):
                try:
                    sock = self._ensure_connected()
                    sock.sendall(encode_message(request))
                    response = decode_message(sock)
                    if response.get("id") != req_id:
                        raise RuntimeError(
                            f"RPC id mismatch: sent {req_id}, got {response.get('id')}"
                        )
                    if not response.get("ok", False):
                        raise RuntimeError(str(response.get("error", "RPC failed")))
                    result = response.get("result")
                    if not isinstance(result, dict):
                        raise TypeError("RPC result must be an object")
                    return result
                except (ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                    last_error = exc
                    self.close()
                    if attempt == 2:
                        break
                    time.sleep(0.05)
            raise RuntimeError(f"RPC {method} failed: {last_error!r}") from last_error


def ping_daemon(url: str, *, default_port: int = 15100, timeout_sec: float = 5.0) -> bool:
    client = JsonRpcClient(url, default_port=default_port, timeout_sec=timeout_sec)
    try:
        client.call("ping")
        return True
    except Exception:
        return False
    finally:
        client.close()


def serve_forever(
    *,
    bind_host: str,
    bind_port: int,
    handler: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_host, bind_port))
    server.listen(32)
    print(f"live_sdk_daemon LISTEN {bind_host}:{bind_port}", flush=True)

    while True:
        conn, addr = server.accept()
        threading.Thread(
            target=_serve_client_connection,
            args=(conn, addr, handler),
            daemon=True,
        ).start()


def _serve_client_connection(
    conn: socket.socket,
    addr: tuple[str, int],
    handler: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    conn.settimeout(120.0)
    try:
        while True:
            try:
                request = decode_message(conn)
            except ConnectionError:
                break
            req_id = request.get("id")
            method = str(request.get("method", ""))
            params = request.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            started = time.perf_counter()
            try:
                result = handler(method, params)
                wall_ms = (time.perf_counter() - started) * 1000.0
                if isinstance(result, dict) and "daemon_wall_ms" not in result:
                    result = {**result, "daemon_wall_ms": float(wall_ms)}
                response = {"id": req_id, "ok": True, "result": result}
            except Exception as exc:
                wall_ms = (time.perf_counter() - started) * 1000.0
                response = {
                    "id": req_id,
                    "ok": False,
                    "error": repr(exc),
                    "daemon_wall_ms": float(wall_ms),
                }
            try:
                conn.sendall(encode_message(response))
            except OSError:
                break
            if method == "shutdown":
                break
    finally:
        try:
            conn.close()
        except OSError:
            pass
