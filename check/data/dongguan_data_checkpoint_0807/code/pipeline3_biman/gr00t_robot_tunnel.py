#!/usr/bin/env python3
"""Local TCP tunnel: 127.0.0.1:15051 -> jump host -> robot SDK gRPC.

Topology (2026-07):
  dev machine   LOCAL  127.0.0.1:15051   (x2robot SDK connects here)
  jump host     SSH    192.168.36.177    user yichu
  robot         SDK    192.168.36.246:50051  (robot listens on 50051, NOT 15051)

Usage:
  /home/ubuntu/anaconda3/bin/python3 gr00t_robot_tunnel.py
  # or symlink/copy to /tmp/gr00t_robot_tunnel.py

Keep this terminal open; tunnel may die after ~1h — restart if SDK times out.
"""

from __future__ import annotations

import select
import socket
import threading

import paramiko

LOCAL_BIND = "127.0.0.1"
LOCAL_PORT = 15051

JUMP_HOST = "192.168.36.177"
JUMP_USER = "yichu"
JUMP_PASSWORD = "123123"

REMOTE_HOST = "192.168.36.246"
REMOTE_PORT = 50051


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        JUMP_HOST,
        username=JUMP_USER,
        password=JUMP_PASSWORD,
        timeout=15,
        allow_agent=False,
        look_for_keys=False,
    )
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError(f"SSH transport unavailable to jump host {JUMP_HOST}")
    transport.set_keepalive(30)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LOCAL_BIND, LOCAL_PORT))
    server.listen(50)
    print(
        f"LISTEN ok  local={LOCAL_BIND}:{LOCAL_PORT} "
        f"-> jump={JUMP_HOST} -> robot={REMOTE_HOST}:{REMOTE_PORT}",
        flush=True,
    )

    def handle(client_sock: socket.socket) -> None:
        try:
            chan = transport.open_channel(
                "direct-tcpip",
                (REMOTE_HOST, REMOTE_PORT),
                client_sock.getpeername(),
            )
        except Exception as exc:
            print(f"chan {exc}", flush=True)
            client_sock.close()
            return
        try:
            while True:
                readable, _, _ = select.select([client_sock, chan], [], [], 60)
                if client_sock in readable:
                    data = client_sock.recv(65536)
                    if not data:
                        break
                    chan.sendall(data)
                if chan in readable:
                    data = chan.recv(65536)
                    if not data:
                        break
                    client_sock.sendall(data)
        finally:
            try:
                client_sock.close()
            except OSError:
                pass
            try:
                chan.close()
            except Exception:
                pass

    while True:
        server.settimeout(1.0)
        try:
            conn, _ = server.accept()
        except socket.timeout:
            if not transport.is_active():
                raise RuntimeError("SSH transport dead; restart tunnel") from None
            continue
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
