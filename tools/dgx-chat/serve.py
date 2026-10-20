#!/usr/bin/env python3
"""
Локальный прокси для чата с DeepSeek V4.1 на dgx-forge.

Зачем: ds4-server не отдаёт статику и не ставит CORS-заголовки, поэтому
открыть index.html через file:// и ходить напрямую в туннель браузер не даст.
Этот прокси отдаёт UI с того же origin и перекладывает /v1/* в туннель.

Запуск:
    python3 serve.py                 # UI на http://127.0.0.1:8787
    python3 serve.py --port 9000
    python3 serve.py --upstream http://127.0.0.1:8000

Туннель поднимается отдельно:
    ssh -N -L 8000:127.0.0.1:8000 -i ~/.ssh/id_ed25519_gpuwerk \
        root@dgx-forge.gpuwerk.com
"""
import argparse
import http.server
import os
import socketserver
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = "http://127.0.0.1:8000"


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # Одна строка на запрос, без шума по умолчанию.
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self):
        path = os.path.join(HERE, "index.html")
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError as exc:
            self._send(500, ("index.html: %s" % exc).encode(), "text/plain; charset=utf-8")
            return
        self._send(200, body, "text/html; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_index()
            return
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_OPTIONS(self):
        # Прокси и браузер на одном origin, но preflight дешёвый — отвечаем.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length) if length else None
        url = UPSTREAM.rstrip("/") + self.path
        req = urllib.request.Request(url, data=payload, method=self.command)
        for key in ("Content-Type", "Accept", "Authorization"):
            if self.headers.get(key):
                req.add_header(key, self.headers[key])
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as exc:
            body = exc.read() or b""
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            msg = ("нет связи с %s (%s).\n"
                   "Туннель поднят? ssh -N -L 8000:127.0.0.1:8000 "
                   "-i ~/.ssh/id_ed25519_gpuwerk root@dgx-forge.gpuwerk.com\n"
                   % (UPSTREAM, exc)).encode()
            self._send(502, msg, "text/plain; charset=utf-8")


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream", default=UPSTREAM)
    args = ap.parse_args()
    UPSTREAM = args.upstream

    with Server((args.host, args.port), Handler) as srv:
        print("UI:       http://%s:%d" % (args.host, args.port))
        print("upstream: %s" % UPSTREAM)
        print("Ctrl-C — остановить")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nостановлено")


if __name__ == "__main__":
    main()
