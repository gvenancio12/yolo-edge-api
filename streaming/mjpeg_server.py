"""Servidor MJPEG com OSD para abrir diretamente no navegador."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from streaming.pipeline import LivePipeline, StreamConfig


class MjpegHandler(BaseHTTPRequestHandler):
    server_version = "EdgeAI-MJPEG/1.0"

    @property
    def pipeline(self) -> LivePipeline:
        return self.server.pipeline  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._write_html()
        elif self.path == "/health":
            self._write_health()
        elif self.path == "/snapshot.jpg":
            self._write_snapshot()
        elif self.path == "/stream.mjpg":
            self._write_stream()
        else:
            self.send_error(404, "Use /, /stream.mjpg, /snapshot.jpg ou /health.")

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _write_html(self) -> None:
        content = """<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>
<title>Edge AI MJPEG</title></head><body><h1>Edge AI — MJPEG com OSD</h1>
<img src='/stream.mjpg' alt='Stream da camera CSI'></body></html>""".encode(
            "utf-8"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _write_health(self) -> None:
        content = json.dumps(
            {"status": "ok", "service": "mjpeg", "stream": "/stream.mjpg"}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _write_snapshot(self) -> None:
        _, jpeg = self.pipeline.wait_for_jpeg(0, timeout_s=10)
        if jpeg is None:
            self.send_error(503, "A câmera ainda não produziu um frame.")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _write_stream(self) -> None:
        self.send_response(200)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        version = 0
        try:
            while True:
                version, jpeg = self.pipeline.wait_for_jpeg(version, timeout_s=10)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--infer-every", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StreamConfig(
        api_url=args.api_url,
        confidence=args.confidence,
        width=args.width,
        height=args.height,
    )
    pipeline = LivePipeline(
        config, infer_every=args.infer_every, target_fps=args.fps
    )
    server = ThreadingHTTPServer((args.host, args.port), MjpegHandler)
    server.pipeline = pipeline  # type: ignore[attr-defined]
    pipeline.start()
    print(f"Abra http://<ip-da-raspberry>:{args.port}/ no navegador.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Encerrando servidor MJPEG.")
    finally:
        server.server_close()
        pipeline.close()


if __name__ == "__main__":
    main()
