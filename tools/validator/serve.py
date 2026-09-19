#!/usr/bin/env python3
"""Servidor mínimo para validar dataset_v2.json a mano.

Sirve la UI, el JSON y las imágenes (desde --images, por defecto las fotos de
piu_yolo/images/new), y guarda los cambios del validador con POST /save.

Uso:
  python3 tools/validator/serve.py [--port 8765] [--images DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGES = Path("/Users/rodrigo/dev/python_projects/piu_yolo/images/new")
DATASET = HERE / "dataset_v2.json"
DATASET_TXT = HERE / "dataset_v2.txt"
CATALOG = HERE.parent.parent / "src/main/assets/piu_ocr/catalog.json"
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def write_txt(data: dict):
    """Vuelca lo validado a texto plano: imagen<TAB>canción<TAB>score<TAB>tipo<TAB>nivel<TAB>ok."""
    lines = ["# imagen\tsong\tscore\ttipo\tnivel\tvalidado"]
    for it in data["items"]:
        song = it.get("song_final") or it.get("song_ocr") or ""
        score = it.get("score_final")
        if score is None:
            score = it.get("score_ocr")
        ctype = it.get("chart_final") or ""
        level = it.get("level_final")
        lines.append(f"{it['file']}\t{song}\t{'' if score is None else score}\t{ctype}"
                     f"\t{'' if level is None else level}\t{'1' if it.get('validated') else '0'}")
    DATASET_TXT.write_text("\n".join(lines) + "\n")

IMAGES = DEFAULT_IMAGES


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/dataset_v2.json":
            return self._send(200, DATASET.read_bytes(), "application/json; charset=utf-8")
        if path == "/catalog.json":
            return self._send(200, CATALOG.read_bytes(), "application/json; charset=utf-8")
        if path.startswith("/img/"):
            name = Path(path[5:]).name
            f = IMAGES / name
            if f.is_file():
                return self._send(200, f.read_bytes(), MIME.get(f.suffix.lower(), "application/octet-stream"))
            return self._send(404, b"imagen no encontrada", "text/plain")
        return self._send(404, b"no existe", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/save":
            return self._send(404, b"no existe", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n))
        except Exception as e:
            return self._send(400, f"json inválido: {e}".encode(), "text/plain")
        if not isinstance(data.get("items"), list):
            return self._send(400, b"falta items[]", "text/plain")
        if DATASET.exists():
            shutil.copy(DATASET, DATASET.with_suffix(".json.bak"))
        DATASET.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        write_txt(data)
        done = sum(1 for i in data["items"] if i.get("validated"))
        return self._send(200, json.dumps({"ok": True, "validated": done, "total": len(data["items"]),
                                           "txt": str(DATASET_TXT)}).encode(),
                          "application/json")

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


def main():
    global IMAGES
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    args = ap.parse_args()
    IMAGES = Path(args.images)
    if not IMAGES.is_dir():
        raise SystemExit(f"no existe el directorio de imágenes: {IMAGES}")
    print(f"validador en http://localhost:{args.port}  (imágenes: {IMAGES})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
