#!/usr/bin/env python3
"""Infiere tipo+nivel de cada imagen leyendo el badge (visión) y lo cruza contra
los charts del catálogo. Escribe chart_final/level_final en dataset_v2.json y
regenera dataset_v2.txt, preservando la validación manual ya hecha.

El orden de CHART sigue el orden de items de dataset_v2.json (= nombres de
archivo ordenados). None = no se pudo leer el badge (queda sin inferir).

Uso:
  python3 tools/validator/infer_charts.py
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import serve

HERE = Path(__file__).resolve().parent
CATALOG = HERE.parent.parent / "src/main/assets/piu_ocr/catalog.json"
DATASET = HERE / "dataset_v2.json"

# (tipo, nivel) leídos del badge. s=SINGLE d=DOUBLE hd=HALFDOUBLE c=COOP.
CHART = [
    ("d", 23), ("d", 25), ("d", 26), ("s", 23), ("s", 21), ("d", 25), ("d", 23), ("d", 26),
    ("s", 23), ("d", 26), ("s", 22), ("s", 25), ("s", 25), ("d", 25), ("s", 23), ("d", 28),
    ("s", 22), ("s", 24), ("s", 21), ("s", 23), ("d", 24), ("s", 22), ("s", 19), ("s", 22),
    ("s", 19), ("s", 19), ("s", 22), ("s", 19), ("d", 24), ("d", 22), ("d", 24), ("s", 19),
    ("s", 19), ("s", 24), ("s", 25), ("s", 24), ("s", 23), ("s", 21), ("s", 18), ("d", 23),
    ("s", 15), ("s", 23), ("d", 27), ("d", 25), ("s", 20), ("d", 24), ("d", 16), ("d", 16),
    ("s", 18), ("s", 20), None,       ("s", 18), ("s", 18), ("s", 21), ("s", 24), ("s", 21),
    ("s", 15), ("d", 17), ("s", 16), ("s", 20), ("s", 19), ("s", 21), ("d", 22),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def charts_by_song() -> dict:
    cat = json.loads(CATALOG.read_text())
    return {norm(v["n"]): [(c[0], int(c[1])) for c in v.get("c", []) if len(c) > 1]
            for v in cat.values()}


def main():
    data = json.loads(DATASET.read_text())
    items = data["items"]
    assert len(items) == len(CHART), f"{len(items)} items vs {len(CHART)} lecturas"
    charts = charts_by_song()
    ok = bad = unknown = 0
    for it, chart in zip(items, CHART):
        if chart is None:
            it["chart_final"] = it.get("chart_final")
            it["level_final"] = it.get("level_final")
            it["chart_source"] = "sin leer (badge fuera de encuadre)"
            unknown += 1
            continue
        ctype, level = chart
        it["chart_final"], it["level_final"] = ctype, level
        names = [it.get("song_final"), (it.get("catalog_match") or {}).get("name"), it["song_ocr"]]
        song = next((n for n in names if n and charts.get(norm(n))), it["song_ocr"])
        valid = charts.get(norm(song))
        if valid is None:
            it["chart_source"] = f"visión (canción fuera del catálogo: {song})"
            unknown += 1
        elif (ctype, level) in valid:
            it["chart_source"] = "visión + catálogo ✓"
            ok += 1
        else:
            it["chart_source"] = f"visión, NO está en catálogo ({valid})"
            bad += 1
    DATASET.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    serve.DATASET_TXT = HERE / "dataset_v2.txt"
    serve.write_txt(data)
    print(f"inferidos: {ok} validados contra catálogo | {bad} fuera | {unknown} sin dato")
    if bad:
        for it in items:
            if str(it.get("chart_source", "")).startswith("visión, NO"):
                print("  ⚠", it["file"][:34], it["song_ocr"], it["chart_source"])


if __name__ == "__main__":
    main()
