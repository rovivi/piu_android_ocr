#!/usr/bin/env python3
"""End-to-end del NATIVO (CLI de host: detector NCNN + OCR C++ + réplica de los gates de Kotlin)
sobre las 63 fotos nuevas validadas a mano (tools/validator/dataset_v2.txt).

Complementa a parity.py, que mide sobre las 45 fotos con GT de piu_ocr/dataset_v2. Ninguna de
estas 63 está en los atlas de OCR. Las imágenes viven en piu_yolo/images/train (renombradas;
el mapa es piu_yolo/rename_map.csv).

    python3 piu_ocr/android_v2/tools/parity/new63.py            # assets del repo
    PIU_ASSETS=/otra/carpeta python3 .../new63.py                # A/B de assets
    python3 .../new63.py --device docs/device_results_new63_v4_tuned.json   # + lo que dio el teléfono
"""
import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parity as P  # noqa: E402  (se relanza con el venv si hace falta)

ROOT = P.ROOT
GT = os.path.join(P.ANDROID, "tools", "validator", "dataset_v2.txt")
RENAME = os.path.join(ROOT, "piu_yolo", "rename_map.csv")
IMGDIR = os.path.join(ROOT, "piu_yolo", "images", "train")
CT = {"s": "single", "d": "double", "h": "halfdouble", "c": "coop"}


def load_gt():
    gt = {}
    for line in open(GT, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        img, song, score, tipo, nivel, *_ = line.rstrip("\n").split("\t")
        gt[os.path.splitext(img)[0]] = {
            "song": song, "score": int(score) if score.strip().isdigit() else None,
            "chart_type": CT.get(tipo.strip().lower()),
            "level": int(nivel) if nivel.strip().isdigit() else None}
    return gt


def report(name, rows, gt, norm):
    print(f"\n== {name}  (n={len(rows)})")
    print(f"{'campo':11s} {'acierto':>8} {'cobertura':>10} {'precisión':>10} {'error aceptado':>15}")
    for f in ("song", "score", "chart_type", "level"):
        n = ok = res = 0
        for k, o in rows.items():
            g = gt[k][f]
            if g is None:
                continue
            n += 1
            v = o.get(f)
            if v is None:
                continue
            res += 1
            ok += (norm(v) == norm(g)) if f == "song" else (v == g)
        print(f"{f:11s} {ok/n:8.3f} {res/n:10.3f} {ok/max(res,1):10.3f} {(res-ok)/n:15.3f}")
    esc = sum(any(o.get(f) is None for f in ("song", "score", "level")) for o in rows.values())
    print(f"fotos con al menos un campo escalado: {esc}/{len(rows)} = {esc/len(rows):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", help="results.json del teléfono para comparar")
    ap.add_argument("--augs", help='pasadas del TTA del detector, p. ej. "1,0.83f" (como parity.py)')
    a = ap.parse_args()
    if a.augs:
        P.AUGS = a.augs
    P.build_cli()
    gt = load_gt()
    ren = {os.path.splitext(r["old_image"])[0]: r["new_image"] for r in csv.DictReader(open(RENAME))}
    catalog = P.Catalog(os.path.join(P.D2, "catalog.json"))  # mismo catálogo que parity.main
    from piu_ocr.song_match import normalize
    rows, ms = {}, []
    for i, k in enumerate(sorted(gt)):
        t = time.time()
        native = P.run_native(os.path.join(IMGDIR, ren[k]))
        ms.append((time.time() - t) * 1000)
        rows[k] = P.interpret(native.get("result", native), catalog)
        print(f"\r{i+1}/{len(gt)}", end="", flush=True)
    print()
    report(f"nativo host ({P.ASSETS})", rows, gt, normalize)
    ms.sort()
    print(f"latencia host: mediana {ms[len(ms)//2]:.0f} ms, max {ms[-1]:.0f} ms")
    if a.device:
        d = json.load(open(a.device))["rows"]
        dev = {k: {"song": v.get("song"), "score": v.get("score"), "chart_type": v.get("chart_type"),
                   "level": v.get("level")} for k, v in d.items() if k in gt}
        report(f"teléfono ({os.path.basename(a.device)})", dev, gt, normalize)


if __name__ == "__main__":
    main()
