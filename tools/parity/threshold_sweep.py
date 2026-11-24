#!/usr/bin/env python3
"""Barrido de umbrales (gates) del interpret() sobre el resultado real de device.

Re-corre la réplica de PiuOcr.interpret con distintos valores de cada gate sobre
el JSON crudo del device (detector v4, docs/device_results_new63_v4_tuned.json)
y evalúa contra el ground truth validado a mano (tools/validator/dataset_v2.txt).
No hace inferencia ni corre el detector: son funciones puras del JSON + catálogo.

Para cada gate barre sus valores uno a uno (el resto en el valor actual) y busca
el mejor total de acierto con un piso de precisión de canción. Luego prueba el
grid alrededor de los mejores y reporta los falsos positivos del config elegido.

Uso:
  python3 tools/parity/threshold_sweep.py
  python3 tools/parity/threshold_sweep.py --json docs/device_results_new63_v4_tuned.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import parity  # noqa: E402  (usa sus globals como gates, ver interpret)
from song_match_ref import Catalog  # noqa: E402
from parity import interpret, normalize  # noqa: E402

ASSETS = os.path.join(REPO, "src", "main", "assets", "piu_ocr", "catalog.json")
DEFAULT_JSON = os.path.join(REPO, "docs", "device_results_new63_v4_tuned.json")
DEFAULT_GT = os.path.join(REPO, "tools", "validator", "dataset_v2.txt")

# Valores actuales del repo (parity.py / PiuOcr.kt).
CURRENT = {
    "MIN_SONG_MARGIN": parity.MIN_SONG_MARGIN,
    "MIN_RAW_SIMILARITY": parity.MIN_RAW_SIMILARITY,
    "MIN_RAW_SIMILARITY_SHORT": parity.MIN_RAW_SIMILARITY_SHORT,
    "MIN_SCORE_MARGIN": parity.MIN_SCORE_MARGIN,
    "MIN_BADGE_CONF": parity.MIN_BADGE_CONF,
    "MIN_RAW_LEN": parity.MIN_RAW_LEN,
    "MIN_TITLE_CONF": parity.MIN_TITLE_CONF,
}

# (gate, valores). Barrido enfocado en bajar umbrales y matar falsos positivos.
# MIN_RAW_LEN es el tope del piso RELATIVO al largo del nombre ganador
# (floor = min(MIN_RAW_LEN, len(nombre))). 0 = gate apagado.
AXES = [
    ("MIN_SONG_MARGIN", [0.0000, 0.0003, 0.001, 0.003, 0.005, 0.010, 0.030]),
    ("MIN_RAW_SIMILARITY", [0.00, 0.20, 0.35, 0.45, 0.55, 0.65]),
    ("MIN_RAW_SIMILARITY_SHORT", [0.00, 0.50, 0.60, 0.70, 0.80, 0.95]),
    ("MIN_SCORE_MARGIN", [0.000, 0.001, 0.003, 0.005, 0.010, 0.020]),
    ("MIN_BADGE_CONF", [0.00, 0.05, 0.10, 0.15, 0.25, 0.35]),
    ("MIN_RAW_LEN", [0, 2, 3, 4, 5, 6]),
    ("MIN_TITLE_CONF", [0.0, 0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]),
]

CT = {"single": "s", "double": "d", "halfdouble": "hd", "coop": "c"}
FIELDS = [  # (campo del interpret, campo del GT, igualdad)
    ("song", "song", lambda v, g: normalize(v) == normalize(g)),
    ("chart_type", "chart", lambda v, g: CT.get(v) == g),
    ("level", "level", lambda v, g: v == g),
    ("score", "score", lambda v, g: v == g),
]


def load(dev_json, gt_txt):
    dev = json.load(open(dev_json))
    gt = {}
    for line in open(gt_txt, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        gt[p[0].rsplit(".", 1)[0]] = {
            "song": p[1], "score": int(p[2]) if p[2] else None,
            "chart": p[3], "level": int(p[4]) if p[4] else None,
            "validated": p[5],
        }
    catalog = Catalog(ASSETS)
    rows = []
    for key, r in dev["rows"].items():
        if key not in gt or gt[key]["validated"] != "1":
            continue
        rows.append({"key": key, "gt": gt[key], "native": r["native"]})
    return rows, catalog


def set_gates(gates):
    for k, v in gates.items():
        setattr(parity, k, v)


def evaluate(rows, catalog, gates):
    set_gates(gates)
    res = []
    for r in rows:
        g = r["gt"]
        o = interpret(r["native"], catalog)
        res.append((r["key"], g, o))
    total = 0
    m = {}
    for of, gf, eq in FIELDS:
        n = cov = ok = 0
        fplist = []
        for key, g, o in res:
            gv = g[gf]
            if gv is None:
                continue
            n += 1
            v = o[of]
            if v is not None:
                cov += 1
                if eq(v, gv):
                    ok += 1
                else:
                    fplist.append((key, gv, v))
            else:
                fplist.append((key, gv, None))
        m[of] = {"n": n, "cov": cov, "ok": ok,
                 "coverage": cov / n, "precision": ok / cov if cov else 0.0,
                 "accuracy": ok / n, "misses": fplist}
        total += ok
    m["total"] = total
    return m


def fmt(m):
    def one(f):
        d = m[f]
        return f"{f[:5]} {d['cov']:>3}/{d['n']:<3} c{d['coverage']:.3f} p{d['precision']:.3f} a{d['accuracy']:.3f}"
    return "  ".join(one(f) for f, _, _ in FIELDS) + f"  total {m['total']}/{sum(m[f]['n'] for f, _, _ in FIELDS)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--min-song-prec", type=float, default=0.85,
                    help="piso de precisión de canción para elegir el mejor config")
    a = ap.parse_args()

    rows, catalog = load(a.json, a.gt)
    print(f"{len(rows)} fotos de device + GT validado (catálogo {os.path.basename(ASSETS)})")
    print("actual (umbrales del repo): " + fmt(evaluate(rows, catalog, CURRENT)))

    for gate, values in AXES:
        print(f"\n— {gate} (actual {CURRENT[gate]}) —")
        best_val, best_m = None, None
        for v in values:
            gates = dict(CURRENT, **{gate: v})
            m = evaluate(rows, catalog, gates)
            mark = ""
            if best_m is None or m["total"] > best_m["total"]:
                best_val, best_m = v, m
                mark = "  <=="
            print(f"  {v!s:<8} {fmt(m)}{mark}")
        print(f"mejor de la corrida: {gate} = {best_val}  ({fmt(best_m)})")

    grid_vals = {}
    for gate, values in AXES:
        cur = CURRENT[gate]
        scored = sorted(
            ((evaluate(rows, catalog, dict(CURRENT, **{gate: v}))["total"], abs(v - cur), v)
             for v in values if v != cur), key=lambda t: (-t[0], t[1]))
        grid_vals[gate] = sorted({scored[0][2] if scored else cur, cur})

    print("\n— grid conjunto alrededor del mejor de cada gate —")
    from itertools import product
    combo_best = None
    for deltas in product(*tuple(([0, 1] for _ in AXES))):
        gates = {}
        for (gate, _), d in zip(AXES, deltas):
            base = grid_vals[gate][d]
            gates[gate] = base
        m = evaluate(rows, catalog, gates)
        good = (gates["MIN_SONG_MARGIN"] >= 0 or True) and (
            m["song"]["precision"] >= a.min_song_prec)
        if good and (combo_best is None or m["total"] > combo_best[1]):
            combo_best = (gates, m["total"])
    if combo_best:
        gates, tot = combo_best
        m = evaluate(rows, catalog, gates)
        print("  config conjunto: " + fmt(m))
        for k in CURRENT:
            print(f"    {k} = {gates[k]} (actual {CURRENT[k]})")

    # reporte de falsos positivos con los umbrales actuales
    m = evaluate(rows, catalog, CURRENT)
    print("\n— falsos positivos con los umbrales actuales —")
    for f, _, _ in FIELDS:
        fl = [fp for fp in m[f]["misses"] if fp[2] is not None]
        for key, gv, v in fl:
            print(f"  {f:10s} {key[:38]}: GT={gv!r}  dijo={v!r}")
        print(f"  -> {f}: {len(fl)} falsos positivos, "
              f"{len(m[f]['misses']) - len(fl)} rechazados")
    print("\nnota: replica los números de BASELINE_DEVICE.md sobre 63 fotos.")
    if combo_best and tot > evaluate(rows, catalog, CURRENT)["total"]:
        print(f"el grid encontró {tot}/252 (> {evaluate(rows, catalog, CURRENT)['total']}).")
        print("si querés aplicarlos: cambia los valores en parity.py, PiuOcr.kt y corre")
        print("parity.py --regen-expected && ./gradlew testReleaseUnitTest")


if __name__ == "__main__":
    main()