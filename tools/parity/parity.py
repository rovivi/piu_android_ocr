#!/usr/bin/env python3
"""Test de paridad: el C++ del .so contra el pipeline Python, sobre las fotos.

Corre las MISMAS 58 fotos de `piu_ocr/dataset_v2` por los dos caminos y las
compara entre sí y contra el ground truth leído a mano:

  python   ResultReader.read(img, boxes)          (piu_ocr/pipeline.py)
  nativo   piuocr_cli --box ... (mismo C++ del .so) + réplica de PiuOcr.interpret

Los dos usan las mismas plantillas (build_mobile/chars.npz == chars.bin) y el
mismo catálogo, así que cualquier diferencia es del port, no de los datos.

Modos:
  (default)   cajas de dataset_v2/boxes.json para los dos: aísla el OCR.
  --detect    además corre el YOLO de NCNN y lo compara con las cajas de
              PyTorch+TTA, y mide el pipeline nativo end-to-end.

Salidas:
  tabla por campo (cobertura / precisión / acierto) para python y nativo, más
  el acuerdo nativo-vs-python; --json con el detalle por foto; --fixture
  graba src/test/resources/parity_fixture.json para el test JVM de Kotlin
  (ParityTest.kt), que verifica que interpret() de verdad da lo mismo que la
  réplica de acá.

Regresión: --baseline compara las métricas nativas con tools/parity/baseline.json
y falla (exit 1) si alguna cae más de --tol. --update-baseline las graba.

    piu_yolo/venv/bin/python android/tools/parity/parity.py
    python3 android/tools/parity/parity.py --detect --fixture --baseline
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ANDROID = os.path.abspath(os.path.join(HERE, "..", ".."))
PIU_OCR = os.path.dirname(ANDROID)                 # piu_ocr/
ROOT = os.path.dirname(PIU_OCR)                    # python_projects/
D2 = os.path.join(PIU_OCR, "dataset_v2")
PHOTOS = os.path.join(PIU_OCR, "DATASET")
MOBILE = os.path.join(PIU_OCR, "build_mobile")
ASSETS = os.path.join(ANDROID, "src", "main", "assets", "piu_ocr")
CLI = os.path.join(ANDROID, "build_host", "piuocr_cli")
FIXTURE = os.path.join(ANDROID, "src", "test", "resources", "parity_fixture.json")
FIXTURE_NOTE = (
    "generado por tools/parity/parity.py (--fixture o --regen-expected). "
    "'expected' es la réplica de PiuOcr.interpret: usa el Catalog de piu_ocr "
    "si está al lado, si no tools/parity/song_match_ref.py. ParityTest.kt "
    "exige igualdad con el interpret() real de Kotlin."
)
VENV_PY = os.path.join(ROOT, "piu_yolo", "venv", "bin", "python")

CLS = {"difficulty": 0, "fullscore": 1, "rank": 2, "score": 3, "song_name": 4}
CLS_NAME = {v: k for k, v in CLS.items()}

# Mismos gates que PiuOcr.kt. Si cambian allá tienen que cambiar acá, y el
# test JVM (ParityTest) es el que lo detecta.
MIN_SONG_MARGIN = 0.010
MIN_BADGE_CONF = 0.15
MIN_SCORE_MARGIN = 0.010
MAX_SONG_BOXES = 5
MIN_RAW_SIMILARITY = 0.55
MIN_RAW_SIMILARITY_SHORT = 0.80
SHORT_NAME = 3


def looks_like(raws, name):
    """difflib ratio sobre lo normalizado, sin espacios: el gate `looksLike` de
    PiuOcr.kt. Un nombre de <= SHORT_NAME caracteres exige casi coincidencia."""
    n = normalize(name).replace(" ", "")
    if not n:
        return False
    need = MIN_RAW_SIMILARITY_SHORT if len(n) <= SHORT_NAME else MIN_RAW_SIMILARITY
    for raw in raws:
        r = normalize(raw).replace(" ", "")
        if r and difflib.SequenceMatcher(None, r, n, autojunk=False).ratio() >= need:
            return True
    return False


def _reexec_with_venv():
    """cv2 vive en el venv de piu_yolo; si falta, relanzar con ese python."""
    try:
        import cv2  # noqa: F401
        return
    except ImportError:
        pass
    if os.path.exists(VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PY):
        os.execv(VENV_PY, [VENV_PY] + sys.argv)
    sys.exit("falta cv2: correr con piu_yolo/venv/bin/python")


# El matcher: el paquete Python de referencia si está al lado, si no la réplica
# local (song_match_ref.py). Hace falta en todos los modos, incluso sin fotos.
try:
    sys.path.insert(0, ROOT)
    from piu_ocr.song_match import Catalog, normalize  # noqa: E402
except ImportError:
    from song_match_ref import Catalog, normalize  # noqa: E402

# El pipeline Python (cv2 + ResultReader) solo hace falta para la capa 1 sobre
# fotos. Se importa de forma diferida para que --from-device, --synth y
# --regen-expected corran aunque el proyecto piu_ocr no esté al lado.
cv2 = np = ResultReader = TemplateDigits = None


def ensure_python_ref():
    """Carga cv2 y el ResultReader de referencia, relanzando con el venv."""
    global cv2, np, ResultReader, TemplateDigits
    if ResultReader is not None:
        return
    _reexec_with_venv()
    import cv2 as _cv2
    import numpy as _np
    cv2, np = _cv2, _np
    sys.path.insert(0, ROOT)
    from piu_ocr.pipeline import ResultReader as _RR
    from piu_ocr.recognize import TemplateDigits as _TD
    ResultReader, TemplateDigits = _RR, _TD


# ── nativo ───────────────────────────────────────────────────────────────────

def build_cli():
    if os.path.exists(CLI):
        return
    host = os.path.join(ANDROID, "third_party", "host")
    if not os.path.isdir(host) or len(os.listdir(host)) < 2:
        subprocess.check_call([os.path.join(ANDROID, "tools", "host", "fetch.sh")])
    bd = os.path.join(ANDROID, "build_host")
    subprocess.check_call(["cmake", "-S", os.path.join(ANDROID, "tools", "host"),
                           "-B", bd, "-DCMAKE_BUILD_TYPE=Release"])
    subprocess.check_call(["cmake", "--build", bd, "-j"])


AUGS = None          # --augs: pasadas del TTA para el detector nativo
EXTRA_FLAGS = []     # flags opt-in (--rectify, --bin-mode, ...) para el CLI


def run_native(image, boxes=None):
    """boxes: {clase: [{"box":[...], "conf":f}]} o None para usar el detector."""
    cmd = [CLI, "--assets", ASSETS, image] + EXTRA_FLAGS
    if AUGS and boxes is None:
        cmd += ["--augs", AUGS]
    if boxes is not None:
        for name, lst in boxes.items():
            if name not in CLS:
                continue
            for b in (lst if isinstance(lst, list) else [lst]):
                x1, y1, x2, y2 = (int(v) for v in b["box"])
                cmd += ["--box", f"{CLS[name]},{x1},{y1},{x2},{y2},{b.get('conf', 1.0):.6f}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"piuocr_cli falló en {image}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def interpret(native, catalog):
    """Réplica 1:1 de PiuOcr.interpret (Kotlin). Cualquier divergencia entre
    esto y Kotlin la detecta ParityTest.kt sobre el fixture."""
    ct_raw = native.get("chart_type") or None
    ct_conf = float(native.get("chart_conf", 0.0))
    chart = ct_raw if (ct_raw and ct_conf >= MIN_BADGE_CONF) else None

    pooled, raws = {}, []
    for t in native.get("titles", [])[:MAX_SONG_BOXES]:
        raw = t.get("raw", "")
        if not raw:
            continue
        raws.append(raw)
        w = math.sqrt(max(float(t.get("conf", 1.0)), 0.02))
        # F4.9: la 2ª lectura de la caja (raw2) entra al pool con peso menor y
        # también a `raws` para que el gate looksLike lo vea.
        raw2 = t.get("raw2") or None
        if raw2 == raw:
            raw2 = None
        if raw2:
            raws.append(raw2)
        texts = [raw] + ([raw2] if raw2 else [])
        for ti, text in enumerate(texts):
            tw = w if ti == 0 else w * 0.85
            for c in catalog.match(text, chart_type=chart, topk=8):
                sc = c["score"] * tw
                e = pooled.get(c["name"], 0.0)
                pooled[c["name"]] = max(e, sc) + 0.15 * min(e, sc)
    ranked = sorted(pooled.items(), key=lambda kv: -kv[1])
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
    clear = bool(ranked) and margin >= MIN_SONG_MARGIN
    alike = clear and looks_like(raws, ranked[0][0])
    song = ranked[0][0] if alike else None

    level = None
    sc = native.get("level_scores") or []
    if sc:
        legal = [l for l in (catalog.levels_for(song, chart) if song else [])
                 if len(str(l)) == len(sc)]
        if legal:
            scored = sorted(((sum(sc[i][int(ch)] for i, ch in enumerate(str(cand))), cand)
                             for cand in legal), reverse=True)
            level = scored[0][1]
        else:
            digits = "".join(str(d) for d in native.get("level_digits", []))
            try:
                v = int(digits)
                level = v if 1 <= v <= 28 else None
            except ValueError:
                level = None
    sv = native.get("score", -1)
    score = sv if (sv >= 0 and float(native.get("score_margin", 0.0)) >= MIN_SCORE_MARGIN) else None

    return {"song": song, "level": level, "chart_type": chart, "score": score,
            "raw": " | ".join(raws), "song_margin": round(margin, 4),
            "chart_conf": round(ct_conf, 3)}


# ── python ───────────────────────────────────────────────────────────────────

def make_reader():
    """ResultReader con LAS MISMAS plantillas que viajan en el .so."""
    ensure_python_ref()
    r = ResultReader(catalog=os.path.join(D2, "catalog.json"))
    r.chars = TemplateDigits.load_packed(os.path.join(MOBILE, "chars.npz"))
    # `digits` ya sale de build_mobile/digits.npz por el default de
    # ResultReader, que es el mismo atlas int8 que digits.bin.
    r.level = TemplateDigits.load_packed(os.path.join(MOBILE, "level.npz"))
    # level.npz guarda los dígitos como str; scores() indexa por int(c).
    r.level.classes = np.array([int(c) for c in r.level.classes])
    return r


def run_python(reader, img, boxes):
    out = reader.read(img, boxes)
    return {"song": out["song"]["value"], "level": out["level"]["value"],
            "chart_type": out["chart_type"]["value"],
            "score": out["score"]["value"],
            "raw": out["song"].get("raw", ""),
            "song_margin": out["song"].get("confidence"),
            "chart_conf": out["chart_type"].get("confidence")}


# ── métricas ─────────────────────────────────────────────────────────────────

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def field_metrics(rows, side, field, gt_key, eq):
    n = cov = ok = 0
    for r in rows:
        gt = r["gt"].get(gt_key)
        if gt is None:
            continue
        n += 1
        v = r[side].get(field)
        if v is not None:
            cov += 1
            ok += int(eq(v, gt))
    return {"n": n, "coverage": cov / n if n else 0.0,
            "precision": ok / cov if cov else 0.0, "accuracy": ok / n if n else 0.0}


def agreement(rows, field):
    same = sum(1 for r in rows if r["python"].get(field) == r["native"].get(field))
    return same / len(rows) if rows else 0.0


# ── buckets por condición (F0.2) ─────────────────────────────────────────────

def photo_buckets(img, rec):
    """Etiquetas de condición de la foto. 'Mejoró en ángulo pero empeoró en
    luz' se pierde si solo se mira el acierto global, así que se parte la
    métrica por condición. Ver PLAN §3.2."""
    tags = set()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tags.add("dark" if float(gray.mean()) < 90 else "bright")

    fs = None
    for name, lst in (rec or {}).items():
        if name != "fullscore":
            continue
        b = lst[0] if isinstance(lst, list) and lst else lst
        if isinstance(b, dict):
            fs = b.get("box")
    if fs:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in fs)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
            v = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
            if float((v > 250).mean()) > 0.02:
                tags.add("glare")
            if ((x2 - x1) * (y2 - y1)) / float(w * h) < 0.4:
                tags.add("far")
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = crop.shape[0] * crop.shape[1]
                if cv2.contourArea(c) > 0.2 * area:
                    _, _, ang = cv2.minAreaRect(c)
                    # distancia al eje más cercano: 0 = recto, crece al sesgar.
                    ang = min(ang % 90.0, 90.0 - ang % 90.0)
                    if ang > 8:
                        tags.add("rotated")
    return tags


def print_buckets(rows, side, detect_rows):
    """Tabla de acierto por campo dentro de cada bucket."""
    eq_song = lambda v, g: normalize(v) == normalize(g)
    eq = lambda v, g: v == g
    eq_score = lambda v, g: v in g if isinstance(g, list) else v == g
    fields = [("song", "song", eq_song), ("level", "level", eq),
              ("chart_type", "chart_type", eq), ("score", "score", eq_score)]
    all_tags = sorted({t for r in rows for t in r.get("buckets", set())})
    for tag in all_tags:
        sub = [r for r in rows if tag in r.get("buckets", set())]
        rowset = sub
        if side == "native_e2e" and detect_rows:
            rown = {r["key"] for r in sub}
            rowset = [r for r in detect_rows if r["key"] in rown]
        s = " (native_e2e)" if side == "native_e2e" else ""
        print(f"bucket {tag}{s}: n={len(rowset)}")
        for f, gt_key, e in fields:
            d = field_metrics(rowset, "native", f, gt_key, e)
            print(f"  {f:12s} n={d['n']:3d}  cob {d['coverage']:.3f}  "
                  f"prec {d['precision']:.3f}  acierto {d['accuracy']:.3f}")


def summarize(rows, detect_rows):
    eq_song = lambda v, g: normalize(v) == normalize(g)
    eq = lambda v, g: v == g
    # Una pantalla 2P tiene dos scores y el lector devuelve el de la caja con
    # más confianza, así que el GT de esas trae los dos y cualquiera vale.
    eq_score = lambda v, g: v in g if isinstance(g, list) else v == g
    fields = [("song", "song", eq_song), ("level", "level", eq),
              ("chart_type", "chart_type", eq), ("score", "score", eq_score)]
    m = {}
    for side in ("python", "native"):
        m[side] = {f: field_metrics(rows, side, f, g, e) for f, g, e in fields}
    m["agreement"] = {f: agreement(rows, f) for f, _, _ in fields}
    m["agreement"]["raw"] = agreement(rows, "raw")
    if detect_rows:
        m["detector"] = {}
        for name, cid in CLS.items():
            hits = tot = 0
            for r in detect_rows:
                ref = r["ref_boxes"].get(name)
                if not ref:
                    continue
                tot += 1
                best = ref[0]["box"] if isinstance(ref, list) else ref["box"]
                det = [b for b in r["det_boxes"] if b["cls"] == cid]
                hits += int(any(iou(b["box"], best) >= 0.5 for b in det))
            m["detector"][name] = {"n": tot, "recall@0.5": hits / tot if tot else 0.0}
        m["native_e2e"] = {f: field_metrics(detect_rows, "native", f, g, e)
                           for f, g, e in fields}
        ms = [r["ms"] for r in detect_rows]
        m["native_ms"] = {"mean": sum(ms) / len(ms), "max": max(ms)}
    return m


def print_table(m):
    def row(label, d):
        return f"  {label:12s} n={d['n']:3d}  cob {d['coverage']:.3f}  prec {d['precision']:.3f}  acierto {d['accuracy']:.3f}"
    for side in ("python", "native"):
        print(f"{side} (cajas de boxes.json):")
        for f in ("song", "level", "chart_type", "score"):
            print(row(f, m[side][f]))
    print("acuerdo nativo == python:  " +
          "  ".join(f"{k} {v:.3f}" for k, v in m["agreement"].items()))
    if "detector" in m:
        print("detector NCNN vs PyTorch+TTA (recall IoU>=0.5 de la mejor caja):")
        print("  " + "  ".join(f"{k} {v['recall@0.5']:.3f}" for k, v in m["detector"].items()))
        print("nativo end-to-end (detector + OCR + gates):")
        for f in ("song", "level", "chart_type", "score"):
            print(row(f, m["native_e2e"][f]))
        print(f"  latencia nativa (host): media {m['native_ms']['mean']:.0f} ms, "
              f"max {m['native_ms']['max']:.0f} ms")


def flatten(m, prefix=""):
    out = {}
    for k, v in m.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)):
            out[prefix + k] = v
    return out


def check_baseline(m, path, tol):
    if not os.path.exists(path):
        print(f"sin baseline en {path}; --update-baseline para crearla")
        return True
    base = flatten(json.load(open(path)))
    cur = flatten(m)
    bad = []
    for k, v in base.items():
        if k.endswith(".n") or k.startswith("native_ms") or k not in cur:
            continue
        if cur[k] < v - tol:
            bad.append(f"  {k}: {v:.3f} -> {cur[k]:.3f}")
    if bad:
        print("REGRESIÓN contra baseline:")
        print("\n".join(bad))
        return False
    print("baseline OK (ninguna métrica cayó)")
    return True


# ── device ───────────────────────────────────────────────────────────────────

def from_device(path, tol):
    """results.json de DeviceParityTest: el .so arm64 real + Kotlin real.
    Compara (1) Kotlin en device vs la réplica Python sobre el MISMO JSON,
    (2) el JSON del device vs el del CLI de host (misma foto, mismo código,
    otra CPU), (3) acierto contra GT y contra baseline.native_e2e."""
    res = json.load(open(path))
    boxes = {b["key"]: b for b in json.load(open(os.path.join(D2, "boxes.json")))}
    gt_song = json.load(open(os.path.join(D2, "gt_song.json")))
    gt_lvl = json.load(open(os.path.join(D2, "gt_level.json")))
    gt_score = json.load(open(os.path.join(D2, "gt_score.json")))
    catalog = Catalog(os.path.join(D2, "catalog.json"))
    build_cli()
    print(f"device: {res.get('device')} [{res.get('abi')}]  carga {res.get('load_ms', 0):.0f} ms")

    rows, kt_diff, host_diff, ms = [], [], [], []
    for stem, r in sorted(res["rows"].items()):
        key = stem.split("_")[0]
        if key not in boxes or not gt_song.get(key):
            continue
        ct, lvl = (gt_lvl.get(key) or [None, None])
        gt = {"song": gt_song[key], "chart_type": ct, "level": lvl,
              "score": gt_score.get(key)}
        replica = interpret(r["native"], catalog)
        rows.append({"key": key, "gt": gt, "native": replica})
        ms.append(r["ms"])
        for f in ("song", "level", "chart_type", "score", "raw"):
            if r.get(f) != replica.get(f):
                kt_diff.append(f"  {key} {f}: device={r.get(f)!r} replica={replica.get(f)!r}")
        host = run_native(os.path.join(PHOTOS, boxes[key]["file"]), None)["result"]
        hr = interpret(host, catalog)
        d = [f for f in ("song", "level", "chart_type", "score", "raw") if hr.get(f) != replica.get(f)]
        if d:
            host_diff.append(f"  {key}: " + "; ".join(
                f"{f} device={replica.get(f)!r} host={hr.get(f)!r}" for f in d))

    eq_song = lambda v, g: normalize(v) == normalize(g)
    def _eq(f):
        if f == "song":
            return lambda v, g: normalize(v) == normalize(g)
        if f == "score":
            return lambda v, g: v in g if isinstance(g, list) else v == g
        return lambda v, g: v == g
    m = {f: field_metrics(rows, "native", f, f, _eq(f))
         for f in ("song", "level", "chart_type", "score")}
    print(f"{len(rows)} fotos, latencia device media {sum(ms) / len(ms):.0f} ms, max {max(ms):.0f} ms")
    print("device end-to-end contra GT:")
    for f, d in m.items():
        print(f"  {f:12s} n={d['n']:3d}  cob {d['coverage']:.3f}  prec {d['precision']:.3f}  acierto {d['accuracy']:.3f}")
    print(f"Kotlin en device vs réplica Python: {len(kt_diff)} diferencias")
    print("\n".join(kt_diff))
    print(f".so arm64 vs CLI de host (mismo código, otra CPU): {len(host_diff)} fotos difieren")
    print("\n".join(host_diff))

    bp = os.path.join(HERE, "baseline.json")
    ok = True
    if os.path.exists(bp):
        base = json.load(open(bp)).get("native_e2e", {})
        for f, d in m.items():
            for k in ("coverage", "precision", "accuracy"):
                b = base.get(f, {}).get(k)
                if b is not None and d[k] < b - tol - 0.03:   # 0.03 = 1 foto de 45
                    print(f"REGRESIÓN device vs baseline host: {f}.{k} {b:.3f} -> {d[k]:.3f}")
                    ok = False
        if ok:
            print("device dentro de baseline host (±1 foto)")
    return ok and not kt_diff


# ── synth (F0.3) ─────────────────────────────────────────────────────────────

SYNTH = os.path.join(ANDROID, "build", "synth")


def run_synth(a):
    """Corre el CLI sobre las pantallas sintéticas (tools/synth.py --build).
    Valida dos cosas: (1) que con los flags puestos se lea el GT sintético bajo
    cada condición, (2) que el modo default no cambie respecto de la primera
    corrida (que representa 'antes del cambio')."""
    gt_path = os.path.join(SYNTH, "gt.json")
    if not os.path.exists(gt_path):
        print(f"falta {gt_path}: correr python3 tools/synth.py --build")
        return False
    build_cli()
    cases = json.load(open(gt_path))
    catalog = Catalog(os.path.join(ASSETS, "catalog.json"))

    def read(path, flags):
        global EXTRA_FLAGS
        saved = EXTRA_FLAGS
        EXTRA_FLAGS = flags if flags is not None else []
        try:
            return interpret(run_native(path, None)["result"], catalog)
        finally:
            EXTRA_FLAGS = saved

    # default (sin flags) contra la baseline grabada, si existe.
    base_path = os.path.join(SYNTH, "default_baseline.json")
    fresh = not os.path.exists(base_path)
    defaults = {}
    for c in cases:
        path = os.path.join(SYNTH, c["file"])
        if os.path.exists(path):
            defaults[c["file"]] = read(path, [])
    if not fresh:
        base = json.load(open(base_path))
        diff = [f for f in defaults if base.get(f) != defaults[f]]
        print(f"default vs baseline synth: {len(diff)} diferencias de {len(defaults)}")
        for f in diff[:10]:
            print(f"  {f}: baseline={base.get(f)} ahora={defaults[f]}")
        if diff:
            return False
    else:
        json.dump(defaults, open(base_path, "w"), indent=1)
        print(f"baseline synth grabada en {base_path}")

    n = ok = 0
    by_cond = {}
    for c in cases:
        path = os.path.join(SYNTH, c["file"])
        if not os.path.exists(path):
            continue
        gt = c["gt"]
        got = read(path, EXTRA_FLAGS)
        n += 1
        hits = (normalize(got["song"] or "") == normalize(gt["song"] or ""),
                got["level"] == gt["level"], got["chart_type"] == gt["chart_type"],
                got["score"] == gt["score"])
        ok += all(hits)
        cond = c.get("cond", {}).get("angle", "?")
        d = by_cond.setdefault(cond, [0, 0])
        d[0] += 1
        d[1] += all(hits)
        if not all(hits):
            print(f"  {c['file']}: got song={got['song']!r} lvl={got['level']} "
                  f"ct={got['chart_type']!r} score={got['score']}  gt={gt}")
    print(f"synth: {ok}/{n} completos")
    for cond, (tot, good) in sorted(by_cond.items(), key=lambda kv: str(kv[0])):
        print(f"  ángulo {cond}: {good}/{tot}")
    # La primera corrida solo graba la baseline (representa "antes del cambio");
    # exigir además que el default lea el GT sintético bloquearía el arranque.
    return n > 0 and (ok == n or fresh)


# ── regen de expected (sin fotos) ───────────────────────────────────────────

def regen_expected():
    """Reescribe el `expected` del fixture a partir del `native` ya guardado.

    `expected` es función pura de `native` + catálogo, así que se puede
    recalcular sin las fotos ni el pipeline Python. Sirve cuando cambia un gate
    de Kotlin y el fixture quedó desfasado pero no está la máquina con datos.
    Usa el catálogo de assets, que es el que lee `ParityTest.kt`.
    """
    if not os.path.exists(FIXTURE):
        print(f"falta {FIXTURE}")
        return False
    fx = json.load(open(FIXTURE))
    catalog = Catalog(os.path.join(ASSETS, "catalog.json"))
    fx["_note"] = FIXTURE_NOTE
    for row in fx["rows"]:
        rep = interpret(row["native"], catalog)
        row["expected"] = {k: rep[k]
                           for k in ("song", "level", "chart_type", "score", "raw")}
    json.dump(fx, open(FIXTURE, "w"), indent=1, ensure_ascii=False)
    print(f"expected regenerado: {FIXTURE} ({len(fx['rows'])} filas)")
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detect", action="store_true", help="correr también el YOLO nativo")
    ap.add_argument("--json", help="detalle por foto")
    ap.add_argument("--fixture", action="store_true",
                    help="grabar src/test/resources/parity_fixture.json")
    ap.add_argument("--baseline", action="store_true",
                    help="comparar con tools/parity/baseline.json")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--diff", action="store_true", help="listar fotos donde difieren")
    ap.add_argument("--augs", help='pasadas TTA del detector, ej. "1,0.83,1f" (default del .so)')
    ap.add_argument("--from-device", metavar="RESULTS_JSON",
                    help="comparar results.json bajado del teléfono (tools/parity/device.sh)")
    # Flags opt-in (F0.1): se reenvían al CLI. A/B por flag.
    ap.add_argument("--rectify", action="store_true", help="F1: enderezar la pantalla")
    ap.add_argument("--bin-mode", choices=["legacy", "clahe", "adaptive", "all"],
                    help="F2: binarización del OCR")
    ap.add_argument("--badge-mode", choices=["color", "adaptive", "fusion"],
                    help="F3: voto del chart_type")
    ap.add_argument("--title-variants", type=int, choices=[1, 2],
                    help="F4: 2 emite raw2 por caja")
    ap.add_argument("--title-boxes", type=int, help="F4: máximo de cajas de título")
    ap.add_argument("--buckets", action="store_true",
                    help="F0.2: acierto por campo dentro de cada condición")
    ap.add_argument("--synth", action="store_true",
                    help="F0.3: valida sobre pantallas sintéticas (tools/synth.py)")
    ap.add_argument("--regen-expected", action="store_true",
                    help="reescribe 'expected' del fixture desde el 'native' guardado (sin fotos)")
    a = ap.parse_args()
    global AUGS, EXTRA_FLAGS
    AUGS = a.augs
    if a.rectify:
        EXTRA_FLAGS.append("--rectify")
    if a.bin_mode:
        EXTRA_FLAGS += ["--bin-mode", a.bin_mode]
    if a.badge_mode:
        EXTRA_FLAGS += ["--badge-mode", a.badge_mode]
    if a.title_variants:
        EXTRA_FLAGS += ["--title-variants", str(a.title_variants)]
    if a.title_boxes:
        EXTRA_FLAGS += ["--title-boxes", str(a.title_boxes)]
    if a.from_device:
        sys.exit(0 if from_device(a.from_device, a.tol) else 1)
    if a.synth:
        sys.exit(0 if run_synth(a) else 1)
    if a.regen_expected:
        sys.exit(0 if regen_expected() else 1)

    build_cli()
    boxes = {b["key"]: b for b in json.load(open(os.path.join(D2, "boxes.json")))}
    gt_song = json.load(open(os.path.join(D2, "gt_song.json")))
    gt_lvl = json.load(open(os.path.join(D2, "gt_level.json")))
    gt_score = json.load(open(os.path.join(D2, "gt_score.json")))
    catalog = Catalog(os.path.join(D2, "catalog.json"))
    reader = make_reader()

    rows, detect_rows = [], []
    keys = [k for k in sorted(boxes) if not k.startswith("_") and gt_song.get(k)]
    if a.limit:
        keys = keys[:a.limit]
    t0 = time.time()
    for key in keys:
        rec = boxes[key]
        path = os.path.join(PHOTOS, rec["file"])
        img = cv2.imread(path)
        if img is None:
            continue
        ct, lvl = (gt_lvl.get(key) or [None, None])
        gt = {"song": gt_song[key], "chart_type": ct, "level": lvl,
              "score": gt_score.get(key)}
        buckets = photo_buckets(img, rec["boxes"])
        if isinstance(gt["score"], list):
            buckets.add("twoP")

        nat = run_native(path, rec["boxes"])
        row = {"key": key, "file": rec["file"], "gt": gt, "buckets": sorted(buckets),
               "python": run_python(reader, img, rec["boxes"]),
               "native": interpret(nat["result"], catalog),
               "native_json": nat["result"]}
        rows.append(row)

        if a.detect:
            det = run_native(path, None)
            detect_rows.append({"key": key, "gt": gt, "buckets": sorted(buckets),
                                "native": interpret(det["result"], catalog),
                                "native_json": det["result"],
                                "det_boxes": det["boxes"], "ref_boxes": rec["boxes"],
                                "ms": det["ms"]["read"]})
        sys.stderr.write(f"\r{len(rows)}/{len(keys)} {key}   ")
    sys.stderr.write(f"\r{len(rows)} fotos en {time.time() - t0:.0f} s\n")

    m = summarize(rows, detect_rows)
    print_table(m)

    if a.buckets:
        print("\nF0.2 buckets (nativo con cajas de boxes.json):")
        print_buckets(rows, "native", detect_rows)
        if detect_rows:
            print("\nF0.2 buckets (nativo end-to-end):")
            print_buckets(rows, "native_e2e", detect_rows)

    if a.diff:
        print("\nfotos donde nativo != python:")
        for r in rows:
            d = [f for f in ("song", "level", "chart_type", "score", "raw")
                 if r["python"].get(f) != r["native"].get(f)]
            if d:
                print(f"  {r['key']}: {', '.join(d)}")
                for f in d:
                    print(f"      {f}: py={r['python'].get(f)!r}  nat={r['native'].get(f)!r}  gt={r['gt'].get(f)!r}")

    if a.json:
        json.dump({"metrics": m, "rows": rows, "detect": detect_rows},
                  open(a.json, "w"), indent=1, ensure_ascii=False)
    if a.fixture:
        fx = FIXTURE
        os.makedirs(os.path.dirname(fx), exist_ok=True)
        json.dump({
            "_note": FIXTURE_NOTE,
            "rows": [{"key": r["key"], "native": r["native_json"], "gt": r["gt"],
                      "expected": {k: r["native"][k] for k in ("song", "level", "chart_type", "score", "raw")}}
                     for r in rows + detect_rows],
        }, open(fx, "w"), indent=1, ensure_ascii=False)
        print(f"fixture: {fx} ({len(rows) + len(detect_rows)} filas)")

    bp = os.path.join(HERE, "baseline.json")
    ok = True
    if a.update_baseline:
        json.dump(m, open(bp, "w"), indent=1)
        print(f"baseline grabada en {bp}")
    elif a.baseline:
        ok = check_baseline(m, bp, a.tol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
