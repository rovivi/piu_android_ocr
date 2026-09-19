#!/usr/bin/env python3
"""Arma dataset_v2.json: lecturas OCR (visión) de images/new cruzadas contra
catalog.json del módulo piu-ocr.

Las lecturas `RAW` son el ground truth "a ojo" de las 48 imágenes de
piu_yolo/images/new, en el orden alfabético del directorio. Se cruzan con el
catálogo usando el MISMO matcher de referencia que los tests de paridad
(tools/parity/song_match_ref.py), así el score de match es comparable.

Uso:
  python3 tools/validator/build_dataset.py [--images DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools" / "parity"))
from song_match_ref import Catalog, normalize  # noqa: E402

CATALOG = REPO / "src/main/assets/piu_ocr/catalog.json"
DEFAULT_IMAGES = Path("/Users/rodrigo/dev/python_projects/piu_yolo/images/new")

# (song leído, score leído, notas, score_alt del segundo panel si hay)
RAW = [
    ("Enjoy The Show", 991013, "", None),
    ("The Last Rebellion", 975862, "", None),
    ("QUATTUORUX", 897837, "", None),
    ("Gargoyle - FULL SONG -", 991755, "", None),
    ("The Last Rebellion", 992985, "versus: panel izq 926770", 926770),
    ("The Last Rebellion", 980427, "overlay de streaming", None),
    ("Final Audition Ep. 2-2", 993568, "versus: panel izq 980190", 980190),
    ("Shub Niggurath", 957163, "", None),
    ("Viyella's Nightmare", 992329, "versus: panel izq 985723", 985723),
    ("Aragami", 949199, "", None),
    ("Super Akuma Emperor", 1000000, "perfect game", None),
    ("FREEDOM DiVE", 970681, "", None),
    ("Shub Sothoth", 967831, "", None),
    ("Ghroth", 981195, "", None),
    ("Telling Fortune Flower", 990697, "versus: panel izq 971097", 971097),
    ("Dead End", 875075, "", None),
    ("Mahika", 971105, "", None),
    ("Crimson hood", 944793, "", None),
    ("Cleaner", 987127, "", None),
    ("ERRORCODE: 0", 964810, "", None),
    ("OVERNIGHT FLOWER", 990882, "", None),
    ("iRELLIA", 982376, "", None),
    ("Hello William", 966997, "", None),
    ("VECTOR", 965362, "", None),
    ("Athena's Shield", 997040, "", None),
    ("The Last Rebellion", 997612, "", None),
    ("BLAZOR", 947878, "", None),
    ("Caprice of Otada", 981598, "", None),
    ("L (PIU Edit)", 918113, "", None),
    ("Rock the house", 905003, "", None),
    ("Digitalis", 946061, "", None),
    ("Smile Diary", 984265, "", None),
    ("Conflict", 988264, "", None),
    ("Super Akuma Emperor", 992237, "", None),
    ("Dead End", 976268, "", None),
    ("L (PIU Edit)", 984798, "", None),
    ("That Kitty (PIU Edit.)", 958180, "", None),
    ("Catastrophe", 994796, "", None),
    ("+DOOF+SENC+", 995683, "", None),
    ("Cynical", 990892, "", None),
    ("Do the Dance", 1000000, "versus: panel der 999393", 999393),
    ("The Last Rebellion", 933570, "", None),
    ("FREEDOM DiVE", 830064, "", None),
    ("INFINITE ENERGY -Overdose-", 994024, "versus: panel der 978207", 978207),
    ("BLAZOR", 995757, "", None),
    ("Imprinting", 974253, "", None),
    ("The Stranger", 997611, "", None),
    ("The Stranger", 997611, "", None),
    ("Dizzy Dance, Street Light", 970763, "", None),
    ("Bluish Rose", 973583, "", None),
    ("IVE", 891043, "título cortado (banda fuera del encuadre); BGA dice IVE", None),
    ("404 (New Era)", 886033, "versus: panel izq 820087", 820087),
    ("NeLiME", 905772, "versus: panel der 890131", 890131),
    ("NightTheater", 918901, "", None),
    ("Extreme Music School 2nd period feat. Nanahira", 886074, "screenshot de stream", None),
    ("BANG BANG", 930214, "", None),
    ("Mopemope", 995516, "título cortado: se ve 'emope'", None),
    ("Crash-Landing Rendezvous", 981295, "", None),
    ("Super Capriccio", 942861, "", None),
    ("The Devil", 811308, "", None),
    ("Can I friend you?", 937272, "título cortado: 'Can I friend...'", None),
    ("Ghost Bloody Train", 939213, "", None),
    ("CALL ME BACK", 914165, "", None),
]

MATCH_MIN = 0.55


def match_status(ocr: str, top: dict | None) -> str:
    if not top:
        return "none"
    if normalize(ocr) == normalize(top["name"]):
        return "exact"
    return "fuzzy" if top["score"] >= MATCH_MIN else "none"


def build(images_dir: Path, out: Path) -> dict:
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(images) != len(RAW):
        raise SystemExit(f"esperaba {len(RAW)} imágenes, hay {len(images)} en {images_dir}")

    catalog = Catalog(str(CATALOG))
    items = []
    for img, (song, score, note, alt) in zip(images, RAW):
        cands = catalog.match(song, topk=3)
        top = cands[0] if cands else None
        items.append({
            "file": img.name,
            "song_ocr": song,
            "score_ocr": score,
            "score_alt": alt,
            "notes": note,
            "catalog_match": top,
            "match_status": match_status(song, top),
            "candidates": cands,
            "validated": False,
            "song_final": None,
            "score_final": None,
        })

    data = {
        "version": 2,
        "source": str(images_dir),
        "catalog": "src/main/assets/piu_ocr/catalog.json",
        "match_min": MATCH_MIN,
        "items": items,
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    counts = {s: sum(1 for i in items if i["match_status"] == s) for s in ("exact", "fuzzy", "none")}
    print(f"escrito {out} | {len(items)} ítems | {counts}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--out", default=str(HERE / "dataset_v2.json"))
    args = ap.parse_args()
    build(Path(args.images), Path(args.out))


if __name__ == "__main__":
    main()
