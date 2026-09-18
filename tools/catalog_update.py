#!/usr/bin/env python3
"""Actualiza el catálogo del teléfono desde master_db.json (dump de piugame.com).

Formato verificado contra el catalog.json actual:
  clave  = normalize(display)  (NFD + ascii + lower + espacios colapsados)
  valor  = {'n': display_name, 'c': [[step_type, level], ...]}
  c      = sorted(set((step_type, level)) de la DB), orden: tipo alfabético, nivel asc

Estrategia conservadora: las entradas existentes NO se tocan (comportamiento
validado por los tests de paridad); solo se AGREGAN las canciones que faltan.
Las comillas escapadas del dump ('""' → '"') se limpian al agregar.

Uso:
  python3 tools/catalog_update.py [--db ruta/master_db.json] [--dry-run]

El master_db.json vive en el repo ligas-piu-api (sync de piugame.com).
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
CATALOG = HERE.parent / 'src/main/assets/piu_ocr/catalog.json'


def norm(s: str) -> str:
    s = unicodedata.normalize('NFD', str(s)).encode('ascii', 'ignore').decode()
    return ' '.join(s.lower().split())


def clean_name(s: str) -> str:
    # El dump escapa comillas estilo CSV y envuelve el nombre en comillas:
    # '"Phalanx ""RS2018 edit"""' -> 'Phalanx "RS2018 edit"'
    s = s.replace('""', '"').strip()
    if len(s) > 1 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='../../piu/ligas-piu-api/master_db.json')
    ap.add_argument('--out', default=str(CATALOG))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    catalog = json.loads(Path(args.out).read_text())
    db = json.loads(db_path.read_text())

    existing = {norm(v['n']): k for k, v in catalog.items()}
    added, skipped = 0, 0
    for song in db:
        name = clean_name(song['song_name'])
        key = norm(name)
        if key in existing:
            skipped += 1
            continue
        pairs = sorted(set((ch['step_type'], ch['level']) for ch in song['charts']))
        if not pairs:
            continue
        catalog[key] = {'n': name, 'c': [[t, l] for t, l in pairs]}
        existing[key] = key
        added += 1

    print(f"catálogo: {len(catalog) - added} → {len(catalog)} | agregadas: {added} | ya estaban: {skipped}")
    if added:
        for k, v in list(catalog.items())[-added:]:
            print(f"  + {v['n']} ({len(v['c'])} charts)")

    if args.dry_run:
        return

    if added:
        Path(args.out).write_text(json.dumps(catalog, ensure_ascii=False, indent=1))
        print(f"escrito: {args.out}")


if __name__ == '__main__':
    main()
