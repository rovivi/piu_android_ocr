#!/usr/bin/env python3
"""Agrega al catálogo del teléfono el título en modo coreano (clave "k").

El lector compara una lectura con Hangul contra "k" y una latina contra "n"
(SongMatcher.match), así que esto no cambia nada para las fotos de pantallas en
inglés. Hace falta para el día en que chars.bin tenga plantillas Hangul: sin
"k" una lectura coreana no tiene contra qué compararse.

Entrada: JSON {título inglés: título coreano}. Sale de ligas-piu-api, donde el
sync de PIUGame llena piu_songs.song_name_kr (654/654 canciones activas):

  select json_object_agg(song_name, song_name_kr) from piu_songs
   where song_name_kr is not null;

Uso:
  python3 tools/catalog_kr.py kr_titles.json [--dry-run]

Cruza por el normalize del matcher (el mismo de SongMatcher.kt). Un nombre que
normaliza igual para dos canciones distintas no se toca: mejor sin "k" que con
el de otra canción.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'parity'))
from song_match_ref import normalize  # noqa: E402

CATALOG = HERE.parent / 'src/main/assets/piu_ocr/catalog.json'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('titles', help='JSON {título inglés: título coreano}')
    ap.add_argument('--catalog', default=str(CATALOG))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    catalog = json.loads(Path(args.catalog).read_text())
    korean = defaultdict(set)
    for english, kr in json.loads(Path(args.titles).read_text()).items():
        korean[normalize(english)].add(kr)

    added = changed = ambiguous = 0
    missing = []
    for key, entry in catalog.items():
        if key.startswith('_'):
            continue
        found = korean.get(normalize(entry['n']), set())
        if len(found) > 1:
            ambiguous += 1
            continue
        if not found:
            missing.append(entry['n'])
            continue
        kr = found.pop()
        if entry.get('k') != kr:
            changed += 'k' in entry
            added += 'k' not in entry
            entry['k'] = kr

    print(f'agregados {added}, cambiados {changed}, ambiguos {ambiguous}, '
          f'sin título coreano {len(missing)}')
    for name in missing[:40]:
        print('  -', name)
    if not args.dry_run:
        Path(args.catalog).write_text(
            json.dumps(catalog, ensure_ascii=False, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
