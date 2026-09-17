#!/usr/bin/env python3
"""Réplica LOCAL de `piu_ocr/song_match.py` para cuando el proyecto Python no está.

`tools/parity/parity.py` usa el paquete Python de referencia (`piu_ocr`) cuando
está al lado. Si no está —por ejemplo en una máquina que solo tiene este repo—
cae a este módulo, que reproduce `Catalog.match`, `Catalog.levels_for` y
`normalize` tal como los porta `SongMatcher.kt`.

Ojo: es una referencia, no la fuente de verdad. El día que haya que regenerar
los datos contra el pipeline Python real, correr `parity.py` con `piu_ocr`
disponible; este módulo existe para que el fixture y los tests de gates se
puedan regenerar sin él.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata

# Mismo orden de operaciones que normalize() de Kotlin/song_match.py: NFD, sin
# diacríticos, minúsculas, todo lo que no sea [a-z0-9] es espacio, colapsar.
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")

# Pesos del matcher y tamaño del prefiltro: ver SongMatcher.kt y ARQUITECTURA §8.
W_TRIGRAM, W_RATIO, W_PARTIAL = 0.45, 0.35, 0.20
PREFILTER = 120
CHART_KEY = {"single": "s", "double": "d", "halfdouble": "hd", "coop": "c"}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return _SPACES.sub(" ", _NON_ALNUM.sub(" ", s.lower())).strip()


def _trigrams(n: str) -> set[str]:
    s = "  " + n.replace(" ", "") + "  "
    return {s} if len(s) < 3 else {s[i:i + 3] for i in range(len(s) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    inter = sum(1 for x in small if x in big)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def _longest_common(a: str, b: str) -> int:
    """Corrida común más larga (substring, no subsecuencia)."""
    if not a or not b:
        return 0
    best = 0
    dp = [0] * (len(b) + 1)
    for i in range(len(a)):
        prev = 0
        for j in range(len(b)):
            tmp = dp[j + 1]
            dp[j + 1] = prev + 1 if a[i] == b[j] else 0
            best = max(best, dp[j + 1])
            prev = tmp
    return best


class _Song:
    __slots__ = ("name", "charts", "norm", "trigrams", "compact")

    def __init__(self, name: str, charts: list[tuple[str, int]]):
        self.name = name
        self.charts = charts
        self.norm = normalize(name)
        self.trigrams = _trigrams(self.norm)
        self.compact = self.norm.replace(" ", "")


class Catalog:
    """Catálogo cerrado de canciones. Ver piu_ocr/song_match.py."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            root = json.load(f, object_pairs_hook=dict)   # preserva el orden
        self.songs = []
        for key, o in root.items():
            if key.startswith("_"):
                continue
            charts = [(c[0], int(c[1])) for c in (o.get("c") or [])
                      if len(c) > 1 and c[1] is not None]
            self.songs.append(_Song(o["n"], charts))

    def match(self, text: str, chart_type: str | None = None,
              topk: int = 2) -> list[dict]:
        # chart_type no filtra: en inferencia el nivel nunca se pasa y el step
        # type solo actúa junto con el nivel (ver SongMatcher.kt).
        q = normalize(text)
        if not q:
            return []
        qt, qc = _trigrams(q), q.replace(" ", "")
        pool = self.songs
        if len(pool) > PREFILTER:
            pool = sorted(pool, key=lambda s: -_jaccard(qt, s.trigrams))[:PREFILTER]
        out = []
        for s in pool:
            tri = _jaccard(qt, s.trigrams)
            ratio = difflib.SequenceMatcher(None, qc, s.compact, autojunk=False).ratio()
            partial = _longest_common(qc, s.compact) / max(min(len(qc), len(s.compact)), 1)
            score = round(W_TRIGRAM * tri + W_RATIO * ratio + W_PARTIAL * partial, 4)
            out.append({"name": s.name, "score": score,
                        "levels": sorted({lv for _, lv in s.charts})})
        out.sort(key=lambda c: -c["score"])
        return out[:topk]

    def levels_for(self, name: str | None, chart_type: str | None) -> list[int]:
        if not name:
            return []
        s = next((x for x in self.songs
                  if x.name == name or x.norm == normalize(name)), None)
        if s is None:
            return []
        want = CHART_KEY.get(chart_type)
        lv = sorted({lv for ct, lv in s.charts if want is None or ct == want or ct == ""})
        return lv or sorted({lv for _, lv in s.charts})
