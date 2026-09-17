#!/usr/bin/env python3
"""Genera las animaciones del README (docs/img/*.gif).

Son assets de documentación, no código de producto. Se regeneran con:

    python3 tools/parity/animate.py

- `synth_conditions.gif` — la misma pantalla PIU sintética barriendo ángulo,
  brillo y glare. Muestra lo que el módulo tiene que leer.
- `field_scan.gif` — qué campos lee y qué valor saca, sobre una **foto real de
  cabina** (`imagtest.jpg` en la raíz del repo): canción, chart/nivel, score y
  rank, revelados a medida que recorre cada zona. Si la foto no está, cae a la
  pantalla sintética canónica de Phoenix.

Solo necesita PIL + numpy (los de `tools/synth.py`).
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ANDROID = os.path.dirname(TOOLS)
IMG = os.path.join(ANDROID, "docs", "img")
sys.path.insert(0, TOOLS)
import synth  # noqa: E402

BAR = 42
BG = (18, 18, 22)
FG = (238, 238, 242)
ACCENT = (255, 196, 0)
PANEL_BG = (14, 16, 22)


def resize(img: Image.Image, width: int) -> Image.Image:
    h = round(img.height * width / img.width)
    return img.resize((width, h), Image.LANCZOS)


def with_caption(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + BAR), BG)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((14, img.height + BAR // 2), text, font=synth.find_font(22),
           fill=FG, anchor="lm")
    return out


def save_gif(frames: list[Image.Image], name: str, duration: int,
             colors: int = 96) -> None:
    os.makedirs(IMG, exist_ok=True)
    path = os.path.join(IMG, name)
    # Paleta acotada: el ruido fotográfico infla el GIF si se guarda en RGB.
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:],
                duration=duration, loop=0, optimize=True, disposal=2)
    print(f"{name}: {len(frames)} frames, {os.path.getsize(path) / 1024:.0f} KB")


# ── 1. condiciones: ángulo × luz × glare ────────────────────────────────────

# (ángulo, brillo, glare, etiqueta). Curada: no es el producto cartesiano
# completo sino la progresión que cuenta la historia.
CONDS = [
    (0, 1.0, False, "pantalla recta, bien iluminada"),
    (0, 0.4, False, "a oscuras · brillo 40%"),
    (10, 0.7, False, "10° de ángulo"),
    (20, 0.7, False, "20°"),
    (30, 0.7, False, "30° · la red ya no ve la pantalla recta"),
    (30, 1.0, True, "30° + glare de neón"),
    (20, 0.4, True, "20° + oscuro + glare"),
]


def synth_conditions(width=560):
    song, chart, level, score = synth.SONGS[0]
    frames = []
    for i, (angle, brightness, glare, label) in enumerate(CONDS):
        run = random.Random(f"anim-{angle}-{brightness}-{glare}")
        nprng = np.random.default_rng(1000 + i)
        screen = synth.draw_screen(song, chart, level, score, "S")
        quad = synth.perspective_quad(angle, run)
        photo = synth.composite(screen, quad)
        photo = synth.photometric(photo, brightness, glare, nprng)
        frames.append(with_caption(resize(photo, width), label))
    save_gif(frames, "synth_conditions.gif", duration=1100)


# ── 2. reconocimiento: qué lee y qué valor saca ─────────────────────────────
#
# (clave, caja, etiqueta, valor). fullscore no produce valor: es la pantalla.

REAL_IMAGE = os.path.join(ANDROID, "imagtest.jpg")
# Recorte de la pantalla dentro de la foto (sin cabina ni marquesina).
REAL_SCREEN = (350, 585, 2490, 1945)

# Cajas a mano sobre la foto real; la pantalla está en perspectiva, así que
# siguen la posición visible. El resultado es el de `imagtest.jpg`:
# Overnight Flower · DOUBLE nivel 20 · score 971041 · rank S.
REAL_FIELDS = [
    ("fullscore", (12, 12, 2128, 1348), None, None),
    ("song_name", (150, 265, 1860, 425), "canción", "OVERNIGHT FLOWER"),
    ("difficulty", (1395, 435, 1660, 690), "chart", "DOUBLE · nivel 20"),
    ("score", (1540, 510, 2115, 760), "score", "971041"),
    ("rank", (1765, 745, 2040, 1005), "rank", "S"),
]

SYNTH_FIELDS = [
    ("fullscore", (8, 8, 1272, 712), None, None),
    ("song_name", (0, 20, 1280, 140), "canción", "LA CAMPANELLA"),
    ("difficulty", (70, 250, 330, 510), "chart", "SINGLE · nivel 10"),
    ("score", (420, 300, 1240, 470), "score", "761729756"),
    ("rank", (1000, 520, 1240, 680), "rank", "S"),
]


def boxed(base: Image.Image, box, alpha: int) -> Image.Image:
    """Caja de foco (sin etiqueta: el valor va en el panel)."""
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rectangle(box, outline=ACCENT + (alpha,), width=6)
    return Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")


def scanline(base: Image.Image, y: int) -> Image.Image:
    """Línea de barrido: la pantalla todavía no tiene resultados."""
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rectangle([0, y - 2, base.width, y + 2], fill=ACCENT + (210,))
    return Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")


def panel(img: Image.Image, results, revealed: int, current: int) -> Image.Image:
    """Imagen + panel al pie; las filas < revealed ya muestran su valor."""
    row_h, top = 30, 18
    out = Image.new("RGB", (img.width, img.height + top * 2 + row_h * len(results)),
                    PANEL_BG)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    f, fs = synth.find_font(25), synth.find_font(22)
    y = img.height + top + row_h // 2
    for i, (lab, val) in enumerate(results):
        if i < revealed:
            v, col = val, (ACCENT if i == current else FG)
        else:
            v, col = "· · ·", (95, 100, 112)
        d.text((16, y), lab, font=fs, fill=(130, 135, 148), anchor="lm")
        d.text((140, y), v, font=f, fill=col, anchor="lm")
        y += row_h
    return out


def field_scan(width=560):
    if os.path.exists(REAL_IMAGE):
        base = Image.open(REAL_IMAGE).convert("RGB").crop(REAL_SCREEN)
        fields, colors = REAL_FIELDS, 64
    else:
        song, chart, level, score = synth.SONGS[0]
        base = synth.draw_screen(song, chart, level, score, "S")
        fields, colors = SYNTH_FIELDS, 96

    small = resize(base, width)
    s = width / base.width
    boxes = {k: tuple(round(v * s) for v in box) for k, box, _, _ in fields}
    results = [(lab, val) for _, _, lab, val in fields if lab]
    full = boxes["fullscore"]

    frames = []
    for frac in (0.15, 0.45, 0.75, 0.90):
        frames.append(panel(scanline(small, int(small.height * frac)), results, 0, -1))
    for a in (110, 210, 140):
        frames.append(panel(boxed(small, full, a), results, 0, -1))

    # Cada zona: pulso de la caja y se revela su valor en el panel.
    rev = 0
    for key, _, lab, _ in fields:
        if lab is None:
            continue
        for a in (130, 255, 170):
            frames.append(panel(boxed(small, boxes[key], a), results, rev + 1, rev))
        rev += 1

    for a in (130, 190):
        frames.append(panel(boxed(small, full, a), results, rev, -1))
    save_gif(frames, "field_scan.gif", duration=230, colors=colors)


if __name__ == "__main__":
    synth_conditions()
    field_scan()
    print(f"ok: {IMG}")
