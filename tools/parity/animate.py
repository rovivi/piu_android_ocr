#!/usr/bin/env python3
"""Genera las animaciones del README (docs/img/*.gif).

Son assets de documentación, no código de producto. Se regeneran con:

    python3 tools/parity/animate.py

- `synth_conditions.gif` — la misma pantalla PIU sintética barriendo ángulo,
  brillo y glare. Muestra lo que el módulo tiene que leer.
- `field_scan.gif` — qué campos lee y dónde, sobre la pantalla canónica de
  Phoenix (título, bolita/nivel, score, rank).

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


def save_gif(frames: list[Image.Image], name: str, duration: int) -> None:
    os.makedirs(IMG, exist_ok=True)
    path = os.path.join(IMG, name)
    # Paleta acotada: el ruido fotográfico infla el GIF si se guarda en RGB.
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
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


# ── 2. qué campos lee ────────────────────────────────────────────────────────

FIELDS = [
    ("song_name", "título → catálogo (canción)"),
    ("difficulty", "bolita → chart type + nivel"),
    ("score", "score"),
    ("rank", "rank (solo se ubica)"),
]


def boxed(screen, box, label, alpha):
    base = screen.convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle(box, outline=ACCENT + (alpha,), width=7)
    tw = 20 + len(label) * 15
    ty = max(0, box[1] - 52)
    d.rectangle([box[0], ty, box[0] + tw, ty + 46], fill=BG + (min(240, alpha + 40),))
    d.text((box[0] + 12, ty + 23), label, font=synth.find_font(28),
           fill=FG + (min(255, alpha + 40),), anchor="lm")
    return Image.alpha_composite(base, ov).convert("RGB")


def field_scan(width=560):
    song, chart, level, score = synth.SONGS[0]
    screen = synth.draw_screen(song, chart, level, score, "S")
    frames = []
    # Pantalla limpia al principio.
    for _ in range(3):
        frames.append(with_caption(resize(screen, width), "lo que lee el módulo"))
    for key, label in FIELDS:
        box = synth.FIELD_BOXES[key]
        for a in (80, 160, 255, 255, 160, 80):
            frames.append(with_caption(resize(boxed(screen, box, label, a), width), label))
    save_gif(frames, "field_scan.gif", duration=150)


if __name__ == "__main__":
    synth_conditions()
    field_scan()
    print(f"ok: {IMG}")
