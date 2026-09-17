#!/usr/bin/env python3
"""F0.3: pantallas PIU Phoenix sintéticas para validar SIN las fotos reales.

Dibuja el layout de la pantalla de resultado (título, bolita con el color del
chart y el nivel, score, rank) y la deforma de forma conocida: perspectiva
(0/10/20/30°), brillo (0.4–1.0), parche de glare gaussiano y ruido. Graba cada
caso en `build/synth/` con su ground truth en `gt.json`.

Lo consume `tools/parity/parity.py --synth`, que corre el CLI de host sobre
cada imagen y chequea (a) que con los flags se lea el GT sintético y (b) que el
modo default no cambie. Solo necesita PIL+numpy (no cv2 ni el dataset).

    python3 tools/synth.py --build
    python3 tools/parity/parity.py --synth --rectify
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ANDROID = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ANDROID, "build", "synth")

SCREEN_W, SCREEN_H = 1280, 720
PHOTO_W, PHOTO_H = 1600, 1000
CABINET = (18, 18, 20)

SONGS = [
    ("la campanella", "single", 10, 761729756),
    ("bee", "double", 21, 900123456),
    ("point break", "halfdouble", 17, 784233147),
    ("see", "coop", 6, 1000000),
]
RANKS = "SABCD"


def hsv_to_rgb(h, s, v):
    """h en grados OpenCV (0..179 -> 0..360)."""
    r, g, b = colorsys.hsv_to_rgb((h / 179.0) % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


CHART_HUE = {"single": 10, "coop": 28, "double": 60, "halfdouble": 105}


def find_font(size):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_screen(song, chart, level, score, rank):
    img = Image.new("RGB", (SCREEN_W, SCREEN_H), (28, 30, 40))
    d = ImageDraw.Draw(img)
    # Banda del título (clara, como en Phoenix) + título oscuro encima.
    d.rectangle([0, 20, SCREEN_W, 140], fill=(235, 235, 240))
    f_title = find_font(84)
    d.text((40, 34), song, font=f_title, fill=(20, 20, 25))

    # Bolita: disco con el color del chart, sub-label y nivel encima.
    cx, cy, r = 200, 380, 130
    hue = CHART_HUE.get(chart, 0)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=hsv_to_rgb(hue, 0.85, 0.95))
    f_small = find_font(34)
    d.text((cx, cy - 72), chart.upper(), font=f_small, fill=(255, 255, 255), anchor="mm")
    d.text((cx, cy + 20), str(level), font=find_font(110), fill=(255, 255, 255),
           anchor="mm")

    # Score en una franja clara, dígitos grandes y parejos.
    d.rectangle([420, 300, SCREEN_W - 40, 470], fill=(240, 240, 245))
    d.text((440, 385), str(score), font=find_font(120), fill=(15, 15, 20), anchor="lm")
    d.text((440, 500), "SCORE", font=f_small, fill=(120, 120, 130), anchor="lm")

    # Rank.
    d.text((SCREEN_W - 160, 600), rank, font=find_font(150), fill=(240, 200, 40),
           anchor="mm")
    return img


def perspective_quad(angle_deg, rng):
    """Cuadrilátero donde cae la pantalla dentro de la foto. A más ángulo, más
    keystone horizontal (una foto en diagonal)."""
    inset = math.sin(math.radians(angle_deg)) * 0.5 * PHOTO_W
    jx = rng.uniform(-0.02, 0.02) * PHOTO_W
    jy = rng.uniform(-0.02, 0.02) * PHOTO_H
    x0, y0 = 60 + jx, 60 + jy
    x1, y1 = PHOTO_W - 60 + jx, PHOTO_H - 60 + jy
    return [(x0 + inset, y0), (x1 - inset, y0), (x1, y1), (x0, y1)]


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(pb, dtype=float).reshape(8)
    return np.linalg.solve(A, B).tolist()


def composite(screen, quad):
    """Pega la pantalla (flat) dentro de la foto en el cuadrilátero `quad`."""
    photo = Image.new("RGB", (PHOTO_W, PHOTO_H), CABINET)
    src = [(0, 0), (SCREEN_W, 0), (SCREEN_W, SCREEN_H), (0, SCREEN_H)]
    coeffs = find_coeffs(quad, src)
    warped = screen.transform((PHOTO_W, PHOTO_H), Image.PERSPECTIVE, coeffs,
                              Image.BICUBIC)
    mask = Image.new("L", (PHOTO_W, PHOTO_H), 0)
    ImageDraw.Draw(mask).polygon(quad, fill=255)
    photo.paste(warped, (0, 0), mask)
    return photo


def photometric(img, brightness, glare, nprng):
    a = np.asarray(img, dtype=np.float32) * brightness
    if glare:
        yy, xx = np.mgrid[0:PHOTO_H, 0:PHOTO_W]
        gx, gy = nprng.uniform(0.2, 0.8) * PHOTO_W, nprng.uniform(0.2, 0.8) * PHOTO_H
        blob = 255.0 * np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) /
                                (2.0 * (0.18 * PHOTO_W) ** 2)))
        a += blob[:, :, None]
    noise = nprng.normal(0, 6.0, a.shape)
    return Image.fromarray(np.clip(a + noise, 0, 255).astype(np.uint8))


def build(out_dir, seed=7):
    os.makedirs(out_dir, exist_ok=True)
    cases = []
    rng = random.Random(seed)
    for song, chart, level, score in SONGS:
        for angle in (0, 10, 20, 30):
            for brightness in (0.4, 0.7, 1.0):
                for glare in (False, True):
                    run = random.Random(f"{song}-{angle}-{brightness}-{glare}")
                    nprng = np.random.default_rng(run.randint(0, 2**31 - 1))
                    rank = run.choice(RANKS)
                    # El score solo se conoce a medias en una foto 2P; acá va uno.
                    screen = draw_screen(song, chart, level, score, rank)
                    photo = composite(screen, perspective_quad(angle, run))
                    photo = photometric(photo, brightness, glare, nprng)
                    name = f"{song.replace(' ', '_')}_a{angle}_b{int(brightness*100)}" \
                           f"_g{int(glare)}.png"
                    photo.save(os.path.join(out_dir, name))
                    cases.append({
                        "file": name,
                        "gt": {"song": song, "chart_type": chart, "level": level,
                               "score": score},
                        "cond": {"angle": angle, "brightness": brightness,
                                 "glare": glare},
                    })
    with open(os.path.join(out_dir, "gt.json"), "w") as f:
        json.dump(cases, f, indent=1)
    print(f"synth: {len(cases)} casos en {out_dir}")
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="generar las pantallas")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build(a.out, a.seed)


if __name__ == "__main__":
    main()
