"""Draw the app icons for the installable web app (PWA): frontend/icons/*.png.

The mark is the same as the favicon: a phone outline in the violet -> cyan brand gradient with a
glowing 'lens'. `maskable` icons keep the mark inside the 80 % safe zone on a full-bleed background,
so Android can crop them to any shape.   python scripts/make_icons.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[1] / "frontend" / "icons"
A1, A2, BG = (139, 92, 246), (6, 182, 212), (11, 11, 20)


def gradient(size: int) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size] / max(size - 1, 1)
    t = ((x + y) / 2)[..., None]
    rgb = np.array(A1) * (1 - t) + np.array(A2) * t
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def icon(size: int, maskable: bool = False) -> Image.Image:
    S = size * 4                                     # draw large, downsample = smooth edges
    im = Image.new("RGBA", (S, S), BG + (255,))
    if not maskable:                                 # rounded app-tile background
        mask = Image.new("L", (S, S), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=255)
        im.putalpha(mask)
    scale = 0.62 if maskable else 0.78
    pw, ph = int(S * scale * 0.62), int(S * scale)
    x0, y0 = (S - pw) // 2, (S - ph) // 2
    phone = Image.new("L", (S, S), 0)
    ImageDraw.Draw(phone).rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=int(pw * 0.25), fill=255)
    im.paste(gradient(S), (0, 0), phone)
    cx, cy, r = S // 2, y0 + int(ph * 0.45), int(pw * 0.28)
    glow = Image.new("L", (S, S), 0)
    ImageDraw.Draw(glow).ellipse([cx - r * 1.5, cy - r * 1.5, cx + r * 1.5, cy + r * 1.5], fill=120)
    glow = glow.filter(ImageFilter.GaussianBlur(r * 0.5))
    white = Image.new("RGBA", (S, S), (255, 255, 255, 255))
    im.paste(white, (0, 0), Image.fromarray((np.asarray(glow) * (np.asarray(phone) / 255)).astype(np.uint8)))
    d = ImageDraw.Draw(im)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 235))
    bw, bh = int(pw * 0.34), max(2, int(ph * 0.05))
    d.rounded_rectangle([cx - bw // 2, y0 + int(ph * 0.82), cx + bw // 2, y0 + int(ph * 0.82) + bh],
                        radius=bh // 2, fill=(255, 255, 255, 180))
    return im.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in (192, 512):
        icon(n).save(OUT / f"icon-{n}.png")
        icon(n, maskable=True).save(OUT / f"maskable-{n}.png")
    icon(180, maskable=True).convert("RGB").save(OUT / "apple-touch-icon.png")
    print("icons ->", OUT)


if __name__ == "__main__":
    main()
