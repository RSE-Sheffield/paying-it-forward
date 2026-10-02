#!/usr/bin/env python3
"""
Data x Software: a 3 x 3 typology of research outputs.

Builds the figure as a standalone SVG from a declarative spec (palette + grid
content), then optionally rasterises it to PNG with cairosvg.

    python3 typology_diagram.py --out figure   ->  figure.svg, figure.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

INK = "#141428"  # near-black for headings
BODY = "#4b5468"  # muted grey for descriptions

RED = "#e05c4b"

# Single 5-step light-to-dark yellow/beige ramp. Body cells are graded by
# (row + column) index into this ramp, so it trends lightest at top-left to
# deepest at bottom-right; the row/column headers use steps 0/2/4 of the
# very same ramp (the diagonal cells of the matching level) so every part
# of the figure darkens in the same direction.
_TONE_STEPS = [
    ("#f8f5ed", "#e7d5ac"),
    ("#f4eede", "#e6d09b"),
    ("#f1e7d0", "#e6ca89"),
    ("#efe1c0", "#e7c576"),
    ("#eedbaf", "#e9c163"),
]
CELL_TINTS = [[_TONE_STEPS[i + j] for j in range(3)] for i in range(3)]
_H0, _H1, _H2 = _TONE_STEPS[0], _TONE_STEPS[2], _TONE_STEPS[4]

PALETTE = {
    # One yellow/beige family used everywhere; "gen" is the lightest step,
    # "use" a step darker, "none" the darkest. Software mirrors data exactly.
    "dt_new": dict(
        dark="#c9922e", mid="#e0b563", light=_H0[0], line=_H0[1], ink="#6b4a12"
    ),
    "dt_existing": dict(
        dark="#b98a3e", mid="#d9b579", light=_H1[0], line=_H1[1], ink="#5c4423"
    ),
    "dt_none": dict(
        dark="#a89a82", mid="#c2b7a3", light=_H2[0], line=_H2[1], ink="#5a5040"
    ),
    "sw_new": dict(
        dark="#c9922e",
        mid="#e0b563",
        light=_H0[0],
        line=_H0[1],
        ink="#6b4a12",
        glyph="#c9922e",
        badge="#c9922e",
    ),
    "sw_existing": dict(
        dark="#b98a3e",
        mid="#d9b579",
        light=_H1[0],
        line=_H1[1],
        ink="#5c4423",
        glyph="#b98a3e",
        badge="#b98a3e",
    ),
    "sw_none": dict(
        dark="#a89a82",
        mid="#c2b7a3",
        light=_H2[0],
        line=_H2[1],
        ink="#5a5040",
        glyph="#5a5040",
        badge=RED,
    ),
}

AXIS_BAND = dict(fill="#f8f5ed", line="#e7d5ac", ink="#6b4a12")  # left axis (data)
TOP_BAND = dict(fill="#f8f5ed", line="#e7d5ac", ink="#6b4a12")  # top axis (software)

# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------

TITLE = "PAYING IT FORWARD"
TOP_AXIS = "RESEARCH SOFTWARE"
LEFT_AXIS = "RESEARCH DATA"

COLUMNS = [
    dict(key="sw_new", num="S1", label="RESEARCH SOFTWARE GEN"),
    dict(key="sw_existing", num="S2", label="RESEARCH SOFTWARE USE"),
    dict(key="sw_none", num="S3", label="NO RESEARCH SOFTWARE"),
]

ROWS = [
    dict(key="dt_new", num="D1", label="RESEARCH DATA GEN"),
    dict(key="dt_existing", num="D2", label="RESEARCH DATA USE"),
    dict(key="dt_none", num="D3", label="NO RESEARCH DATA"),
]

CELLS = [
    [
        dict(
            title="RESEARCH DATA GEN + RESEARCH SOFTWARE GEN",
            lines=[
                "Collecting a novel dataset and processing it with a new software tool",
            ],
        ),
        dict(
            title="RESEARCH DATA GEN + RESEARCH SOFTWARE USE",
            lines=[
                "Collecting a novel dataset and processing it with an existing software tool",
            ],
        ),
        dict(
            title="RESEARCH DATA GEN + NO RESEARCH SOFTWARE",
            lines=[
                "Collecting a novel dataset and processing it manually or with generic software tools",
            ],
        ),
    ],
    [
        dict(
            title="RESEARCH DATA USE + RESEARCH SOFTWARE GEN",
            lines=[
                "New algorithm, package, pipeline or workflow developed and tested on public datasets",
            ],
        ),
        dict(
            title="RESEARCH DATA USE + RESEARCH SOFTWARE USE",
            lines=[
                "Analysis of public datasets using existing software tools",
            ],
        ),
        dict(
            title="RESEARCH DATA USE + NO RESEARCH SOFTWARE",
            lines=[
                "Analysis of public datasets using manual or generic software tools",
            ],
        ),
    ],
    [
        dict(
            title="NO RESEARCH DATA + RESEARCH SOFTWARE GEN",
            lines=[
                "New piece of research software developed but not applied to any dataset",
            ],
        ),
        dict(
            title=" NO RESEARCH DATA + RESEARCH SOFTWARE USE",
            lines=[
                "evaluating or benchmarking existing research software",
            ],
        ),
        dict(
            title="NO RESEARCH DATA + NO RESEARCH SOFTWARE",
            lines=[
                "Letter to the editor, editorial or commentary",
            ],
        ),
    ],
]

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass
class Layout:
    width: int = 1480
    height: int = 832
    margin: int = 26
    axis_w: int = 46  # left vertical band
    rowhead_w: int = 178  # row label column
    gap: int = 16
    top_band_y: int = 74
    top_band_h: int = 36
    colhead_y: int = 128
    colhead_h: int = 58
    grid_y: int = 202
    row_h: int = 188
    row_gap: int = 18
    radius: int = 10

    col_x: list = field(default_factory=list)
    col_w: int = 0

    def __post_init__(self):
        left = self.margin + self.axis_w + 10 + self.rowhead_w + self.gap
        right = self.width - self.margin
        self.col_w = (right - left - 2 * self.gap) // 3
        self.col_x = [left + i * (self.col_w + self.gap) for i in range(3)]
        self.grid_left = left
        self.grid_right = right
        self.rowhead_x = self.margin + self.axis_w + 10
        self.row_y = [self.grid_y + i * (self.row_h + self.row_gap) for i in range(3)]
        self.grid_bottom = self.row_y[-1] + self.row_h
        self.height = self.grid_bottom + self.margin


L = Layout()

FONT = "Inter, 'Helvetica Neue', Helvetica, 'DejaVu Sans', Arial, sans-serif"

# --------------------------------------------------------------------------
# Text helpers (approximate metrics so text can be shrunk / wrapped to fit)
# --------------------------------------------------------------------------


def text_width(
    s: str, size: float, bold: bool = False, tracking: float = 0.0
) -> float:
    """Rough advance width of a sans-serif string."""
    narrow = "iljtfIr.,:;'!|()[] "
    wide = "mMwWQ@"
    w = 0.0
    for ch in s:
        if ch in narrow:
            f = 0.33
        elif ch in wide:
            f = 0.88
        elif ch.isupper() or ch.isdigit():
            f = 0.68
        else:
            f = 0.545
        w += f * size
    if bold:
        w *= 1.07
    return (w + tracking * max(len(s) - 1, 0)) * 1.04  # safety margin


def wrap(s: str, size: float, max_w: float, bold=False, tracking=0.0):
    words, lines, cur = s.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and text_width(trial, size, bold, tracking) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def fit(
    s: str,
    max_w: float,
    size: float,
    min_size: float,
    bold=False,
    tracking=0.0,
    max_lines=1,
):
    """Shrink font until the string fits in `max_lines`; then wrap."""
    size_ = size
    while size_ >= min_size:
        lines = wrap(s, size_, max_w, bold, tracking)
        if len(lines) <= max_lines:
            return size_, lines
        size_ -= 0.5
    return min_size, wrap(s, min_size, max_w, bold, tracking)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(
    x,
    y,
    s,
    size=14,
    fill=BODY,
    weight="400",
    anchor="middle",
    tracking=0.0,
    opacity=1.0,
):
    ls = f' letter-spacing="{tracking}"' if tracking else ""
    op = f' opacity="{opacity}"' if opacity != 1.0 else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
        f'font-size="{size:g}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{ls}{op}>{esc(s)}</text>'
    )


def rrect(x, y, w, h, r, fill, stroke=None, sw=1.5):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{r}" ry="{r}" fill="{fill}"{st}/>'
    )


# --------------------------------------------------------------------------
# Icons — all drawn from primitives, no external assets
# --------------------------------------------------------------------------


def database(cx, cy, w, dark, mid, muted=False):
    """Classic three-disc database cylinder, vertically centred on (cx, cy)."""
    rx = w / 2
    ry = rx * 0.34
    h = w * 0.78  # straight side height
    top = cy - h / 2
    bot = cy + h / 2
    body = "#b6bcc8" if muted else mid
    cap = "#cfd4dd" if muted else dark
    side = "#9aa2b1" if muted else dark
    o = []
    o.append(
        f'<path d="M{cx - rx:.1f},{top:.1f} L{cx - rx:.1f},{bot:.1f} '
        f"A{rx:.1f},{ry:.1f} 0 0 0 {cx + rx:.1f},{bot:.1f} "
        f'L{cx + rx:.1f},{top:.1f} Z" fill="{side}"/>'
    )
    o.append(
        f'<ellipse cx="{cx:.1f}" cy="{top:.1f}" rx="{rx:.1f}" '
        f'ry="{ry:.1f}" fill="{cap}"/>'
    )
    for k in (0.34, 0.68):
        yy = top + h * k
        o.append(
            f'<path d="M{cx - rx:.1f},{yy:.1f} '
            f'A{rx:.1f},{ry:.1f} 0 0 0 {cx + rx:.1f},{yy:.1f}" '
            f'fill="none" stroke="{body}" stroke-width="{w * 0.075:.1f}" '
            f'stroke-linecap="round" opacity="0.95"/>'
        )
    return "".join(o)


def code_glyph(cx, cy, w, color):
    """The </> mark, stroked so it is font-independent."""
    h = w * 0.56
    sw = max(w * 0.085, 2.2)
    x0, x1 = cx - w / 2, cx + w / 2
    inset = w * 0.30
    o = [
        f'<g fill="none" stroke="{color}" stroke-width="{sw:.1f}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
    ]
    o.append(
        f'<path d="M{x0 + inset:.1f},{cy - h / 2:.1f} L{x0:.1f},{cy:.1f} '
        f'L{x0 + inset:.1f},{cy + h / 2:.1f}"/>'
    )
    o.append(
        f'<path d="M{x1 - inset:.1f},{cy - h / 2:.1f} L{x1:.1f},{cy:.1f} '
        f'L{x1 - inset:.1f},{cy + h / 2:.1f}"/>'
    )
    o.append(
        f'<path d="M{cx - w * 0.10:.1f},{cy + h * 0.52:.1f} '
        f'L{cx + w * 0.10:.1f},{cy - h * 0.52:.1f}"/>'
    )
    o.append("</g>")
    return "".join(o)


def sparkle(cx, cy, s, color):
    k = s * 0.26
    d = (
        f"M{cx:.1f},{cy - s:.1f} C{cx:.1f},{cy - k:.1f} {cx + k:.1f},{cy:.1f} "
        f"{cx + s:.1f},{cy:.1f} C{cx + k:.1f},{cy:.1f} {cx:.1f},{cy + k:.1f} "
        f"{cx:.1f},{cy + s:.1f} C{cx:.1f},{cy + k:.1f} {cx - k:.1f},{cy:.1f} "
        f"{cx - s:.1f},{cy:.1f} C{cx - k:.1f},{cy:.1f} {cx:.1f},{cy - k:.1f} "
        f"{cx:.1f},{cy - s:.1f} Z"
    )
    return f'<path d="{d}" fill="{color}"/>'


def gear(cx, cy, r, color, teeth=8):
    pts = []
    for i in range(teeth * 2):
        a0 = (i / (teeth * 2)) * 2 * math.pi
        a1 = ((i + 1) / (teeth * 2)) * 2 * math.pi
        rad = r if i % 2 == 0 else r * 0.70
        for a in (a0, a1):
            pts.append(f"{cx + rad * math.cos(a):.2f},{cy + rad * math.sin(a):.2f}")
    poly = " ".join(pts)
    return (
        f'<polygon points="{poly}" fill="{color}"/>'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * 0.30:.1f}" '
        f'fill="#ffffff"/>'
    )


def play_badge(cx, cy, r, color):
    t = r * 0.46
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{color}"/>'
        f'<path d="M{cx - t * 0.55:.1f},{cy - t:.1f} L{cx + t * 0.85:.1f},{cy:.1f} '
        f'L{cx - t * 0.55:.1f},{cy + t:.1f} Z" fill="#ffffff"/>'
    )


def cross_badge(cx, cy, r, color=RED):
    a = r * 0.45
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{color}"/>'
        f'<g stroke="#ffffff" stroke-width="{r * 0.28:.1f}" '
        f'stroke-linecap="round">'
        f'<path d="M{cx - a:.1f},{cy - a:.1f} L{cx + a:.1f},{cy + a:.1f}"/>'
        f'<path d="M{cx + a:.1f},{cy - a:.1f} L{cx - a:.1f},{cy + a:.1f}"/></g>'
    )


def recycle_badge(cx, cy, r, color):
    """Two chasing arrows = re-use of existing material."""
    o = [f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{color}"/>']
    rr, sw, hh = r * 0.52, r * 0.20, r * 0.34

    def pt(a):
        return (cx + rr * math.cos(a), cy + rr * math.sin(a))

    for a0_deg, a1_deg in ((25, 145), (205, 325)):
        a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
        x0, y0 = pt(a0)
        x1, y1 = pt(a1)
        o.append(
            f'<path d="M{x0:.1f},{y0:.1f} A{rr:.1f},{rr:.1f} 0 0 1 '
            f'{x1:.1f},{y1:.1f}" fill="none" stroke="#ffffff" '
            f'stroke-width="{sw:.1f}" stroke-linecap="butt"/>'
        )
        tx, ty = -math.sin(a1), math.cos(a1)  # tangent (direction)
        nx, ny = math.cos(a1), math.sin(a1)  # radial (normal)
        tip = (x1 + tx * hh, y1 + ty * hh)
        b1 = (x1 + nx * hh * 0.62, y1 + ny * hh * 0.62)
        b2 = (x1 - nx * hh * 0.62, y1 - ny * hh * 0.62)
        o.append(
            f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{b1[0]:.1f},'
            f'{b1[1]:.1f} L{b2[0]:.1f},{b2[1]:.1f} Z" fill="#ffffff"/>'
        )
    return "".join(o)


def data_icon(cx, cy, w, row_key):
    p = PALETTE[row_key]
    o = []
    if row_key == "dt_new":
        o.append(database(cx - w * 0.10, cy, w, p["dark"], p["light"]))
        o.append(sparkle(cx + w * 0.52, cy - w * 0.34, w * 0.26, p["mid"]))
        o.append(sparkle(cx + w * 0.34, cy + w * 0.30, w * 0.14, p["mid"]))
    elif row_key == "dt_existing":
        o.append(database(cx - w * 0.10, cy, w, p["dark"], p["light"]))
        o.append(recycle_badge(cx + w * 0.48, cy + w * 0.30, w * 0.34, p["dark"]))
    else:
        o.append(database(cx - w * 0.10, cy, w, p["dark"], p["light"], muted=True))
        o.append(cross_badge(cx + w * 0.46, cy + w * 0.30, w * 0.30))
    return "".join(o)


def software_icon(cx, cy, w, col_key):
    p = PALETTE[col_key]
    glyph, badge = p["glyph"], p["badge"]
    o = [code_glyph(cx - w * 0.08, cy, w, glyph)]
    bx, by, br = cx + w * 0.46, cy + w * 0.24, w * 0.20
    if col_key == "sw_new":
        o.append(gear(bx, by, br, badge))
    elif col_key == "sw_existing":
        o.append(play_badge(bx, by, br, badge))
    else:
        o.append(cross_badge(bx, by, br, badge))
    return "".join(o)


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def build_svg() -> str:
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{L.width}" '
        f'height="{L.height}" viewBox="0 0 {L.width} {L.height}">',
        f'<rect width="{L.width}" height="{L.height}" fill="#ffffff"/>',
    ]

    # ---- title
    o.append(
        text(L.width / 2, 48, TITLE, size=27, fill=INK, weight="800", tracking=0.4)
    )

    # ---- top axis band
    o.append(
        rrect(
            L.grid_left,
            L.top_band_y,
            L.grid_right - L.grid_left,
            L.top_band_h,
            8,
            TOP_BAND["fill"],
            TOP_BAND["line"],
        )
    )
    o.append(
        text(
            (L.grid_left + L.grid_right) / 2,
            L.top_band_y + L.top_band_h / 2 + 6,
            TOP_AXIS,
            size=17,
            fill=TOP_BAND["ink"],
            weight="700",
            tracking=0.8,
        )
    )

    # ---- left axis band
    ax_y, ax_h = L.grid_y, L.grid_bottom - L.grid_y
    o.append(
        rrect(L.margin, ax_y, L.axis_w, ax_h, 8, AXIS_BAND["fill"], AXIS_BAND["line"])
    )
    cx, cy = L.margin + L.axis_w / 2, ax_y + ax_h / 2
    o.append(
        f'<g transform="rotate(-90 {cx:.1f} {cy:.1f})">'
        + text(
            cx,
            cy + 6,
            LEFT_AXIS,
            size=17,
            fill=AXIS_BAND["ink"],
            weight="700",
            tracking=1.0,
        )
        + "</g>"
    )

    # ---- column headers (same light fill as the matching row header, so
    # e.g. S1 and D1 read as identical colours)
    for j, col in enumerate(COLUMNS):
        p = PALETTE[col["key"]]
        x, w = L.col_x[j], L.col_w
        y, h = L.colhead_y, L.colhead_h
        o.append(rrect(x, y, w, h, L.radius, p["light"], p["line"]))
        ix = x + 46
        o.append(software_icon(ix, y + h / 2, 40, col["key"]))
        tx0 = x + 86
        avail = w - 86 - 18
        label = f"{col['num']}  ·  {col['label']}"
        size, lines = fit(label, avail, 14, 11, bold=True, tracking=0.6, max_lines=2)
        start = y + h / 2 - (len(lines) - 1) * (size + 4) / 2 + size * 0.36
        for k, ln in enumerate(lines):
            o.append(
                text(
                    tx0,
                    start + k * (size + 4),
                    ln,
                    size=size,
                    fill=p["ink"],
                    weight="700",
                    anchor="start",
                    tracking=0.6,
                )
            )

    # ---- rows
    for i, row in enumerate(ROWS):
        rp = PALETTE[row["key"]]
        y, h = L.row_y[i], L.row_h

        # row header
        o.append(
            rrect(L.rowhead_x, y, L.rowhead_w, h, L.radius, rp["light"], rp["line"])
        )
        o.append(
            data_icon(L.rowhead_x + L.rowhead_w / 2, y + h * 0.33, 44, row["key"])
        )
        row_label = f"{row['num']}  ·  {row['label']}"
        size, lines = fit(
            row_label,
            L.rowhead_w - 26,
            15,
            11.5,
            bold=True,
            tracking=0.5,
            max_lines=3,
        )
        base = y + h * 0.63
        for k, ln in enumerate(lines):
            o.append(
                text(
                    L.rowhead_x + L.rowhead_w / 2,
                    base + k * (size + 5),
                    ln,
                    size=size,
                    fill=rp["ink"],
                    weight="800",
                    tracking=0.5,
                )
            )

        # cells
        for j, col in enumerate(COLUMNS):
            cell = CELLS[i][j]
            x, w = L.col_x[j], L.col_w
            tint_fill, tint_line = CELL_TINTS[i][j]
            o.append(rrect(x, y, w, h, L.radius, tint_fill, tint_line))
            o.append(
                text(
                    x + 14,
                    y + 22,
                    f"{col['num']}{row['num']}",
                    size=13,
                    fill=INK,
                    weight="800",
                    anchor="start",
                    tracking=0.3,
                )
            )

            # icon pair
            ccx = x + w / 2
            icy = y + 46
            o.append(data_icon(ccx - 52, icy, 38, row["key"]))
            o.append(software_icon(ccx + 52, icy, 44, col["key"]))

            # title + description, vertically centred under the icons
            avail = w - 34
            tsize, tlines = fit(
                cell["title"], avail, 16.5, 11, bold=True, tracking=0.4, max_lines=1
            )
            if len(tlines) > 1:  # only wrap if it truly cannot fit
                tsize, tlines = fit(
                    cell["title"], avail, 15, 11, bold=True, tracking=0.4, max_lines=2
                )
            body = [fit(s_, avail, 14, 11, max_lines=1) for s_ in cell["lines"]]
            bsize = min(b[0] for b in body)
            flat = [ln for _, lns in body for ln in lns]

            tlh, blh = tsize + 5, bsize + 9
            block_h = len(tlines) * tlh + 12 + len(flat) * blh
            r_top, r_bot = y + 76, y + h - 14
            start = r_top + max(0.0, (r_bot - r_top - block_h) / 2)

            for k, ln in enumerate(tlines):
                o.append(
                    text(
                        ccx,
                        start + tsize + k * tlh,
                        ln,
                        size=tsize,
                        fill=INK,
                        weight="800",
                        tracking=0.4,
                    )
                )
            btop = start + len(tlines) * tlh + 12 + bsize
            for k, ln in enumerate(flat):
                o.append(text(ccx, btop + k * blh, ln, size=bsize, fill=BODY))

    o.append("</svg>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="typology")
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()

    svg = build_svg()
    with open(f"{args.out}.svg", "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"wrote {args.out}.svg  ({L.width}x{L.height})")

    try:
        import cairosvg

        cairosvg.svg2png(
            bytestring=svg.encode(),
            write_to=f"{args.out}.png",
            scale=args.scale,
            background_color="#ffffff",
        )
        print(f"wrote {args.out}.png (scale {args.scale})")
    except ImportError:
        print("cairosvg not installed - SVG only (pip install cairosvg)")


if __name__ == "__main__":
    main()
