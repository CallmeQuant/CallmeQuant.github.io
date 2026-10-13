"""Shared plotting style for the figures on this site (Python / matplotlib).

Usage inside figures/<slug>/fig.py:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
    from style import use_style, PALETTE, save

    use_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ...
    save(fig, __file__, "posterior")      # -> assets/figures/<slug>/posterior.svg

The Python scripts produce the images that the website shows.
"""
from __future__ import annotations

import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "ink": "#1a2233",      # text, axes, reference curves
    "blue": "#2b4c9b",     # primary series
    "red": "#b5482f",      # secondary series / contrast
    "green": "#2e7d5b",    # third series
    "gold": "#c2902a",     # fourth series / highlights
    "violet": "#6a4c93",   # fifth series
    "grey": "#8a93a4",     # priors, reference, de-emphasised elements
    "light": "#dde2ea",    # grids, fills
}
CYCLE = [PALETTE[k] for k in ("blue", "red", "green", "gold", "violet", "grey")]

REPO = pathlib.Path(__file__).resolve().parents[2]


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.titleweight": "normal",
        "axes.titlepad": 8,
        "axes.labelsize": 11,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["light"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.handlelength": 1.8,
        "figure.constrained_layout.use": True,
        "svg.fonttype": "path",
        "text.color": PALETTE["ink"],
    })


def outdir(script: str) -> pathlib.Path:
    slug = pathlib.Path(script).resolve().parent.name
    d = REPO / "assets" / "figures" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, script: str, name: str, fmt: str = "svg") -> pathlib.Path:
    """Save to assets/figures/<slug>/<name>.<fmt>. Use fmt='png' for dense scatter plots."""
    path = outdir(script) / f"{name}.{fmt}"
    kw = {"metadata": {"Date": None}} if fmt == "svg" else {}
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig)
    return path
