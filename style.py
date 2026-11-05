"""Shared plotting style so every figure in figures/ looks like one system."""
import matplotlib.pyplot as plt

from data import SUBBASINS  # canonical order: Rustlers, Copper, EAQ, Gothic_ME, PH_LT

DPI = 300
SAVE_KW = dict(dpi=DPI, bbox_inches="tight")

FONT = {
    "suptitle": 13,
    "title": 10,
    "label": 9,
    "tick": 8,
    "legend": 8,
    "annot": 8,
}

# One color per recurring series, reused across every figure.
C = {
    "obs": "#333333",
    "single": "#4c72b0",   # single-C_in model
    "split": "#dd8452",    # split-C_in model
    "Quf": "#8172b2",
    "Qf": "#dd8452",
    "Qs": "#4c72b0",
    "Sf": "#dd8452",
    "Ss": "#4c72b0",
    "Cin_fast": "#55a868",
    "Cin_gw": "#c44e52",
    "benchmark": "#999999",
    "highlight": "#c44e52",
}


def apply_base_style():
    plt.rcParams.update({
        "font.size": FONT["label"],
        "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "figure.titlesize": FONT["suptitle"],
        "savefig.dpi": DPI,
    })


def write_caption(png_path, caption_md):
    md_path = png_path.rsplit(".", 1)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(caption_md.strip() + "\n")


def savefig(fig, png_path, caption_md, **kw):
    kw = {**SAVE_KW, **kw}
    fig.savefig(png_path, **kw)
    write_caption(png_path, caption_md)
    print(f"wrote {png_path}")
