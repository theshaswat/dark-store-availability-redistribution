"""Professional matplotlib theme for finance/analytics figures.

Colours come from published design systems: IBM Carbon, PostHog, Supabase, Linear and Stripe.

    from chartstyle import apply, C, save
    apply(); ...; save(fig, "out.png")
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

C = {
    # IBM Carbon
    "primary":  "#0f62fe",
    "ink":      "#161616",
    "ink_mut":  "#525252",
    "ink_sub":  "#8c8c8c",
    "canvas":   "#ffffff",
    "surface":  "#f4f4f4",
    "hairline": "#e0e0e0",
    # finance semantics, IBM Carbon
    "pos":      "#24a148",
    "neg":      "#da1e28",
    "warn":     "#f1c21b",
    # categorical sequence, chosen for separability; each traceable to one system
    "cat": [
        "#0f62fe",  # ibm      blue
        "#f7a501",  # posthog  amber
        "#3ecf8e",  # supabase green
        "#5e6ad2",  # linear   lavender
        "#da1e28",  # ibm      red
        "#0d253d",  # stripe   navy
        "#8c8c8c",  # ibm      grey
    ],
}

# ₹ renders as a black box in matplotlib's default font stack, same failure as reportlab.
_RUPEE_SAFE = ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"]


def apply(grid: str = "y") -> None:
    """Set rcParams globally. grid: 'y' (default), 'x', 'both' or 'none'."""
    mpl.rcParams.update({
        "figure.facecolor":  C["canvas"],
        "axes.facecolor":    C["canvas"],
        "savefig.facecolor": C["canvas"],
        "font.family":       "sans-serif",
        "font.sans-serif":   _RUPEE_SAFE,
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlelocation": "left",
        "axes.titlepad":     12,
        "axes.labelsize":    10,
        "axes.labelcolor":   C["ink_mut"],
        "axes.edgecolor":    C["hairline"],
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.prop_cycle":   mpl.cycler(color=C["cat"]),
        "text.color":        C["ink"],
        "xtick.color":       C["ink_sub"],
        "ytick.color":       C["ink_sub"],
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "grid.color":        C["hairline"],
        "grid.linewidth":    0.7,
        "axes.axisbelow":    True,   # grid behind bars, never through them
        "axes.grid":         grid != "none",
        "axes.grid.axis":    "y" if grid == "y" else ("x" if grid == "x" else "both"),
        "legend.frameon":    False,
        "legend.fontsize":   9,
        "lines.linewidth":   2.0,
        "lines.solid_capstyle": "round",
        "figure.dpi":        110,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
    })


def save(fig, path: str, dpi: int = 200) -> str:
    """Save and return the path. Always look at the file afterwards."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=C["canvas"])
    return path


def demo() -> None:
    """assert-based self-check: theme applies, ₹ has a real glyph, file is written."""
    import tempfile, os
    apply()
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.axisbelow"] is True, "grid would draw over bars"
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"][0] == C["cat"][0]

    fp = font_manager.findfont(font_manager.FontProperties(family=_RUPEE_SAFE))
    from matplotlib.ft2font import FT2Font
    assert FT2Font(fp).get_char_index(ord("₹")) != 0, f"no rupee glyph in {fp}"

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([1, 2, 3], [3, 1, 4], color=C["primary"])
    ax.set_ylabel("Revenue (₹ crore)")
    ax.set_title("Self-check")
    out = os.path.join(tempfile.mkdtemp(), "t.png")
    save(fig, out)
    assert os.path.getsize(out) > 5000, "figure did not render"
    plt.close(fig)
    print("demo OK ->", out)


if __name__ == "__main__":
    demo()
