"""Static PNG charts for reports (matplotlib, imported lazily).

Colors come from a validated categorical palette, assigned by entity and never by
rank: the target language is always slot 1 and the reference language slot 2.
"""

from __future__ import annotations

from pathlib import Path

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TARGET_COLOR = "#2a78d6"  # slot 1, blue
REFERENCE_COLOR = "#eb6834"  # slot 2, orange
BAR = 0.36


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on installed extras
        raise ImportError("charts need the report extra: pip install 'zubaan[report]'") from e
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": TEXT,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT,
            "axes.edgecolor": GRID,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    return plt


def _style(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _short(model: str) -> str:
    return model.split("/")[-1]


def tokens_per_word_chart(
    rows: list[dict], path: Path, target: str = "Urdu", reference: str = "English"
) -> Path:
    """Grouped horizontal bars: tokens per word in each language, one group per tokenizer."""
    plt = _plt()
    rows = [r for r in rows if r["status"] != "skipped"]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(rows) + 1.2))
    ys = list(range(len(rows)))[::-1]
    for offset, key, label, color in (
        (BAR / 2, "target_tokens_per_word", target, TARGET_COLOR),
        (-BAR / 2, "reference_tokens_per_word", reference, REFERENCE_COLOR),
    ):
        values = [r[key] for r in rows]
        pos = [y + offset for y in ys]
        ax.barh(pos, values, height=BAR - 0.04, color=color, label=label)
        for p, v in zip(pos, values, strict=True):
            ax.text(v + 0.05, p, f"{v:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(ys, [_short(r["model"]) for r in rows])
    ax.set_xlabel("Tokens per word (lower is cheaper)")
    ax.set_xlim(0, max(r["target_tokens_per_word"] for r in rows) * 1.15)
    ax.legend(loc="upper right", frameon=False)
    _style(ax)
    ax.set_title("Tokens per word on FLORES-200 devtest", loc="left", fontsize=11)
    return _save(fig, path)


def cost_multiplier_chart(rows: list[dict], path: Path, highlight: str | None = None) -> Path:
    """Horizontal bars: target-language token cost relative to the reference language."""
    plt = _plt()
    rows = [r for r in rows if r["status"] != "skipped"]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
    ys = list(range(len(rows)))[::-1]
    values = [r["cost_multiplier"] for r in rows]
    ax.barh(ys, values, height=0.6, color=TARGET_COLOR)
    for y, v in zip(ys, values, strict=True):
        ax.text(v + 0.04, y, f"{v:.2f}x", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.axvline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.text(1.04, ys[0] + 0.45, "parity (1.0x)", fontsize=8, color=TEXT_SECONDARY)
    labels = [_short(r["model"]) for r in rows]
    ax.set_yticks(ys, labels)
    for tick, row in zip(ax.get_yticklabels(), rows, strict=True):
        if row["model"] == highlight:
            tick.set_fontweight("bold")
    ax.set_xlabel("Urdu tokens / English tokens for the same sentences")
    ax.set_xlim(0, max(values) * 1.15)
    _style(ax)
    ax.set_title("Cost multiplier: how much more Urdu costs", loc="left", fontsize=11)
    return _save(fig, path)


def acceptance_chart(
    languages: dict[str, dict], names: dict[str, str], path: Path, title: str = ""
) -> Path:
    """Acceptance rate and greedy agreement per language, with 95% intervals as whiskers."""
    plt = _plt()
    metrics = (
        ("alpha", "alpha_ci_low", "alpha_ci_high", "Acceptance rate (alpha)"),
        ("greedy_match", "greedy_ci_low", "greedy_ci_high", "Greedy top-1 match"),
    )
    colors = [TARGET_COLOR, REFERENCE_COLOR]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ys = [1, 0]
    langs = list(languages)
    for i, lang in enumerate(langs):
        r = languages[lang]
        offset = BAR / 2 - i * BAR
        pos = [y + offset for y in ys]
        vals = [r[m] for m, *_ in metrics]
        err = [
            [r[m] - r[lo] for m, lo, _, _ in metrics],
            [r[hi] - r[m] for m, _, hi, _ in metrics],
        ]
        ax.barh(pos, vals, height=BAR - 0.04, color=colors[i % 2], label=names.get(lang, lang))
        ax.errorbar(vals, pos, xerr=err, fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
        for p, v, hi in zip(pos, vals, err[1], strict=True):
            ax.text(v + hi + 0.01, p, f"{v:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(ys, [label for *_, label in metrics])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Rate (1.0 = every drafted token accepted); whiskers: 95% interval")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
    _style(ax)
    ax.set_title(title, loc="left", fontsize=11)
    return _save(fig, path)


# Slots 3, 7 and 5 of the validated categorical palette: aqua, violet, magenta. They stay
# apart under colour-vision simulation (worst adjacent pair: Delta E 19.9). Aqua and magenta
# are below 3:1 against the surface, so every bar carries a visible value label.
METHOD_COLORS = {"mean": "#1baf7a", "focus": "#6250d6", "random": "#e87ba4"}
METHOD_LABELS = {
    "mean": "Mean of pieces",
    "focus": "FOCUS-style",
    "random": "Random (control)",
}


def extension_cost_chart(rows: list[dict], path: Path) -> Path:
    """Urdu tokens per word against the number of new tokens, with English as the floor."""
    plt = _plt()
    xs = [r["n_new_tokens"] for r in rows]
    urdu = [r["urdu_tokens_per_word"] for r in rows]
    english = rows[0]["english_tokens_per_word"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(xs, urdu, color=TARGET_COLOR, linewidth=2, marker="o", markersize=7)
    for x, y in zip(xs, urdu, strict=True):
        ax.annotate(
            f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
            fontsize=8, color=TEXT_SECONDARY,
        )  # fmt: skip
    ax.axhline(english, color=REFERENCE_COLOR, linewidth=2)
    ax.text(xs[0], english + 0.12, f"English, {english:.2f}", ha="left", fontsize=8,
            color=TEXT_SECONDARY)  # fmt: skip
    ax.text(xs[0] + 900, urdu[0], "Urdu", ha="left", va="center", fontsize=8,
            color=TEXT_SECONDARY)  # fmt: skip
    ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.set_xlabel("New Urdu tokens added to the vocabulary")
    ax.set_ylabel("Tokens per word (lower is cheaper)")
    ax.set_ylim(0, max(urdu) * 1.2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.set_title("Token cost of Urdu falls fast, then flattens", loc="left", fontsize=11)
    return _save(fig, path)


def extension_quality_chart(
    variants: dict, path: Path, set_name: str = "urdu_heldout", set_label: str = "held-out Urdu"
) -> Path:
    """Bits per character per variant: one group per size, one bar per init method."""
    plt = _plt()
    sizes = sorted({v["n_new_tokens"] for v in variants.values() if v["n_new_tokens"]})
    methods = [m for m in METHOD_COLORS if any(v["method"] == m for v in variants.values())]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(sizes) * len(methods) + 1.6))
    height, y = 0.22, 0.0
    ticks, tick_labels = [], []
    for n in sizes:
        start = y
        for m in methods:
            v = variants.get(f"n{n}-{m}")
            if v is None:
                continue
            bpc = v["bpc"][set_name]["bits_per_character"]
            ax.barh(y, bpc, height=height - 0.03, color=METHOD_COLORS[m],
                    label=METHOD_LABELS[m] if n == sizes[0] else None)  # fmt: skip
            ax.text(bpc + 0.05, y, f"{bpc:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY)
            y -= height
        ticks.append((start + y + height) / 2)
        tick_labels.append(f"{n:,} new tokens")
        y -= 0.25
    base = variants.get("base")
    if base:
        b = base["bpc"][set_name]["bits_per_character"]
        ax.axvline(b, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.text(b + 0.05, 0.2, f"base model, {b:.2f}", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(ticks, tick_labels)
    ax.set_xlabel(f"Bits per character on {set_label} (lower is better)")
    ax.set_xlim(0, max(v["bpc"][set_name]["bits_per_character"] for v in variants.values()) * 1.15)
    ax.set_ylim(y + 0.1, 0.32)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    _style(ax)
    ax.set_title("Untrained new tokens hurt Urdu quality", loc="left", fontsize=11)
    return _save(fig, path)


# Stage 3: one color per model, fixed by entity. The extended model before training is the
# mean-of-pieces variant, so it keeps that variant's aqua from the chart above.
ADAPT_MODEL_COLORS = {"base": "#2a78d6", "extended-untrained": "#1baf7a", "adapted": "#eb6834"}
ADAPT_MODEL_LABELS = {
    "base": "Base model",
    "extended-untrained": "Extended, untrained",
    "adapted": "Extended, adapted (stage 3)",
}
ADAPT_SETS = (
    ("urdu_flores", "FLORES Urdu"),
    ("urdu_heldout", "Held-out Urdu"),
    ("english_flores", "FLORES English"),
)


def adaptation_quality_chart(variants: dict, order: list[str], path: Path) -> Path:
    """Bits per character per text set (groups) and model (bars), every bar labeled.

    Bits per character is per character, so the bars compare across the two tokenizers.
    """
    plt = _plt()
    models = [m for m in order if m in variants]
    unknown = [m for m in models if m not in ADAPT_MODEL_COLORS]
    if unknown:
        raise ValueError(f"no chart color for {unknown}; known: {sorted(ADAPT_MODEL_COLORS)}")
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(models) * len(ADAPT_SETS) + 1.8))
    height, y = 0.3, 0.0
    ticks, labels, top = [], [], 0.0
    for key, label in ADAPT_SETS:
        start = y
        for m in models:
            bpc = variants[m]["bpc"][key]["bits_per_character"]
            top = max(top, bpc)
            ax.barh(y, bpc, height=height - 0.04, color=ADAPT_MODEL_COLORS[m],
                    label=ADAPT_MODEL_LABELS[m] if key == ADAPT_SETS[0][0] else None)  # fmt: skip
            ax.text(bpc + 0.04, y, f"{bpc:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY)
            y -= height
        ticks.append((start + y + height) / 2)
        labels.append(label)
        y -= 0.3
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Bits per character (lower is better)")
    ax.set_xlim(0, top * 1.15)
    ax.set_ylim(y + 0.15, 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    _style(ax)
    ax.set_title("Bits per character before and after stage 3 training", loc="left", fontsize=11)
    return _save(fig, path)
