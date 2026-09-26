"""Prompt ablation: default, strict and loose instructions on the same items.

    default   the instruction every run in the paper uses
    strict    default plus one line forbidding world knowledge
    loose     no definitions, no mention of premises: "here is some information, answer"

Every comparison is paired on the rows all three arms judged, and every delta is against default
with an anchor-level bootstrap interval. The question is whether the instruction changes how much
a model leans on memory, so the row breakdown matters more than the aggregate: memory can only
help on P+ and can only hurt on P- and P0.

    eval/out/ablation_prompt/prompt_ablation.csv   per model per arm: Acc6, per-row and per-cell accuracy
    eval/out/ablation_prompt/prompt_ablation.tex   the tabular
    eval/out/ablation_prompt/prompt_acc6.png       Acc6 per arm, one group per model

    uv run python eval/ablation_prompt.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from eval.process import CELLS, OMIT, display, frame, models
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.process import CELLS, OMIT, display, frame, models

from eval.rq1 import is_probe  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "ablation_prompt"
ARMS = ["default", "strict", "loose"]
COLOUR = {"default": "#2a4d9b", "strict": "#d55e00", "loose": "#009e73"}
ROWS = {"P+": ["P+C+", "P+C-"], "P-": ["P-C+", "P-C-"], "P0": ["P0C+", "P0C-"]}

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 11, "axes.linewidth": 0.8})
FS, FS_SMALL = 11, 9


def arms(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """{model: {arm: column}} for every model with the default arm and at least one other."""
    out: dict[str, dict[str, str]] = {}
    for c in models(df):
        if is_probe(c):
            continue
        base, _, rest = c.partition("/")
        if base in OMIT:
            continue
        if rest == "":
            out.setdefault(base, {})["default"] = c
        elif rest in ("strict", "loose"):
            out.setdefault(base, {})[rest] = c
    return {m: a for m, a in out.items() if "default" in a and len(a) > 1}


def _shared(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    core = df[df["cell"].isin(CELLS)]
    return core[core[list(cols.values())].notna().all(axis=1)]


def _acc(part: pd.DataFrame, col: str, cells: list[str] | None = None) -> float:
    p = part if cells is None else part[part["cell"].isin(cells)]
    return float(p[col].eq(p["gold"]).mean()) if len(p) else float("nan")


def _delta(part: pd.DataFrame, a: str, b: str, cells: list[str] | None = None,
           draws: int = 2000) -> tuple[float, bool]:
    p = part if cells is None else part[part["cell"].isin(cells)]
    d = (p[b].eq(p["gold"]).astype(float) - p[a].eq(p["gold"]).astype(float)).groupby(p["anchor"]).mean()
    keys, rng = list(d.index), random.Random(0)
    bs = sorted(sum(d[keys[rng.randrange(len(keys))]] for _ in keys) / len(keys) for _ in range(draws))
    return float(d.mean()), bool(bs[int(.025 * draws)] * bs[int(.975 * draws) - 1] > 0)


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, cols in arms(df).items():
        part = _shared(df, cols)
        for arm, col in cols.items():
            r = {"model": model, "arm": arm, "n": len(part), "acc6": _acc(part, col)}
            for name, cells in ROWS.items():
                r[name] = _acc(part, col, cells)
            for cell in CELLS:
                r[cell] = _acc(part, col, [cell])
            if arm != "default":
                r["d_acc6"], r["s_acc6"] = _delta(part, cols["default"], col)
                for name, cells in ROWS.items():
                    r[f"d_{name}"], r[f"s_{name}"] = _delta(part, cols["default"], col, cells)
            rows.append(r)
    t = pd.DataFrame(rows).set_index(["model", "arm"])
    order = sorted({m for m, _ in t.index}, key=lambda m: -t.loc[(m, "default"), "acc6"])
    return t.reindex([(m, a) for m in order for a in ARMS if (m, a) in t.index])


def latex(t: pd.DataFrame) -> str:
    pct = lambda x: "--" if pd.isna(x) else f"{100 * x:.1f}"
    def dlt(r, k):
        if f"d_{k}" not in r or pd.isna(r.get(f"d_{k}")):
            return ""
        v = f"{100 * r[f'd_{k}']:+.1f}"
        return f" ($\\mathbf{{{v}}}$)" if r.get(f"s_{k}") else f" (${v}$)"
    head = [r"\begin{tabularx}{\linewidth}{@{}Xl rrrr@{}}", r"\toprule",
            r"Model & Prompt & $\mathrm{Acc}_6$ & $P^{+}$ row & $P^{-}$ row & $P^{0}$ row\\", r"\midrule"]
    body, last = [], None
    for (m, a), r in t.iterrows():
        name = display(m) if m != last else ""
        if last is not None and m != last:
            body.append(r"\addlinespace[2pt]")
        body.append(f"{name} & {a} & {pct(r['acc6'])}{dlt(r, 'acc6')} & {pct(r['P+'])}{dlt(r, 'P+')} & "
                    f"{pct(r['P-'])}{dlt(r, 'P-')} & {pct(r['P0'])}{dlt(r, 'P0')}\\\\")
        last = m
    return "\n".join([*head, *body, r"\bottomrule", r"\end{tabularx}",
                      "% parentheses: change from default, bold where an anchor bootstrap interval excludes zero"])


def _tidy(ax, ylabel, lo, hi):
    ax.set_ylim(lo, hi); ax.set_ylabel(ylabel, fontsize=FS); ax.tick_params(labelsize=FS)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="#ffffff", bbox_inches="tight")
    print(f"wrote {out}")


def plot_acc6(t: pd.DataFrame, out: Path) -> None:
    names = list(dict.fromkeys(m for m, _ in t.index))
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 1.8, 3.8), dpi=300)
    w = 0.26
    for k, arm in enumerate(ARMS):
        xs = [i + (k - 1) * w for i, m in enumerate(names) if (m, arm) in t.index]
        ys = [100 * t.loc[(m, arm), "acc6"] for m in names if (m, arm) in t.index]
        ax.bar(xs, ys, width=w * 0.92, color=COLOUR[arm], edgecolor="#000000", linewidth=0.5,
               label=arm, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}", ha="center", fontsize=FS_SMALL, color="#52514e")
    ax.set_xticks(range(len(names)), [display(m) for m in names], fontsize=FS, rotation=15, ha="right")
    _tidy(ax, "$\\mathrm{Acc}_6$ (%)", 30, 90)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False, fontsize=FS)
    _save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prompt ablation: default / strict / loose.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    df = frame()
    t = table(df)
    args.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out / "prompt_ablation.csv")
    (args.out / "prompt_ablation.tex").write_text(latex(t) + "\n", encoding="utf-8")
    plot_acc6(t, args.out / "prompt_acc6.png")
    show = (100 * t[["acc6", "P+", "P-", "P0"]]).round(1)
    for k in ("acc6", "P+", "P-", "P0"):
        if f"d_{k}" in t:
            show[f"Δ{k}"] = t.apply(lambda r: "" if pd.isna(r.get(f"d_{k}")) else
                                    f"{100 * r[f'd_{k}']:+.1f}{'*' if r.get(f's_{k}') else ''}", axis=1)
    print(f"\nPROMPT ABLATION, paired on shared rows (* = anchor bootstrap 95% CI excludes 0)\n")
    print(show.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
