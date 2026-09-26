"""RQ1 — do judgments track evidential relations?

Scores every model on the six-cell core of every anchor (Section 5, RQ1):

    eval/out/rq1/rq1_core.csv        per model: Nat., Acc6, Macro-F1, GC_premise, GC_claim, GC6
    eval/out/rq1/rq1_core.tex        tab:rq1_core
    eval/out/rq1/rq1_grid.png        fig:rq1_grid, the six-cell grid per model
    eval/out/rq1/rq1_necessity.csv   decisive removal (P0C+) against redundant removal (P=C+)
    eval/out/rq1/rq1_necessity.png   fig:rq2_necessity

Measures:

    Nat.        accuracy on the natural cell (P+,C+)
    Acc6        mean over the six cells, anchor-balanced
    Macro-F1    unweighted mean F1 over the three labels
    GC_premise  a premise row is right when both claims are, over 3N rows
    GC_claim    a claim column is right when all three premises are, over 2N columns
    GC6         an anchor is right when all six cells are
    both        paired correctness on (P0C+, P=C+), over the anchors that carry P=

    uv run python eval/rq1.py
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from eval.process import CELLS, LABELS, OMIT, FAMILY, display, frame, models
except ImportError:  # running as `python eval/rq1.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.process import CELLS, LABELS, OMIT, FAMILY, display, frame, models

OUT = Path(__file__).resolve().parent / "out" / "rq1"

GOLD = {"P+C+": "supported", "P+C-": "contradicted", "P-C+": "contradicted",
        "P-C-": "supported", "P0C+": "uncertain", "P0C-": "uncertain"}
SHORT = {"supported": "S", "contradicted": "C", "uncertain": "U"}

#  Decisive removal (P0, the evidence cut) against redundant removal (P=, as much text cut from
#  where the evidence is not). A model right on both is reading what was removed, not how much.
NECESSITY = [("P0C+", "P=C+")]


def is_probe(name: str) -> bool:
    """An evidence-free probe measures a PRIOR, not evidence use. It is used by RQ2 and never
    joins the model comparison here."""
    return "no-premise" in name or "baseline" in name


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.8,
})
HUES = ["#2a4d9b", "#d55e00", "#009e73", "#8e44ad", "#56b4e9", "#cc79a7", "#666666", "#b8860b"]
FS = 11


def _ok(df: pd.DataFrame, model: str) -> pd.Series:
    """Correct/incorrect per row, NaN where the model did not judge the row."""
    return df[model].eq(df["gold"]).where(df[model].notna()).astype("Float64")


def _paired(df: pd.DataFrame, model: str, pairs: list[tuple[str, str]]) -> tuple[float, int]:
    """(score, anchors) for a set of cell pairs: both members right, averaged within an anchor."""
    ok = _ok(df, model)
    wide = pd.DataFrame({"anchor": df["anchor"], "cell": df["cell"], "ok": ok})
    table = wide.pivot_table(index="anchor", columns="cell", values="ok", aggfunc="first")
    per_anchor = []
    for _, row in table.iterrows():
        got = [row[a] * row[b] for a, b in pairs
               if a in row.index and b in row.index and pd.notna(row[a]) and pd.notna(row[b])]
        if got:
            per_anchor.append(sum(got) / len(got))     # weights sum to one within the anchor
    if not per_anchor:
        return float("nan"), 0
    return float(sum(per_anchor) / len(per_anchor)), len(per_anchor)


def _macro_f1(df: pd.DataFrame, model: str) -> float:
    judged = df[df[model].notna()]
    scores = []
    for label in LABELS:
        tp = int(((judged["gold"] == label) & (judged[model] == label)).sum())
        fp = int(((judged["gold"] != label) & (judged[model] == label)).sum())
        fn = int(((judged["gold"] == label) & (judged[model] != label)).sum())
        if tp + fn == 0:
            continue
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(sum(scores) / len(scores)) if scores else float("nan")


def _gc6(df: pd.DataFrame, model: str) -> tuple[float, int]:
    ok = _ok(df, model)
    per = pd.DataFrame({"anchor": df["anchor"], "ok": ok}).groupby("anchor")["ok"].agg(["sum", "count"])
    complete = per[per["count"] == len(CELLS)]
    if complete.empty:
        return float("nan"), 0
    return float((complete["sum"] == len(CELLS)).mean()), len(complete)


def grouped(core: pd.DataFrame, model: str) -> dict:
    """The core read by row and by column, each group correct only if every cell in it is."""
    w = core.pivot_table(index="anchor", columns="cell", values=model, aggfunc="first")
    if not set(CELLS) <= set(w.columns):
        return {}
    g = core.pivot_table(index="anchor", columns="cell", values="gold", aggfunc="first")
    w, g = w[CELLS], g.reindex(index=w.index)[CELLS]
    ok = ((w == g) & w.notna())[w.notna().all(axis=1)]
    if ok.empty:
        return {}
    rows = [ok[f"{r}C+"] & ok[f"{r}C-"] for r in ("P+", "P-", "P0")]
    cols = [ok["P+C+"] & ok["P-C+"] & ok["P0C+"], ok["P+C-"] & ok["P-C-"] & ok["P0C-"]]
    return {"gc_premise": float(pd.concat(rows).mean()), "gc_claim": float(pd.concat(cols).mean())}


def native_models(df: pd.DataFrame) -> list[str]:
    """The paper's model set: native runs that judged the whole core."""
    core = df[df["cell"].isin(CELLS)]
    return [m for m in models(df) if "/" not in m and m not in OMIT and not is_probe(m)
            and core.loc[core[m].notna(), "cell"].nunique() == len(CELLS)]


def core_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: the columns of Table 2, plus the denominators behind them."""
    core = df[df["cell"].isin(CELLS)]
    rows = []
    for model in native_models(df):
        ok = _ok(core, model)
        gc6, n_cores = _gc6(core, model)
        nat = _ok(core[core["cell"] == "P+C+"], model)
        #  Acc6 is anchor-balanced: average an anchor's cells, then average anchors.
        per_anchor = pd.DataFrame({"anchor": core["anchor"], "ok": ok}).groupby("anchor")["ok"].mean()
        rows.append({"system": model, "rows_judged": int(ok.notna().sum()),
                     "nat": float(nat.mean()), "acc6": float(per_anchor.mean()),
                     "macro_f1": _macro_f1(core, model), "gc6": gc6, "gc6_cores": n_cores,
                     **grouped(core, model)})
    return pd.DataFrame(rows).set_index("system").sort_values("acc6", ascending=False)


def necessity_table(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Per model: correct on decisive removal, on redundant removal, and on both."""
    rows = {}
    for model in order:
        both, n = _paired(df, model, NECESSITY)
        p0 = df[(df["cell"] == "P0C+") & df[model].notna()]
        pe = df[df["cell"] == "P=C+"]
        rows[model] = {"decisive": float((p0[model] == "uncertain").mean()),
                       "redundant": float(_ok(pe, model).mean()), "both": both, "anchors": n}
    return pd.DataFrame(rows).T


def _pct(x) -> str:
    return "--" if x is None or pd.isna(x) else f"{100 * float(x):.1f}"


def _escape(text: str) -> str:
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def latex_table(table: pd.DataFrame) -> str:
    """The tabular of Table 2: single-cell scores, then the grouped scores, strictest last."""
    head = [r"\begin{tabularx}{\linewidth}{@{}Xrrr rrr@{}}", r"\toprule",
            r"System & Nat. & Acc$_6$ & Macro-F1 & \multicolumn{3}{c}{Grouped correctness}\\",
            r"\cmidrule(lr){5-7}",
            r" & & & & GC$_{\mathrm{premise}}$ & GC$_{\mathrm{claim}}$ & GC$_6$\\", r"\midrule"]
    body = [f"{_escape(display(m))} & {_pct(r['nat'])} & {_pct(r['acc6'])} & {_pct(r['macro_f1'])} & "
            f"{_pct(r['gc_premise'])} & {_pct(r['gc_claim'])} & {_pct(r['gc6'])}\\\\"
            for m, r in table.iterrows()]
    return "\n".join([*head, *body, r"\bottomrule", r"\end{tabularx}"])


def grid_order(df: pd.DataFrame, ncol: int = 5) -> list[str]:
    """Native systems grouped by family, families and members both ordered by Acc6, packed into
    rows of `ncol` without splitting a family where a permutation allows it."""
    core = df[df["cell"].isin(CELLS)]
    names = native_models(df)
    acc = {m: float(_ok(core, m).mean()) for m in names}
    fams: dict[str, list[str]] = {}
    for m in sorted(names, key=lambda m: -acc[m]):
        fams.setdefault(FAMILY.get(m, m), []).append(m)
    ranked = sorted(fams, key=lambda f: -sum(acc[m] for m in fams[f]) / len(fams[f]))
    nrow = -(-len(names) // ncol)

    def rows_of(order):
        rows, cur = [], []
        for f in order:
            if sum(len(fams[g]) for g in cur) + len(fams[f]) > ncol:
                rows.append(cur); cur = []
            cur.append(f)
        rows.append(cur)
        return rows

    best = ranked
    for perm in itertools.permutations(ranked):
        rows = rows_of(perm)
        if len(rows) == nrow and all(sum(len(fams[g]) for g in r) == ncol for r in rows[:-1]):
            best = list(perm); break
    return [m for f in best for m in fams[f]]


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="#ffffff", bbox_inches="tight")
    print(f"wrote {out}")


def plot_grid(df: pd.DataFrame, out: Path, ncol: int = 5) -> None:
    """The six-cell matrix per model, in two rows grouped by family."""
    df = df[df["cell"].isin(CELLS)]
    names = grid_order(df, ncol)
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.6, 3.1 * nrow), dpi=300, squeeze=False)
    cmap = plt.get_cmap("Blues")
    for k, model in enumerate(names):
        ax = axes[k // ncol][k % ncol]
        ok = _ok(df, model)
        cell = pd.DataFrame({"premise": df["premise_state"], "claim": df["claim_side"], "ok": ok})
        table = cell.pivot_table(index="premise", columns="claim", values="ok", aggfunc="mean")
        for r, premise in enumerate(["P+", "P-", "P0"]):
            for c, claim in enumerate(["C+", "C-"]):
                v = 100 * float(table.loc[premise, claim])
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=cmap(v / 100),
                                           edgecolor="#000000", linewidth=0.6))
                ink = "#ffffff" if v > 55 else "#0b0b0b"
                ax.text(c + .5, r + .56, f"{v:.0f}%", ha="center", va="center", fontsize=11, color=ink)
                ax.text(c + .5, r + .25, SHORT[GOLD[premise + claim]], ha="center", va="center",
                        fontsize=8, color="#e8e8e8" if v > 55 else "#52514e")
        ax.set_xlim(0, 2), ax.set_ylim(3, 0)
        ax.set_xticks([.5, 1.5], ["$C^{+}$", "$C^{-}$"], fontsize=11)
        ax.set_yticks([.5, 1.5, 2.5], ["$P^{+}$", "$P^{-}$", "$P^{0}$"] if k % ncol == 0 else [""] * 3,
                      fontsize=11)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(length=0, pad=2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel(f"{display(model)}\n{100 * float(ok.mean()):.1f}%", fontsize=10.5, labelpad=5)
    for k in range(len(names), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.subplots_adjust(wspace=0.12, hspace=0.45)
    _save(fig, out)


def plot_necessity(t: pd.DataFrame, out: Path) -> None:
    with plt.rc_context({"font.size": FS}):
        fig, ax = plt.subplots(figsize=(1.35 * max(len(t), 3) + 1.8, 3.6), dpi=300)
        width, xs = 0.26, range(len(t))
        for k, (col, colour, label) in enumerate((("decisive", "#009e73", "decisive removal"),
                                                  ("redundant", "#2a4d9b", "redundant removal"),
                                                  ("both", "#d55e00", "both"))):
            ax.bar([x + (k - 1) * width for x in xs], [100 * t.loc[m, col] for m in t.index],
                   width=width * 0.9, color=colour, edgecolor="#000000", linewidth=0.5, label=label,
                   zorder=2)
        ax.set_xticks(list(xs), [display(m) for m in t.index], fontsize=FS, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Correct (%)", fontsize=FS)
        ax.tick_params(labelsize=FS)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False, fontsize=FS)
        _save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RQ1: do judgments track evidential relations?")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    df = frame()
    table = core_table(df)
    nec = necessity_table(df, list(table.index))

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "rq1_core.csv")
    (args.out / "rq1_core.tex").write_text(latex_table(table) + "\n", encoding="utf-8")
    nec.to_csv(args.out / "rq1_necessity.csv")
    plot_grid(df, args.out / "rq1_grid.png")
    plot_necessity(nec, args.out / "rq1_necessity.png")

    print(f"\n{df['anchor'].nunique()} anchors · {len(df)} rows\n")
    print((table[["nat", "acc6", "macro_f1", "gc_premise", "gc_claim", "gc6"]] * 100).round(1).to_string())
    print(f"\nDECISIVE AGAINST REDUNDANT REMOVAL (over {int(nec['anchors'].max())} anchors with P=)\n")
    print((nec[["decisive", "redundant", "both"]] * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
