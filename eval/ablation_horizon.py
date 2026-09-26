"""Horizon ablations (Appendix, "Horizon Ablations"): the RQ3 renderings, taken apart.

    eval/out/ablation_horizon/horizon_pplus.tex        the natural row P+ under three views
    eval/out/ablation_horizon/horizon_pminus_cells.tex the counterfactual row, one block per claim
    eval/out/ablation_horizon/horizon.png              per-model gain over native against horizon
    eval/out/ablation_horizon/horizon_slopes.csv       Spearman correlation of that gain with horizon
    eval/out/ablation_horizon/horizon_regression.tex   logit of correctness on premise properties

The regression needs statsmodels.

    uv run python eval/ablation_horizon.py
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
    from eval.process import CELLS, OMIT, display, frame
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.process import CELLS, OMIT, display, frame

from eval.rq1 import HUES  # noqa: E402
from eval.rq3 import (DETERMINATE, VIEWS, _escape, _pct, conditions, context,  # noqa: E402
                      latex_row_table, probe_of, row_table)

OUT = Path(__file__).resolve().parent / "out" / "ablation_horizon"

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 11, "axes.linewidth": 0.8})
FS, FS_SMALL = 11, 9
PALETTE = HUES + ["#1b9e77", "#e6ab02"]          # ten systems

#  (field, bin edges, label options, axis label)
HORIZONS = (("chars", [0, 40_000, 70_000, 120_000, 10**9], dict(scale=1000, unit="k"),
             "Premise length (characters)"),
            ("n_statements", [0, 8, 16, 32, 10**6], {}, "Evidence statements"),
            ("covered", [0, .04, .09, .15, 1.0], {}, "Share of the document changed"))


def pminus_table(df: pd.DataFrame, draws: int = 2000) -> pd.DataFrame:
    """The counterfactual row, cell by cell, under all three views, beside each model's prior."""
    rows = []
    for model, cols in conditions(df).items():
        if set(VIEWS) - set(cols):
            continue
        probe = probe_of(df, model)
        pc = df[df["cell"] == "P+C+"]
        r = {"model": model,
             "prior": float((pc[probe] == "supported").mean()) if probe in pc else float("nan")}
        for cell in ("P-C+", "P-C-"):
            d = df[df["cell"] == cell]
            d = d[d[[cols[v] for v in VIEWS]].notna().all(axis=1)]
            ok = {v: d[cols[v]].eq(d["gold"]).astype(float) for v in VIEWS}
            r[f"{cell}_native"] = float(ok["native"].mean())
            for v in ("highlighted", "oracle"):
                diff = (ok[v] - ok["native"]).groupby(d["anchor"]).mean()
                keys, rng = list(diff.index), random.Random(0)
                bs = sorted(sum(diff[keys[rng.randrange(len(keys))]] for _ in keys) / len(keys)
                            for _ in range(draws))
                lo, hi = bs[int(.025 * draws)], bs[int(.975 * draws) - 1]
                r[f"{cell}_{v}"] = float(ok[v].mean())
                r[f"{cell}_d{v}"] = float(diff.mean())
                r[f"{cell}_s{v}"] = bool(lo * hi > 0)
        rows.append(r)
    return pd.DataFrame(rows).set_index("model").sort_values("prior", ascending=False)


def latex_pminus(t: pd.DataFrame) -> str:
    def d(r, cell, v):
        x = 100 * r[f"{cell}_d{v}"]
        return (f"$\\mathbf{{{x:+.1f}}}$" if r[f"{cell}_s{v}"] else f"${x:+.1f}$")
    head = [r"\begin{tabularx}{\linewidth}{@{}Xr rrrr rrrr@{}}", r"\toprule",
            r"Model & Prior & \multicolumn{4}{c}{$P^{-}C^{+}$, reference $\MLC$} & "
            r"\multicolumn{4}{c}{$P^{-}C^{-}$, reference $\MLS$}\\",
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
            r" & says $\MLS$ & native & $\Delta$highl. & $\Delta$oracle & oracle & "
            r"native & $\Delta$highl. & $\Delta$oracle & oracle\\", r"\midrule"]
    body = []
    for m, r in t.iterrows():
        body.append(f"{_escape(m)} & {_pct(r['prior'])} & "
                    f"{_pct(r['P-C+_native'])} & {d(r, 'P-C+', 'highlighted')} & {d(r, 'P-C+', 'oracle')} & "
                    f"{_pct(r['P-C+_oracle'])} & "
                    f"{_pct(r['P-C-_native'])} & {d(r, 'P-C-', 'highlighted')} & {d(r, 'P-C-', 'oracle')} & "
                    f"{_pct(r['P-C-_oracle'])}\\\\")
    return "\n".join([*head, *body, r"\bottomrule", r"\end{tabularx}"])


def by_context(df: pd.DataFrame, ctx: pd.DataFrame, a: str, b: str, field: str,
               bins) -> pd.DataFrame:
    """Two views against a property of the native premise, on the rows both judged."""
    cells = DETERMINATE if "oracle" in (a, b) else CELLS
    d = df.copy()
    d["_x"] = d["anchor"].map(ctx[field])
    d["bin"] = pd.cut(d["_x"], bins)
    rows = []
    for model, cols in conditions(df).items():
        if a not in cols or b not in cols:
            continue
        part = d[d["cell"].isin(cells) & d[cols[a]].notna() & d[cols[b]].notna()]
        for bucket, chunk in part.groupby("bin", observed=True):
            rows.append({"model": model, "bin": str(bucket), "n": len(chunk),
                         a: float(chunk[cols[a]].eq(chunk["gold"]).mean()),
                         b: float(chunk[cols[b]].eq(chunk["gold"]).mean())})
    return pd.DataFrame(rows)


def horizon_slopes(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Per model and view, Spearman correlation between an anchor's paired gain over native
    (averaged over its determinate cells) and each horizon measure."""
    from scipy.stats import spearmanr
    d = df[df["cell"].isin(DETERMINATE)]
    rows = []
    for model, cols in conditions(df).items():
        for view in ("oracle", "highlighted"):
            if view not in cols:
                continue
            part = d[d[[cols["native"], cols[view]]].notna().all(axis=1)]
            gain = (part[cols[view]].eq(part["gold"]).astype(float)
                    - part[cols["native"]].eq(part["gold"]).astype(float)).groupby(part["anchor"]).mean()
            row = {"model": model, "view": view, "mean_gain": 100 * gain.mean()}
            for field, *_ in HORIZONS:
                rho, pv = spearmanr(ctx.loc[gain.index, field], gain)
                row[field], row[f"{field}_p"] = rho, pv
            rows.append(row)
    return pd.DataFrame(rows).set_index(["view", "model"]).sort_index()


def _label(bucket: str, scale: float = 1.0, unit: str = "") -> str:
    lo, hi = (float(x) for x in bucket.strip("()[]").split(","))
    lo, hi = lo / scale, hi / scale
    fmt = (lambda v: f"{v:g}")
    if lo <= 0:
        return f"$\\leq${fmt(hi)}{unit}"
    if hi >= 1e6 or (hi > 1 and lo < 1):
        return f"$>${fmt(lo)}{unit}"
    return f"{fmt(lo)}--{fmt(hi)}{unit}"


def _order(long: pd.DataFrame) -> list[str]:
    return sorted(set(long["bin"]), key=lambda b: float(b.strip("(").split(",")[0]))


def plot_horizon(df: pd.DataFrame, ctx: pd.DataFrame, out: Path) -> None:
    """Per-model gain over native, oracle and highlighted, against three horizon measures."""
    big, small = FS + 3, FS + 1
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5), dpi=300, sharey=True)
    for row, (field, bins, kw, xlabel) in enumerate(HORIZONS):
        for col, view in enumerate(("oracle", "highlighted")):
            ax = axes[row][col]
            long = by_context(df, ctx, "native", view, field, bins)
            long = long.assign(gain=long[view] - long["native"])
            order = _order(long)
            for i, (model, part) in enumerate(long.groupby("model")):
                v = part.set_index("bin")["gain"].reindex(order)
                ax.plot(range(len(order)), 100 * v.values, color=PALETTE[i % len(PALETTE)],
                        linewidth=1.8, marker="o", markersize=5, alpha=0.85, zorder=2,
                        label=display(model))
            ax.axhline(0, color="#000000", linewidth=0.9, zorder=1)
            ax.set_ylim(-30, 30)
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel(xlabel, fontsize=big)
            ax.tick_params(labelsize=small)
            ax.set_xticks(range(len(order)), [_label(b, **kw) for b in order], fontsize=small)
            ax.set_ylabel("gain (pp)" if col == 0 else "", fontsize=big)
            if row == 0:
                ax.set_title(view, fontsize=big, pad=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.90, 0.5), ncol=1,
               frameon=False, fontsize=big, handlelength=1.8, labelspacing=1.1)
    fig.subplots_adjust(left=0.07, right=0.89, top=0.98, bottom=0.06, wspace=0.08, hspace=0.36)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="#ffffff", bbox_inches="tight")
    print(f"wrote {out}")


def regression(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Per row of the core, a logit of correctness on the premise properties, with model and cell
    fixed effects. Coefficients are log-odds per standard deviation, abstract per unit."""
    import warnings

    import statsmodels.formula.api as smf
    warnings.filterwarnings("ignore")
    long = []
    for model, cols in conditions(df).items():
        for view, col in cols.items():
            part = df[df["cell"].isin(CELLS)][["anchor", "cell", "premise_state", "gold", col]].dropna()
            long.append(part.assign(model=model, view=view, ok=(part[col] == part["gold"]).astype(int))
                        .drop(columns=[col]))
    long = pd.concat(long).join(ctx[["log_chars", "log_stmt", "covered", "abstract"]], on="anchor")
    long = long.rename(columns={"covered": "removed"})
    for k in ("log_chars", "log_stmt", "removed"):
        long[k] = (long[k] - long[k].mean()) / long[k].std()
    out = {}
    for label, view, state in (("native P+", "native", "P+"), ("native P-", "native", "P-"),
                               ("native P0", "native", "P0"), ("oracle P-", "oracle", "P-"),
                               ("highlighted P-", "highlighted", "P-")):
        sub = long[(long.view == view) & (long.premise_state == state)]
        fit = smf.logit("ok ~ log_chars + log_stmt + removed + abstract + C(model) + C(cell)", sub).fit(disp=0)
        t = fit.summary2().tables[1]
        out[label] = {k: (t.loc[k, "Coef."], t.loc[k, "P>|z|"]) for k in ("log_chars", "log_stmt", "removed", "abstract")}
    return pd.DataFrame(out)


def latex_regression(t: pd.DataFrame) -> str:
    names = {"log_chars": "Document length ($\\log_{10}$ chars)", "log_stmt": "Evidence statements ($\\log_2$)",
             "removed": "Fraction removed in $P^{0}$", "abstract": "Evidence in the abstract"}
    head = [r"\begin{tabularx}{\linewidth}{@{}X" + "r" * len(t.columns) + "@{}}", r"\toprule",
            "Premise property & " + " & ".join(t.columns) + r"\\", r"\midrule"]
    body = []
    for k in ("log_chars", "log_stmt", "removed", "abstract"):
        cells = []
        for c in t.columns:
            coef, p = t.loc[k, c]
            star = "$^{***}$" if p < .001 else "$^{**}$" if p < .01 else "$^{*}$" if p < .05 else ""
            cells.append(f"${coef:+.2f}${star}")
        body.append(f"{names[k]} & " + " & ".join(cells) + r"\\")
    return "\n".join([*head, *body, r"\bottomrule", r"\end{tabularx}"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Horizon ablations for RQ3.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    df = frame()
    ctx = context()
    args.out.mkdir(parents=True, exist_ok=True)

    plus = row_table(df, ["P+C+", "P+C-"], exclude=tuple(OMIT))
    plus.to_csv(args.out / "horizon_pplus.csv")
    (args.out / "horizon_pplus.tex").write_text(latex_row_table(plus, "$P^{+}$") + "\n", encoding="utf-8")
    pm = pminus_table(df)
    pm.to_csv(args.out / "horizon_pminus_cells.csv")
    (args.out / "horizon_pminus_cells.tex").write_text(latex_pminus(pm) + "\n", encoding="utf-8")
    plot_horizon(df, ctx, args.out / "horizon.png")
    horizon_slopes(df, ctx).to_csv(args.out / "horizon_slopes.csv")
    reg = regression(df, ctx)
    (args.out / "horizon_regression.tex").write_text(latex_regression(reg) + "\n", encoding="utf-8")

    print("\nLOGIT COEFFICIENTS (log-odds per SD; abstract per unit)\n")
    print(reg.map(lambda v: f"{v[0]:+.2f}{'***' if v[1] < .001 else '**' if v[1] < .01 else '*' if v[1] < .05 else ''}").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
