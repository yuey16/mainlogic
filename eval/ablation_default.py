"""Default behaviour study (Appendix, "Default Behaviour Study"): what each system reaches for.

Decomposes the six-cell core by the model's own answer distribution rather than by accuracy.

    Answers           the share of core rows answered S, C, U
    Answers when wrong  the same, over the rows answered incorrectly
    FD                committed to S or C where the reference label is U
    FU                answered U where the reference label is S or C

    eval/out/ablation_default/default_profile.csv       per model: every measure above
    eval/out/ablation_default/default_profile.tex       the answer-distribution table
    eval/out/ablation_default/default_trend.png         Acc6 against the rate of answering S
    eval/out/ablation_default/default_determination.png FD against FU

    uv run python eval/ablation_default.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from eval.process import CELLS, LABELS, display, frame
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.process import CELLS, LABELS, display, frame

from eval.rq1 import HUES, SHORT, _ok, native_models  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "ablation_default"

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 11, "axes.linewidth": 0.8})
FS, FS_SMALL = 11, 9
PALETTE = HUES + ["#1b9e77", "#e6ab02"]   # ten systems


def profile(df: pd.DataFrame, model: str) -> dict:
    """One model's disposition: what it says, what it says when wrong, FD and FU."""
    core = df[df["cell"].isin(CELLS)]
    judged = core[core[model].notna()]
    out = {"acc": float(_ok(core, model).mean())}
    for label in LABELS:
        out[f"says_{SHORT[label]}"] = float((judged[model] == label).mean())
    wrong = judged[judged[model] != judged["gold"]]
    out["n_wrong"] = int(len(wrong))
    shares = {label: float((wrong[model] == label).mean()) for label in LABELS}
    top = max(shares, key=shares.get)
    out["err_mode"] = f"{SHORT[top]} {100 * shares[top]:.0f}"
    for label in LABELS:
        out[f"err_{SHORT[label]}"] = shares[label]
    gold_u, gold_d = judged[judged["gold"] == "uncertain"], judged[judged["gold"] != "uncertain"]
    out["FD"] = float(gold_u[model].isin(["supported", "contradicted"]).mean())
    out["FU"] = float((gold_d[model] == "uncertain").mean())
    return out


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {m: profile(df, m) for m in native_models(df)}
    return pd.DataFrame(rows).T.sort_values("acc", ascending=False)


def _pct(x) -> str:
    return "--" if x is None or pd.isna(x) else f"{100 * float(x):.1f}"


def latex_table(t: pd.DataFrame) -> str:
    """Acc6 beside the answer distribution, overall and when wrong. A wrong-answer share is bold
    where it exceeds the model's overall share of that label."""
    head = [r"\begin{tabularx}{\linewidth}{@{}Xr rrr rrr r@{}}", r"\toprule",
            r"Model & Acc$_6$ & \multicolumn{3}{c}{Answers} & "
            r"\multicolumn{3}{c}{Answers when wrong} & Errors\\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r" & & $\MLS$ & $\MLC$ & $\MLU$ & $\MLS$ & $\MLC$ & $\MLU$ & "
            r"$\rightarrow$\\",
            r"\midrule"]
    body = []
    for m, r in t.iterrows():
        cells = []
        for label in ("S", "C", "U"):
            said, when_wrong = r[f"says_{label}"], r[f"err_{label}"]
            cells.append((_pct(said), _pct(when_wrong), when_wrong > said))
        body.append(
            f"{display(m).replace('_', chr(92) + '_')} & {_pct(r['acc'])} & "
            + " & ".join(c[0] for c in cells) + " & "
            + " & ".join((r"\textbf{" + c[1] + "}") if c[2] else c[1] for c in cells)
            + f" & {r['err_mode']}\\\\")
    return "\n".join([*head, *body, r"\bottomrule", r"\end{tabularx}"])


def _style(ax) -> None:
    ax.tick_params(labelsize=FS)
    ax.grid(color="#d9d9d9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=FS_SMALL,
              handletextpad=0.4, labelspacing=0.7)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="#ffffff", bbox_inches="tight")
    print(f"wrote {out}")


def plot_determination(t: pd.DataFrame, out: Path) -> None:
    """FD against FU: over-committing against over-hedging."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=300)
    for i, (m, r) in enumerate(t.iterrows()):
        ax.scatter(100 * r["FD"], 100 * r["FU"], s=90, color=PALETTE[i % len(PALETTE)],
                   edgecolor="#000000", linewidth=0.6, zorder=3, label=display(m))
    ax.set_xlabel("FD: committed when the premises do not settle it (%)", fontsize=FS)
    ax.set_ylabel("FU: hedged when they do (%)", fontsize=FS)
    ax.set_xlim(0, 100), ax.set_ylim(0, 40)
    _style(ax)
    _save(fig, out)


def plot_trend(t: pd.DataFrame, out: Path, outlier: str = "gemini-3.8-flash") -> None:
    """Acc6 against how often the model answers `supported`, with the fit drawn without the
    outlier so the exception stays visible."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=300)
    rest = t.drop(index=outlier, errors="ignore")
    x, y = 100 * rest["says_S"].astype(float), 100 * rest["acc"].astype(float)
    m, c = np.polyfit(x, y, 1)
    xs = np.array([x.min() - 3, x.max() + 3])
    ax.plot(xs, m * xs + c, color="#b0b0b0", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(0.97, 0.04, f"$r={y.corr(x):.2f}$ excluding {display(outlier)}", transform=ax.transAxes,
            fontsize=FS_SMALL, color="#8a8a8a", ha="right", va="bottom")
    for i, (name, row) in enumerate(t.iterrows()):
        ax.scatter(100 * float(row["says_S"]), 100 * float(row["acc"]), s=90,
                   color=PALETTE[i % len(PALETTE)], edgecolor="#000000", linewidth=0.6, zorder=3,
                   label=display(name))
    ax.set_xlabel("Answers that are \"supported\" (%)", fontsize=FS)
    ax.set_ylabel("Acc$_6$ (%)", fontsize=FS)
    _style(ax)
    _save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Default behaviour study.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    df = frame()
    t = table(df)
    args.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out / "default_profile.csv")
    (args.out / "default_profile.tex").write_text(latex_table(t) + "\n", encoding="utf-8")
    plot_trend(t, args.out / "default_trend.png")
    plot_determination(t, args.out / "default_determination.png")

    cols = ["acc", "says_S", "says_C", "says_U", "err_S", "err_C", "err_U", "FD", "FU"]
    print("\nANSWER DISTRIBUTION, OVERALL AND WHEN WRONG (%)\n")
    print((t[cols].astype(float) * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
