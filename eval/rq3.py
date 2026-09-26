"""RQ3 — how does long context change evidence dependence?

Three renderings of the same items, differing only in how hard the evidence is to find
(Section 5, RQ3):

    native       the cited papers in full, unmarked
    highlighted  the same papers, with the located evidence bracketed in place
    oracle       the located evidence alone, the papers discarded

Each comparison is paired on the rows every view judged, and each gain over native carries an
anchor-level bootstrap interval. Prior is the share of C+ claims the evidence-free probe calls
supported.

    eval/out/rq3/rq3_pminus.csv    the counterfactual row P-, pooled over both claims
    eval/out/rq3/rq3_pminus.tex    tab:rq4_pminus

    uv run python eval/rq3.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd

try:
    from eval.process import CELLS, DATASET, OMIT, display, frame, models
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.process import CELLS, DATASET, OMIT, display, frame, models

from eval.rq1 import is_probe  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "rq3"
VIEWS = ["native", "highlighted", "oracle"]
DETERMINATE = [c for c in CELLS if not c.startswith("P0")]   # the cells oracle can carry


def view_of(condition: str) -> str:
    if condition.endswith("/oracle"):
        return "oracle"
    if condition.endswith("/highlighted"):
        return "highlighted"
    return "native" if "/" not in condition else ""


def conditions(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """{model: {view: column}} for every model that has at least the native view."""
    out: dict[str, dict[str, str]] = {}
    for c in models(df):
        if is_probe(c):
            continue
        v = view_of(c)
        if v:
            out.setdefault(c.split("/")[0], {})[v] = c
    return {m: v for m, v in out.items() if "native" in v}


def probe_of(df: pd.DataFrame, model: str) -> str | None:
    probes = {c.split("/")[0]: c for c in models(df) if "baseline" in c}
    return probes.get(model) or probes.get({"fable": "fable-5.1"}.get(model, model))


def context(dataset: Path = DATASET) -> pd.DataFrame:
    """Per anchor, the properties of the NATIVE premise, which the other views alter."""
    rows = [json.loads(l) for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for r in rows:
        if r["cell"] != "P+C+":
            continue
        st = r.get("ref_statements") or []
        chars = sum(len(t) for t in r["premise"])
        lens = [len(t) for t in r["premise"]]
        offs = [sum(lens[:i]) for i in range(len(lens))]
        starts = [(offs[s["premise_index"]] + s["char_start"]) / max(1, chars) for s in st]
        out[r["anchor_id"]] = {"chars": chars, "n_statements": len(st),
                               "log_chars": math.log10(max(1, chars)),
                               "log_stmt": math.log2(max(1, len(st))),
                               #  evidence in the first 2% of the text is in the abstract
                               "abstract": float(any(x < 0.02 for x in starts)),
                               "covered": sum(s["char_end"] - s["char_start"] for s in st) / max(1, chars)}
    return pd.DataFrame(out).T


def row_table(df: pd.DataFrame, cells: list[str], exclude: tuple[str, ...] = (),
              draws: int = 2000) -> pd.DataFrame:
    """One premise row pooled over its cells, under all three views, beside each model's prior."""
    rows = []
    for model, cols in conditions(df).items():
        if set(VIEWS) - set(cols) or model in exclude:
            continue
        probe = probe_of(df, model)
        pc = df[df["cell"] == "P+C+"]
        d = df[df["cell"].isin(cells)]
        d = d[d[[cols[v] for v in VIEWS]].notna().all(axis=1)]
        ok = {v: d[cols[v]].eq(d["gold"]).astype(float) for v in VIEWS}
        r = {"model": model, "n": len(d),
             "prior": float((pc[probe] == "supported").mean()) if probe in pc else float("nan")}
        for v in VIEWS:
            r[v] = float(ok[v].mean())
        for v in ("highlighted", "oracle"):
            diff = (ok[v] - ok["native"]).groupby(d["anchor"]).mean()
            keys, rng = list(diff.index), random.Random(0)
            bs = sorted(sum(diff[keys[rng.randrange(len(keys))]] for _ in keys) / len(keys)
                        for _ in range(draws))
            r[f"d{v}"] = float(diff.mean())
            r[f"s{v}"] = bool(bs[int(.025 * draws)] * bs[int(.975 * draws) - 1] > 0)
        rows.append(r)
    return pd.DataFrame(rows).set_index("model").sort_values("prior", ascending=False)


def _pct(x) -> str:
    return "--" if x is None or pd.isna(x) else f"{100 * float(x):.1f}"


def _escape(text) -> str:
    return display(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def latex_row_table(t: pd.DataFrame, row_label: str) -> str:
    def d(r, v):
        x = 100 * r[f"d{v}"]
        return f"$\\mathbf{{{x:+.1f}}}$" if r[f"s{v}"] else f"${x:+.1f}$"
    head = [r"\begin{tabularx}{\linewidth}{@{}Xr rrr rr@{}}", r"\toprule",
            rf"Model & Prior & \multicolumn{{3}}{{c}}{{Accuracy on {row_label}}} & "
            r"\multicolumn{2}{c}{Gain over native}\\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
            r" & says $\MLS$ & native & highlighted & oracle & highlighted & oracle\\", r"\midrule"]
    body = [f"{_escape(m)} & {_pct(r['prior'])} & {_pct(r['native'])} & {_pct(r['highlighted'])} & "
            f"{_pct(r['oracle'])} & {d(r, 'highlighted')} & {d(r, 'oracle')}\\\\"
            for m, r in t.iterrows()]
    return "\n".join([*head, *body, r"\bottomrule", r"\end{tabularx}"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RQ3: how does long context change evidence dependence?")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    df = frame()
    minus = row_table(df, ["P-C+", "P-C-"], exclude=tuple(OMIT))
    args.out.mkdir(parents=True, exist_ok=True)
    minus.to_csv(args.out / "rq3_pminus.csv")
    (args.out / "rq3_pminus.tex").write_text(latex_row_table(minus, "$P^{-}$") + "\n", encoding="utf-8")

    print("\nCOUNTERFACTUAL ROW P-, BY VIEW (gains over native in pp, * = interval excludes 0)\n")
    show = (minus[["prior", *VIEWS]] * 100).round(1)
    for v in ("highlighted", "oracle"):
        show[f"Δ{v}"] = minus.apply(lambda r: f"{100 * r[f'd{v}']:+.1f}{'*' if r[f's{v}'] else ''}", axis=1)
    print(show.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
