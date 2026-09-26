"""One wide DataFrame: a row per dataset item, a column per model.

    from eval.process import frame, row, core, accuracy
    df = frame()                       # 648 rows x (item columns + one per model)
    row(df, "1810.04805:15/P-C+")      # what every model answered on one item
    core(df, "1810.04805:15")          # the whole six-cell family
    accuracy(df, by="cell")

Every model's answer for an item sits on that item's row, so comparing models is a column
operation and needs no joins. Nothing here calls a model: `data/run.py` writes
`data/results/<run>/verdicts.jsonl`, and this reads those files.

A model that never judged an item leaves NaN there rather than dropping the row — an oracle run has
no P0 rows, and that absence is information. `long(df)` gives the tidy form when a groupby wants it.

Standalone by design: no relative imports, so `python eval/process.py`, `python -m eval.process`
and `from eval.process import frame` all work from the repo root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "final" / "mainlogic.jsonl"
RESULTS = ROOT / "data" / "results"
BUILT = ROOT / "data" / "built"

CELLS = ["P+C+", "P+C-", "P-C+", "P-C-", "P0C+", "P0C-"]
LABELS = ["supported", "contradicted", "uncertain"]

#  Models that BUILT the benchmark. They read the sources, wrote the contrasts and located the
#  evidence, so their score on it is not a measurement of anything - they are being asked to agree
#  with themselves. A run of one may exist on disk for construction-side checks; it never reaches a
#  frame, and therefore never reaches a table, a figure or a paper.
CONSTRUCTION = {"gpt-5.6-sol", "sol"}

#  Canonical short names, keyed by the model id the provider reports.
ALIAS = {"claude-fable-5-1": "fable", "claude-opus-5": "opus-5", "claude-sonnet-5": "sonnet-5",
         "gpt-5.6-luna": "luna", "gpt-5.6-terra": "terra", "gpt-4.1": "4.1", "gpt-6-astra": "astra"}

#  How a short name is printed in a figure or a table, and which family it belongs to.
DISPLAY = {"fable": "Claude Fable 5.1", "opus-5": "Claude Opus 5", "sonnet-5": "Claude Sonnet 5",
           "luna": "GPT-5.6 Luna", "terra": "GPT-5.6 Terra", "4.1": "GPT-4.1", "astra": "GPT-6 Astra",
           "gemini-3.8-flash": "Gemini 3.8 Flash", "gemini-2.5-flash": "Gemini 2.5 Flash",
           "deepseek-flash": "DeepSeek Flash"}
FAMILY = {"fable": "Claude", "opus-5": "Claude", "sonnet-5": "Claude",
          "luna": "GPT", "terra": "GPT", "4.1": "GPT", "astra": "GPT",
          "gemini-3.8-flash": "Gemini", "gemini-2.5-flash": "Gemini", "deepseek-flash": "DeepSeek"}
#  Systems that are on disk but not in the paper's model set.
OMIT: set[str] = set()


def display(name: str) -> str:
    """The printed name for a condition column: model name mapped, ablation suffix kept."""
    base, _, rest = str(name).partition("/")
    out = DISPLAY.get(base, base)
    return f"{out} / {rest}" if rest else out

#  What describes the item, before any model sees it. Everything else in the table is a model.
ITEM_COLS = ["id", "anchor", "paper", "cell", "premise_state", "claim_side", "gold", "conclusion",
             "n_docs", "premise_chars", "n_statements", "route", "negation", "build_status", "n_edits"]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _construction() -> dict[str, dict]:
    """anchor -> what data/augment.py recorded while building it. Absent for anchors built without it."""
    out = {}
    for path in BUILT.glob("*.json") if BUILT.is_dir() else []:
        rec = json.loads(path.read_text(encoding="utf-8"))
        out[rec["id"]] = {"route": rec.get("route"), "negation": rec.get("type"),
                          "build_status": rec.get("status"), "n_edits": len(rec.get("edits") or [])}
    return out


def run_dirs(results: Path) -> list[Path]:
    """Every run directory under `results`, including nested ones such as results/baseline/."""
    if not results.is_dir():
        return []
    return sorted(p.parent for p in results.rglob("summary.json")
                  if (p.parent / "verdicts.jsonl").is_file() and not _is_construction(p.parent))


def _is_construction(run: Path) -> bool:
    """True if this run belongs to a model that built the benchmark. See CONSTRUCTION."""
    try:
        meta = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool({meta.get("model"), meta.get("alias")} & CONSTRUCTION)


def condition_name(meta: dict) -> str:
    """A column name for one run: model, plus whatever makes it a different condition.

    `luna`, `luna/oracle`, `luna/strict`, `luna/evidence` are four columns, not one — they are four
    different questions asked of the same items.
    """
    #  One name per model id, whatever alias the run was launched under. The same model run as
    #  `-m fable` one day and `-m fable-5.1` the next is one column, not two.
    name = ALIAS.get(meta.get("model"), meta.get("alias") or meta.get("model") or "?")
    name = name.replace("gpt-", "")
    if meta.get("ablation", "none") != "none":
        name += f"/{meta['ablation']}"
    if meta.get("strict"):
        name += "/strict"
    if meta.get("prompt", "default") != "default":
        name += f"/{meta['prompt']}"
    return name


def runs(results: Path = RESULTS, dataset: Path = DATASET) -> pd.DataFrame:
    """What is on disk: one row per run directory. Read this before trusting a column."""
    ids = {r["id"] for r in _jsonl(dataset)}
    rows = []
    for run in run_dirs(results):
        summary, verdicts = run / "summary.json", run / "verdicts.jsonl"
        meta = json.loads(summary.read_text(encoding="utf-8"))
        judged = [v for v in _jsonl(verdicts) if v["id"] in ids]
        rows.append({"run": run.name, "condition": condition_name(meta), "model": meta.get("model"),
                     "ablation": meta.get("ablation", "none"), "strict": bool(meta.get("strict")),
                     "prompt": meta.get("prompt", "default"), "rows": len(judged),
                     "answered": sum(1 for v in judged if v.get("verdict")),
                     "generated_at": meta.get("generated_at")})
    return pd.DataFrame(rows)


def frame(dataset: Path = DATASET, results: Path = RESULTS, conditions: list[str] | None = None,
          value: str = "pred", ablations: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The table: one row per dataset item, one column per condition.

    value       "pred"    each condition's column holds the label it returned
                "correct" each column holds True/False
                "both"    both, correctness suffixed " ok"
    conditions  keep only these column names (None keeps all)
    ablations   keep only these views, e.g. ("none",) to exclude oracle and no-premise columns

    Where a condition has several runs, the one that ANSWERED the most rows wins — a run that
    errored on half its rows has the same row count as a clean one and must not displace it.
    """
    if value not in ("pred", "correct", "both"):
        raise ValueError("value must be 'pred', 'correct' or 'both'")

    items = _jsonl(dataset)
    built = _construction()
    base = []
    for r in items:
        extra = built.get(r["anchor_id"], {})
        base.append({
            "id": r["id"], "anchor": r["anchor_id"], "paper": r["paper"], "cell": r["cell"],
            "premise_state": r["cell"][:2], "claim_side": r["cell"][2:], "gold": r["relation"],
            "conclusion": r["conclusion"], "n_docs": r.get("no_premise") or len(r["premise"]),
            "premise_chars": sum(len(t) for t in r["premise"]),
            "n_statements": len(r.get("ref_statements") or []),
            "route": extra.get("route"), "negation": extra.get("negation"),
            "build_status": extra.get("build_status"), "n_edits": extra.get("n_edits"),
        })
    df = pd.DataFrame(base, columns=ITEM_COLS)
    ids = set(df["id"])

    best: dict[str, tuple[int, dict]] = {}
    for run in run_dirs(results):
        summary, verdicts = run / "summary.json", run / "verdicts.jsonl"
        meta = json.loads(summary.read_text(encoding="utf-8"))
        if ablations and meta.get("ablation", "none") not in ablations:
            continue
        name = condition_name(meta)
        if conditions and name not in conditions:
            continue
        answers = {v["id"]: v.get("verdict") for v in _jsonl(verdicts)
                   if v["id"] in ids and v.get("verdict")}
        if answers and len(answers) > best.get(name, (0, {}))[0]:
            best[name] = (len(answers), answers)

    for name, (_, answers) in best.items():
        pred = df["id"].map(answers)
        if value in ("pred", "both"):
            df[name] = pred
        if value in ("correct", "both"):
            ok = pred.eq(df["gold"]).where(pred.notna())
            df[f"{name} ok" if value == "both" else name] = ok

    order = {c: i for i, c in enumerate(CELLS)}
    df = df.sort_values(["paper", "anchor", "cell"],
                        key=lambda s: s.map(order) if s.name == "cell" else s)
    return df.reset_index(drop=True)


def models(df: pd.DataFrame) -> list[str]:
    """The model columns: everything that is not part of the item schema."""
    return [c for c in df.columns if c not in ITEM_COLS and not c.endswith(" ok")]


def row(df: pd.DataFrame, row_id: str) -> pd.DataFrame:
    """One item: what each model answered, and whether it was right."""
    hit = df.loc[df["id"] == row_id]
    if hit.empty:
        raise KeyError(f"{row_id} is not in this frame")
    item = hit.iloc[0]
    names = models(df)
    out = pd.DataFrame({"answered": [item[c] for c in names],
                        "correct": [None if pd.isna(item[c]) else item[c] == item["gold"]
                                    for c in names]}, index=names)
    out.index.name = f"{row_id}  ·  gold: {item['gold']}"
    return out


def item(df: pd.DataFrame, row_id: str) -> pd.Series:
    """The item's own fields, without the model columns — the thing that was shown."""
    return df.loc[df["id"] == row_id, ITEM_COLS].iloc[0]


def core(df: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """A whole six-cell family: cells down the side, gold then each model across the top."""
    part = df.loc[df["anchor"] == anchor].set_index("cell")
    return part.reindex([c for c in CELLS if c in part.index])[["gold", *models(df)]]


def accuracy(df: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Accuracy per model, optionally split by any item column ('cell', 'route', 'paper')."""
    names = models(df)
    ok = pd.DataFrame({c: df[c].eq(df["gold"]).where(df[c].notna()) for c in names}).astype("Float64")
    if by is None:
        return ok.mean().to_frame("accuracy").astype(float)
    keys = [by] if isinstance(by, str) else by
    return ok.groupby([df[k] for k in keys], dropna=False).mean().astype(float)


def grid(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """One model's six-cell grid: premise state down, claim side across."""
    ok = df[model].eq(df["gold"]).where(df[model].notna()).astype("Float64")
    return ok.groupby([df["premise_state"], df["claim_side"]]).mean().unstack().astype(float)


def all_correct(df: pd.DataFrame) -> pd.DataFrame:
    """Per model, the share of complete cores where every one of the six cells is right."""
    names = models(df)
    out = {}
    for c in names:
        ok = df[c].eq(df["gold"]).where(df[c].notna())
        per = ok.groupby(df["anchor"]).agg(["sum", "count"])
        complete = per.loc[per["count"] == len(CELLS)]
        out[c] = (complete["sum"] == len(CELLS)).mean() if len(complete) else float("nan")
    return pd.Series(out).to_frame("all six correct")


def agreement(df: pd.DataFrame) -> pd.DataFrame:
    """How often each pair of models returned the same label, on the rows both judged."""
    names = models(df)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            both = df[df[a].notna() & df[b].notna()]
            out.loc[a, b] = (both[a] == both[b]).mean() if len(both) else float("nan")
    return out


def hardest(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The items fewest models got right — where the benchmark is actually biting."""
    names = models(df)
    ok = pd.DataFrame({c: df[c].eq(df["gold"]).where(df[c].notna()) for c in names})
    # `.where` on a bool column yields object dtype, which nsmallest refuses; make it numeric
    out = df.assign(**{"share_correct": ok.astype("Float64").mean(axis=1).astype(float)})
    return out.nsmallest(n, "share_correct")[
        ["id", "cell", "gold", "route", "negation", "share_correct", *names]]


def confusion(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """gold x predicted counts for one model."""
    return pd.crosstab(df["gold"], df[model]).reindex(index=LABELS, columns=LABELS, fill_value=0)


def long(df: pd.DataFrame) -> pd.DataFrame:
    """The tidy form: one row per (item, model), for groupby and plotting libraries."""
    names = models(df)
    out = df.melt(id_vars=ITEM_COLS, value_vars=names, var_name="condition", value_name="pred")
    out = out.loc[out["pred"].notna()].copy()
    out["correct"] = out["pred"] == out["gold"]
    return out.reset_index(drop=True)


def main() -> int:
    print(runs().to_string(index=False))
    df = frame()
    print(f"\n{len(df)} rows x {len(models(df))} model column(s)\n")
    print(accuracy(df).sort_values("accuracy", ascending=False).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
