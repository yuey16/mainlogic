"""RQ2 — when do prior beliefs compete with evidence?

Joins each model's evidence-free probe (its prior on the claim, asked with no premises) to its
contextual answers (Section 5, RQ2). The frame keeps DETERMINATE core cells the model judged and
holds a settled prior on:

    alpha = 1   the prior points the same way as the reference label
    alpha = 0   it points the opposite way

Measures, anchor-balanced (average within an anchor, then across anchors), with paired
anchor-level bootstrap intervals (B = 2000):

    Gap        accuracy aligned - accuracy conflicting, paired within claims
    ErrDir     on conflict errors, the share that repeats the prior rather than answering U
    DirComp    on undetermined rows with a settled prior, the share of commitments that follow it

GLM: logit(correct) ~ beta0 + beta1 * alpha per model. With one binary covariate the model is
saturated, so the estimates are the two group logits in closed form. Standard errors are
anchor-cluster-robust (CR1 sandwich); the model-based ones stay in the CSV with suffix `_ise`.

Stratified gaps: the Gap recomputed within the claims a model believes true (prior supported)
and believes false (prior contradicted). A fixed label preference enters the two with opposite
signs and prior reliance with the same sign, so their half-sum is the memory share and their
half-difference the disposition share. Both groups share one set of anchor resamples.

    eval/out/rq2/rq2_measures.csv     tab:rq3_measures_main, tab:rq3_measures: Gap, ErrDir, DirComp with intervals
    eval/out/rq2/rq2_glm.csv          tab:rq3_glm: n, group accuracies, betas, SEs, z, p, OR
    eval/out/rq2/rq2_stratified.csv   tab:rq3_stratified, tab:rq3_stratified_cis: stratified gaps, memory, disposition

    uv run python eval/rq2.py
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

try:
    from eval.process import CELLS, frame, models, runs
    from eval.rq1 import is_probe
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.process import CELLS, frame, models, runs
    from eval.rq1 import is_probe

OUT = Path(__file__).resolve().parent / "out" / "rq2"

SETTLED = ["supported", "contradicted"]     # a prior of "uncertain" is b = bottom: no prior


def probe_columns(df: pd.DataFrame) -> dict[str, str]:
    """native model column -> its baseline probe column, where both exist. Matched by the
    model id in the run manifests, not by column-name prefix: a native run and its probe
    can carry different aliases (e.g. `fable` vs `fable-5.1/...` for claude-fable-5-1)."""
    by_cond = dict(zip(runs()["condition"], runs()["model"]))
    probes = [c for c in models(df) if is_probe(c)]
    out = {}
    for m in models(df):
        if is_probe(m) or "/" in m:
            continue
        mid = by_cond.get(m)
        out[m] = next((p for p in probes if by_cond.get(p) == mid), "") if mid else ""
    return out


def prior_lookup(df: pd.DataFrame, probe: str) -> pd.DataFrame:
    """(anchor, claim_side) -> the model's prior on that claim. One row per key; the no-premise
    run asks each claim once, so this is a lookup, not an aggregation."""
    seen = df[["anchor", "claim_side", probe]].dropna(subset=[probe])
    return seen.drop_duplicates(["anchor", "claim_side"])


def glm_frame(df: pd.DataFrame, model: str, probe: str) -> pd.DataFrame:
    """One row per evaluated determinate instance with a settled prior: the GLM 1 frame."""
    core = df[df["cell"].isin(CELLS) & df["gold"].isin(SETTLED) & df[model].notna()]
    joined = core.merge(prior_lookup(df, probe).rename(columns={probe: "prior"}),
                        on=["anchor", "claim_side"], how="inner")
    settled = joined[joined["prior"].isin(SETTLED)]
    out = settled[["id", "anchor", "claim_side", "cell", "gold", model, "prior"]].rename(columns={model: "pred"})
    out["model"] = model
    out["alpha"] = (out["prior"] == out["gold"]).astype(int)
    out["correct"] = (out["pred"] == out["gold"]).astype(int)
    return out


def u_frame(df: pd.DataFrame, model: str, probe: str) -> pd.DataFrame:
    """Prior-settled rows on undetermined (P0) cells: the DirCompletion frame."""
    und = df[df["cell"].isin(["P0C+", "P0C-"]) & df[model].notna()]
    joined = und.merge(prior_lookup(df, probe).rename(columns={probe: "prior"}),
                       on=["anchor", "claim_side"], how="inner")
    settled = joined[joined["prior"].isin(SETTLED)]
    return settled[["id", "anchor", "claim_side", "cell", "gold", model, "prior"]].rename(
        columns={model: "pred"})


def fit_binary(glm: pd.DataFrame) -> dict:
    """Saturated logit(correct) ~ alpha: closed-form betas, OR; anchor-clustered (CR1) SEs, z, p."""
    out = {}
    for arm, key in ((0, "misaligned"), (1, "aligned")):
        part = glm[glm["alpha"] == arm]
        acc = float(part["correct"].mean()) if len(part) else float("nan")
        out[f"n_{key}"], out[f"acc_{key}"] = len(part), acc
        if 0 < acc < 1:
            out[f"se_{key}"] = 1.0 / math.sqrt(len(part) * acc * (1 - acc))
    if "se_misaligned" not in out or "se_aligned" not in out:
        return out                            # a degenerate arm: MLE off at +-inf, no variance
    p0, p1 = out["acc_misaligned"], out["acc_aligned"]
    out["beta0"] = math.log(p0 / (1 - p0))
    out["beta1"] = math.log(p1 / (1 - p1)) - out["beta0"]
    out["beta0_ise"] = out["se_misaligned"]
    out["beta1_ise"] = math.sqrt(out["se_misaligned"] ** 2 + out["se_aligned"] ** 2)
    out["or"] = math.exp(out["beta1"])
    out["n"] = len(glm)

    #  Cluster-robust variance (sandwich over anchors, CR1): sigmoid of the same closed-form betas.
    def sig(x):
        return 1.0 / (1.0 + math.exp(-x))
    prob = [sig(out["beta0"] + out["beta1"] * a) for a in glm["alpha"]]
    res = [y - p for y, p in zip(glm["correct"], prob)]
    w = [p * (1 - p) for p in prob]
    sww = sum(w)
    swa = sum(wi * a for wi, a in zip(w, glm["alpha"]))
    bread = [[sww, swa], [swa, swa]]        # alpha is 0/1 here, so sum(w.a^2) == sum(w.a)
    scores = {}
    for a_, r_, g in zip(glm["alpha"], res, glm["anchor"]):
        s = scores.setdefault(g, [0.0, 0.0])
        s[0] += r_
        s[1] += r_ * a_
    meat = [[0.0, 0.0], [0.0, 0.0]]
    for s0, s1 in scores.values():
        meat[0][0] += s0 * s0
        meat[0][1] += s0 * s1
        meat[1][1] += s1 * s1
    meat[1][0] = meat[0][1]
    det = bread[0][0] * bread[1][1] - bread[0][1] ** 2
    ibread = [[bread[1][1] / det, -bread[0][1] / det], [-bread[0][1] / det, bread[0][0] / det]]
    cov = [[sum(ibread[i][k] * sum(meat[k][j] * ibread[j][l] for j in (0, 1)) for k in (0, 1))
            for l in (0, 1)] for i in (0, 1)]
    g_, n_, k_ = len(scores), len(glm), 2
    cr1 = (g_ / (g_ - 1)) * ((n_ - 1) / (n_ - k_))
    out["anchors"] = g_
    out["beta0_cse"] = math.sqrt(max(cr1 * cov[0][0], 0.0))
    out["beta1_cse"] = math.sqrt(max(cr1 * cov[1][1], 0.0))
    out["z"] = out["beta1"] / out["beta1_cse"]
    out["p"] = 2 * (1 - NormalDist().cdf(abs(out["z"])))
    return out


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _boot_ci(vals: list[float], rng: random.Random, reps: int) -> tuple[float, float]:
    """Percentile CI of the anchor-balanced mean, resampling anchor values with replacement."""
    n, draws = len(vals), []
    for _ in range(reps):
        draws.append(_mean(vals[rng.randrange(n)] for _ in range(n)))
    draws.sort()
    return draws[int(0.025 * reps)], draws[min(int(0.975 * reps), reps - 1)]


def measures(df: pd.DataFrame, model: str, probe: str,
             boot: int = 2000, seed: int = 811) -> dict:
    """The RQ2 measures for one model: eligibility, anchor-balanced Gap, ErrDir,
    DirCompletion, and a bootstrap interval for the GLM's beta1 - all with paired
    anchor-level resampling so matched pairs stay together."""
    glm = glm_frame(df, model, probe)
    und = u_frame(df, model, probe)
    priors = prior_lookup(df, probe)[probe]
    out = {"claims": len(priors), "settled": int(priors.isin(SETTLED).sum())}

    #  Gap: complete (aligned, conflicting) pairs within each claim, averaged within
    #  anchors, then across anchors.
    per_anchor: dict[str, list[float]] = {}
    for (anchor, _), g in glm.groupby(["anchor", "claim_side"]):
        if set(g["alpha"]) == {0, 1}:
            a = float(g.loc[g["alpha"] == 1, "correct"].iloc[0])
            c = float(g.loc[g["alpha"] == 0, "correct"].iloc[0])
            per_anchor.setdefault(anchor, []).append(a - c)
    gap_vals = [_mean(v) for v in per_anchor.values()]
    out["gap"] = _mean(gap_vals)
    out["gap_pairs"] = sum(len(v) for v in per_anchor.values())

    #  ErrDir: among conflict errors, the share repeating the prior - per anchor, then mean.
    errs = glm[(glm["alpha"] == 0) & (glm["correct"] == 0)]
    out["n_err"] = len(errs)
    errdir_vals = [float((g["pred"] == g["prior"]).mean())
                   for _, g in errs.groupby("anchor")]
    out["errdir"] = _mean(errdir_vals)

    #  DirCompletion: among determinations on undetermined rows with a settled prior,
    #  the share going the prior's way - per anchor, then mean.
    det = und[und["pred"].isin(SETTLED)]
    out["n_det"] = len(det)
    dc_vals = [float((g["pred"] == g["prior"]).mean())
               for _, g in det.groupby("anchor")]
    out["dircomp"] = _mean(dc_vals)

    rng = random.Random(seed)
    out["gap_lo"], out["gap_hi"] = _boot_ci(gap_vals, rng, boot)
    out["errdir_lo"], out["errdir_hi"] = _boot_ci(errdir_vals, rng, boot)
    out["dircomp_lo"], out["dircomp_hi"] = _boot_ci(dc_vals, rng, boot)

    #  Bootstrap interval for beta1 as well (replicate = refit on resampled anchors).
    by_anchor = {a: g for a, g in glm.groupby("anchor")}
    names = list(by_anchor)
    reps = []
    for _ in range(boot):
        g = pd.concat(by_anchor[names[rng.randrange(len(names))]]
                      for _ in range(len(names)))
        p0, p1 = (g.loc[g["alpha"] == a, "correct"].mean() for a in (0, 1))
        if 0 < p0 < 1 and 0 < p1 < 1:          # skip degenerate replicates
            reps.append(math.log(p1 / (1 - p1)) - math.log(p0 / (1 - p0)))
    reps.sort()
    out["beta1_lo"], out["beta1_hi"] = reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]
    return out


def table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(coefficient table, the regression frame over every model with a baseline probe)."""
    rows, frames = {}, []
    for model, probe in probe_columns(df).items():
        if not probe:
            continue
        glm = glm_frame(df, model, probe)
        frames.append(glm)
        rows[model] = {"acc_overall": float(glm["correct"].mean()), **fit_binary(glm)}
    t = pd.DataFrame(rows).T.sort_values("acc_overall", ascending=False)
    return t, pd.concat(frames, ignore_index=True)

# -- belief-sign stratified gaps -------------------------------------------------------------

B, SEED, ALPHA = 2000, 0, 0.05
PAPER_ORDER = ["4.1", "luna", "terra", "sonnet-5", "gemini-3.8-flash",
               "deepseek-flash", "opus-5", "gemini-2.5-flash", "fable", "astra"]
GROUPS = ("supported", "contradicted")


def anchor_diffs(dfm, mask):
    """Per-anchor (aligned-conflict) correctness difference over masked rows."""
    d = dfm[mask & dfm.prior.notna()] if callable(mask) else dfm[mask]
    per = d.groupby(["anchor", "alpha"], sort=False)["correct"].mean().unstack("alpha")
    per = per.dropna()
    if len(per) < 2:
        return per.index.values, np.zeros(0)
    return per.index.values, (per[1] - per[0]).to_numpy()


def bootstrap_gaps(dfm, rng):
    """One coupled resample run for both groups and the pooled set.

    Returns dict name -> (point_estimate, replicate array). Anchor resamples
    are shared across groups, so derived contrasts (sym/asym) use the same
    draws.
    """
    anchors_map = {}
    diffs = {}
    valid_anchors = dfm.anchor.unique()
    for gname in GROUPS:
        anchors_map[gname], diffs[gname] = anchor_diffs(dfm, dfm.prior == gname)
    anchors_map["all"], diffs["all"] = anchor_diffs(dfm, dfm.prior.notna())

    idx_of = {a: i for i, a in enumerate(valid_anchors)}
    pos = {g: np.array([idx_of[a] for a in anchors_map[g]])
           for g in anchors_map}

    def point(g):
        return diffs[g].mean() if len(diffs[g]) else np.nan

    reps = {g: np.full(B, np.nan) for g in anchors_map}
    K = len(valid_anchors)
    for i in range(B):
        cnt = np.bincount(rng.integers(0, K, K), minlength=K)
        for g in anchors_map:
            w = cnt[pos[g]]
            if w.sum() > 0:
                reps[g][i] = w @ diffs[g] / w.sum()

    out = {g: (point(g), reps[g]) for g in anchors_map}
    out["memory"] = ((out["supported"][0] + out["contradicted"][0]) / 2,
                     (reps["supported"] + reps["contradicted"]) / 2)
    out["disposition"] = ((out["supported"][0] - out["contradicted"][0]) / 2,
                          (reps["supported"] - reps["contradicted"]) / 2)
    return out


def ci(reps):
    reps = reps[~np.isnan(reps)]
    if len(reps) < B // 2:
        return np.nan, np.nan
    return tuple(np.percentile(reps, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))


def stratified(glm: pd.DataFrame) -> pd.DataFrame:
    """Per model and group (believes-true, believes-false, memory, disposition): estimate and CI."""
    rng = np.random.default_rng(SEED)
    rows = []
    for model in PAPER_ORDER:
        dfm = glm[glm.model == model]
        bs = bootstrap_gaps(dfm, rng)
        for gname in (*GROUPS, "memory", "disposition"):
            est, r = bs[gname]
            lo, hi = ci(r)
            n = int((dfm.prior == gname).sum() // 2) if gname in GROUPS else pd.NA
            rows.append(dict(model=model, prior_group=gname, n_claims=n,
                             est=round(est, 3), lo=round(lo, 3), hi=round(hi, 3)))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RQ2: when do prior beliefs compete with evidence?")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    df = frame()
    t, glm = table(df)
    meas = pd.DataFrame({m: measures(df, m, p) for m, p in probe_columns(df).items() if p}).T
    meas["elig"] = meas["settled"] / meas["claims"]
    meas = meas.reindex(t.index)          # same model order as the GLM table
    strat = stratified(glm)

    args.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out / "rq2_glm.csv")
    meas.to_csv(args.out / "rq2_measures.csv")
    strat.to_csv(args.out / "rq2_stratified.csv", index=False)

    print(f"\n{len(t)} model(s) with a baseline probe · {len(glm)} frame rows\n")
    print(t[["n", "acc_misaligned", "acc_aligned", "beta0", "beta1", "beta1_cse", "z", "p"]]
          .astype(float).round(3).to_string())
    cols = ["elig", "gap", "gap_lo", "gap_hi", "errdir", "errdir_lo", "errdir_hi",
            "dircomp", "dircomp_lo", "dircomp_hi"]
    print("\nPRIOR-CONDITIONED MEASURES (anchor-balanced, 2000 paired bootstrap replicates)\n")
    print(meas[cols].astype(float).round(3).to_string())
    print("\nSTRATIFIED GAPS\n")
    print(strat.pivot(index="model", columns="prior_group", values="est")
          .reindex(PAPER_ORDER)[[*GROUPS, "memory", "disposition"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
