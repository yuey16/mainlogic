# MainLogic: Evidence Over Memory

MainLogic is a claim-verification benchmark for long-context language models. Each item pairs a
one-sentence claim taken from a scientific paper with the full text of the works it cites, and asks
whether the documents **support**, **contradict**, or leave **undetermined** the claim. Every
natural anchor is expanded into a matched six-cell core: the original documents, a counterfactual
rewrite that reverses the evidence, and a version with the evidence removed, each paired with the
claim and a contrasting claim. A model that answers from memory rather than from the documents
fails the counterfactual and removed-evidence cells.

Everything is run from the scripts under `data/` (construction and evaluation runs) and `eval/`
(the paper's analyses). There is no package to install and no other entry point.

## Layout

```
data/            construction pipeline, released dataset, model runs  (see data/README.md)
  final/mainlogic.jsonl   the benchmark: 812 rows, 108 six-cell cores, 27 citing papers
  results/                every model run the paper reports, one directory per model and condition
  audit/                  the blinded human audit, one file per anonymised reviewer
eval/            one script per research question and ablation; outputs in eval/out/
ingest.sh        fetches papers into data/clean/ (runs the acquisition scripts in data/)
```

The supplementary archive leaves out the full text of cited works that no claim uses (only needed to
screen new sentences); `./ingest.sh` fetches them again.

## Setup

```bash
uv sync
```

Model calls need `OPENAI_API_KEY` and/or `OPENROUTER_API_KEY` in a `.env` file at the repository root.
Reproducing the paper's tables and figures from the committed runs needs no keys.

## Reproduce the paper's results

The dataset and every model run are included, so the analyses run offline. `data/export.py` rebuilds
`data/final/mainlogic.jsonl` from the construction records; it is deterministic and reproduces the
released file byte for byte:

```bash
uv run python data/export.py              # -> data/final/mainlogic.jsonl (812 rows, no model calls)
uv run python eval/rq1.py                 # RQ1: six-cell core, grouped correctness, decisive vs redundant removal
uv run python eval/rq2.py                 # RQ2: prior-conditioned measures, GLM, stratified gaps
uv run python eval/rq3.py                 # RQ3: counterfactual row under native / highlighted / oracle
uv run python eval/ablation_horizon.py    # appendix: horizon ablations
uv run python eval/ablation_default.py    # appendix: default behaviour study
uv run python eval/ablation_prompt.py     # appendix: prompt ablation
```

## Evaluate a model

```bash
uv run python data/run.py --model <alias>                          # native six-cell core
uv run python data/run.py --model <alias> --ablation highlighted   # evidence marked in place
uv run python data/run.py --model <alias> --ablation oracle        # evidence alone
uv run python data/run.py --model <alias> --baseline               # evidence-free prior probe
uv run python data/run.py --model <alias> --strict                 # prompt ablation (also --prompt loose)
```

Runs land in `data/results/` and are picked up by `eval/` automatically.

## Rebuild the dataset

Each stage is a script in `data/`; `data/README.md` documents every stage, its prompts, and the row
schema.

```bash
./ingest.sh                                   # 1. acquisition: sources, citations, cited full text
uv run python data/filter.py --write          # 2. screen citing sentences into data/claims.yaml
                                              #    (then read claims.yaml and mark bad claims drop: true)
uv run python data/augment.py --workers 3     # 3. six-cell augmentation -> data/built/
uv run python data/preserve.py --write        # 4. the P= redundant-removal control -> data/preserved/
uv run python data/export.py                  # 5. assemble the triaged cores -> data/final/mainlogic.jsonl
uv run python data/human_audit.py --reviewer reviewer-a --n 10   # 6. blinded audit -> data/audit/
uv run python data/export.py                  #    re-export: flags every row with its core's human_audit
```

Construction calls are cached by prompt and model, so stages re-run cheaply.
