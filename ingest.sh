#!/usr/bin/env bash
# Fetch papers so their lines can be used in data/claims.yaml. No model calls.
#
#   ./ingest.sh 2205.14135v2 1706.03762v7    # these papers (added to data/papers.txt)
#   ./ingest.sh                              # every paper in data/papers.txt
#
# clean     arXiv source -> data/clean/arXiv-<id>/main.tex + bibitems.json
# resolve   \cite keys -> Semantic Scholar -> refs.json
# fulltext  cited works' full text -> references/*.md + fulltext.json
# Every stage skips work already done, so re-running is cheap.

set -euo pipefail
cd "$(dirname "$0")"

IDS=data/papers.txt
if [ $# -gt 0 ]; then
    for id in "$@"; do
        grep -q "'$id'" "$IDS" || uv run python - "$id" <<'PY'
import ast, sys
path = "data/papers.txt"
ids = [i.strip() for i in ast.literal_eval(open(path).read())]
ids.append(sys.argv[1])
open(path, "w").write("[" + ", ".join(repr(i) for i in ids) + "]\n")
PY
    done
    IDS=$(mktemp)
    printf '%s\n' "$@" > "$IDS"
fi

failed=0
for stage in clean resolve fulltext; do
    printf '\n\033[1m== %s\033[0m\n' "$stage"
    uv run python "data/$stage.py" "$IDS" || { failed=1; printf '\033[33m!! %s reported failures; continuing\033[0m\n' "$stage"; }
done
printf '\nnext: uv run python data/filter.py\n'
exit $failed
