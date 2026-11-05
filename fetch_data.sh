#!/usr/bin/env bash
# Fetch the data the tests read but this repository does not include, at the versions used, and check every file
# against SHA256SUMS. Needs git, curl and sha256sum.
#   - Section 1: the NoLiMa haystack books (their licences are listed in data/haystack/LICENSES.md), and the two
#     joined haystacks built from them.
#   - Sections 2-3: three benchmark repositories pinned to the commits used, and the IMDB test split
#     (Maas et al., ACL 2011).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

H="$ROOT/data/haystack"
HF=https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack
mkdir -p "$H/rand_shuffle" "$H/rand_shuffle_long" "$H/joined_long"
curl -sSfL -o "$H/rand_shuffle/rand_book_1.txt" "$HF/rand_shuffle/rand_book_1.txt"
for i in 1 2 3 4; do curl -sSfL -o "$H/rand_shuffle_long/rand_book_$i.txt" "$HF/rand_shuffle_long/rand_book_$i.txt"; done
cat "$H/rand_shuffle_long/rand_book_1.txt" "$H/rand_shuffle_long/rand_book_2.txt" > "$H/joined_long/rand_book_1+2.txt"
cat "$H/rand_shuffle_long/rand_book_"{1,2,3,4}.txt > "$H/joined_long/rand_book_1-4.txt"

E="$ROOT/data/external"
mkdir -p "$E"
fetch() {  # url dir commit
  [ -d "$E/$2/.git" ] || git clone -q "$1" "$E/$2"
  git -C "$E/$2" checkout -q "$3"
}
fetch https://github.com/Y0oMu/LLM-Judge-Bias-Dataset LLM-Judge-Bias-Dataset f6248e9f9a93c20c2059bbb46ce3096ca414ad19
fetch https://github.com/infobip/llm-judge-bias llm-judge-bias 6b5b2efc7c34102763e36b9561b9e73468651540
fetch https://github.com/cjj826/LongJudgeBench LongJudgeBench 18b172116184e7b31b524e764077a605a32dd17e
mkdir -p "$E/imdb"
[ -f "$E/imdb/test.parquet" ] || curl -sSfL -o "$E/imdb/test.parquet" \
  https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet

cd "$ROOT" && sha256sum -c --quiet SHA256SUMS && echo "all hashes match"
