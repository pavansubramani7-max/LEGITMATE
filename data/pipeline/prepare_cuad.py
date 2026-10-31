"""
data/pipeline/prepare_cuad.py
Repeatable preparation pipeline for LegalBenchRAG-CUAD dataset.

Input:  data/raw/cuad/legalbenchrag_cuad.parquet  (original, never modified)
Output: data/processed/cuad/
          train.jsonl   -- 80% split, (query, positive) pairs
          val.jsonl     -- 10% split
          test.jsonl    -- 10% split
          stats.json    -- row counts and split sizes

Each output line: {"query": "...", "positive": "..."}
Multiple answer spans per row are each emitted as a separate training pair.
Split is deterministic (random_state=42) and stratified by question type
to prevent the same query appearing in both train and val/test.
"""

import json, re, pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = pathlib.Path("data/raw/cuad/legalbenchrag_cuad.parquet")
OUT = pathlib.Path("data/processed/cuad")
OUT.mkdir(parents=True, exist_ok=True)


def parse_spans(s):
    try:
        return json.loads(s)
    except Exception:
        return []


def qtype(q):
    m = re.match(r".*?;\s*(Does|Is|What|Who|When|Can|How|Which|Are)", q)
    return m.group(1) if m else "Other"


df = pd.read_parquet(RAW)

records = []
for _, row in df.iterrows():
    spans = parse_spans(row["expected_output"])
    qt = qtype(row["input"])
    for span in spans:
        ans = span.get("answer", "").strip()
        if ans:
            records.append({
                "id": row["id"],
                "query": row["input"].strip(),
                "positive": ans,
                "qtype": qt,
            })

pairs = pd.DataFrame(records)
print(f"Total (query, positive) pairs: {len(pairs)}")
print(f"Unique queries: {pairs['id'].nunique()}")

# Split at query level to prevent leakage
query_ids = pairs[["id", "qtype"]].drop_duplicates("id")
train_ids, temp_ids = train_test_split(
    query_ids, test_size=0.20, random_state=42, stratify=query_ids["qtype"]
)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.50, random_state=42, stratify=temp_ids["qtype"]
)

train_set = set(train_ids["id"])
val_set   = set(val_ids["id"])
test_set  = set(test_ids["id"])


def write_split(name, id_set):
    subset = pairs[pairs["id"].isin(id_set)][["query", "positive"]]
    path = OUT / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for _, r in subset.iterrows():
            f.write(json.dumps({"query": r["query"], "positive": r["positive"]}, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(subset)} pairs ({len(id_set)} queries) -> {path}")
    return len(subset)


print("\nWriting splits:")
n_train = write_split("train", train_set)
n_val   = write_split("val",   val_set)
n_test  = write_split("test",  test_set)

stats = {
    "source": "adityaraghuvanshi999/legalbenchrag-cuad-retrieval-dataset",
    "license": "CC BY 4.0",
    "upstream": "orgrctera/legalbenchrag_cuad (Hugging Face), lineage to LegalBenchRAG and CUAD",
    "total_pairs": len(pairs),
    "unique_queries": int(pairs["id"].nunique()),
    "train_pairs": n_train,
    "val_pairs": n_val,
    "test_pairs": n_test,
    "split_seed": 42,
    "split_strategy": "query-level stratified by question type (80/10/10)",
}
with open(OUT / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\nStats written to data/processed/cuad/stats.json")
print(json.dumps(stats, indent=2))
