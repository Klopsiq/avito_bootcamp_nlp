"""Validate the exact CSV contract against the original benchmark inventory."""

import argparse
from pathlib import Path
import re
import pandas as pd


def validate(path, data):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    queries = pd.read_parquet(data / "benchmark_queries.parquet", columns=["query_id"])
    items = pd.read_parquet(data / "benchmark_items.parquet", columns=["item_id"])
    if list(frame.columns) != ["query_id", "answer"]:
        raise ValueError("Expected only query_id,answer")
    if (
        not frame.query_id.is_unique
        or set(frame.query_id) != set(queries.query_id)
        or len(frame) != len(queries)
    ):
        raise ValueError("Query inventory mismatch")
    allowed = set(items.item_id)
    for row in frame.itertuples(index=False):
        ids = row.answer.split(" ")
        if (
            len(row.query_id) != 16
            or not 1 <= len(ids) <= 50
            or len(set(ids)) != len(ids)
            or any(not re.fullmatch("[0-9a-f]{16}", i) for i in ids)
            or not set(ids) <= allowed
        ):
            raise ValueError("Invalid answer for " + row.query_id)
    print("Validated:", len(frame), "queries; <=50 unique allowed item IDs per query")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--answer", default="answer.csv")
    p.add_argument("--data-dir", required=True)
    a = p.parse_args()
    validate(Path(a.answer), Path(a.data_dir))
