"""Strict protocol-v1 retrieval evaluator."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from src.protocol import CONTEXT

KS = (10, 50, 100, 200)


def _mean(values):
    return float(sum(values) / len(values)) if values else None


def evaluate(protocol, predictions, query_set, corpus, output_dir, allow_holdout=False):
    protocol, output_dir = Path(protocol), Path(output_dir)
    if corpus not in ("expanded", "benchmark"):
        raise ValueError("corpus must be expanded or benchmark")
    manifest = json.loads((protocol / "manifest.json").read_text())
    if manifest.get("version") != "protocol-v1":
        raise ValueError("unsupported protocol")
    queries = pd.read_parquet(protocol / "queries.parquet")
    selected = pd.read_parquet(query_set)
    if (
        selected.empty
        or not selected.query_key.is_unique
        or not set(selected.query_key).issubset(set(queries.query_key))
    ):
        raise ValueError("unknown or duplicate query-set")
    expected = queries.set_index("query_key").loc[selected.query_key]
    for col in ["text_group", "split", *CONTEXT]:
        if col not in selected or not expected[col].reset_index(drop=True).equals(
            selected[col].reset_index(drop=True)
        ):
            raise ValueError(f"foreign or modified query-set: {col}")
    if not allow_holdout and selected.split.eq("holdout").any():
        raise ValueError("holdout evaluation requires --allow-holdout")
    items = pd.read_parquet(
        protocol / "items.parquet", columns=["item_id", "source_file"]
    )
    corpus_ids = set(
        items.item_id
        if corpus == "expanded"
        else items.loc[items.source_file.eq("benchmark_items"), "item_id"]
    )
    pred = pd.read_parquet(predictions)
    required = ["query_key", "item_id", "rank", "score", "source"]
    if list(pred.columns) != required:
        raise ValueError("predictions columns/order invalid")
    query_ids = set(selected.query_key)
    if pred.query_key.isna().any() or not set(pred.query_key).issubset(query_ids):
        raise ValueError("unknown prediction query")
    if pred.item_id.isna().any() or not set(pred.item_id).issubset(corpus_ids):
        raise ValueError("unknown or out-of-corpus item")
    if pred.source.isna().any() or pred.source.astype(str).str.len().eq(0).any():
        raise ValueError("invalid source")
    if pred.duplicated(["query_key", "item_id"]).any():
        raise ValueError("duplicate predicted item")
    if (
        pred["rank"].dtype.kind not in "iu"
        or pred["rank"].isna().any()
        or pred["rank"].lt(1).any()
    ):
        raise ValueError("invalid rank")
    if pred.score.dtype.kind not in "iuf" or not pred.score.map(math.isfinite).all():
        raise ValueError("invalid score")
    if not pred.sort_values(["query_key", "rank"]).index.equals(pred.index):
        raise ValueError("predictions must be sorted by query_key, rank")
    for _, group in pred.groupby("query_key", sort=False):
        if group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError("ranks must be consecutive")
    qrels = pd.read_parquet(protocol / "qrels.parquet")
    qrels = qrels[qrels.query_key.isin(query_ids)]
    known = qrels.groupby("query_key").item_id.agg(set).to_dict()
    predicted = pred.groupby("query_key", sort=False).item_id.agg(list).to_dict()
    rows = []
    for row in selected.itertuples(index=False):
        key = row.query_key
        positives = known.get(key, set())
        available = positives & corpus_ids
        # The benchmark diagnostic is conditional on at least one positive in that corpus.
        reference = positives if corpus == "expanded" else available
        ranked = predicted.get(key, [])
        record = {
            "query_key": key,
            "split": row.split,
            "text_group": row.text_group,
            "search_category": row.search_category,
            "empty_filters": not bool(str(row.search_infm_params_text).strip()),
            "text_length": len(row.text_group),
            "known_positives": len(positives),
            "available_positives": len(available),
            "evaluated": bool(reference),
            "predicted_count": len(ranked),
            "ceiling_recall_at_50": min(50, len(reference)) / len(reference)
            if reference
            else None,
        }
        for k in KS:
            record[f"recall_at_{k}"] = (
                len(set(ranked[:k]) & reference) / len(reference) if reference else None
            )
        record["hit_at_50"] = (
            float(bool(set(ranked[:50]) & reference)) if reference else None
        )
        rows.append(record)
    result = pd.DataFrame(rows).sort_values("query_key").reset_index(drop=True)
    valid = result[result.evaluated]
    metrics = {
        "protocol_version": "protocol-v1",
        "corpus": corpus,
        "query_count": len(result),
        "evaluated_queries": len(valid),
        "excluded_queries": len(result) - len(valid),
        "known_positives": int(result.known_positives.sum()),
        "available_positives": int(result.available_positives.sum()),
        "query_coverage": len(valid) / len(result),
    }
    for k in KS:
        metrics[f"recall_at_{k}"] = _mean(valid[f"recall_at_{k}"].tolist())
    metrics["hit_at_50"] = _mean(valid.hit_at_50.tolist())
    metrics["ceiling_recall_at_50"] = _mean(valid.ceiling_recall_at_50.tolist())
    grouped = valid.groupby("text_group").recall_at_50.mean()
    metrics["text_balanced_recall_at_50"] = _mean(grouped.tolist())
    result["text_length_slice"] = pd.cut(
        result.text_length,
        [-1, 10, 20, 40, float("inf")],
        labels=["0-10", "11-20", "21-40", "41+"],
    ).astype(str)
    result["positives_slice"] = pd.cut(
        result.known_positives,
        [0, 1, 2, 5, 50, float("inf")],
        labels=["1", "2", "3-5", "6-50", "51+"],
    ).astype(str)
    metrics["slices"] = {}
    for col in (
        "search_category",
        "empty_filters",
        "text_length_slice",
        "positives_slice",
    ):
        metrics["slices"][col] = {}
        for value, part in result.groupby(col, dropna=False):
            active = part[part.evaluated]
            metrics["slices"][col][str(value)] = {
                "queries": len(part),
                "evaluated": len(active),
                "recall_at_50": _mean(active.recall_at_50.tolist()),
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(
        output_dir / "per_query.parquet", index=False, engine="fastparquet"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--query-set", required=True)
    parser.add_argument("--corpus", choices=("expanded", "benchmark"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.protocol,
                args.predictions,
                args.query_set,
                args.corpus,
                args.output_dir,
                args.allow_holdout,
            ),
            ensure_ascii=False,
        )
    )
