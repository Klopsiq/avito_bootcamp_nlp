"""Recompute D5 from query features and fitted models; no cached query answers."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from lightgbm import Booster
from threadpoolctl import threadpool_limits
from src.bm25 import BM25Index, tokens
from src.local_retrieval import query_terms
from src.geo_fallback import GeoFallbackRetriever, GeoProxy
from src.geo_transition import TransitionPool, fuse_rankings as lexical_fusion
from src.semantic_fusion import SemanticRetriever, exact_scores, top_docs, fuse_rankings
from src.protocol import query_key
from features import Features


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=40000)
def evidence(title, params):
    return (
        frozenset(tokens(title)),
        frozenset(tokens(params)),
        frozenset(re.findall(r"\d+(?:[.,]\d+)?", title + " " + params)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Original three Parquet files")
    p.add_argument("--assets", default="assets")
    p.add_argument("--output", default="answer.reproduced.csv")
    a = p.parse_args()
    assets = Path(a.assets)
    root = Path(__file__).resolve().parent
    manifest = json.loads((assets / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(assets / name) != digest:
            raise ValueError("Asset hash mismatch: " + name)
    config = json.loads((root / "configs/inference.json").read_text())
    if sha(root / "models/ranker.txt") != manifest["ranker_sha256"]:
        raise ValueError("Ranker hash mismatch")
    queries = pd.read_parquet(Path(a.data_dir) / "benchmark_queries.parquet")
    benchmark = pd.read_parquet(
        Path(a.data_dir) / "benchmark_items.parquet", columns=["item_id"]
    )
    queries["query_key"] = [
        query_key(r._asdict()) for r in queries.itertuples(index=False)
    ]
    # Retrieval uses benchmark IDs. The fitted BM25 weights, IDF and location
    # inventory retain the training+benchmark corpus statistics used in fitting.
    items = pd.read_parquet(assets / "items.parquet")
    index = BM25Index.load(assets / "bm25")
    ids = np.load(assets / "e5/item_ids.npy")
    ev = np.load(assets / "e5/item_vectors.npy", mmap_mode="r")
    tv = np.load(assets / "rubert/item_vectors.npy", mmap_mode="r")
    if not np.array_equal(ids, np.load(assets / "rubert/item_ids.npy")):
        raise ValueError("Encoder ID alignment differs")
    allowed = benchmark.item_id.to_numpy(dtype=ids.dtype)
    mask = np.isin(ids, allowed)
    bids = ids[mask]
    bev = np.asarray(ev[mask])
    btv = np.asarray(tv[mask])
    if len(bids) != len(benchmark):
        raise ValueError("Benchmark inventory differs from fitted assets")
    bm_rows = {str(i): n for n, i in enumerate(index.item_ids)}
    keep = np.isin(index.item_ids, allowed)
    bindex = BM25Index(
        index.matrix[keep].tocsc(),
        index.vocabulary,
        index.item_ids[keep],
        index.k1,
        index.b,
    )
    coords = pd.read_parquet(assets / "coordinates.parquet")
    centers = json.loads((assets / "centers.json").read_text())
    lexical = GeoFallbackRetriever(bindex, items, coords, centers)
    blocations = items.set_index("item_id").reindex(bids).item_location_id.to_numpy()
    transitions = TransitionPool(
        items.set_index("item_id").reindex(bindex.item_ids).item_location_id.to_numpy(),
        json.loads((assets / "transition.json").read_text()),
    )
    semantic = SemanticRetriever(
        bids, blocations, GeoProxy(bids, coords, centers, 100.0, 200)
    )
    base_features = Features(index, items, coords, centers)
    # Compact assets retain the full-corpus IDF and inventory used in fitting.
    if (assets / "feature_idf.npy").exists():
        base_features.idf = np.load(assets / "feature_idf.npy")
        base_features.inventory = {
            int(k): int(v)
            for k, v in json.loads(
                (assets / "feature_inventory.json").read_text()
            ).items()
        }
    meta = (
        pd.read_parquet(assets / "metadata.parquet")
        .set_index("item_id")
        .reindex(index.item_ids)
    )
    titles = meta.item_title_raw.fillna("").astype(str).to_numpy()
    params = meta.item_infm_params_text.fillna("").astype(str).to_numpy()
    rating = meta.item_rating.to_numpy(dtype=float)
    reviews = meta.item_rating_reviews_count.to_numpy(dtype=float)
    ranker = Booster(model_file=str(root / "models/ranker.txt"))
    # Match encoder batch padding and ordering from the original run.
    texts = list(
        dict.fromkeys(queries.sort_values("query_key").search_query.astype(str))
    )
    ti = {t: i for i, t in enumerate(texts)}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
        for name in ["e5", "rubert"]:
            subprocess.run(
                [
                    sys.executable,
                    str(root / "encode_queries.py"),
                    "--texts",
                    str(tmp / "texts.json"),
                    "--model",
                    str(assets / (name + "_model")),
                    "--manifest",
                    str(assets / name / "vectors_manifest.json"),
                    "--output",
                    str(tmp / (name + ".npy")),
                ],
                check=True,
            )
        eq = np.load(tmp / "e5.npy")
        tq = np.load(tmp / "rubert.npy")

    def features(row, docs, values, lr, sr, cosine):
        bm = np.asarray([bm_rows[str(bids[d])] for d in docs])
        base = base_features(row, bm, values)
        base[:, 0] = np.maximum(lr, sr)
        words = set(tokens(row.search_query))
        filters = set(tokens(row.search_infm_params_text))
        numbers = set(
            re.findall(
                r"\d+(?:[.,]\d+)?",
                str(row.search_query) + " " + str(row.search_infm_params_text),
            )
        )
        proofs = [evidence(titles[d], params[d]) for d in bm]
        title_cov = [len(words & t) / max(1, len(words)) for t, _, _ in proofs]
        param_cov = [len(filters & q) / max(1, len(filters)) for _, q, _ in proofs]
        number_cov = [len(numbers & n) / max(1, len(numbers)) for _, _, n in proofs]
        ratings = np.nan_to_num(rating[bm], nan=0)
        counts = np.log1p(np.maximum(np.nan_to_num(reviews[bm], nan=0), 0))
        requested = any("рейтинг" in t for t in filters)
        extra = np.column_stack(
            [
                cosine,
                lr,
                sr,
                title_cov,
                param_cov,
                number_cov,
                counts,
                ratings,
                requested * (ratings >= 4),
                cosine * base[:, 4],
                cosine * base[:, 3],
            ]
        )
        return np.column_stack([base, extra]).astype(np.float32)

    answers = {}
    grouped = {i: [] for i in range(len(texts))}
    for row in queries.itertuples(index=False):
        grouped[ti[str(row.search_query)]].append(row)
    with threadpool_limits(limits=4):
        for start in range(0, len(texts), 32):
            es = exact_scores(eq[start : start + 32], bev)
            ts = exact_scores(tq[start : start + 32], btv)
            for offset in range(len(es)):
                for row in grouped[start + offset]:
                    loc = int(row.search_location_id)
                    scores = lexical.scores(query_terms(row.search_query))
                    global_docs, geo_docs = lexical.branches(scores, loc, None, 200)
                    transition_docs = (
                        lexical.top(scores, docs=transitions.pool(loc), k=200)
                        if loc > 0 and loc not in lexical.location_docs
                        else np.empty(0, dtype=np.int64)
                    )
                    lex = lexical_fusion(
                        bindex.item_ids,
                        global_docs,
                        geo_docs,
                        transition_docs,
                        config["lexical_transition_weight"],
                        200,
                    )
                    ld = np.searchsorted(
                        bids, np.asarray([i for i, _ in lex], dtype=bids.dtype)
                    )
                    ed, _ = semantic.ranking(
                        es[offset], loc, top_docs(es[offset], bids, k=200)
                    )
                    td, _ = semantic.ranking(
                        ts[offset], loc, top_docs(ts[offset], bids, k=200)
                    )
                    d2, _ = fuse_rankings(
                        [ld, ed], [1, config["e5_weight"]], bids, k=200
                    )
                    docs, values = fuse_rankings([ld, ed], [1, 1], bids, k=400)
                    lr = {int(d): 1 / (60 + r) for r, d in enumerate(ld, 1)}
                    sr = {int(d): 1 / (60 + r) for r, d in enumerate(ed, 1)}
                    fullrows = np.searchsorted(ids, bids[docs])
                    vv = np.asarray(ev[fullrows], dtype=np.float32)
                    vv /= np.maximum(np.linalg.norm(vv, axis=1, keepdims=True), 1e-12)
                    cosine = vv @ eq[start + offset]
                    scores = ranker.predict(
                        features(
                            row,
                            docs,
                            values,
                            [lr.get(int(d), 0) for d in docs],
                            [sr.get(int(d), 0) for d in docs],
                            cosine,
                        ),
                        num_threads=2,
                    )
                    learned = docs[np.lexsort((bids[docs], -scores))]
                    r2, _ = fuse_rankings(
                        [d2, learned], [1, config["ranker_weight"]], bids, k=200
                    )
                    final, _ = fuse_rankings(
                        [r2, td], [1, config["trained_rubert_weight"]], bids, k=200
                    )
                    answers[row.query_key] = " ".join(str(bids[d]) for d in final[:50])
            print(
                "queries encoded/retrieved:",
                min(start + 32, len(texts)),
                "/",
                len(texts),
                flush=True,
            )
    result = queries[["query_id", "query_key"]].copy()
    result["answer"] = result.query_key.map(answers)
    result[["query_id", "answer"]].to_csv(a.output, index=False)
    from validate_answer import validate

    validate(Path(a.output), Path(a.data_dir))
    digest = sha(a.output)
    print("SHA256:", digest)
    if digest != config["answer_sha256"]:
        raise ValueError(
            "Output differs from the submitted CSV; inspect environment and input data"
        )
    print("Exact submitted answer reproduced.")


if __name__ == "__main__":
    main()
