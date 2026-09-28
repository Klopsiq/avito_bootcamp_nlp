"""Exact cosine global/local retrieval and deterministic rank fusion."""

import numpy as np


POOLING = "attention-masked mean float32 + L2"


def representation_from_manifest(manifest):
    version = manifest.get("version")
    if manifest.get("pooling") != POOLING or manifest.get("dtype") != "float16":
        raise ValueError("mean-float32-L2 float16 item representation required")
    if version == "merged-frozen-e5-v1":
        dimensions, max_length, prefix = 384, 128, "query: "
    elif version == "merged-trained-rubert-v1":
        dimensions, max_length, prefix = 312, 64, None
        transform = manifest.get("transformation", {})
        if not manifest.get("complete") or manifest.get("no_training") is not False:
            raise ValueError("complete trained RuBERT manifest required")
        if (
            transform.get("query") != "remove exactly one leading query: "
            or transform.get("passage") != "remove exactly one leading passage: "
            or transform.get("query_prefix", "missing") is not None
            or transform.get("passage_prefix", "missing") is not None
        ):
            raise ValueError("trained RuBERT prefix transformation mismatch")
    else:
        raise ValueError("unsupported merged vector representation")
    if (
        manifest.get("dimensions") != dimensions
        or manifest.get("max_length") != max_length
    ):
        raise ValueError("manifest dimension/max_length inconsistent with backbone")
    return {
        "dimensions": dimensions,
        "max_length": max_length,
        "query_prefix": prefix,
        "pooling": POOLING,
        "version": version,
    }


def query_input(text, representation):
    prefix = representation["query_prefix"]
    return (prefix or "") + str(text)


def validate_alignment(item_ids, vectors, metadata, dimensions=None):
    ids = np.asarray(item_ids)
    if (
        vectors.ndim != 2
        or len(ids) != len(vectors)
        or vectors.shape[1] < 1
        or (dimensions is not None and vectors.shape[1] != dimensions)
    ):
        raise ValueError("item vector shape or ID count mismatch")
    if len(ids) != len(np.unique(ids)) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("item IDs must be unique and lexically sorted")
    if not metadata.item_id.is_unique or set(ids) != set(metadata.item_id):
        raise ValueError("vector IDs must exactly match selected protocol corpus")
    aligned = metadata.set_index("item_id").reindex(ids)
    if aligned.item_location_id.isna().any():
        raise ValueError("missing item location")
    return aligned.item_location_id.to_numpy()


def exact_scores(
    query_vectors, item_vectors, item_block=32768, max_bytes=256 * 1024**2
):
    """Float32 cosine matmul; bounded temporary scores + converted item block."""
    queries = np.asarray(query_vectors, dtype=np.float32)
    queries = queries / np.maximum(
        np.linalg.norm(queries, axis=1, keepdims=True), 1e-12
    )
    n, dim = item_vectors.shape
    required = (
        len(queries) * n * 5
        + min(n, item_block) * dim * 8
        + len(queries) * min(n, item_block) * 4
    )
    if required > max_bytes:
        raise ValueError("score batch exceeds 256MB temporary memory budget")
    scores = np.empty((len(queries), n), dtype=np.float32)
    for start in range(0, n, item_block):
        vectors = np.asarray(item_vectors[start : start + item_block], dtype=np.float32)
        vectors = vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
        )
        scores[:, start : start + len(vectors)] = queries @ vectors.T
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite cosine scores")
    return scores


def top_docs(scores, item_ids, docs=None, k=200):
    hits = np.arange(len(scores)) if docs is None else np.asarray(docs)
    if len(hits) > k:
        boundary = np.partition(scores[hits], -k)[-k]
        hits = hits[scores[hits] >= boundary]
    return hits[np.lexsort((item_ids[hits], -scores[hits]))[:k]]


def fuse_rankings(rankings, weights, item_ids, k=200):
    values = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        for rank, doc in enumerate(ranking, 1):
            values[int(doc)] = values.get(int(doc), 0.0) + weight / (60 + rank)
    chosen = sorted(values, key=lambda doc: (-values[doc], item_ids[doc]))[:k]
    return np.asarray(chosen, dtype=np.int64), np.asarray(
        [values[doc] for doc in chosen], dtype=np.float64
    )


class SemanticRetriever:
    def __init__(self, item_ids, locations, geo_proxy=None):
        self.item_ids = np.asarray(item_ids)
        self.geo_proxy = geo_proxy
        self.location_docs = {}
        order = np.argsort(locations, kind="stable")
        boundaries = np.flatnonzero(np.diff(np.asarray(locations)[order])) + 1
        for docs in np.split(order, boundaries):
            self.location_docs[int(locations[docs[0]])] = docs

    def ranking(self, scores, location, global_docs=None, k=200):
        global_docs = (
            top_docs(scores, self.item_ids, k=k) if global_docs is None else global_docs
        )
        docs = (
            self.location_docs.get(int(location))
            if location is not None and int(location) > 0
            else None
        )
        if docs is None and self.geo_proxy is not None:
            docs = self.geo_proxy.pool(location)
        local = (
            top_docs(scores, self.item_ids, docs=docs, k=k) if docs is not None else []
        )
        return fuse_rankings([global_docs, local], [1.0, 2.0], self.item_ids, k=k)
