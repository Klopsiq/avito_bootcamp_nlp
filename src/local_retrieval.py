"""C2: deterministic full-corpus global and exact-location BM25 retrieval."""

import numpy as np

from src.bm25 import tokens
from src.lexical import stem


def query_terms(query):
    return tuple(sorted({stem(token) for token in tokens(query)}))


class LocalGlobalRetriever:
    def __init__(self, index, items):
        self.index = index
        if not items.item_id.is_unique:
            raise ValueError("duplicate metadata item IDs")
        aligned = items.set_index("item_id").reindex(index.item_ids)
        if aligned.item_location_id.isna().any():
            raise ValueError("index item missing location metadata")
        self.locations = aligned.item_location_id.to_numpy()
        self.location_docs = {}
        order = np.argsort(self.locations, kind="stable")
        boundaries = np.flatnonzero(np.diff(self.locations[order])) + 1
        for docs in np.split(order, boundaries):
            self.location_docs[int(self.locations[docs[0]])] = docs

    def scores(self, terms):
        scores = np.zeros(len(self.index.item_ids), dtype=np.float32)
        for term in sorted(set(terms)):
            col = self.index.vocabulary.get(term)
            if col is not None:
                begin, end = self.index.matrix.indptr[col : col + 2]
                scores[self.index.matrix.indices[begin:end]] += self.index.matrix.data[
                    begin:end
                ]
        return scores

    def top(self, scores, docs=None, k=200, fill=False):
        hits = np.flatnonzero(scores > 0) if docs is None else docs[scores[docs] > 0]
        if len(hits) > k:
            boundary = np.partition(scores[hits], -k)[-k]
            hits = hits[scores[hits] >= boundary]
        order = np.lexsort((self.index.item_ids[hits], -scores[hits]))
        chosen = hits[order[:k]]
        if fill and len(chosen) < k:
            extra = self.index.lexical_order[
                ~np.isin(self.index.lexical_order, chosen)
            ][: k - len(chosen)]
            chosen = np.concatenate([chosen, extra])
        return chosen

    def branches(self, scores, location, global_docs=None, k=200):
        global_docs = (
            self.top(scores, k=k, fill=True) if global_docs is None else global_docs
        )
        docs = (
            self.location_docs.get(int(location))
            if location is not None and int(location) > 0
            else None
        )
        local_docs = (
            self.top(scores, docs=docs, k=k)
            if docs is not None
            else np.empty(0, dtype=np.int64)
        )
        return global_docs, local_docs

    def fuse(self, global_docs, local_docs, weight, k=200):
        values = {
            int(doc): 1.0 / (60 + rank) for rank, doc in enumerate(global_docs, 1)
        }
        for rank, doc in enumerate(local_docs, 1):
            values[int(doc)] = values.get(int(doc), 0.0) + weight / (60 + rank)
        ordered = sorted(
            values, key=lambda doc: (-values[doc], self.index.item_ids[doc])
        )[:k]
        return [(str(self.index.item_ids[doc]), values[doc]) for doc in ordered]

    def search(self, query, location, weight=1.0, k=200):
        return self.fuse(
            *self.branches(self.scores(query_terms(query)), location, k=k), weight, k=k
        )
