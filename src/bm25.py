"""Exact sparse BM25 for the fixed B0 text representation."""

import hashlib
import json
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


TOKEN_RE = re.compile(r"(?u)\b\w+\b")


def normalize(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ""
    return unicodedata.normalize("NFC", str(text)).casefold()


def document_text(title, params, description):
    def value(text):
        return (
            ""
            if text is None or (isinstance(text, float) and np.isnan(text))
            else str(text)
        )

    return normalize(
        f"{value(title)} {value(params)[:512]} {value(description)[:1024]}"
    )


def tokens(text):
    return TOKEN_RE.findall(normalize(text))


class BM25Index:
    def __init__(self, matrix, vocabulary, item_ids, k1=1.2, b=0.75):
        self.matrix = matrix
        self.vocabulary = vocabulary
        self.item_ids = np.asarray(item_ids, dtype="U16")
        self.k1 = k1
        self.b = b
        self.lexical_order = np.argsort(self.item_ids)

    @classmethod
    def build(cls, item_ids, texts, k1=1.2, b=0.75):
        vectorizer = CountVectorizer(
            tokenizer=tokens, token_pattern=None, lowercase=False, dtype=np.int32
        )
        counts = vectorizer.fit_transform(texts)
        n = counts.shape[0]
        if n != len(item_ids):
            raise ValueError("item_ids and texts length mismatch")
        lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(lengths.mean())
        if avgdl <= 0:
            raise ValueError("all documents have zero tokens")
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        row_length = np.repeat(lengths, np.diff(counts.indptr))
        counts = counts.astype(np.float32)
        tf = counts.data
        denominator = tf + k1 * (1 - b + b * row_length / avgdl)
        counts.data = (idf[counts.indices] * tf * (k1 + 1) / denominator).astype(
            np.float32
        )
        return cls(counts.tocsc(), vectorizer.vocabulary_, item_ids, k1, b)

    def search(self, query, top_k=200):
        scores = np.zeros(len(self.item_ids), dtype=np.float32)
        for term in set(tokens(query)):
            col = self.vocabulary.get(term)
            if col is None:
                continue
            begin, end = self.matrix.indptr[col : col + 2]
            scores[self.matrix.indices[begin:end]] += self.matrix.data[begin:end]
        hits = np.flatnonzero(scores > 0)
        if len(hits) > top_k:
            # Partition keeps the tied boundary; deterministic item-ID sort follows.
            boundary = np.partition(scores[hits], -top_k)[-top_k]
            hits = hits[scores[hits] >= boundary]
        order = np.lexsort((self.item_ids[hits], -scores[hits]))
        chosen = list(hits[order[:top_k]])
        if len(chosen) < top_k:
            selected = set(chosen)
            for doc in self.lexical_order:
                if int(doc) not in selected:
                    chosen.append(int(doc))
                    if len(chosen) == top_k:
                        break
        return [(str(self.item_ids[i]), float(scores[i])) for i in chosen]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "weights.npz", self.matrix, compressed=False)
        np.save(directory / "item_ids.npy", self.item_ids)
        with open(directory / "vocabulary.pkl", "wb") as stream:
            pickle.dump(self.vocabulary, stream, protocol=pickle.HIGHEST_PROTOCOL)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "documents": len(self.item_ids),
                    "terms": len(self.vocabulary),
                    "nonzero": int(self.matrix.nnz),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        with open(directory / "vocabulary.pkl", "rb") as stream:
            vocabulary = pickle.load(stream)
        return cls(
            sparse.load_npz(directory / "weights.npz").tocsc(),
            vocabulary,
            np.load(directory / "item_ids.npy"),
            metadata["k1"],
            metadata["b"],
        )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
