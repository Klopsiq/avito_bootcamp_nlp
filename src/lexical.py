"""Fixed lexical variants of the B0 BM25 baseline."""

from functools import lru_cache

import numpy as np
import snowballstemmer

from src.bm25 import document_text, tokens


_russian = snowballstemmer.stemmer("russian")


@lru_cache(maxsize=250_000)
def stem(token):
    # Snowball's Russian algorithm leaves Latin words and numbers unchanged.
    return (
        _russian.stemWord(token)
        if any("а" <= c <= "я" or c == "ё" for c in token)
        else token
    )


def stem_text(text):
    return " ".join(stem(token) for token in tokens(text))


def variant_text(variant, title, params, description):
    if variant == "L1":
        return document_text(title, params, "")
    if variant == "L2":
        return stem_text(document_text(title, params, description))
    raise ValueError(variant)


def variant_query(variant, query):
    return stem_text(query) if variant == "L2" else query


def grouped_bootstrap(differences, groups, repeats=1000, seed=42):
    """Paired query metric difference, resampling normalized text groups."""
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    values = np.asarray(differences, dtype=np.float64)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for repeat in range(repeats):
        chosen = rng.integers(0, len(unique), size=len(unique))
        samples[repeat] = sums[chosen].sum() / counts[chosen].sum()
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "text_groups": len(unique),
        "queries": len(values),
        "repeats": repeats,
        "seed": seed,
    }
