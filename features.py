"""The nine lexical/geo features used to fit the submitted ranker."""

import numpy as np
from src.local_retrieval import query_terms
from src.geo_fallback import haversine_km, valid_coordinates

FEATURES = [
    "reciprocal_rank",
    "fusion_score",
    "bm25_score",
    "idf_coverage",
    "exact_location",
    "log_distance_km",
    "has_distance",
    "query_terms",
    "log_location_inventory",
]


class Features:
    def __init__(self, index, items, coordinates, centers):
        self.index = index
        index.matrix.sort_indices()
        self.locations = (
            items.set_index("item_id")
            .reindex(index.item_ids)
            .item_location_id.to_numpy()
        )
        aligned = coordinates.set_index("item_id").reindex(index.item_ids)
        self.lat = aligned.latitude.to_numpy()
        self.lon = aligned.longitude.to_numpy()
        self.centers = {int(k): v for k, v in centers.items()}
        unique, counts = np.unique(self.locations, return_counts=True)
        self.inventory = dict(zip(unique, counts))
        df = np.diff(index.matrix.indptr)
        self.idf = np.log1p((len(index.item_ids) - df + 0.5) / (df + 0.5))

    def __call__(self, row, docs, scores):
        terms = query_terms(row.search_query)
        columns = [
            self.index.vocabulary[t] for t in terms if t in self.index.vocabulary
        ]
        raw = np.zeros(len(docs))
        coverage = np.zeros(len(docs))
        matrix = self.index.matrix
        for col in columns:
            begin, end = matrix.indptr[col : col + 2]
            postings = matrix.indices[begin:end]
            positions = np.searchsorted(postings, docs)
            valid = positions < len(postings)
            hit = np.flatnonzero(valid)
            hit = hit[postings[positions[hit]] == docs[hit]]
            raw[hit] += matrix.data[begin:end][positions[hit]]
            coverage[hit] += self.idf[col]
        coverage /= max(float(self.idf[columns].sum()), 1e-12)
        loc = int(row.search_location_id)
        distance = np.full(len(docs), np.log1p(20000.0))
        has = np.zeros(len(docs))
        center = self.centers.get(loc)
        if center:
            good = valid_coordinates(self.lat[docs], self.lon[docs])
            has[good] = 1
            distance[good] = np.log1p(
                haversine_km(
                    center["latitude"],
                    center["longitude"],
                    self.lat[docs[good]],
                    self.lon[docs[good]],
                )
            )
        return np.column_stack(
            [
                1 / (60 + np.arange(1, len(docs) + 1)),
                scores,
                raw,
                coverage,
                (self.locations[docs] == loc) & (loc > 0),
                distance,
                has,
                np.full(len(docs), len(terms)),
                np.full(len(docs), np.log1p(self.inventory.get(loc, 0))),
            ]
        ).astype(np.float32)
