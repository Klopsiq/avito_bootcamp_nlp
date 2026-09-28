"""Train-only query-location coordinate priors for sparse geographic fallback."""

import numpy as np
import pandas as pd


def valid_coordinates(latitudes, longitudes):
    latitudes, longitudes = (
        np.asarray(latitudes, dtype=float),
        np.asarray(longitudes, dtype=float),
    )
    return (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & (np.abs(latitudes) <= 90)
        & (np.abs(longitudes) <= 180)
        & ((latitudes != 0) | (longitudes != 0))
    )


def fit_location_centers(queries, qrels, coordinates):
    """Median over unique (train search location, positive item), never dev targets."""
    train = queries.loc[queries.split.eq("train"), ["query_key", "search_location_id"]]
    if not train.query_key.is_unique or not coordinates.item_id.is_unique:
        raise ValueError("query keys and coordinate item IDs must be unique")
    positives = qrels.loc[
        qrels.query_key.isin(train.query_key), ["query_key", "item_id"]
    ]
    observations = positives.merge(train, on="query_key", validate="many_to_one")
    observations = observations.merge(coordinates, on="item_id", validate="many_to_one")
    observations = observations.loc[
        observations.search_location_id.gt(0)
        & valid_coordinates(observations.latitude, observations.longitude)
    ]
    observations = observations.drop_duplicates(["search_location_id", "item_id"])
    grouped = observations.groupby("search_location_id")[
        ["latitude", "longitude"]
    ].median()
    counts = observations.groupby("search_location_id").size()
    return {
        int(location): dict(
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            positive_items=int(counts.loc[location]),
        )
        for location, row in grouped.iterrows()
    }


def haversine_km(latitude, longitude, latitudes, longitudes):
    lat1, lon1, lat2, lon2 = map(
        np.radians, (latitude, longitude, latitudes, longitudes)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class GeoProxy:
    def __init__(
        self, item_ids, coordinates, centers, radius_km=100.0, minimum_pool=200
    ):
        self.item_ids = np.asarray(item_ids)
        if radius_km <= 0 or minimum_pool < 1 or not coordinates.item_id.is_unique:
            raise ValueError("invalid geographic pool settings/coordinate item IDs")
        aligned = coordinates.set_index("item_id").reindex(self.item_ids)
        self.latitudes = aligned.latitude.to_numpy(dtype=float)
        self.longitudes = aligned.longitude.to_numpy(dtype=float)
        self.valid_docs = np.flatnonzero(
            valid_coordinates(self.latitudes, self.longitudes)
        )
        self.centers = {int(key): value for key, value in centers.items()}
        self.radius_km, self.minimum_pool = radius_km, minimum_pool
        self.geo_cache = {}

    def pool(self, location):
        location = int(location) if location is not None else 0
        if location <= 0 or location not in self.centers:
            return np.empty(0, dtype=np.int64)
        if location not in self.geo_cache:
            center = self.centers[location]
            if not valid_coordinates([center["latitude"]], [center["longitude"]])[0]:
                self.geo_cache[location] = np.empty(0, dtype=np.int64)
            else:
                docs = self.valid_docs
                distances = haversine_km(
                    center["latitude"],
                    center["longitude"],
                    self.latitudes[docs],
                    self.longitudes[docs],
                )
                selected = docs[distances <= self.radius_km]
                if len(selected) < self.minimum_pool:
                    count = min(self.minimum_pool, len(docs))
                    nearest = np.lexsort((self.item_ids[docs], distances))[:count]
                    selected = docs[nearest]
                self.geo_cache[location] = selected
        return self.geo_cache[location]


def GeoFallbackRetriever(
    index, items, coordinates, centers, radius_km=100.0, minimum_pool=200
):
    """Lazy sparse adapter; importing GeoProxy requires only numpy/pandas."""
    from src.local_retrieval import LocalGlobalRetriever

    class Retriever(LocalGlobalRetriever):
        def __init__(self):
            super().__init__(index, items)
            self.proxy = GeoProxy(
                index.item_ids, coordinates, centers, radius_km, minimum_pool
            )

        def geography_docs(self, location):
            return self.proxy.pool(location)

        def branches(self, scores, location, global_docs=None, k=200):
            global_docs, local_docs = super().branches(scores, location, global_docs, k)
            # Only missing inventory triggers geographic fallback, not zero lexical hits.
            if (
                location is not None
                and int(location) > 0
                and int(location) not in self.location_docs
            ):
                local_docs = self.top(scores, docs=self.geography_docs(location), k=k)
            return global_docs, local_docs

    return Retriever()
