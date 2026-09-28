"""Categorical train-only query-location to item-location retrieval prior."""

import numpy as np


def fit_transition_prior(queries, qrels, items):
    train = queries.loc[queries.split.eq("train"), ["query_key", "search_location_id"]]
    if not train.query_key.is_unique or not items.item_id.is_unique:
        raise ValueError("train query keys and item IDs must be unique")
    links = qrels.loc[qrels.query_key.isin(train.query_key), ["query_key", "item_id"]]
    links = links.merge(train, on="query_key", validate="many_to_one").merge(
        items[["item_id", "item_location_id"]], on="item_id", validate="many_to_one"
    )
    links = links.loc[links.search_location_id.gt(0) & links.item_location_id.gt(0)]
    links = links.drop_duplicates(["query_key", "item_location_id"])
    counts = links.groupby(["search_location_id", "item_location_id"]).size()
    contexts = links.groupby("search_location_id").query_key.nunique()
    prior = {}
    for (source, target), count in counts.items():
        entry = prior.setdefault(
            int(source), dict(contexts=int(contexts.loc[source]), counts={}, total=0)
        )
        entry["counts"][int(target)] = int(count)
        entry["total"] += int(count)
    return prior


class TransitionPool:
    def __init__(
        self, item_locations, prior, min_contexts=5, max_locations=8, mass=0.9
    ):
        if min_contexts < 1 or max_locations < 1 or not 0 < mass <= 1:
            raise ValueError("invalid prior settings")
        locations = np.asarray(item_locations)
        self.location_docs = {}
        order = np.argsort(locations, kind="stable")
        for docs in np.split(order, np.flatnonzero(np.diff(locations[order])) + 1):
            if len(docs) and int(locations[docs[0]]) > 0:
                self.location_docs[int(locations[docs[0]])] = docs
        self.prior = {int(key): value for key, value in prior.items()}
        self.min_contexts, self.max_locations, self.mass = (
            min_contexts,
            max_locations,
            mass,
        )
        self.cache = {}
        self.selected_locations = {}

    def pool(self, query_location):
        location = int(query_location) if query_location is not None else 0
        if location in self.cache:
            return self.cache[location]
        entry = self.prior.get(location, {})
        chosen = []
        if location > 0 and entry.get("contexts", 0) >= self.min_contexts:
            ranked = sorted(
                (
                    (int(target), int(count))
                    for target, count in entry["counts"].items()
                    if int(target) in self.location_docs
                ),
                key=lambda row: (-row[1], row[0]),
            )
            covered = 0
            for target, count in ranked[: self.max_locations]:
                chosen.append(target)
                covered += count
                if covered >= self.mass * entry["total"]:
                    break
        self.selected_locations[location] = chosen
        docs = (
            np.sort(np.concatenate([self.location_docs[target] for target in chosen]))
            if chosen
            else np.empty(0, dtype=np.int64)
        )
        self.cache[location] = docs
        return docs


def fuse_rankings(
    item_ids, global_docs, geographic_docs, transition_docs, prior_weight, k=200
):
    scores = {}
    for docs, weight in [
        (global_docs, 1.0),
        (geographic_docs, 2.0),
        (transition_docs, prior_weight),
    ]:
        for rank, doc in enumerate(docs, 1):
            scores[int(doc)] = scores.get(int(doc), 0.0) + weight / (60 + rank)
    ordered = sorted(scores, key=lambda doc: (-scores[doc], item_ids[doc]))[:k]
    return [(str(item_ids[doc]), scores[doc]) for doc in ordered]
