"""Train-only shared-encoder contrastive primitives; no validation labels are read."""

import torch
import torch.nn.functional as F


def pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=1)


def positive_mask(known_positives, candidate_ids, device=None):
    return torch.tensor(
        [
            [item in positives for item in candidate_ids]
            for positives in known_positives
        ],
        dtype=torch.bool,
        device=device,
    )


def exact_location_mask(query_locations, passage_locations, device=None):
    """Match categorical location IDs; missing/nonpositive IDs never match."""
    import math

    def known_id(value):
        if value is None or isinstance(value, bool):
            return -1
        try:
            number = int(value)
            if (
                number <= 0
                or (not isinstance(value, str) and not math.isfinite(value))
                or number != float(value)
            ):
                return -1
            return number
        except (ValueError, TypeError, OverflowError):
            return -1

    queries = torch.tensor(
        [known_id(x) for x in query_locations], dtype=torch.long, device=device
    )
    passages = torch.tensor(
        [known_id(x) for x in passage_locations], dtype=torch.long, device=device
    )
    return (
        (queries[:, None] == passages[None, :])
        & (queries[:, None] > 0)
        & (passages[None, :] > 0)
    )


class GeoBias(torch.nn.Module):
    """A learned additive exact-location preference in cosine-score units."""

    def __init__(self, initial=0.0):
        super().__init__()
        self.bias = torch.nn.Parameter(
            torch.tensor(float(initial), dtype=torch.float32)
        )

    def forward(self, scores, location_matches):
        if location_matches.shape != scores.shape:
            raise ValueError("location mask must match score shape")
        return scores + self.bias * location_matches.to(scores.dtype)


def contrastive_loss(
    query_vectors,
    passage_vectors,
    positives,
    temperature=0.05,
    geo_bias=None,
    location_matches=None,
):
    """Marginal likelihood of any known positive, never a false-negative target.

    Candidates known positive for a query belong to its numerator as well as its
    denominator. Gradient accumulation changes optimizer batch size only: each
    microbatch has its own candidate matrix and in-batch negatives.
    """
    if temperature <= 0 or positives.shape != (
        len(query_vectors),
        len(passage_vectors),
    ):
        raise ValueError("invalid temperature or positive mask")
    if not positives.any(1).all():
        raise ValueError("every query must have at least one positive candidate")
    scores = query_vectors.float() @ passage_vectors.float().T
    if geo_bias is not None:
        if location_matches is None:
            raise ValueError("geo bias requires a location match mask")
        scores = geo_bias(scores, location_matches)
    logits = scores / temperature
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    ).mean()


def candidates_for_batch(records, positive_limit, step, negative_limit=None):
    """Rotate sampled positives, retaining all known positives for label masks."""
    ids = []
    if negative_limit is not None and negative_limit < 0:
        raise ValueError("negative_limit must be nonnegative")
    for record in records:
        positives = record["positive_ids"]
        offset = step % len(positives)
        ids.extend(
            (positives + positives)[
                offset : offset + min(positive_limit, len(positives))
            ]
        )
        negatives = record.get("negative_ids")
        if negatives is None:
            negatives = [record["negative_id"]]
        # A malformed mined negative must never displace a known positive.
        negatives = list(
            dict.fromkeys(item for item in negatives if item not in set(positives))
        )
        ids.extend(negatives if negative_limit is None else negatives[:negative_limit])
    return list(dict.fromkeys(ids))


def linear_schedule(optimizer, total_steps, warmup_steps=0):
    """Serializable linear warmup/decay measured in optimizer updates."""
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "warmup_steps must be nonnegative and smaller than total_steps"
        )

    def multiplier(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def prune_checkpoints(run_dir, keep):
    """Only remove old numbered checkpoints inside this explicitly selected run."""
    from pathlib import Path
    import re
    import shutil

    if keep <= 0:
        return
    checkpoints = []
    for path in Path(run_dir).iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir() and not path.is_symlink():
            checkpoints.append((int(match.group(1)), path))
    for _, path in sorted(checkpoints)[:-keep]:
        shutil.rmtree(path)
