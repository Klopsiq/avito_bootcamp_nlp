"""Frozen protocol-v1 keys and text split."""

import hashlib
import json
import re
import unicodedata

CONTEXT = (
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
)
INTEGER_FIELDS = {"search_location_id", "search_is_delivery_search", "search_category"}


def text_group(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).casefold()).strip()


def query_key(row):
    values = [
        int(row[field]) if field in INTEGER_FIELDS else row[field] for field in CONTEXT
    ]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_for_group(group):
    digest = hashlib.sha256(("42:" + group).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    return "train" if bucket < 7000 else "dev" if bucket < 8500 else "holdout"


def pilot_digest(key):
    return hashlib.sha256(("pilot-v1:" + key).encode("utf-8")).hexdigest()
