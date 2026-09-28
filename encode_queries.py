"""Local-only encoder worker, isolated from the LightGBM OpenMP runtime."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from src.semantic_fusion import representation_from_manifest, query_input

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["texts", "model", "manifest", "output"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    rep = representation_from_manifest(json.loads(Path(a.manifest).read_text()))
    texts = json.loads(Path(a.texts).read_text())
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModel.from_pretrained(a.model, local_files_only=True).eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tokenizer(
                [query_input(t, rep) for t in texts[start : start + 16]],
                padding=True,
                truncation=True,
                max_length=rep["max_length"],
                return_tensors="pt",
            )
            states = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
            result.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
    np.save(a.output, np.concatenate(result).astype(np.float32), allow_pickle=False)
