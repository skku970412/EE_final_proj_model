"""CTC decoding utilities for license plate recognition."""

from __future__ import annotations

import numpy as np


def ctc_greedy_decode(logits: np.ndarray, blank: int = 0) -> list[int]:
    """Collapse a sequence of per-timestep logits (T x C) into label indices."""
    seq = logits.argmax(axis=1).tolist()
    out = []
    prev = None
    for idx in seq:
        if idx != blank and idx != prev:
            out.append(idx)
        prev = idx
    return out


def beam_search_decode(logits: np.ndarray, beam_width: int = 5, blank: int = 0) -> list[int]:
    """Small beam-search decoder for CTC logits. Returns best label index path."""
    time_steps, num_classes = logits.shape
    log_probs = np.log_softmax(logits, axis=1) if hasattr(np, "log_softmax") else np.log(
        np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    )
    beams: dict[tuple[int, ...], float] = {(): 0.0}
    for t in range(time_steps):
        new_beams: dict[tuple[int, ...], float] = {}
        for prefix, score in beams.items():
            for cls in range(num_classes):
                new_score = score + log_probs[t, cls]
                if cls == blank:
                    key = prefix
                else:
                    key = prefix if prefix and prefix[-1] == cls else prefix + (cls,)
                if key not in new_beams or new_score > new_beams[key]:
                    new_beams[key] = new_score
        beams = dict(sorted(new_beams.items(), key=lambda item: item[1], reverse=True)[:beam_width])
    best = max(beams.items(), key=lambda item: item[1])[0] if beams else ()
    return list(best)
