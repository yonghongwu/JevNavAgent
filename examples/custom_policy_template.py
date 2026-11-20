"""Minimal template showing how to plug any custom VLM into JevNavAgent benchmarks.

Run with:
    jevnav-eval --benchmark wsivqa_351 \
        --policy examples/custom_policy_template.py:MyCustomPathologyPolicy \
        --data-dir ./data \
        --output results/custom_policy.jsonl
"""
from __future__ import annotations

from jevnav.eval import BaseWSINavPolicy, NavigationAction, NavigationObservation


class MyCustomPathologyPolicy(BaseWSINavPolicy):
    """Replace the heuristic body below with your own VLM / RL policy forward pass."""

    def __init__(self) -> None:
        # Load your model / weights here once at startup.
        pass

    def reset(self) -> None:
        # Reset any per-episode memory (e.g., KV cache or visited-region state).
        pass

    def act(self, obs: NavigationObservation) -> NavigationAction:
        # obs.overview_image: PIL.Image (slide overview thumbnail)
        # obs.crop_images: list[PIL.Image] (512x512 crops acquired so far)
        # obs.candidates: list[dict] with id, downsample (64/16/4/1), center, bbox_level0, features
        # obs.question / obs.choices / obs.remaining_budget
        if obs.remaining_budget > 0 and obs.candidates:
            # Example: zoom into the highest-magnification candidate with highest tissue fraction
            best_idx = max(
                range(len(obs.candidates)),
                key=lambda i: (
                    float(obs.candidates[i]["features"][7]) if len(obs.candidates[i].get("features", [])) > 7 else 0.0,
                    -int(obs.candidates[i]["downsample"]),
                ),
            )
            return NavigationAction(type="read", candidate_index=best_idx, stop_prob=0.0)

        # When ready to answer (or remaining_budget == 0), return STOP + answer_index
        n = max(1, len(obs.choices))
        return NavigationAction(
            type="stop",
            answer_index=0,
            answer_probs=[1.0 / n] * n,
            stop_prob=1.0,
        )
