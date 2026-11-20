"""Annotation-blind navigation controls sharing a frozen observed-image answerer."""
from __future__ import annotations

import math
import random


class ForcedAnswerer:
    """Force an answer from a shared trained policy without consulting its type head."""
    def __init__(self, policy):
        self.policy, self.last_metrics = policy, {}

    def act(self, state):
        answer = self.policy.answer(state)
        self.last_metrics = dict(getattr(self.policy, "last_metrics", {}))
        return {"type": "stop", "answer_index": answer}


class NavigationBaseline:
    """Read the common candidate set, then answer once with the frozen answerer.

    ``coverage`` visits roots in raster order, refining each center 64→16→4→1
    before moving to the next root. Thus budgets 4 and 8 include genuinely
    high-resolution reads rather than making qualified CAM success impossible.
    ``tissue``
    favors thumbnail tissue and refinement; neither inspects reference labels.
    ``thumbnail`` answers from the overview without issuing a read.
    """

    def __init__(self, kind, answerer, seed=17):
        if kind not in {"random", "coverage", "tissue", "thumbnail", "question-only", "overview-only"}:
            raise ValueError(f"Unknown baseline {kind}")
        self.kind, self.answerer = kind, answerer
        self.reset(seed)

    def reset(self, seed=17):
        self.rng = random.Random(seed)
        self.last_metrics = {}

    @staticmethod
    def _tissue(candidate):
        features = candidate.get("features", [])
        value = candidate.get("tissue_fraction", features[7] if len(features) >= 12 else 0.)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.

    def act(self, state):
        candidates = state.get("candidates", [])
        if self.kind in {"thumbnail", "question-only", "overview-only"} or state.get("remaining_budget", 0) <= 0 or not candidates:
            answer_state = state
            if self.kind == "question-only":
                from PIL import Image
                answer_state = dict(state, images=[Image.new("RGB", (512, 512), "white")],
                                    views=[], dimensions=[512, 512], mpp=[None, None],
                                    remaining_budget=0, candidates=[], decisions_remaining=1)
                answer_state.pop("overview", None)
            elif self.kind in {"thumbnail", "overview-only"}:
                overview = state.get("overview")
                images = [overview] if overview is not None else list(state.get("images", []))[:1]
                answer_state = dict(state, images=images, views=[], candidates=[], remaining_budget=0)
                answer_state.pop("overview", None)
            action = self.answerer.act(answer_state)
            self.last_metrics = dict(getattr(self.answerer, "last_metrics", {}))
            if self.kind == "question-only":
                self.last_metrics.update(blank_image_placeholders=1, observed_image_count=0)
            elif self.kind in {"thumbnail", "overview-only"}:
                self.last_metrics.update(blank_image_placeholders=0, observed_image_count=len(answer_state["images"]))
            return action
        self.last_metrics = {"model_calls": 0, "model_forwards": 0, "input_tokens": 0,
                             "generated_tokens": 0, "latency_seconds": 0.}
        if self.kind == "random":
            index = self.rng.randrange(len(candidates))
        elif self.kind == "coverage":
            roots = [i for i, c in enumerate(candidates)
                     if len(c.get("features", [])) >= 12 and c["features"][10] == 1]
            current = state.get("views", [])[-1] if state.get("views") else None
            finer = []
            if current and float(current["downsample"]) > 1:
                target_scale = max(1., float(current["downsample"]) / 4)
                finer = [i for i, c in enumerate(candidates) if float(c["downsample"]) == target_scale]
            if finer:
                box = current["bbox_level0"]
                width, height = state["dimensions"]
                center = [(box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)]
                index = min(finer, key=lambda i: (math.dist(candidates[i]["center"], center), i))
            else:
                pool = roots or list(range(len(candidates)))
                index = min(pool, key=lambda i: (candidates[i]["center"][1], candidates[i]["center"][0], i))
        else:
            # Finer scale first among candidates retaining substantial tissue.
            tissue = [i for i, c in enumerate(candidates) if self._tissue(c) >= .15]
            pool = tissue or list(range(len(candidates)))
            index = min(pool, key=lambda i: (float(candidates[i]["downsample"]),
                                             -self._tissue(candidates[i]), i))
        return {"type": "read", "candidate_index": index}

    __call__ = act


def make_baseline(kind, answerer, seed=17):
    return NavigationBaseline(kind, answerer, seed)
