"""Pluggable policy interface for gigapixel WSI active visual navigation.

External researchers can evaluate any local VLM or remote OpenAI-compatible Vision API
by subclassing `BaseWSINavPolicy` and implementing a single method: `act(obs)`.
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class NavigationObservation:
    """Annotation-blind observation presented to a navigation policy at step t."""

    question: str
    choices: list[str]
    overview_image: Image.Image
    crop_images: list[Image.Image]
    views: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    dimensions: tuple[int, int]
    mpp: Any
    remaining_budget: int
    decisions_remaining: int
    status: str
    raw_state: dict[str, Any] = field(default_factory=dict)

    @property
    def step_index(self) -> int:
        return len(self.views)

    @property
    def current_downsample(self) -> int:
        if not self.views:
            return 64
        return int(self.views[-1].get("downsample", 64))

    @property
    def latest_image(self) -> Image.Image:
        return self.crop_images[-1] if self.crop_images else self.overview_image


@dataclass
class NavigationAction:
    """Decision returned by a policy at step t."""

    type: str  # "read" or "stop"
    candidate_index: int | None = None
    answer_index: int | None = None
    answer_probs: list[float] | None = None
    stop_prob: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseWSINavPolicy(ABC):
    """Abstract base class for pluggable WSI navigation policies."""

    def reset(self) -> None:
        """Called at the start of each episode (override if your policy maintains state)."""
        return None

    @abstractmethod
    def act(self, obs: NavigationObservation) -> NavigationAction:
        """Select either `NavigationAction(type="read", candidate_id=...)`
        or `NavigationAction(type="stop", answer_index=...)`.
        """


class OverviewOnlyPolicy(BaseWSINavPolicy):
    """Zero-read control baseline that stops immediately at Step 0."""

    def __init__(self, default_answer_index: int = 0) -> None:
        self.default_answer_index = int(default_answer_index)

    def act(self, obs: NavigationObservation) -> NavigationAction:
        idx = min(self.default_answer_index, max(0, len(obs.choices) - 1))
        probs = [0.0] * len(obs.choices)
        if probs:
            probs[idx] = 1.0
        return NavigationAction(type="stop", answer_index=idx, answer_probs=probs, stop_prob=1.0)


class RandomNavPolicy(BaseWSINavPolicy):
    """Random candidate explorer baseline (reads until budget exhausted, then answers)."""

    def __init__(self, seed: int = 17) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: NavigationObservation) -> NavigationAction:
        if obs.remaining_budget > 0 and obs.candidates:
            cidx = self._rng.randrange(len(obs.candidates))
            return NavigationAction(type="read", candidate_index=cidx, stop_prob=0.0)
        idx = self._rng.randrange(len(obs.choices)) if obs.choices else 0
        probs = [1.0 / max(1, len(obs.choices))] * len(obs.choices)
        return NavigationAction(type="stop", answer_index=idx, answer_probs=probs, stop_prob=1.0)


class TissuePeakNavPolicy(BaseWSINavPolicy):
    """Heuristic baseline that zooms into the highest-tissue-fraction candidate each step."""

    def act(self, obs: NavigationObservation) -> NavigationAction:
        if obs.remaining_budget > 0 and obs.candidates:
            # Candidate feature index 7 is `thumbnail_tissue_fraction`
            best_idx = max(
                range(len(obs.candidates)),
                key=lambda i: (
                    float(obs.candidates[i].get("features", [0.0] * 8)[7])
                    if len(obs.candidates[i].get("features", [])) > 7
                    else 0.0,
                    -int(obs.candidates[i].get("downsample", 64)),
                ),
            )
            return NavigationAction(type="read", candidate_index=best_idx, stop_prob=0.0)
        probs = [1.0 / max(1, len(obs.choices))] * len(obs.choices)
        return NavigationAction(type="stop", answer_index=0, answer_probs=probs, stop_prob=1.0)


def _pil_to_base64_jpeg(img: Image.Image, max_side: int = 512, quality: int = 88) -> str:
    im = img.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OpenAICompatiblePolicy(BaseWSINavPolicy):
    """Zero-code adapter for any OpenAI-compatible Vision API (GPT-4o, Claude/Gemini proxies, vLLM, SGLang).

    Sends the slide overview + latest observed high-resolution crop together with the
    candidate coordinate table, and parses a structured JSON navigation decision.
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 90,
        max_image_side: int = 512,
    ) -> None:
        self.model = model
        self.api_base = (api_base or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.max_image_side = int(max_image_side)

    def _build_messages(self, obs: NavigationObservation) -> list[dict[str, Any]]:
        choices_str = "\n".join(f"  {i}: {c}" for i, c in enumerate(obs.choices))
        cand_lines = []
        for idx, c in enumerate(obs.candidates[:29]):
            feats = c.get("features", [])
            tissue = feats[7] if len(feats) > 7 else 0.0
            cx, cy = c.get("center", [0.0, 0.0])
            cand_lines.append(
                f"  index={idx} | ds={c['downsample']} | center=({cx:.3f}, {cy:.3f}) | tissue={tissue:.2f}"
            )
        cands_str = "\n".join(cand_lines) if cand_lines else "  (none)"

        prompt = (
            "You are an expert digital pathologist navigating a gigapixel whole-slide image.\n"
            f"Question: {obs.question}\n"
            f"Choices:\n{choices_str}\n"
            f"Current step: {obs.step_index} | Current downsample: DS={obs.current_downsample} "
            f"| Remaining read budget: {obs.remaining_budget}\n"
            f"Available next-view candidates (lower ds = higher magnification; ds=1 is 40x):\n{cands_str}\n\n"
            "Respond ONLY with a valid JSON object:\n"
            '- To zoom/pan to a candidate: {"action": "read", "candidate_index": <int>}\n'
            '- To stop and answer now: {"action": "stop", "answer_index": <int>}\n'
            "If remaining read budget is 0, you MUST choose action=\"stop\"."
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_base64_jpeg(obs.overview_image, self.max_image_side)}"},
            },
        ]
        if obs.crop_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_base64_jpeg(obs.crop_images[-1], self.max_image_side)}"},
                }
            )
        return [{"role": "user", "content": content}]

    def _call_api(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])

    @staticmethod
    def _parse_response(text: str, obs: NavigationObservation) -> NavigationAction:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        parsed = {}
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = {}

        action = str(parsed.get("action", "stop")).strip().lower()
        n_cands = len(obs.candidates)

        if action == "read" and obs.remaining_budget > 0 and n_cands > 0:
            cidx = parsed.get("candidate_index", parsed.get("candidate_id"))
            if cidx is not None and 0 <= int(cidx) < n_cands:
                return NavigationAction(
                    type="read", candidate_index=int(cidx), stop_prob=0.0, metadata={"raw_response": text}
                )
            return NavigationAction(
                type="read",
                candidate_index=0,
                stop_prob=0.0,
                metadata={"raw_response": text, "fallback": "first_candidate"},
            )

        ans = parsed.get("answer_index", 0)
        try:
            ans_idx = max(0, min(len(obs.choices) - 1, int(ans)))
        except (TypeError, ValueError):
            ans_idx = 0
        probs = [0.0] * len(obs.choices)
        if probs:
            probs[ans_idx] = 1.0
        return NavigationAction(
            type="stop",
            answer_index=ans_idx,
            answer_probs=probs,
            stop_prob=1.0,
            metadata={"raw_response": text},
        )

    def act(self, obs: NavigationObservation) -> NavigationAction:
        messages = self._build_messages(obs)
        text = self._call_api(messages)
        return self._parse_response(text, obs)


def resolve_checkpoint_path(checkpoint_spec: str) -> Path:
    """Resolve either a local checkpoint path or a HuggingFace spec `<repo_id>:<subfolder>`."""
    p = Path(checkpoint_spec)
    if p.exists():
        return p
    if ":" in checkpoint_spec:
        repo_id, subfolder = checkpoint_spec.split(":", 1)
    else:
        repo_id, subfolder = checkpoint_spec, ""
    from huggingface_hub import snapshot_download

    allow = [f"{subfolder}/*"] if subfolder else None
    local_root = Path(snapshot_download(repo_id=repo_id, repo_type="model", allow_patterns=allow))
    return local_root / subfolder if subfolder else local_root


class JevNavLocalPolicy(BaseWSINavPolicy):
    """Single-forward JevNavAgent policy (Qwen3-VL-2B + LoRA + Typed Decision Heads)."""

    def __init__(
        self,
        checkpoint: str,
        backbone: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        import jevnav.core  # noqa: F401
        from jev_model import JEVPolicy

        ckpt_path = resolve_checkpoint_path(checkpoint)
        self._policy = JEVPolicy.from_checkpoint(
            str(ckpt_path),
            backbone_path=backbone,
            device=device,
            dtype=dtype,
        )

    def act(self, obs: NavigationObservation) -> NavigationAction:
        out = self._policy.decide(obs.raw_state)
        act_type = str(out.get("type", "stop")).lower()
        cidx = out.get("candidate_index", out.get("candidate_id"))
        return NavigationAction(
            type=act_type,
            candidate_index=int(cidx) if act_type == "read" and cidx is not None else None,
            answer_index=int(out["answer_index"]) if out.get("answer_index") is not None else None,
            answer_probs=[float(x) for x in out.get("answer_probs", [])] or None,
            stop_prob=float(out.get("stop_prob", 1.0 if act_type == "stop" else 0.0)),
            metadata={k: v for k, v in out.items() if k not in ("type", "candidate_index", "candidate_id", "answer_index", "answer_probs", "stop_prob")},
        )
