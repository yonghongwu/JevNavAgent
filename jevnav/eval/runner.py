"""Unified closed-loop WSI navigation evaluation runner (`jevnav-eval`)."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from jevnav.core import WSINavigationEnvironment, score_point
from jevnav.data.dataset import load_manifest_cases
from jevnav.eval.interface import (
    BaseWSINavPolicy,
    JevNavLocalPolicy,
    NavigationObservation,
    OpenAICompatiblePolicy,
    OverviewOnlyPolicy,
    RandomNavPolicy,
    TissuePeakNavPolicy,
)
from jevnav.eval.metrics import score_cam16_predictions, score_wsivqa_predictions


def load_custom_policy(spec: str, kwargs: dict[str, Any] | None = None) -> BaseWSINavPolicy:
    """Load a user policy from `path/to/file.py:ClassName` or `module.path:ClassName`."""
    if ":" not in spec:
        raise ValueError(f"Custom policy spec must be `file_or_module:ClassName`, got: {spec}")
    mod_part, cls_name = spec.split(":", 1)
    if mod_part.endswith(".py") or Path(mod_part).exists():
        p = Path(mod_part).resolve()
        spec_obj = importlib.util.spec_from_file_location(p.stem, str(p))
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Cannot load module from {p}")
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_part)
    cls = getattr(mod, cls_name)
    policy = cls(**(kwargs or {}))
    if not isinstance(policy, BaseWSINavPolicy):
        raise TypeError(f"{cls_name} must subclass BaseWSINavPolicy")
    return policy


def _resolve_slide_file(case: dict[str, Any], data_dir: Path, extra_slide_dirs: list[Path]) -> Path:
    candidates = [data_dir / case["slide_relpath"]]
    fname = Path(case["slide_relpath"]).name
    for d in extra_slide_dirs:
        candidates.append(d / fname)
        candidates.append(d / case["slide_relpath"])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Slide not found for {case['slide_id']}. Looked in: {[str(c) for c in candidates]}. "
        "Run `jevnav-download --mode eval-slides --out <data-dir>` or pass `--slides-dir`."
    )


def _resolve_annotation_file(case: dict[str, Any], data_dir: Path, extra_ann_dirs: list[Path]) -> Path | None:
    rel = case.get("annotation_relpath")
    if not rel:
        return None
    candidates = [data_dir / rel]
    fname = Path(rel).name
    for d in extra_ann_dirs:
        candidates.append(d / fname)
        candidates.append(d / rel)
    for p in candidates:
        if p.exists():
            return p
    return None


def _wrap_observation(raw_obs: dict[str, Any]) -> NavigationObservation:
    overview = raw_obs["overview"]
    crops = list(raw_obs.get("images", []))
    raw_state = {
        "question": raw_obs["question"],
        "choices": raw_obs["choices"],
        "images": [overview] + crops,
        "views": raw_obs.get("views", []),
        "dimensions": raw_obs["dimensions"],
        "mpp": raw_obs.get("mpp"),
        "remaining_budget": raw_obs["remaining_budget"],
        "decisions_remaining": raw_obs["decisions_remaining"],
        "candidates": raw_obs.get("candidates", []),
        "status": raw_obs.get("status", "active"),
    }
    return NavigationObservation(
        question=raw_obs["question"],
        choices=raw_obs["choices"],
        overview_image=overview,
        crop_images=crops,
        views=raw_obs.get("views", []),
        candidates=raw_obs.get("candidates", []),
        dimensions=tuple(raw_obs["dimensions"]),
        mpp=raw_obs.get("mpp"),
        remaining_budget=int(raw_obs["remaining_budget"]),
        decisions_remaining=int(raw_obs["decisions_remaining"]),
        status=str(raw_obs.get("status", "active")),
        raw_state=raw_state,
    )


def run_case_episode(
    policy: BaseWSINavPolicy,
    case: dict[str, Any],
    slide_path: Path,
    annotation_path: Path | None,
    budget: int,
) -> dict[str, Any]:
    """Execute one closed-loop navigation episode on a gigapixel WSI."""
    t0 = time.perf_counter()
    policy.reset()
    env = WSINavigationEnvironment(
        {"slide_path": str(slide_path)},
        budget=budget,
        question=case["question"],
        choices=case["choices"],
    )
    try:
        raw_obs = env.reset()
        step_trace: list[dict[str, Any]] = []
        any_qualified_hit = False
        final_answer_idx = 0
        final_answer_probs: list[float] | None = None

        while True:
            obs = _wrap_observation(raw_obs)
            action = policy.act(obs)

            # Evaluate spatial localization on positive CAM16 slides at every high-mag step
            in_tumor_qualified = False
            if case["benchmark"] == "cam16" and case["is_positive"] and annotation_path and obs.views:
                last_v = obs.views[-1]
                if int(last_v.get("downsample", 64)) <= 4:
                    box = last_v["bbox_level0"]
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    ev = score_point(str(annotation_path), cx, cy, window_size=512)
                    in_tumor_qualified = bool(ev.get("in_tumor") and ev.get("tumor_fraction", 0.0) >= 0.25)
                    if in_tumor_qualified:
                        any_qualified_hit = True

            step_trace.append(
                {
                    "step_index": obs.step_index,
                    "downsample": obs.current_downsample,
                    "action_type": action.type,
                    "candidate_index": action.candidate_index,
                    "answer_index": action.answer_index,
                    "answer_probs": action.answer_probs,
                    "stop_prob": action.stop_prob,
                    "in_tumor_qualified": in_tumor_qualified,
                }
            )

            if action.type == "read" and obs.remaining_budget > 0 and action.candidate_index is not None:
                raw_obs, done, _info = env.step({"type": "read", "candidate_index": int(action.candidate_index)})
                if done:
                    break
            else:
                final_answer_idx = int(action.answer_index) if action.answer_index is not None else 0
                final_answer_probs = action.answer_probs
                env.step({"type": "stop", "answer_index": final_answer_idx})
                break

        wall_seconds = time.perf_counter() - t0
        views = env.views
        gold = int(case["answer_index"])
        correct = final_answer_idx == gold

        evidence: dict[str, Any] = {"any_qualified_hit": any_qualified_hit}
        cam_hit = False
        if case["benchmark"] == "cam16" and case["is_positive"] and annotation_path and views:
            last_v = views[-1]
            if int(last_v.get("downsample", 64)) <= 4:
                box = last_v["bbox_level0"]
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                ev = score_point(str(annotation_path), cx, cy, window_size=512)
                qualified = bool(ev.get("in_tumor") and ev.get("tumor_fraction", 0.0) >= 0.25)
                evidence.update(
                    {
                        "in_tumor": bool(ev.get("in_tumor", False)),
                        "tumor_fraction": float(ev.get("tumor_fraction", 0.0)),
                        "qualified_field_success": qualified,
                    }
                )
                cam_hit = bool(final_answer_idx == 0 and qualified)

        return {
            "benchmark": case["benchmark"],
            "slide_id": case["slide_id"],
            "question_id": case["question_id"],
            "split": case["split"],
            "is_positive": case.get("is_positive"),
            "gold_answer_index": gold,
            "answer_index": final_answer_idx,
            "answer_probs": final_answer_probs,
            "correct": correct,
            "cam_hit": cam_hit,
            "visited_hit": any_qualified_hit,
            "evidence": evidence,
            "views": views,
            "steps": step_trace,
            "costs": {
                "read_count": len(views),
                "wall_seconds": round(wall_seconds, 4),
            },
        }
    finally:
        env.close()


def _manifests_for_benchmark(benchmark: str, data_dir: Path) -> list[Path]:
    m = data_dir / "manifests"
    mapping = {
        "wsivqa_val": [m / "wsivqa_clean_visual_val_179.json"],
        "wsivqa_test": [m / "wsivqa_clean_visual_test_172.json"],
        "wsivqa_351": [
            m / "wsivqa_clean_visual_val_179.json",
            m / "wsivqa_clean_visual_test_172.json",
        ],
        "cam16_dev": [m / "cam16_verified_dev_46.json"],
        "cam16_test": [m / "cam16_verified_test_129.json"],
    }
    return mapping[benchmark]


def build_policy_from_args(args: argparse.Namespace) -> BaseWSINavPolicy:
    if args.policy == "overview":
        return OverviewOnlyPolicy()
    if args.policy == "random":
        return RandomNavPolicy(seed=args.seed)
    if args.policy == "tissue_peak":
        return TissuePeakNavPolicy()
    if args.policy == "openai_api":
        if not args.api_model:
            raise ValueError("--api-model is required when --policy openai_api")
        return OpenAICompatiblePolicy(
            model=args.api_model,
            api_base=args.api_base,
            api_key=args.api_key,
        )
    if args.policy == "jevnav":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --policy jevnav")
        return JevNavLocalPolicy(
            checkpoint=args.checkpoint,
            backbone=args.backbone,
            device=args.device,
        )
    return load_custom_policy(args.policy)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local or API policy on JevNavAgent benchmarks.")
    parser.add_argument(
        "--benchmark",
        choices=["wsivqa_val", "wsivqa_test", "wsivqa_351", "cam16_dev", "cam16_test"],
        required=True,
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Built-in (`jevnav`, `openai_api`, `overview`, `random`, `tissue_peak`) or `path/to/file.py:PolicyClass`.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Directory containing downloaded metadata & slides.")
    parser.add_argument("--slides-dir", type=Path, action="append", default=[], help="Optional extra slide search directory.")
    parser.add_argument("--annotations-dir", type=Path, action="append", default=[], help="Optional extra CAM16 XML directory.")
    parser.add_argument("--budget", type=int, default=None, help="Read step budget (default: 4 for WSI-VQA, 12 for CAM16).")
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate only the first N cases (for smoke tests).")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint", type=str, default=None, help="Local path or HF `renyimen/JevNavAgent:<subdir>` for `jevnav`.")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--api-model", type=str, default=None, help="Model name for `--policy openai_api`.")
    parser.add_argument("--api-base", type=str, default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", type=str, default=None, help="API key (or set OPENAI_API_KEY).")
    parser.add_argument("--output", type=Path, required=True, help="Output trajectory JSONL path.")
    args = parser.parse_args(argv)

    budget = args.budget if args.budget is not None else (4 if args.benchmark.startswith("wsivqa") else 12)
    manifests = _manifests_for_benchmark(args.benchmark, args.data_dir)
    cases = load_manifest_cases(manifests)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    policy = build_policy_from_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[jevnav-eval] Running {args.policy} on {args.benchmark} ({len(cases)} cases, budget={budget})")
    with args.output.open("w", encoding="utf-8") as fout:
        for i, case in enumerate(cases, 1):
            slide_path = _resolve_slide_file(case, args.data_dir, args.slides_dir)
            ann_path = _resolve_annotation_file(case, args.data_dir, args.annotations_dir)
            ep = run_case_episode(policy, case, slide_path, ann_path, budget=budget)
            fout.write(json.dumps(ep, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 10 == 0 or i == len(cases):
                print(f"  [{i}/{len(cases)}] {case['question_id']} | reads={ep['costs']['read_count']} | correct={ep['correct']}")

    if args.benchmark.startswith("wsivqa"):
        summary = score_wsivqa_predictions(manifests, [args.output])
    else:
        summary = score_cam16_predictions(manifests, [args.output])

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
