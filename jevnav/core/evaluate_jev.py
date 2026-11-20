"""Auditable budgeted evaluation; references never enter policy observations.

JSON/JSONL manifests contain slide rows, optionally with a ``questions`` list.
Each question has question/choices/answer_index and optionally question_id.
Test runs require a freeze JSON with protocol_frozen and model_selection_frozen.
No examples are discarded following slide, policy, parser, or scoring errors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time


PUBLIC_KEYS = ("question", "choices", "images", "views", "dimensions", "mpp",
               "remaining_budget", "decisions_remaining", "candidates", "status")
VIEW_KEYS = ("index", "candidate_id", "center", "downsample", "bbox_level0",
             "requested_downsample", "native_bbox", "native_level", "native_level_downsample",
             "read_pixels", "returned_pixels", "physical_span_um")
CANDIDATE_KEYS = ("id", "action", "center", "downsample", "bbox_level0", "features")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path):
    """File SHA or SHA of canonical JSON {relative_path:file_SHA} for a directory."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Artifact does not exist: {path}")
    if path.is_file():
        return sha256(path)
    files = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError(f"Artifact directory is empty: {path}")
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_test_freeze(args, code_files):
    freeze = json.loads(Path(args.freeze).read_text()) if args.freeze else {}
    if not freeze.get("protocol_frozen") or not freeze.get("model_selection_frozen"):
        raise ValueError("Test evaluation requires protocol_frozen/model_selection_frozen=true")
    if sha256(args.manifest) not in freeze.get("allowed_manifest_sha256", []):
        raise ValueError("Test manifest SHA256 is not authorized by evaluation freeze")
    selection = freeze.get("selected_checkpoints", {}).get(args.method)
    if not selection:
        raise ValueError(f"No selected checkpoint frozen for method {args.method}")
    if not args.checkpoint or Path(args.checkpoint).resolve() != Path(selection["path"]).resolve():
        raise ValueError("Selected checkpoint path differs from evaluation freeze")
    if artifact_sha256(args.checkpoint) != selection["sha256"]:
        raise ValueError("Selected checkpoint SHA256 differs from evaluation freeze")
    frozen_code = freeze.get("code_sha256", freeze.get("source_files", {}))
    frozen_paths = {}
    for name, expected in frozen_code.items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        expected = expected["sha256"] if isinstance(expected, dict) else expected
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen code SHA256 differs: {path}")
        frozen_paths[str(path.resolve())] = expected
    for path in code_files:
        if str(path.resolve()) not in frozen_paths:
            raise ValueError(f"Evaluation dependency is not frozen: {path.name}")
    frozen_answerer = selection.get("answerer")
    if frozen_answerer is not None and frozen_answerer != args.answerer:
        raise ValueError("Baseline answerer differs from evaluation freeze")
    if args.method == "cnn":
        frozen_cnn = selection.get("cnn_checkpoint", {})
        if not args.cnn_checkpoint or Path(args.cnn_checkpoint).resolve() != Path(frozen_cnn["path"]).resolve():
            raise ValueError("CNN checkpoint path differs from evaluation freeze")
        if sha256(args.cnn_checkpoint) != frozen_cnn["sha256"]:
            raise ValueError("CNN checkpoint SHA256 differs from evaluation freeze")
        if args.cnn_threshold != selection.get("cnn_threshold"):
            raise ValueError("CNN threshold differs from evaluation freeze")
    return freeze


def json_safe(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "size") and hasattr(value, "mode"):
        return {"image_size": list(value.size), "mode": value.mode}
    return {"unsupported_type": type(value).__name__}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def public_observation(observation):
    state = {key: observation[key] for key in PUBLIC_KEYS if key in observation}
    state["views"] = [{k: v[k] for k in VIEW_KEYS if k in v} for v in observation.get("views", [])]
    state["candidates"] = [{k: c[k] for k in CANDIDATE_KEYS if k in c}
                           for c in observation.get("candidates", [])]
    if "overview" in observation:
        state["images"] = [observation["overview"]] + list(observation.get("images", []))
    return state


def load_cases(path, split):
    path = Path(path)
    raw = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
           if path.suffix == ".jsonl" else json.loads(path.read_text()))
    rows = raw if isinstance(raw, list) else raw.get("cases", raw.get("slides", raw.get("rows", [])))
    cases = []
    for row in rows:
        if row.get("split") != split:
            continue
        for q_index, question in enumerate(row.get("questions", [row])):
            case = {k: v for k, v in row.items() if k != "questions"}
            case.update(question)
            case["slide_id"] = str(row["slide_id"])
            case["question_id"] = str(question.get("question_id", question.get("id", q_index)))
            # Split/slide identity must not be overridden by a nested question.
            case["split"] = row["split"]
            cases.append(case)
    ids = [(c["slide_id"], c["question_id"]) for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate slide/question identity in manifest")
    if not cases:
        raise ValueError(f"Manifest contains no cases for split {split}")
    return cases


def episode_seed(seed, case):
    key = json.dumps([seed, case["slide_id"], case["question_id"]], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def validate_action(action, state):
    if not isinstance(action, dict):
        raise ValueError("action_must_be_object")
    kind = action.get("type")
    key = "candidate_index" if kind == "read" else "answer_index"
    if kind not in {"read", "stop"} or set(action) != {"type", key}:
        raise ValueError("invalid_action_schema")
    count = len(state.get("candidates", [])) if kind == "read" else len(state.get("choices", []))
    if type(action[key]) is not int or not 0 <= action[key] < count:
        raise ValueError("invalid_action_index")
    if kind == "read" and state.get("remaining_budget", 0) <= 0:
        raise ValueError("read_budget_exhausted")
    return action


def cam_evidence(case, dimensions, views, score_fn=None):
    if not case.get("annotation_path"):
        return {"available": False, "reason": "no_trustworthy_region_reference"}
    if score_fn is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
        from wsi_recovery.environment import score_point
        score_fn = score_point
    def score(view):
        box = view.get("bbox_level0", view.get("native_bbox"))
        point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        # The current field must itself supply high-resolution support. Passing
        # the whole history lets a later coarse field borrow an earlier fine
        # observation, which violates the frozen last-field definition.
        return score_fn(case["annotation_path"], dimensions, point, [view])
    last = score(views[-1]) if views else {"success": False, "reason": "no_observed_field"}
    # Reference-selected best field is an oracle diagnostic, never primary success.
    oracle = any(score(v).get("success", False) for v in views)
    return {"available": True, "selection": "last_visited_field_center", "last_field": last,
            "success": bool(last.get("success", False)), "oracle_best_visited_success": oracle}


def run_episode(case, policy, budget, seed, env_factory, output_dir=None, evidence_scorer=cam_evidence):
    started = time.perf_counter()
    stable_seed = episode_seed(seed, case)
    result = {"slide_id": case["slide_id"], "patient_id": case.get("patient_id", case["slide_id"]),
              "question_id": case["question_id"], "split": case["split"], "seed": seed,
              "episode_seed": stable_seed, "budget": budget, "status": "error", "correct": False,
              "answer_index": None, "reference_answer_index": case.get("answer_index"),
              "trace": [], "views": [], "evidence": {"available": False},
              "observations_dir": str(Path(output_dir).resolve()) if output_dir else None,
              "costs": {"policy_calls": 0, "model_calls": 0, "model_forwards": 0,
                        "input_tokens": 0, "generated_tokens": 0, "image_tokens": 0,
                        "model_latency_seconds": 0., "read_count": 0, "stop_count": 0,
                        "raw_decode_pixels_known_reads": 0, "overview_returned_pixels": 0,
                        "cnn_crop_count": 0, "cnn_source_pixels": 0, "cnn_input_pixels": 0,
                        "cnn_forwards": 0, "cnn_latency_seconds": 0.}}
    env = None
    try:
        if hasattr(policy, "reset"):
            policy.reset(stable_seed)
        env = env_factory(case, budget=budget, output_dir=output_dir)
        observation = env.reset(seed=stable_seed)
        if "overview" in observation:
            w, h = observation["overview"].size
            result["costs"]["overview_returned_pixels"] = w * h
        for step in range(budget + 1):
            state = public_observation(observation)
            trace = {"step": step, "state": json_safe(state)}
            if output_dir:
                root = Path(output_dir).resolve()
                trace["state"]["images"] = [str(root / "overview.png")] + [
                    str(root / f"view_{index:02d}.png") for index in range(len(state.get("views", [])))]
            result["trace"].append(trace)
            result["costs"]["policy_calls"] += 1
            # Clear stale counters: failed calls must not repeat the prior call's metrics.
            if hasattr(policy, "last_metrics"):
                policy.last_metrics = {}
            call_start = time.perf_counter()
            try:
                action = policy.act(state)
            finally:
                metrics = dict(getattr(policy, "last_metrics", {}))
                trace["policy_metrics"] = json_safe(metrics)
                trace["policy_wall_seconds"] = time.perf_counter() - call_start
                result["costs"]["model_calls"] += metrics.get("model_calls", int(metrics.get("model_forwards", 0) > 0))
                for key in ("model_forwards", "input_tokens", "generated_tokens", "image_tokens"):
                    result["costs"][key] += metrics.get(key, 0)
                for key in ("cnn_crop_count", "cnn_source_pixels", "cnn_input_pixels", "cnn_forwards", "cnn_latency_seconds"):
                    result["costs"][key] += metrics.get(key, 0)
                result["costs"]["model_latency_seconds"] += metrics.get("latency_seconds", 0.)
            trace["action"] = json_safe(action)
            try:
                validate_action(action, state)
            except ValueError as exc:
                result["status"], result["error"] = "invalid_action", str(exc)
                break
            observation, done, info = env.step(action)
            trace["tool_info"] = json_safe(info)
            if action["type"] == "stop":
                result["costs"]["stop_count"] += 1
            if done:
                result["status"] = getattr(env, "status", "finished")
                break
        else:
            result["status"] = "decision_budget_exhausted"
        final = getattr(env, "final", None)
        if final and result["status"] != "invalid_action":
            result["answer_index"] = final.get("answer_index")
        gold = case.get("answer_index")
        result["reference_valid"] = type(gold) is int and 0 <= gold < len(observation.get("choices", []))
        result["correct"] = bool(result["reference_valid"] and type(result["answer_index"]) is int
                                  and result["answer_index"] == gold)
    except Exception as exc:
        result["status"], result["error"] = "error", f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            views = list(getattr(env, "views", []))
            result["views"] = json_safe(views)
            result["tool_records"] = json_safe(getattr(env, "records", []))
            result["costs"]["read_count"] = len(views)
            result["costs"]["raw_decode_pixels_known_reads"] = sum(
                v.get("read_pixels", [0, 0])[0] * v.get("read_pixels", [0, 0])[1] for v in views)
            tool_costs = getattr(env, "costs", {})
            result["costs"]["tool_costs"] = json_safe(tool_costs() if callable(tool_costs) else tool_costs)
            try:
                result["evidence"] = evidence_scorer(case, env.dimensions, views)
                if result["evidence"].get("available"):
                    result["evidence"]["qualified_field_success"] = bool(result["evidence"].get("success", False))
                    result["evidence"]["success"] = bool(
                        result["evidence"]["qualified_field_success"] and
                        result["answer_index"] == 0 and result["status"] == "stopped")
            except Exception as exc:
                result["evidence"] = {"available": bool(case.get("annotation_path")), "success": False,
                                      "scoring_error": f"{type(exc).__name__}: {exc}"}
            try:
                env.close()
            except Exception as exc:
                result["close_error"] = f"{type(exc).__name__}: {exc}"
        result["costs"]["wall_seconds"] = time.perf_counter() - started
    return json_safe(result)


def load_policy(args):
    if args.method == "codex":
        from codex_policy import CodexPolicy
        return CodexPolicy(model=args.codex_model, reasoning_effort=args.codex_effort,
                           timeout=args.codex_timeout, max_pixels=args.max_pixels)
    kwargs = {"device": args.device, "max_pixels": args.max_pixels}
    if args.method == "jev":
        from jev_model import JEVPolicy
        return JEVPolicy.from_pretrained(args.base_path, args.checkpoint, **kwargs)
    if args.method == "ar":
        from ar_policy import ARPolicy
        return ARPolicy.from_pretrained(args.base_path, args.checkpoint, **kwargs)
    from baselines import ForcedAnswerer, make_baseline
    if args.answerer == "jev":
        from jev_model import JEVPolicy
        answerer = ForcedAnswerer(JEVPolicy.from_pretrained(args.base_path, args.checkpoint, **kwargs))
    else:
        from choice_answerer import ChoiceLogitAnswerer
        answerer = ChoiceLogitAnswerer(args.base_path, **kwargs, history_limit=args.history_limit)
    if args.method == "cnn":
        from cnn_policy import CNNNavigationPolicy
        return CNNNavigationPolicy(answerer, args.cnn_checkpoint, args.cnn_threshold)
    return make_baseline(args.method, answerer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "dev", "val", "test"], default="dev")
    parser.add_argument("--method", choices=["jev", "ar", "codex", "random", "coverage", "tissue", "thumbnail", "question-only", "overview-only", "cnn"], required=True)
    parser.add_argument("--codex-model", default="gpt-6.1-terra")
    parser.add_argument("--codex-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--codex-timeout", type=float, default=120.0)
    parser.add_argument("--cnn-checkpoint", default=str(Path(__file__).resolve().parents[3] / "experiments/llm_baseline_v3/training/scorer_seed17/best.pt"))
    parser.add_argument("--cnn-threshold", type=float, choices=[.5, .8, .95], default=.8)
    parser.add_argument("--answerer", choices=["jev", "frozen"], default="jev")
    parser.add_argument("--history-limit", type=int, default=3)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--freeze")
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    if args.method not in {"jev", "ar", "codex"} and args.answerer == "jev" and not args.checkpoint:
        parser.error("Fixed navigation with the trained JEV answerer requires --checkpoint")
    if args.method == "cnn" and args.answerer != "jev":
        parser.error("CNN navigation requires the shared trained JEV answerer")
    if len(set(args.budgets)) != len(args.budgets) or any(not 1 <= b <= 64 for b in args.budgets):
        parser.error("Budgets must be unique integers in 1..64")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    cases = load_cases(args.manifest, args.split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists() or (output / "results.jsonl").exists():
        parser.error("Output already contains a run; choose a new output directory")
    code_files = [Path(__file__), Path(__file__).with_name("navigation_env.py"),
                  Path(__file__).with_name("baselines.py"), Path(__file__).with_name("jev_model.py"),
                  Path(__file__).with_name("ar_policy.py"), Path(__file__).with_name("choice_answerer.py"),
                  Path(__file__).with_name("cnn_policy.py")]
    if args.method == "cnn":
        from cnn_policy import scorer_source_path
        code_files.append(scorer_source_path())
    if args.split == "test":
        try:
            validate_test_freeze(args, code_files)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
    checkpoints = ([Path(args.checkpoint)] if args.checkpoint and Path(args.checkpoint).is_file() else
                   sorted(p for p in Path(args.checkpoint).rglob("*") if p.is_file()) if args.checkpoint else [])
    identity = {"manifest_sha256": sha256(args.manifest), "code_sha256": {str(p): sha256(p) for p in code_files},
                "checkpoint_sha256": {str(p): sha256(p) for p in checkpoints},
                "checkpoint_artifact_sha256": artifact_sha256(args.checkpoint) if args.checkpoint else None,
                "freeze_sha256": sha256(args.freeze) if args.freeze else None}
    if args.method == "cnn":
        identity["cnn_checkpoint_sha256"] = sha256(args.cnn_checkpoint)
        identity["cnn_threshold"] = args.cnn_threshold
    plan = [{"slide_id": c["slide_id"], "patient_id": c.get("patient_id", c["slide_id"]),
             "question_id": c["question_id"], "seed": s, "budget": b, "method": args.method,
             "has_region_reference": bool(c.get("annotation_path"))}
            for b in args.budgets for s in args.seeds for c in cases]
    run = {"args": vars(args), "identity": identity, "expected_attempts": plan,
           "started_utc": datetime.now(timezone.utc).isoformat(), "status": "initializing"}
    write_json(output / "run.json", run)
    start = time.perf_counter()
    try:
        policy = load_policy(args)
    except Exception as exc:
        run.update(status="initialization_error", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", run)
        raise
    run.update(initialization_seconds=time.perf_counter()-start, status="running")
    write_json(output / "run.json", run)
    from navigation_env import NavigationEnvironment
    with (output / "results.jsonl").open("w") as ledger:
        for budget in args.budgets:
            for seed in args.seeds:
                for case in cases:
                    token = hashlib.sha256(json.dumps([case["slide_id"], case["question_id"], budget, seed]).encode()).hexdigest()[:20]
                    image_dir = output / "observations" / token if args.save_images else None
                    result = run_episode(case, policy, budget, seed, NavigationEnvironment, image_dir)
                    result.update(method=args.method, identity=identity)
                    ledger.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    ledger.flush()
                    print(json.dumps({k: result[k] for k in ("slide_id", "question_id", "budget", "seed", "status", "correct")}), flush=True)
    run.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "run.json", run)


if __name__ == "__main__":
    main()
