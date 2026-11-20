"""Official paper metrics scorer for Clean Visual WSI-VQA and Verified CAMELYON16."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from jevnav.data.dataset import load_manifest_cases


def _load_jsonl(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        with Path(p).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _extract_step_probs(ep: dict[str, Any]) -> tuple[list[float] | None, list[float] | None]:
    """Extract Step-0 and Step-T answer probability vectors from a trajectory record."""
    def _from_step(st: dict[str, Any]) -> list[float] | None:
        for key in ("answer_probs", "answer_probabilities", "probs", "p_ans"):
            if key in st and st[key]:
                return [float(x) for x in st[key]]
        for sub in ("policy_metrics", "policy_output"):
            if isinstance(st.get(sub), dict):
                for key in ("answer_probabilities", "answer_probs"):
                    if st[sub].get(key):
                        return [float(x) for x in st[sub][key]]
        return None

    steps = ep.get("steps") or ep.get("trace") or []
    p0 = _from_step(steps[0]) if steps else None
    pt = _from_step(steps[-1]) if steps else None
    if p0 is None and ep.get("step0_answer_probs"):
        p0 = [float(x) for x in ep["step0_answer_probs"]]
    if pt is None and ep.get("answer_probs"):
        pt = [float(x) for x in ep["answer_probs"]]
    return p0, pt


def score_wsivqa_predictions(
    manifest_paths: list[str | Path],
    prediction_paths: list[str | Path],
) -> dict[str, Any]:
    """Compute official Clean Visual WSI-VQA metrics matching the paper tables."""
    cases = load_manifest_cases(manifest_paths)
    qmeta = {c["question_id"]: c for c in cases}
    preds = _load_jsonl(prediction_paths)
    by_qid = {str(p.get("question_id")): p for p in preds if p.get("question_id") in qmeta}

    def _subset_acc(qids: list[str], use_multiscale: bool = False) -> dict[str, Any]:
        if not qids:
            return {"count": 0, "correct": 0, "acc": 0.0}
        correct = 0
        for qid in qids:
            ep = by_qid[qid]
            m = qmeta[qid]
            gold = int(m["answer_index"])
            if use_multiscale:
                p0, pt = _extract_step_probs(ep)
                if p0 is not None and pt is not None and len(p0) == len(pt):
                    alpha = 0.5 if m.get("diagnostic_scale") == "macro_global" else 1.0
                    fused = [(1.0 - alpha) * a + alpha * b for a, b in zip(p0, pt)]
                    pred = int(np.argmax(fused))
                    correct += int(pred == gold)
                    continue
            if "correct" in ep:
                correct += int(bool(ep["correct"]))
            else:
                pred = ep.get("answer_index", ep.get("predicted_answer_index"))
                correct += int(pred is not None and int(pred) == gold)
        return {
            "count": len(qids),
            "correct": correct,
            "acc": round(100.0 * correct / len(qids), 2),
        }

    eval_qids = [c["question_id"] for c in cases if c["question_id"] in by_qid]
    val_qids = [q for q in eval_qids if qmeta[q].get("split") == "val"]
    test_qids = [q for q in eval_qids if qmeta[q].get("split") == "test"]
    macro_qids = [q for q in eval_qids if qmeta[q].get("diagnostic_scale") == "macro_global"]
    focal_qids = [q for q in eval_qids if qmeta[q].get("diagnostic_scale") == "micro_focal"]

    bucket_names = [
        "A_grade",
        "B_subtype",
        "C_invasion_global",
        "D_focal_presence_exclusion",
        "E_focal_descriptive",
    ]
    buckets = {
        b: _subset_acc([q for q in eval_qids if qmeta[q].get("clinical_bucket") == b])
        for b in bucket_names
    }
    ad_qids = [q for q in eval_qids if qmeta[q].get("clinical_bucket") != "E_focal_descriptive"]
    buckets["subtotal_A_to_D"] = _subset_acc(ad_qids)

    # Within-episode Step-0 vs Step-T diagnostics
    s0_correct = 0
    sT_correct = 0
    n_0_to_1 = 0
    n_1_to_0 = 0
    agree = 0
    l1_shifts = []
    has_step0 = 0
    for qid in eval_qids:
        ep = by_qid[qid]
        gold = int(qmeta[qid]["answer_index"])
        p0, pt = _extract_step_probs(ep)
        if p0 is not None and pt is not None and len(p0) == len(pt):
            has_step0 += 1
            pred0 = int(np.argmax(p0))
            predT = int(np.argmax(pt))
            c0 = pred0 == gold
            cT = predT == gold
            s0_correct += int(c0)
            sT_correct += int(cT)
            if not c0 and cT:
                n_0_to_1 += 1
            elif c0 and not cT:
                n_1_to_0 += 1
            if pred0 == predT:
                agree += 1
            l1_shifts.append(float(np.sum(np.abs(np.asarray(pt) - np.asarray(p0)))))

    reads = [
        int(by_qid[q].get("costs", {}).get("read_count", len(by_qid[q].get("views", []))))
        for q in eval_qids
    ]
    times = [
        float(by_qid[q].get("costs", {}).get("wall_seconds", by_qid[q].get("wall_seconds", 0.0)))
        for q in eval_qids
    ]

    result: dict[str, Any] = {
        "benchmark": "clean_visual_wsivqa",
        "matched_cases": len(eval_qids),
        "expected_cases": len(cases),
        "val": _subset_acc(val_qids),
        "test": _subset_acc(test_qids),
        "total": _subset_acc(eval_qids),
        "macro_global": _subset_acc(macro_qids),
        "micro_focal": _subset_acc(focal_qids),
        "buckets": buckets,
        "multiscale_query_gated": {
            "val": _subset_acc(val_qids, use_multiscale=True),
            "test": _subset_acc(test_qids, use_multiscale=True),
            "total": _subset_acc(eval_qids, use_multiscale=True),
            "macro_global": _subset_acc(macro_qids, use_multiscale=True),
            "micro_focal": _subset_acc(focal_qids, use_multiscale=True),
            "clin_A_to_D": _subset_acc(ad_qids, use_multiscale=True),
        },
        "avg_reads": round(float(np.mean(reads)) if reads else 0.0, 2),
        "avg_latency_sec": round(float(np.mean(times)) if times else 0.0, 2),
    }
    if has_step0 > 0:
        result["step0_vs_stepT"] = {
            "evaluated": has_step0,
            "step0_acc": round(100.0 * s0_correct / has_step0, 2),
            "stepT_acc": round(100.0 * sT_correct / has_step0, 2),
            "n_0_to_1": n_0_to_1,
            "n_1_to_0": n_1_to_0,
            "net_zoom_gain": n_0_to_1 - n_1_to_0,
            "agreement_pct": round(100.0 * agree / has_step0, 2),
            "mean_l1_shift": round(float(np.mean(l1_shifts)), 3),
        }
    return result


def score_cam16_predictions(
    manifest_paths: list[str | Path],
    prediction_paths: list[str | Path],
    calib_stop_th: float | None = None,
    calib_pos_th: float | None = None,
) -> dict[str, Any]:
    """Compute official Verified CAMELYON16 screening and spatial localization metrics."""
    cases = load_manifest_cases(manifest_paths)
    meta = {c["slide_id"]: c for c in cases}
    preds = _load_jsonl(prediction_paths)
    by_sid = {str(p.get("slide_id")): p for p in preds if p.get("slide_id") in meta}

    pos_sids = [c["slide_id"] for c in cases if c["is_positive"] and c["slide_id"] in by_sid]
    neg_sids = [c["slide_id"] for c in cases if not c["is_positive"] and c["slide_id"] in by_sid]

    def _eval_episode(ep: dict[str, Any], is_pos: bool) -> dict[str, Any]:
        # Optional calibrated trace readout over recorded steps
        if calib_stop_th is not None and calib_pos_th is not None and ep.get("steps"):
            chosen_step = None
            for st in ep["steps"]:
                ds = st.get("downsample", 64)
                p_stop = st.get("p_stop", st.get("stop_prob", 0.0))
                p_pos = st.get("p_pos")
                if p_pos is None and st.get("answer_probs"):
                    p_pos = st["answer_probs"][0]
                if ds <= 4 and p_stop >= calib_stop_th and (p_pos or 0.0) >= calib_pos_th:
                    chosen_step = st
                    break
            if chosen_step is not None:
                pred_pos = True
                loc_hit = bool(is_pos and chosen_step.get("in_tumor_qualified", chosen_step.get("qualified_hit", False)))
                reads = int(chosen_step.get("step_index", len(ep.get("views", []))))
            else:
                pred_pos = False
                loc_hit = False
                reads = int(ep.get("costs", {}).get("read_count", len(ep.get("views", []))))
        else:
            pred_idx = ep.get("answer_index", ep.get("predicted_answer_index", 1))
            pred_pos = int(pred_idx) == 0
            ev = ep.get("evidence", {})
            qualified = bool(
                ep.get("cam_hit")
                or ev.get("qualified_field_success", ev.get("success", False))
            )
            loc_hit = bool(is_pos and pred_pos and qualified)
            reads = int(ep.get("costs", {}).get("read_count", len(ep.get("views", []))))

        visited = bool(
            is_pos
            and (
                loc_hit
                or ep.get("visited_hit")
                or ep.get("any_qualified_hit")
                or ep.get("evidence", {}).get("any_qualified_hit", False)
            )
        )
        return {
            "pred_pos": pred_pos,
            "loc_hit": loc_hit,
            "visited": visited,
            "reads": reads,
            "wall_seconds": float(ep.get("costs", {}).get("wall_seconds", ep.get("wall_seconds", 0.0))),
        }

    pos_res = [_eval_episode(by_sid[s], True) for s in pos_sids]
    neg_res = [_eval_episode(by_sid[s], False) for s in neg_sids]

    n_pos = len(pos_res)
    n_neg = len(neg_res)
    n_tot = n_pos + n_neg

    loc_hits = sum(int(r["loc_hit"]) for r in pos_res)
    vis_hits = sum(int(r["visited"]) for r in pos_res)
    pos_rec = sum(int(r["pred_pos"]) for r in pos_res)
    neg_spec = sum(int(not r["pred_pos"]) for r in neg_res)

    return {
        "benchmark": "verified_camelyon16",
        "matched_slides": n_tot,
        "expected_slides": len(cases),
        "positive_slides": n_pos,
        "negative_slides": n_neg,
        "pos_loc_hit_pct": round(100.0 * loc_hits / max(1, n_pos), 2),
        "pos_visited_hit_pct": round(100.0 * vis_hits / max(1, n_pos), 2),
        "stop_conversion_pct": round(100.0 * loc_hits / max(1, vis_hits), 2) if vis_hits else 0.0,
        "pos_slide_recall_pct": round(100.0 * pos_rec / max(1, n_pos), 2),
        "neg_specificity_pct": round(100.0 * neg_spec / max(1, n_neg), 2),
        "strict_loc_plus_spec_pct": round(100.0 * (loc_hits + neg_spec) / max(1, n_tot), 2),
        "slide_accuracy_pct": round(100.0 * (pos_rec + neg_spec) / max(1, n_tot), 2),
        "avg_pos_reads": round(float(np.mean([r["reads"] for r in pos_res])) if pos_res else 0.0, 2),
        "avg_neg_reads": round(float(np.mean([r["reads"] for r in neg_res])) if neg_res else 0.0, 2),
        "avg_latency_sec": round(
            float(np.mean([r["wall_seconds"] for r in pos_res + neg_res])) if n_tot else 0.0, 2
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score JevNavAgent benchmark predictions.")
    parser.add_argument(
        "--benchmark",
        choices=["wsivqa_351", "wsivqa_val", "wsivqa_test", "cam16_dev", "cam16_test"],
        required=True,
    )
    parser.add_argument("--manifest", nargs="+", required=True, help="One or more manifest JSON paths.")
    parser.add_argument("--predictions", nargs="+", required=True, help="One or more prediction JSONL paths.")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to save summary JSON.")
    args = parser.parse_args(argv)

    if args.benchmark.startswith("wsivqa"):
        summary = score_wsivqa_predictions(args.manifest, args.predictions)
    else:
        summary = score_cam16_predictions(args.manifest, args.predictions)

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
