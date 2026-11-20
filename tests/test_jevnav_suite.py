"""End-to-end verification test suite for the JevNavAgent open-source package."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from PIL import Image

from jevnav.data.dataset import CounterfactualTrajectoryDataset, load_manifest_cases
from jevnav.eval.interface import (
    NavigationObservation,
    OpenAICompatiblePolicy,
    OverviewOnlyPolicy,
    RandomNavPolicy,
    TissuePeakNavPolicy,
)
from jevnav.eval.metrics import score_cam16_predictions, score_wsivqa_predictions
from jevnav.eval.runner import load_custom_policy, run_case_episode

STAGED_HF_DS = Path(os.environ.get("JEVNAV_TEST_DATA_DIR", "./data"))


def test_parquet_trajectory_dataset_loads_cleanly():
    pq_path = STAGED_HF_DS / "trajectories/formal_terra_v5_train.parquet"
    if not pq_path.exists():
        pytest.skip(f"Staged parquet not found at {pq_path}")
    ds = CounterfactualTrajectoryDataset(pq_path)
    assert len(ds) == 16635
    item = ds[0]
    assert item["slide_id"]
    assert isinstance(item["choices"], list) and len(item["choices"]) >= 2
    assert isinstance(item["candidates"], list)
    assert isinstance(item["target"], dict)
    assert all("/workspace/" not in p for p in item["images_relpaths"])


def test_official_wsivqa_scorer_matches_paper_numbers():
    val_m = STAGED_HF_DS / "manifests/wsivqa_clean_visual_val_179.json"
    test_m = STAGED_HF_DS / "manifests/wsivqa_clean_visual_test_172.json"
    pred_p = STAGED_HF_DS / "baseline_predictions/wsivqa_351/jevnav_traj_grpo_vqa_kl_B4.jsonl"
    if not pred_p.exists():
        pytest.skip("Staged baseline predictions not available")
    summary = score_wsivqa_predictions([val_m, test_m], [pred_p])
    assert summary["matched_cases"] == 351
    assert summary["val"]["count"] == 179
    assert summary["test"]["count"] == 172
    assert summary["macro_global"]["count"] == 207
    assert summary["micro_focal"]["count"] == 144
    assert summary["buckets"]["subtotal_A_to_D"]["count"] == 296
    assert summary["buckets"]["E_focal_descriptive"]["count"] == 55
    # Verify exact paper metrics for JevNavAgent + Traj-GRPO (VQA+KL)
    assert abs(summary["total"]["acc"] - 50.43) < 0.02
    assert abs(summary["micro_focal"]["acc"] - 51.39) < 0.02
    assert abs(summary["multiscale_query_gated"]["total"]["acc"] - 51.57) < 0.02


def test_official_cam16_scorer_runs_cleanly():
    test_m = STAGED_HF_DS / "manifests/cam16_verified_test_129.json"
    pred_p = STAGED_HF_DS / "baseline_predictions/cam16/jevnav_traj_grpo_cam16_test129_B12.jsonl"
    if not pred_p.exists():
        pytest.skip("Staged CAM16 predictions not available")
    summary = score_cam16_predictions([test_m], [pred_p])
    assert summary["matched_slides"] == 129
    assert summary["positive_slides"] == 49
    assert summary["negative_slides"] == 80


def test_custom_policy_template_loader():
    tmpl = Path(__file__).resolve().parent.parent / "examples/custom_policy_template.py"
    policy = load_custom_policy(f"{tmpl}:MyCustomPathologyPolicy")
    img = Image.new("RGB", (64, 64), color=(200, 150, 180))
    obs = NavigationObservation(
        question="Test question?",
        choices=["A", "B"],
        overview_image=img,
        crop_images=[],
        views=[],
        candidates=[{"id": 3, "downsample": 16, "center": [0.5, 0.5], "features": [0.0] * 7 + [0.9]}],
        dimensions=(10000, 10000),
        mpp=0.25,
        remaining_budget=2,
        decisions_remaining=3,
        status="active",
    )
    act = policy.act(obs)
    assert act.type == "read" and act.candidate_index == 0


def test_openai_compatible_policy_with_local_mock_server():
    responses = [
        '{"action": "read", "candidate_index": 0}',
        '{"action": "stop", "answer_index": 1}',
    ]
    call_idx = {"n": 0}

    class MockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            idx = min(call_idx["n"], len(responses) - 1)
            call_idx["n"] += 1
            body = {"choices": [{"message": {"content": responses[idx]}}]}
            raw = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        pol = OpenAICompatiblePolicy(
            model="mock-vlm",
            api_base=f"http://127.0.0.1:{port}/v1",
            api_key="test-key",
        )
        img = Image.new("RGB", (64, 64), color=(180, 120, 160))
        obs1 = NavigationObservation(
            question="Grade?",
            choices=["Grade I", "Grade III"],
            overview_image=img,
            crop_images=[],
            views=[],
            candidates=[{"id": 7, "downsample": 16, "center": [0.4, 0.5], "features": [0.0] * 7 + [0.8]}],
            dimensions=(50000, 50000),
            mpp=0.25,
            remaining_budget=3,
            decisions_remaining=4,
            status="active",
        )
        a1 = pol.act(obs1)
        assert a1.type == "read" and a1.candidate_index == 0

        obs2 = NavigationObservation(
            question="Grade?",
            choices=["Grade I", "Grade III"],
            overview_image=img,
            crop_images=[img],
            views=[{"downsample": 16}],
            candidates=[],
            dimensions=(50000, 50000),
            mpp=0.25,
            remaining_budget=0,
            decisions_remaining=1,
            status="active",
        )
        a2 = pol.act(obs2)
        assert a2.type == "stop" and a2.answer_index == 1
    finally:
        server.shutdown()
        server.server_close()


def test_real_wsi_closed_loop_navigation_smoke():
    """End-to-end real gigapixel WSI test using TissuePeakNavPolicy and RandomNavPolicy."""
    val_m = STAGED_HF_DS / "manifests/wsivqa_clean_visual_val_179.json"
    raw_dir = Path(os.environ.get("JEVNAV_TEST_SLIDES_DIR", str(STAGED_HF_DS / "slides/wsivqa")))
    if not val_m.exists() or not raw_dir.exists():
        pytest.skip("Real WSI slides not present in this environment")
    cases = load_manifest_cases([val_m])
    # Find first case whose slide is on disk
    found = None
    for c in cases:
        p = raw_dir / c["file_name"]
        if p.exists():
            found = (c, p)
            break
    if found is None:
        pytest.skip("No matching WSI found in raw_dir")
    case, slide_path = found
    ep = run_case_episode(TissuePeakNavPolicy(), case, slide_path, annotation_path=None, budget=2)
    assert ep["question_id"] == case["question_id"]
    assert ep["costs"]["read_count"] == 2
    assert len(ep["views"]) == 2
    assert len(ep["steps"]) == 3  # 2 READ + 1 STOP

    ep0 = run_case_episode(OverviewOnlyPolicy(), case, slide_path, annotation_path=None, budget=2)
    assert ep0["costs"]["read_count"] == 0
    assert len(ep0["steps"]) == 1
