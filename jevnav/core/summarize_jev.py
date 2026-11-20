"""Cluster-aware summaries retaining every predeclared evaluation attempt.

Bootstrap sampling units are patients (slide IDs when no patient ID exists),
with all questions and repeated navigation seeds retained inside each cluster.
Missing attempts count as failures for accuracy and qualified localization.
Measured costs report their observed count separately; missing costs are unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path

from evaluate_jev import sha256, write_json


def attempt_key(row):
    return (str(row["slide_id"]), str(row["question_id"]), int(row["budget"]), int(row["seed"]), row["method"])


def checkpoint_metadata(run):
    identity = run.get("identity", {})
    checkpoint = run.get("args", {}).get("checkpoint")
    metadata = {"checkpoint_id": identity.get("checkpoint_artifact_sha256") or checkpoint,
                "checkpoint_path": checkpoint, "training_seed": None, "training_seed_source": None}
    if not checkpoint:
        return metadata
    for filename in ("training_state.json", "training_config.json", "config.json"):
        path = Path(checkpoint) / filename
        if not path.is_file():
            continue
        for recorded_path, expected in identity.get("checkpoint_sha256", {}).items():
            if Path(recorded_path).resolve() == path.resolve() and sha256(path) != expected:
                raise ValueError(f"Training metadata changed since evaluation: {path}")
        value = json.loads(path.read_text())
        seed = value.get("training_seed", value.get("seed") if filename != "config.json" else None)
        if type(seed) is int:
            metadata.update(training_seed=seed, training_seed_source=str(path))
            break
    return metadata


def load_run(directory):
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text())
    metadata = checkpoint_metadata(run)
    run["training_metadata"] = metadata
    result_path = directory / "results.jsonl"
    results = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()] if result_path.exists() else []
    actual = {}
    for row in results:
        key = attempt_key(row)
        if key in actual:
            raise ValueError(f"Duplicate attempt in {directory}: {key}")
        actual[key] = row
    expected = run["expected_attempts"]
    expected_keys = [attempt_key(r) for r in expected]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError(f"Duplicate planned attempt in {directory}")
    extra = set(actual) - set(expected_keys)
    if extra:
        raise ValueError(f"Unplanned evaluation attempts in {directory}: {sorted(extra)}")
    rows = []
    for plan in expected:
        row = dict(plan)
        row.update(actual.get(attempt_key(plan), {"status": "missing", "correct": False,
                                                "evidence": {"success": False}, "costs": {}}))
        row["patient_id"] = plan.get("patient_id", plan["slide_id"])
        row["has_region_reference"] = bool(plan.get("has_region_reference", False))
        row["run_path"] = str(directory)
        row.update(metadata)
        rows.append(row)
    return run, rows


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def cluster_interval(rows, value_fn, samples=2000, seed=17029):
    if not rows:
        return {"estimate": None, "ci95": None, "attempts": 0, "clusters": 0}
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("patient_id", row["slide_id"]))].append(float(value_fn(row)))
    clusters = list(grouped.values())
    all_values = [v for cluster in clusters for v in cluster]
    result = {"estimate": sum(all_values) / len(all_values), "attempts": len(all_values),
              "clusters": len(clusters), "ci95": None}
    if len(clusters) < 2 or samples <= 0:
        result["ci_note"] = "At least two independent clusters are needed for a bootstrap interval."
        return result
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [rng.choice(clusters) for _ in clusters]
        draws.append(sum(sum(c) for c in selected) / sum(len(c) for c in selected))
    result["ci95"] = [percentile(draws, .025), percentile(draws, .975)]
    return result


def patient_macro_interval(rows, value_fn, samples=2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("patient_id", row["slide_id"]))].append(float(value_fn(row)))
    patients = [{"slide_id": patient, "patient_id": patient, "value": sum(values)/len(values)}
                for patient, values in grouped.items()]
    result = cluster_interval(patients, lambda row: row["value"], samples)
    result["source_attempts"] = len(rows)
    return result


def summarize(rows, bootstrap_samples=2000):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = attempt_key(row)
        if key in seen:
            raise ValueError(f"Attempt occurs in more than one supplied run: {key}")
        seen.add(key)
        grouped[(row["method"], row["budget"])].append(row)
    groups = []
    for (method, budget), attempts in sorted(grouped.items()):
        evidence = [r for r in attempts if r.get("has_region_reference")]
        statuses = Counter(r.get("status", "unknown") for r in attempts)
        group = {"method": method, "budget": budget, "planned_attempts": len(attempts),
                 "observed_attempts": len(attempts) - statuses["missing"],
                 "slides": len({r["slide_id"] for r in attempts}),
                 "questions": len({(r["slide_id"], r["question_id"]) for r in attempts}),
                 "seeds": sorted({r["seed"] for r in attempts}), "statuses": dict(statuses),
                 "mcq_accuracy": cluster_interval(attempts, lambda r: bool(r.get("correct")), bootstrap_samples),
                 "patient_macro_accuracy": patient_macro_interval(attempts, lambda r: bool(r.get("correct")), bootstrap_samples),
                 "qualified_localization_success": cluster_interval(evidence,
                        lambda r: bool(r.get("evidence", {}).get("success")), bootstrap_samples),
                 "oracle_best_visited_success_diagnostic": cluster_interval(evidence,
                        lambda r: bool(r.get("evidence", {}).get("oracle_best_visited_success")), bootstrap_samples),
                 "costs": {}}
        group["checkpoint_ids"] = sorted({r["checkpoint_id"] for r in attempts if r.get("checkpoint_id")})
        group["training_seed_unknown_attempts"] = sum(r.get("training_seed") is None for r in attempts)
        group["per_training_seed"] = []
        for training_seed in sorted({r["training_seed"] for r in attempts if r.get("training_seed") is not None}):
            selected = [r for r in attempts if r.get("training_seed") == training_seed]
            selected_evidence = [r for r in selected if r.get("has_region_reference")]
            group["per_training_seed"].append({"training_seed": training_seed,
                "training_seed_sources": sorted({r["training_seed_source"] for r in selected if r.get("training_seed_source")}),
                "checkpoint_ids": sorted({r["checkpoint_id"] for r in selected if r.get("checkpoint_id")}),
                "evaluation_seeds": sorted({r["seed"] for r in selected}),
                "mcq_accuracy": cluster_interval(selected, lambda r: bool(r.get("correct")), bootstrap_samples),
                "patient_macro_accuracy": patient_macro_interval(selected, lambda r: bool(r.get("correct")), bootstrap_samples),
                "qualified_localization_success": cluster_interval(selected_evidence,
                    lambda r: bool(r.get("evidence", {}).get("success")), bootstrap_samples)})
        cost_names = sorted({key for row in attempts for key, value in row.get("costs", {}).items()
                             if type(value) in (int, float)})
        for name in cost_names:
            values = [r["costs"][name] for r in attempts if type(r.get("costs", {}).get(name)) in (int, float)]
            group["costs"][name] = {"measured_attempts": len(values), "unknown_attempts": len(attempts)-len(values),
                                     "total_measured": sum(values), "mean_measured": sum(values)/len(values)}
        group["reference_missing_attempts"] = sum(r.get("reference_valid") is False for r in attempts)
        groups.append(group)
    paired = []
    for budget in sorted({b for _, b in grouped}):
        methods = sorted(m for m, b in grouped if b == budget)
        for left_index, left in enumerate(methods):
            for right in methods[left_index+1:]:
                def pairing(row):
                    return str(row["slide_id"]), str(row["question_id"]), int(row["seed"])
                a = {pairing(r): r for r in grouped[(left, budget)]}
                b = {pairing(r): r for r in grouped[(right, budget)]}
                shared = sorted(set(a) & set(b))
                differences, evidence_differences = [], []
                for key in shared:
                    if a[key].get("patient_id") != b[key].get("patient_id"):
                        raise ValueError(f"Patient identity differs across paired methods: {key}")
                    differences.append({"slide_id": a[key]["slide_id"], "patient_id": a[key].get("patient_id"),
                                        "difference": int(bool(a[key].get("correct"))) - int(bool(b[key].get("correct")))})
                    if a[key].get("has_region_reference") and b[key].get("has_region_reference"):
                        evidence_differences.append({"slide_id": a[key]["slide_id"], "patient_id": a[key].get("patient_id"),
                            "difference": int(bool(a[key].get("evidence", {}).get("success"))) -
                                          int(bool(b[key].get("evidence", {}).get("success")))})
                paired.append({"left": left, "right": right, "budget": budget,
                               "mcq_accuracy_difference": cluster_interval(differences, lambda r: r["difference"], bootstrap_samples),
                               "patient_macro_accuracy_difference": patient_macro_interval(differences, lambda r: r["difference"], bootstrap_samples),
                               "qualified_localization_success_difference": cluster_interval(evidence_differences,
                                    lambda r: r["difference"], bootstrap_samples),
                               "left_unpaired_attempts": len(a)-len(shared), "right_unpaired_attempts": len(b)-len(shared)})
    return {"groups": groups, "paired_comparisons": paired,
            "uncertainty": "95% patient/slide cluster percentile bootstrap; all QA and seeds stay together",
            "missing_policy": "Every planned attempt stays in the denominator; missing/invalid/errors are incorrect.",
            "region_reference_policy": "Region evidence metrics apply only to annotation-backed cases; WSI QA has no region gold.",
            "cost_policy": "Raw native read pixels are known; thumbnail native decode pixels are unavailable unless explicitly instrumented."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    runs, rows = [], []
    for path in args.runs:
        run, loaded = load_run(path)
        runs.append({"path": path, "status": run["status"], "identity": run["identity"],
                     "training_metadata": run["training_metadata"]})
        rows.extend(loaded)
    result = summarize(rows, args.bootstrap_samples)
    result["runs"] = runs
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
