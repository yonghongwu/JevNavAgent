"""Stage-1 Counterfactual Trajectory Alignment SFT entrypoint (`jevnav-train-sft`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevnav.data.dataset import CounterfactualTrajectoryDataset


def export_parquet_to_jsonl(parquet_path: Path, out_jsonl: Path, assets_root: Path | None = None) -> int:
    """Materialize a HuggingFace trajectory Parquet file into JSONL for `train_jev.py`."""
    ds = CounterfactualTrajectoryDataset(parquet_path)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for i in range(len(ds)):
            rec = ds[i]
            imgs = rec.pop("images_relpaths")
            if assets_root is not None:
                imgs = [str(assets_root / x) for x in imgs]
            rec["images"] = imgs
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(ds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage-1 Counterfactual Trajectory Alignment SFT for JevNavAgent."
    )
    parser.add_argument("--train-data", type=Path, required=True, help="Path to formal_terra_v5_train.parquet or .jsonl")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=None, help="Root dir for crop images if using Parquet.")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--overview-dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    train_jsonl = args.train_data
    if train_jsonl.suffix == ".parquet":
        train_jsonl = args.output_dir / "prepared_train.jsonl"
        n = export_parquet_to_jsonl(args.train_data, train_jsonl, assets_root=args.assets_root)
        print(f"[jevnav-train-sft] Exported {n} counterfactual transitions -> {train_jsonl}")

    import jevnav.core  # noqa: F401
    import train_jev

    train_argv = [
        "--train-jsonl", str(train_jsonl),
        "--backbone", args.backbone,
        "--output-dir", str(args.output_dir),
        "--max-steps", str(args.max_steps),
        "--lr", str(args.lr),
        "--grad-accum", str(args.grad_accum),
        "--overview-dropout-prob", str(args.overview_dropout),
        "--seed", str(args.seed),
        "--arch-contextual-choice",
        "--arch-prompt-recency",
    ]
    train_jev.main(train_argv)


if __name__ == "__main__":
    main()
