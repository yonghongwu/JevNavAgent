"""Train matched-state JEV or AR adapters, with deterministic resume and cutoffs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch
from jev_model import (JevModel, canonical_target, choose_decision, decision_loss, encode_state,
                       load_backbone, make_processor, validate_target, backbone_identity)
from ar_policy import ar_loss, encode_ar


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def read_records(path, split):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not rows or any(r.get("split") != split for r in rows):
        raise ValueError(f"{path} must contain nonempty {split}-only records")
    for row in rows:
        validate_target(row)
        if not row.get("slide_id") or not row.get("state_id"):
            raise ValueError("Every record needs slide_id and state_id")
    return rows, hashlib.sha256(raw).hexdigest()


def deadline_timestamp(value):
    if not value:
        return float("inf")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Deadline needs an explicit timezone")
    return parsed.timestamp()


def compute_loss(model, processor, row, mode, device):
    if mode == "jev":
        inputs, extra, meta = encode_state(
            processor,
            row,
            device,
            arch_contextual_choice=getattr(model, "arch_contextual_choice", False),
            arch_prompt_recency=getattr(model, "arch_prompt_recency", False),
        )
        logits = model(inputs, **extra)
        loss, components = decision_loss(logits, row)
        correct = choose_decision(logits, row) == canonical_target(row)
        return loss, components, meta, correct
    inputs, positions, labels, meta = encode_ar(processor, row, device)
    loss = ar_loss(model, inputs, positions, labels)
    return loss, {"ar_loss": float(loss.detach())}, meta, None


@torch.inference_mode()
def evaluate(model, processor, rows, mode, device, deadline):
    model.eval()
    losses, correct = [], []
    for row in rows:
        if time.time() >= deadline:
            return {"complete": False, "count": len(losses)}
        loss, _, _, hit = compute_loss(model, processor, row, mode, device)
        losses.append(float(loss))
        if hit is not None:
            correct.append(hit)
    return {"complete": True, "count": len(losses), "loss": sum(losses)/len(losses),
            "joint_accuracy": sum(correct)/len(correct) if correct else None}


def save_checkpoint(path, model, processor, optimizer, state, mode):
    path = Path(path)
    temporary = path.with_name(path.name+".incomplete")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completed checkpoint {path}")
    temporary.mkdir(parents=True, exist_ok=True)
    if mode == "jev":
        model.save_pretrained(temporary)
    else:
        model.save_pretrained(temporary/"adapter")
    processor.save_pretrained(temporary/"processor")
    rng = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save({"optimizer": optimizer.state_dict(), "rng": rng}, temporary/"optimizer.pt")
    atomic_json(temporary/"training_state.json", state)
    temporary.rename(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["jev", "ar"], default="jev")
    p.add_argument("--model", default="artifacts/models/qwen3-vl-2b")
    p.add_argument("--records", required=True)
    p.add_argument("--dev-records")
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", help="Completed checkpoint directory, restores optimizer and RNG")
    p.add_argument("--init-checkpoint", help="Initialize weights for a new data stage; fresh optimizer and step count")
    p.add_argument("--steps", type=int, default=1000, help="Total update target including resumed updates")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--milestones", default="100,500,1000")
    p.add_argument("--deadline", help="Absolute ISO8601 UTC cutoff; checked before every update/eval state")
    p.add_argument("--max-pixels", type=int, default=262144)
    p.add_argument("--attention", default="eager")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--empty-cache-every", type=int, default=0,
                   help="Release unused CUDA cache periodically for variable-length MetaX workloads")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    deadline = deadline_timestamp(args.deadline)
    if time.time() >= deadline:
        raise RuntimeError("Training deadline has already passed")
    if args.steps < 1 or args.save_every < 1 or args.max_pixels < 1024:
        raise ValueError("Invalid positive step/save/pixel settings")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows, digest = read_records(args.records, "train")
    dev_rows, dev_digest = read_records(args.dev_records, "dev") if args.dev_records else ([], None)
    if {r["slide_id"] for r in rows} & {r["slide_id"] for r in dev_rows}:
        raise ValueError("Train and dev slides overlap")
    if args.dev_limit:
        dev_rows = sorted(dev_rows, key=lambda r: r["state_id"])[:args.dev_limit]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise FileExistsError("Fresh training output must be empty; use --resume for a checkpoint")
    resume_state = json.loads((Path(args.resume)/"training_state.json").read_text()) if args.resume else {}
    if args.resume:
        for k, v in {"data_sha256": digest, "dev_sha256": dev_digest, "mode": args.mode,
                     "base_model": str(Path(args.model).resolve()), "seed": args.seed,
                     "max_pixels": args.max_pixels}.items():
            if resume_state.get(k) != v:
                raise ValueError(f"Resume configuration mismatch: {k}")
    processor = make_processor(args.model, args.max_pixels)
    weight_checkpoint = args.resume or args.init_checkpoint
    model = (JevModel.from_pretrained(args.model, weight_checkpoint, args.device, args.attention, trainable=True)
             if args.mode == "jev" else load_backbone(args.model, args.device, args.attention,
                 str(Path(weight_checkpoint)/"adapter") if weight_checkpoint else None, trainable=True))
    backbone = model.backbone if args.mode == "jev" else model
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
    if args.resume:
        saved = torch.load(Path(args.resume)/"optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"]["torch"])
        random.setstate(saved["rng"]["python"])
        if "cuda" in saved["rng"]:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    config = vars(args) | {"data_sha256": digest, "dev_sha256": dev_digest,
        "backbone_identity": backbone_identity(args.model),
        "train_records": len(rows), "dev_records_count": len(dev_rows),
        "trainable_parameters": sum(p.numel() for p in trainable), "vision_frozen": True,
        "sampling": "uniform slides then uniform states, independent RNG(seed+step*1009)",
        "selection": "minimum dev supervised loss; root selects formal rollout checkpoint independently",
        "candidate_feature_dim": 12, "max_observed_crop_images": 3, "prompt_decimal_places": 4,
        "feature_transform": model.feature_transform if args.mode == "jev" else None,
        "lora_r": 16, "lora_alpha": 32}
    atomic_json(output/("resume_config.json" if args.resume else "config.json"), config)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["slide_id"], []).append(row)
    slide_ids = sorted(grouped)
    milestones = {int(x) for x in args.milestones.split(",") if x.strip()}
    step = int(resume_state.get("step", 0))
    totals = resume_state.get("totals", {"input_tokens": 0, "image_tokens": 0, "action_tokens": 0})
    previous_elapsed = float(resume_state.get("elapsed_seconds", 0))
    start = time.monotonic()
    best_loss = resume_state.get("best_dev_loss")
    latest_checkpoint = args.resume

    def checkpoint():
        nonlocal best_loss, latest_checkpoint
        dev = evaluate(model, processor, dev_rows, args.mode, args.device, deadline) if dev_rows else None
        is_best = bool(dev and dev["complete"] and (best_loss is None or dev["loss"] < best_loss))
        if is_best:
            best_loss = dev["loss"]
        state = {"step": step, "totals": totals, "elapsed_seconds": previous_elapsed+time.monotonic()-start,
                 "data_sha256": digest, "dev_sha256": dev_digest, "mode": args.mode,
                 "base_model": str(Path(args.model).resolve()), "seed": args.seed,
                 "max_pixels": args.max_pixels, "best_dev_loss": best_loss, "dev": dev}
        path = output/f"checkpoint-{step}"
        save_checkpoint(path, model, processor, optimizer, state, args.mode)
        latest_checkpoint = str(path.resolve())
        atomic_json(output/"latest.json", {"checkpoint": latest_checkpoint, "step": step})
        if is_best:
            atomic_json(output/"best_dev.json", {"checkpoint": latest_checkpoint, "step": step, "dev": dev})

    while step < args.steps and time.time() < deadline:
        next_step = step+1
        rng = random.Random(args.seed+next_step*1009)
        sid = rng.choice(slide_ids)
        row = rng.choice(grouped[sid])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_start = time.perf_counter()
        loss, components, meta, correct = compute_loss(model, processor, row, args.mode, args.device)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite loss at step {next_step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError(f"Nonfinite gradients at step {next_step}")
        optimizer.step()
        step = next_step
        for key in totals:
            totals[key] += meta.get(key, 0)
        log = {"step": step, "state_id": row["state_id"], "slide_id": sid, "task": row.get("task"),
               "loss": float(loss.detach()), **components, **meta, "train_joint_correct": correct,
               "gradient_norm": float(norm), "update_seconds": time.perf_counter()-update_start,
               "elapsed_seconds": previous_elapsed+time.monotonic()-start, "totals": dict(totals)}
        if str(args.device).startswith("cuda"):
            log["peak_allocated_gb"] = torch.cuda.max_memory_allocated(args.device)/1e9
        with (output/"learning_curve.jsonl").open("a") as handle:
            handle.write(json.dumps(log)+"\n")
        print(json.dumps(log), flush=True)
        if (args.empty_cache_every and str(args.device).startswith("cuda")
                and step % args.empty_cache_every == 0):
            torch.cuda.empty_cache()
        if step % args.save_every == 0 or step in milestones or step == args.steps:
            checkpoint()
    if step > int(resume_state.get("step", 0)) and not (output/f"checkpoint-{step}").exists():
        checkpoint()
    atomic_json(output/"completion.json", {"steps": step, "requested_steps": args.steps,
         "stopped_at_deadline": time.time() >= deadline, "checkpoint": latest_checkpoint,
         "totals": totals, "elapsed_seconds": previous_elapsed+time.monotonic()-start})


if __name__ == "__main__":
    main()
