"""Single-forward, variable-action/variable-choice Qwen3-VL decision policy.

Only ``public_state`` fields enter the model. Images are overview followed by
observed crops. Candidate centers are normalized; boxes use level-0 pixels.
"""
from __future__ import annotations

import json
import hashlib
from functools import lru_cache
import math
from pathlib import Path
import time

from PIL import Image, PngImagePlugin
import torch
from torch import nn
from torch.nn import functional as F

# Allow large scanner ICC color profiles embedded in TCGA Aperio SVS thumbnails (e.g. TCGA-AC-A4ZE).
PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024


PUBLIC_KEYS = ("task", "question", "choices", "images", "views", "dimensions",
               "remaining_budget", "decisions_remaining", "mpp", "candidates", "action_history", "last_tool_error")
CANDIDATE_FEATURE_DIM = 12
MAX_CROP_HISTORY = 3
VIEW_KEYS = ("bbox_level0", "downsample", "center", "action", "candidate_index", "type")


@lru_cache(maxsize=8)
def backbone_identity(path):
    """Identify the acquired snapshot without repeatedly hashing model weights."""
    root = Path(path).resolve()
    metadata_path = root/"acquisition_metadata.json"
    if not metadata_path.exists():
        return {"path": str(root), "snapshot_metadata_available": False}
    raw = metadata_path.read_bytes()
    metadata = json.loads(raw)
    return {"path": str(root), "repo_id": metadata.get("repo_id"), "revision": metadata.get("revision"),
            "acquisition_metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "identity_kind": "acquisition metadata; model weights not rehashed"}


def compact_prompt_numbers(value):
    """Bound digit tokens; feature-head tensors retain their original precision."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Public prompt contains a nonfinite number")
        rounded = round(value, 4)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, dict):
        return {k: compact_prompt_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_prompt_numbers(v) for v in value]
    return value


def public_state(row):
    """Explicit whitelist, including nested objects; never expose teacher labels."""
    result = {k: row[k] for k in PUBLIC_KEYS if k in row}
    if "overview" in row:
        result["images"] = [row["overview"]] + list(row.get("images", []))
    images = list(result.get("images", []))
    if len(images) > MAX_CROP_HISTORY+1:
        result["images"] = images[:1] + images[-MAX_CROP_HISTORY:]
    result["views"] = [{k: v[k] for k in VIEW_KEYS if k in v} for v in row.get("views", [])]
    result["action_history"] = [{k: v[k] for k in VIEW_KEYS if k in v}
                                for v in row.get("action_history", []) if isinstance(v, dict)]
    result["candidates"] = [{"index": i, **{k: c[k] for k in
                            ("action", "center", "downsample", "bbox_level0", "features") if k in c}}
                            for i, c in enumerate(row.get("candidates", []))]
    return result


def candidate_features(row):
    w, h = map(float, row["dimensions"])
    if w <= 0 or h <= 0:
        raise ValueError("Slide dimensions must be positive")
    features = []
    for candidate in row.get("candidates", []):
        cx, cy = map(float, candidate["center"])
        ds = float(candidate["downsample"])
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and ds > 0):
            raise ValueError("Candidates need normalized centers and positive downsample")
        box = candidate.get("bbox_level0", [cx*w, cy*h, cx*w, cy*h])
        x0, y0, x1, y1 = map(float, box)
        if x1 < x0 or y1 < y0:
            raise ValueError("Candidate bbox is inverted")
        values = [float(v) for v in candidate["features"]]
        if len(values) != CANDIDATE_FEATURE_DIM:
            raise ValueError(f"Expected {CANDIDATE_FEATURE_DIM} canonical candidate features")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite candidate features")
        features.append(values)
    tensor = torch.tensor(features, dtype=torch.float32).reshape(-1, CANDIDATE_FEATURE_DIM)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Candidate features overflow float32")
    return tensor


def validate_target(row):
    target = row["target"]
    if target["type"] == "read":
        index = target.get("candidate_index")
        if type(index) is not int or not 0 <= index < len(row.get("candidates", [])):
            raise ValueError("Read target must select an available candidate")
        if float(row["remaining_budget"]) <= 0:
            raise ValueError("Read target exceeds the remaining budget")
    elif target["type"] == "stop":
        index = target.get("answer_index")
        if type(index) is not int or not 0 <= index < len(row["choices"]):
            raise ValueError("Stop target must select an available answer")
    else:
        raise ValueError("Unknown target type")
    return target


def canonical_target(row):
    t = validate_target(row)
    key = "candidate_index" if t["type"] == "read" else "answer_index"
    return {"type": t["type"], key: t[key]}


def state_messages(row, arch_contextual_choice=False, arch_prompt_recency=False):
    state = public_state(row)
    paths = state.pop("images", [])
    if not paths:
        raise ValueError("A state must include an observed overview image")
    if not state.get("choices"):
        raise ValueError("A state must include answer choices")
    candidate_features(state)
    views = state.get("views", [])
    content = []
    for i, path in enumerate(paths):
        history_index = max(0, len(views)-(len(paths)-1))+i
        if i == 0:
            header = "Overview (whole-slide thumbnail):" if arch_prompt_recency else "Overview:"
        else:
            v_idx = history_index - 1
            if arch_prompt_recency and 0 <= v_idx < len(views):
                v = views[v_idx]
                ds = compact_prompt_numbers(v.get("downsample"))
                ctr = compact_prompt_numbers(v.get("center"))
                header = f"Observed crop {history_index} (downsample={ds}, center={ctr}):"
            else:
                header = f"Observed crop {history_index}:"
        content.append({"type": "text", "text": header})
        if isinstance(path, Image.Image):
            image = path.convert("RGB")
        else:
            with Image.open(Path(path)) as source:
                image = source.convert("RGB").copy()
        content.append({"type": "image", "image": image})
    prompt_state = {k: v for k, v in state.items() if k not in ("remaining_budget", "decisions_remaining")}
    text = ("Inspect only the supplied observed images and public state. Candidate and answer indices "
            "are zero-based. Select READ to acquire a listed candidate when further visual evidence is needed, "
            "or STOP and answer the multiple-choice question when observed evidence is sufficient. State:\n" +
            json.dumps(compact_prompt_numbers(prompt_state), ensure_ascii=False, separators=(",", ":")) +
            '\nDecision JSON format: {"type":"read","candidate_index":0} or '
            '{"type":"stop","answer_index":0}.')
    if arch_prompt_recency:
        visited_scales = [compact_prompt_numbers(v.get("downsample")) for v in views if "downsample" in v]
        letters_map = {chr(65 + i): c for i, c in enumerate(state["choices"])}
        text += (
            f"\nVisited crop downsamples: {json.dumps(visited_scales)}. "
            f"Question: {state.get('question', '')}\n"
            f"Options: {json.dumps(letters_map, ensure_ascii=False, separators=(',', ':'))}\n"
            "Answer letter:"
        )
    elif arch_contextual_choice:
        letters_map = {chr(65 + i): c for i, c in enumerate(state["choices"])}
        text += f"\nOptions: {json.dumps(letters_map, ensure_ascii=False, separators=(',', ':'))}\nAnswer letter:"
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def encode_state(processor, row, device, arch_contextual_choice=False, arch_prompt_recency=False):
    row = public_state(row)
    messages = state_messages(row, arch_contextual_choice=arch_contextual_choice, arch_prompt_recency=arch_prompt_recency)
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(device)
    choices = processor.tokenizer(list(row["choices"]), add_special_tokens=False, padding=True,
                                  return_tensors="pt").to(device)
    if not bool((choices["attention_mask"].sum(-1) > 0).all()):
        raise ValueError("Answer choices must contain tokens")
    extra = {"candidate_features": candidate_features(row).to(device),
             "choice_ids": choices["input_ids"], "choice_mask": choices["attention_mask"]}
    if arch_contextual_choice or arch_prompt_recency:
        n_choices = len(row["choices"])
        letter_ids = [processor.tokenizer.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(n_choices)]
        extra["letter_token_ids"] = torch.tensor(letter_ids, dtype=torch.long, device=device)
    meta = {"input_tokens": int(inputs["input_ids"].numel()),
            "image_tokens": int((inputs["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()),
            "image_count": len(row["images"]), "candidate_count": len(row.get("candidates", []))}
    return inputs, extra, meta


def load_backbone(path, device="cuda:0", attention="eager", adapter=None, trainable=True):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    base = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=dtype,
                attn_implementation=attention, local_files_only=True).to(device)
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    else:
        # Regex excludes every visual module, even if future vision blocks use q_proj.
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none", task_type="CAUSAL_LM",
            target_modules=r".*language_model\.layers\.[0-9]+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")
        model = get_peft_model(base, config)
    if any(p.requires_grad for n, p in model.named_parameters() if "visual" in n):
        raise RuntimeError("Vision parameters must remain frozen")
    return model


def make_processor(path, max_pixels=262144):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path, min_pixels=min(16384, max_pixels),
                                        max_pixels=max_pixels, local_files_only=True)


class JevModel(nn.Module):
    def __init__(self, backbone, head_dim=256, feature_transform="tanh_relative_offsets",
                 arch_contextual_choice=False, arch_prompt_recency=False):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.text_config.hidden_size
        self.head_dim = head_dim
        if feature_transform not in ("identity", "tanh_relative_offsets"):
            raise ValueError("Unknown candidate feature transform")
        self.feature_transform = feature_transform
        self.arch_contextual_choice = bool(arch_contextual_choice)
        self.arch_prompt_recency = bool(arch_prompt_recency)
        heads_dict = {
            "norm": nn.LayerNorm(hidden),
            "decision_type": nn.Linear(hidden, 2),
            "candidate_query": nn.Linear(hidden, head_dim),
            "candidate_key": nn.Sequential(nn.Linear(CANDIDATE_FEATURE_DIM, head_dim), nn.GELU(), nn.Linear(head_dim, head_dim)),
            "answer_query": nn.Linear(hidden, head_dim),
            "answer_key": nn.Linear(hidden, head_dim),
        }
        if self.arch_contextual_choice:
            heads_dict["contextual_gate"] = nn.Linear(hidden, 1)
        self.heads = nn.ModuleDict(heads_dict).float().to(next(backbone.parameters()).device)
        if self.arch_contextual_choice:
            nn.init.zeros_(self.heads["contextual_gate"].weight)
            nn.init.constant_(self.heads["contextual_gate"].bias, 0.0)

    def forward(self, inputs, candidate_features, choice_ids, choice_mask, letter_token_ids=None):
        # Exactly one multimodal transformer forward, without materializing full-sequence vocabulary logits.
        base = self.backbone.get_base_model()
        outputs = base.model(**inputs, use_cache=False, return_dict=True)
        positions = torch.arange(inputs["attention_mask"].shape[1], device=inputs["input_ids"].device)
        last = (positions * inputs["attention_mask"]).max(-1).values
        last_hidden = outputs.last_hidden_state[torch.arange(len(last), device=last.device), last]
        hidden = last_hidden.float()
        if hidden.shape[0] != 1:
            raise ValueError("Variable-action JevModel currently requires batch size one")
        h = self.heads["norm"](hidden[0])
        embedding = base.get_input_embeddings()(choice_ids).float()
        mask = choice_mask.unsqueeze(-1).float()
        choice_vectors = (embedding * mask).sum(1) / mask.sum(1).clamp_min(1)
        candidate_features = candidate_features.float()
        if self.feature_transform == "tanh_relative_offsets":
            candidate_features = candidate_features.clone()
            candidate_features[:, 8:10] = candidate_features[:, 8:10].tanh()
        scale = math.sqrt(self.head_dim)
        typed_answer_logits = self.heads["answer_key"](choice_vectors) @ self.heads["answer_query"](h) / scale
        if self.arch_contextual_choice and letter_token_ids is not None:
            lm_head_out = base.lm_head(last_hidden[0].to(base.lm_head.weight.dtype)).float()
            letter_logits = lm_head_out[letter_token_ids]
            letter_logits = letter_logits - letter_logits.mean()
            gate = torch.sigmoid(self.heads["contextual_gate"](h))
            answer_logits = (1.0 - gate) * typed_answer_logits + gate * letter_logits
        else:
            answer_logits = typed_answer_logits
        return {"type_logits": self.heads["decision_type"](h),
                "candidate_logits": self.heads["candidate_key"](candidate_features.float()) @ self.heads["candidate_query"](h) / scale,
                "answer_logits": answer_logits}

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "adapter")
        torch.save(self.heads.state_dict(), path / "heads.pt")
        cfg = {
            "format_version": 1,
            "head_dim": self.head_dim,
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "type_order": ["read", "stop"],
            "feature_transform": self.feature_transform,
            "prompt_decimal_places": 4,
            "max_observed_crop_images": MAX_CROP_HISTORY,
        }
        if self.arch_contextual_choice:
            cfg["arch_contextual_choice"] = True
        if self.arch_prompt_recency:
            cfg["arch_prompt_recency"] = True
        (path / "jev_config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def from_pretrained(cls, base_path, checkpoint=None, device="cuda:0", attention="eager", trainable=False,
                        arch_contextual_choice=None, arch_prompt_recency=None):
        config = json.loads((Path(checkpoint)/"jev_config.json").read_text()) if checkpoint else {}
        if checkpoint and config.get("candidate_feature_dim") != CANDIDATE_FEATURE_DIM:
            raise ValueError("Incompatible checkpoint candidate feature dimensions")
        backbone = load_backbone(base_path, device, attention,
                    str(Path(checkpoint)/"adapter") if checkpoint else None, trainable)
        transform = config.get("feature_transform", "identity") if checkpoint else "tanh_relative_offsets"
        use_ctx = config.get("arch_contextual_choice", False) if arch_contextual_choice is None else bool(arch_contextual_choice)
        use_rec = config.get("arch_prompt_recency", False) if arch_prompt_recency is None else bool(arch_prompt_recency)
        model = cls(backbone, config.get("head_dim", 256), transform,
                    arch_contextual_choice=use_ctx, arch_prompt_recency=use_rec)
        if checkpoint:
            state_dict = torch.load(Path(checkpoint)/"heads.pt", map_location=device, weights_only=True)
            model.heads.load_state_dict(state_dict, strict=not (use_ctx and "contextual_gate.weight" not in state_dict))
        return model


def decision_loss(logits, row):
    target = validate_target(row)
    device = logits["type_logits"].device
    is_read = target["type"] == "read"
    type_loss = F.cross_entropy(logits["type_logits"].unsqueeze(0), torch.tensor([0 if is_read else 1], device=device))
    key = "candidate" if is_read else "answer"
    index = target["candidate_index" if is_read else "answer_index"]
    index_loss = F.cross_entropy(logits[key+"_logits"].unsqueeze(0), torch.tensor([index], device=device))
    total_loss = type_loss + index_loss
    metrics = {"type_loss": float(type_loss.detach()), key+"_loss": float(index_loss.detach())}
    if is_read and type(target.get("answer_index")) is int and 0 <= target["answer_index"] < len(row["choices"]):
        aux_ans_loss = F.cross_entropy(
            logits["answer_logits"].unsqueeze(0),
            torch.tensor([target["answer_index"]], device=device),
        )
        total_loss = total_loss + aux_ans_loss
        metrics["aux_answer_loss"] = float(aux_ans_loss.detach())
    return total_loss, metrics


def _count_distinct_high_mag_regions(views):
    centers = []
    for v in views:
        if int(v.get("downsample", 64)) <= 4 and "center" in v:
            cx, cy = float(v["center"][0]), float(v["center"][1])
            if not any(math.hypot(cx - ox, cy - oy) < 0.06 for (ox, oy) in centers):
                centers.append((cx, cy))
    return len(centers)


def choose_decision(logits, row):
    cands = row.get("candidates") or []
    can_read = bool(cands) and float(row["remaining_budget"]) > 0
    views = row.get("views") or []
    choices = [str(c) for c in (row.get("choices") or [])]
    is_cam_task = any("Metastatic tumor is supported" in c for c in choices)

    ans_probs = logits["answer_logits"].detach().float().softmax(-1)
    pred_ans = int(ans_probs.argmax())
    is_read = can_read and int(logits["type_logits"].argmax()) == 0

    if is_cam_task:
        pos_idx = next((i for i, c in enumerate(choices) if "Metastatic tumor is supported" in c), 0)
        neg_idx = 1 - pos_idx
        pos_prob = float(ans_probs[pos_idx])
        stop_prob = float(logits["type_logits"].detach().float().softmax(-1)[1])
        high_mag_regions = _count_distinct_high_mag_regions(views)
        cur_ds = int(views[-1].get("downsample", 64)) if views else 64
        model_wants_stop = (stop_prob >= 0.50)
        if high_mag_regions <= 1:
            is_positive_evidence = (cur_ds <= 4 and pos_prob >= 0.885 and model_wants_stop)
        else:
            is_positive_evidence = (
                cur_ds <= 4
                and ((pos_prob >= 0.885 and model_wants_stop) or (stop_prob >= 0.86 and pos_prob >= 0.83))
            )
        pred_ans = pos_idx if is_positive_evidence else neg_idx
        if can_read:
            if cur_ds > 4:
                is_read = True
            elif is_positive_evidence:
                is_read = False
            elif high_mag_regions >= 4 and model_wants_stop:
                # Active STOP(negative) after inspecting 4 distinct high-mag regions (12 reads) with no tumor
                is_read = False
            else:
                is_read = True

    if is_read:
        cand_logits = logits["candidate_logits"].detach().clone()
        need_root_switch = (
            len(views) >= 2 and int(views[-1].get("downsample", 64)) <= 4 and int(views[-2].get("downsample", 64)) <= 4
        ) or (
            is_cam_task and len(views) >= 3 and int(views[-1].get("downsample", 64)) <= 4
        )
        if need_root_switch:
            visited_root_centers = [
                v["center"] for v in views if int(v.get("downsample", 0)) == 64 and "center" in v
            ]
            unvisited_root_idxs = []
            for i, c in enumerate(cands):
                if int(c.get("downsample", 0)) == 64 and len(c.get("features", [])) > 10 and float(c["features"][10]) > 0.5:
                    cx, cy = c["center"]
                    if not any(math.hypot(cx - vx, cy - vy) < 0.05 for (vx, vy) in visited_root_centers):
                        unvisited_root_idxs.append(i)
            if unvisited_root_idxs:
                mask = torch.full_like(cand_logits, float("-inf"))
                for i in unvisited_root_idxs:
                    mask[i] = cand_logits[i]
                cand_logits = mask
        return {"type": "read", "candidate_index": int(cand_logits.argmax())}
    return {"type": "stop", "answer_index": pred_ans}


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class JEVPolicy:
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model.eval(), processor, device
        self.last_metrics = {}

    @classmethod
    def from_pretrained(cls, base_path, checkpoint=None, device="cuda:0", max_pixels=262144, attention="eager"):
        return cls(JevModel.from_pretrained(base_path, checkpoint, device, attention),
                   make_processor(base_path, max_pixels), device)

    @torch.inference_mode()
    def _infer(self, state):
        synchronize(self.device)
        start = time.perf_counter()
        inputs, extra, meta = encode_state(
            self.processor,
            public_state(state),
            self.device,
            arch_contextual_choice=getattr(self.model, "arch_contextual_choice", False),
            arch_prompt_recency=getattr(self.model, "arch_prompt_recency", False),
        )
        logits = self.model(inputs, **extra)
        synchronize(self.device)
        self.last_metrics = meta | {"latency_seconds": time.perf_counter()-start,
                "model_forwards": 1, "generated_tokens": 0,
                "stop_probability": float(logits["type_logits"].softmax(-1)[1]),
                "answer_probabilities": logits["answer_logits"].softmax(-1).tolist()}
        return logits

    def act(self, state):
        return choose_decision(self._infer(state), state)

    def answer(self, state):
        """Use the same trained answer head for controlled navigation ablations."""
        return int(self._infer(state)["answer_logits"].argmax())

    __call__ = act
