"""Same-state autoregressive comparison, sharing the JEV public prompt."""
from __future__ import annotations
import json
from pathlib import Path
import re
import time
import torch
from torch.nn import functional as F
from jev_model import (canonical_target, load_backbone, make_processor, public_state,
                       state_messages, synchronize)


def encode_ar(processor, row, device):
    messages = state_messages(row)
    prefix = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt")
    target = json.dumps(canonical_target(row), separators=(",", ":"))
    complete = processor.apply_chat_template(messages + [{"role": "assistant", "content":
                   [{"type": "text", "text": target}]}], tokenize=True, add_generation_prompt=False,
                   return_dict=True, return_tensors="pt")
    n = prefix["input_ids"].shape[1]
    if not torch.equal(prefix["input_ids"], complete["input_ids"][:, :n]):
        raise ValueError("AR prompt tokens are not a prefix of the target conversation")
    full_len = complete["input_ids"].shape[1]
    # Compute vocabulary logits only for supervised positions, not visual/prompt tokens.
    positions = torch.arange(n-1, full_len-1, device=device)
    labels = complete["input_ids"][:, n:].to(device)
    meta = {"input_tokens": n, "action_tokens": full_len-n,
            "image_tokens": int((prefix["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()),
            "image_count": len(public_state(row)["images"]), "candidate_count": len(row.get("candidates", []))}
    return complete.to(device), positions, labels, meta


def ar_loss(model, inputs, positions, labels):
    result = model(**inputs, logits_to_keep=positions, use_cache=False)
    return F.cross_entropy(result.logits.float().reshape(-1, result.logits.shape[-1]), labels.reshape(-1))


def parse_decision(text, state):
    try:
        # Exact answer letters are accepted for MCQ robustness; never infer an
        # answer from free-form explanations or fuzzily match option text.
        letter = re.fullmatch(r"(?:Answer:\s*)?([A-Z])", text.strip(), re.IGNORECASE)
        if letter:
            action = {"type": "stop", "answer_index": ord(letter.group(1).upper())-65}
        else:
            action = json.loads(text.strip())
        kind = action["type"]
        key = "candidate_index" if kind == "read" else "answer_index"
        if kind not in ("read", "stop") or set(action) != {"type", key}:
            raise ValueError("Invalid action schema")
        index = action[key]
        if kind == "stop" and isinstance(index, str) and re.fullmatch(r"[A-Za-z]", index):
            index = ord(index.upper())-65
            action[key] = index
        count = len(state.get("candidates", [])) if kind == "read" else len(state["choices"])
        if type(index) is not int or not 0 <= index < count:
            raise ValueError("Invalid action index")
        if kind == "read" and float(state["remaining_budget"]) <= 0:
            raise ValueError("No read budget remains")
        return action
    except (ValueError, TypeError, KeyError):
        return {"type": "invalid", "raw_output": text}


class ARPolicy:
    def __init__(self, model, processor, device, max_new_tokens=48):
        self.model, self.processor, self.device = model.eval(), processor, device
        self.max_new_tokens = max_new_tokens
        self.last_metrics = {}

    @classmethod
    def from_pretrained(cls, base_path, checkpoint=None, device="cuda:0", max_pixels=262144,
                        attention="eager", max_new_tokens=48):
        adapter = str(Path(checkpoint)/"adapter") if checkpoint else None
        model = load_backbone(base_path, device, attention, adapter, trainable=False)
        return cls(model, make_processor(base_path, max_pixels), device, max_new_tokens)

    @torch.inference_mode()
    def act(self, state):
        synchronize(self.device)
        start = time.perf_counter()
        inputs = self.processor.apply_chat_template(state_messages(state), tokenize=True,
                   add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                     use_cache=True)
        suffix = output[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(suffix, skip_special_tokens=True)
        action = parse_decision(text, state)
        synchronize(self.device)
        self.last_metrics = {"latency_seconds": time.perf_counter()-start,
            "input_tokens": int(inputs["input_ids"].numel()), "generated_tokens": int(suffix.numel()),
            "model_forwards": int(suffix.numel()), "raw_output": text,
            "image_tokens": int((inputs["input_ids"] == self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()),
            "image_count": len(public_state(state)["images"]), "candidate_count": len(state.get("candidates", []))}
        return action

    __call__ = act


class FrozenAnswerer:
    """Shared one-forward zero-shot answerer for nonlearned navigation baselines."""
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model.eval(), processor, device
        self.last_metrics = {}
        self.last_probabilities = []

    @classmethod
    def from_pretrained(cls, base_path, device="cuda:0", max_pixels=262144, attention="eager"):
        from transformers import Qwen3VLForConditionalGeneration
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        model = Qwen3VLForConditionalGeneration.from_pretrained(base_path, dtype=dtype,
                   attn_implementation=attention, local_files_only=True).to(device)
        model.requires_grad_(False)
        return cls(model, make_processor(base_path, max_pixels), device)

    @torch.inference_mode()
    def answer(self, state):
        n = len(state["choices"])
        if not 1 <= n <= 26:
            raise ValueError("Single-letter answerer supports 1..26 choices")
        messages = state_messages(state)
        letters = [chr(65+i) for i in range(n)]
        instruction = ("The read budget is now exhausted. Answer the question with exactly one letter, "
                       "using this option mapping: " + json.dumps(dict(zip(letters, state["choices"]))) +
                       ". Output only the answer letter.")
        messages[0]["content"].append({"type": "text", "text": instruction})
        ids = [self.processor.tokenizer.encode(x, add_special_tokens=False) for x in letters]
        if any(len(x) != 1 for x in ids):
            raise ValueError("Option letters must each tokenize to a single token")
        token_ids = [x[0] for x in ids]
        if len(set(token_ids)) != n or self.processor.tokenizer.unk_token_id in token_ids:
            raise ValueError("Option token IDs must be distinct and known")
        synchronize(self.device)
        start = time.perf_counter()
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to(self.device)
        logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]
        option_logits = logits[torch.tensor(token_ids, device=self.device)].float()
        choice = int(option_logits.argmax())
        self.last_probabilities = option_logits.softmax(-1).tolist()
        synchronize(self.device)
        self.last_metrics = {"latency_seconds": time.perf_counter()-start,
            "input_tokens": int(inputs["input_ids"].numel()), "model_forwards": 1, "generated_tokens": 0,
            "option_probabilities": self.last_probabilities,
            "image_tokens": int((inputs["input_ids"] == self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()),
            "image_count": len(public_state(state)["images"])}
        return choice

    def act(self, state):
        return {"type": "stop", "answer_index": self.answer(state)}

    __call__ = act
