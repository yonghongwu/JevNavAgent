"""One-prefill restricted-letter MCQ answerer for local train/dev teachers.

The implementation reuses ARPolicy's frozen Qwen loader and scoring backend.
No generation, reference label, candidate geometry, or unread image is supplied.
``history_limit`` caps crops while always retaining the observed overview.
"""
from __future__ import annotations


class ChoiceLogitAnswerer:
    def __init__(self, base_model_path, device="cuda:0", max_pixels=262144, history_limit=3,
                 *, backend=None):
        if type(history_limit) is not int or history_limit < 0:
            raise ValueError("history_limit must be a nonnegative integer")
        if backend is None:
            from ar_policy import FrozenAnswerer
            backend = FrozenAnswerer.from_pretrained(base_model_path, device=device, max_pixels=max_pixels)
        self.backend, self.history_limit = backend, history_limit
        self.last_metrics, self.last_probabilities = {}, []

    def answer(self, state):
        choices = list(state["choices"])
        if not 1 <= len(choices) <= 26 or not all(isinstance(c, str) and c.strip() for c in choices):
            raise ValueError("Expected 1..26 nonempty answer choices")
        tokenizer = self.backend.processor.tokenizer
        ids = [tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(len(choices))]
        if any(len(value) != 1 for value in ids):
            raise ValueError("Each answer letter must have exactly one token")
        flattened = [value[0] for value in ids]
        if len(set(flattened)) != len(flattened) or getattr(tokenizer, "unk_token_id", None) in flattened:
            raise ValueError("Answer letter token IDs must be distinct and known")
        images = list(state.get("images", []))
        if "overview" in state:
            images = [state["overview"]] + images
        if not images:
            raise ValueError("The observed overview is required")
        crops = images[1:]
        selected = [images[0]] + (crops[-self.history_limit:] if self.history_limit else [])
        clean = {"question": state["question"], "choices": choices, "images": selected,
                 "views": [], "dimensions": [1, 1], "remaining_budget": 0, "candidates": []}
        answer = self.backend.answer(clean)
        if type(answer) is not int or not 0 <= answer < len(choices):
            raise ValueError("Answerer returned an invalid choice index")
        self.last_metrics = dict(self.backend.last_metrics)
        self.last_probabilities = list(getattr(self.backend, "last_probabilities", []))
        if len(self.last_probabilities) != len(choices):
            raise ValueError("Answerer did not return per-choice probabilities")
        self.last_metrics.update(option_probabilities=self.last_probabilities,
                                 observed_image_count=len(selected), omitted_history_crops=len(crops)-len(selected)+1)
        return answer

    def act(self, state):
        return {"type": "stop", "answer_index": self.answer(state)}

    __call__ = act
