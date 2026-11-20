"""A+B Hybrid GRPO (Group Relative Policy Optimization) for the JEV Agent.

Key design (combining Option A online episode rollout with Option B exact expectation):
1. Navigation steps (``READ`` vs ``STOP`` and ``candidate_index`` selection) are rolled out
   online in ``WSINavigationEnv`` across a group of ``G`` trajectories per slide/question (Option A).
2. At the terminal ``STOP`` step of each trajectory, instead of randomly sampling an answer choice
   ``m`` and punishing good navigation if a random guess misses, we evaluate the **exact analytic
   probability** ``P(m = y* | s_T, stop)`` from ``answer_logits`` (Option B):
   - Zero answer-sampling variance in the trajectory reward ``R(tau_i)`` credited to navigation steps;
   - Exact cross-entropy / expectation gradient on ``answer_logits`` at the terminal state.
3. Because JEV outputs a compact factorized discrete distribution over ``2`` action types, ``K``
   candidates, and ``M`` choices in a single forward pass, KL regularization ``KL(pi_theta || pi_ref)``
   is computed in closed form with zero Monte-Carlo variance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn import functional as F

import jev  # noqa: F401
from jev.env import WSINavigationEnv
from jev_model import JevModel, encode_state, make_processor, public_state


def masked_type_logits(type_logits: torch.Tensor, can_read: bool) -> torch.Tensor:
    """Mask ``type=read`` (index 0) to a large negative value when no reads are allowed."""
    if can_read:
        return type_logits
    masked = type_logits.clone()
    masked[0] = -1e9
    return masked


def categorical_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Exact analytic KL divergence KL(P || Q) between two categorical logit vectors."""
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def action_log_prob_and_kl(
    logits: dict[str, torch.Tensor],
    ref_logits: dict[str, torch.Tensor] | None,
    action: dict[str, Any],
    can_read: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute exact factorized log pi_theta(a|s) and closed-form KL(pi_theta || pi_ref).

    For ``a = {"type": "read", "candidate_index": k}``:
        log pi(a|s) = log P(type=read | s) + log P(cand=k | s, read)
    For ``a = {"type": "stop", "answer_index": m}``:
        log pi(a|s) = log P(type=stop | s) + log P(ans=m | s, stop)
    """
    t_logits = masked_type_logits(logits["type_logits"], can_read)
    log_p_type = F.log_softmax(t_logits, dim=-1)
    p_type = log_p_type.exp()

    act_type = action["type"]
    if act_type == "read":
        if not can_read:
            raise ValueError("Cannot compute log_prob for READ action when can_read is False")
        cand_idx = int(action["candidate_index"])
        log_p_cand = F.log_softmax(logits["candidate_logits"], dim=-1)
        log_prob = log_p_type[0] + log_p_cand[cand_idx]
    elif act_type == "stop":
        ans_idx = int(action.get("answer_index", 0))
        log_p_ans = F.log_softmax(logits["answer_logits"], dim=-1)
        log_prob = log_p_type[1] + log_p_ans[ans_idx]
    else:
        raise ValueError(f"Unknown action type: {act_type}")

    if ref_logits is None:
        kl = torch.zeros((), dtype=log_prob.dtype, device=log_prob.device)
        kl_parts = {"kl_type": 0.0, "kl_candidate": 0.0, "kl_answer": 0.0, "kl_total": 0.0}
    else:
        ref_t_logits = masked_type_logits(ref_logits["type_logits"].detach(), can_read)
        kl_type = categorical_kl(t_logits, ref_t_logits) if can_read else torch.zeros_like(log_prob)
        if can_read and logits["candidate_logits"].numel() > 0:
            kl_cand = categorical_kl(logits["candidate_logits"], ref_logits["candidate_logits"].detach())
        else:
            kl_cand = torch.zeros_like(log_prob)
        kl_ans = categorical_kl(logits["answer_logits"], ref_logits["answer_logits"].detach())
        # Hierarchical KL decomposition: KL(type) + P(read)*KL(cand) + P(stop)*KL(ans)
        kl = kl_type + p_type[0] * kl_cand + p_type[1] * kl_ans
        kl_parts = {
            "kl_type": float(kl_type.detach()),
            "kl_candidate": float(kl_cand.detach()),
            "kl_answer": float(kl_ans.detach()),
            "kl_total": float(kl.detach()),
        }

    return log_prob, kl, kl_parts


def nav_action_log_prob_and_kl(
    logits: dict[str, torch.Tensor],
    ref_logits: dict[str, torch.Tensor] | None,
    action: dict[str, Any],
    can_read: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Navigation-only log-prob & KL (decoupled from terminal answer choice sampling).

    - When ``action["type"] == "read"``: log P(type=read | s) + log P(cand=k | s, read)
    - When ``action["type"] == "stop"``: log P(type=stop | s)
      (the answer choice itself is trained via Option B exact expectation).
    """
    t_logits = masked_type_logits(logits["type_logits"], can_read)
    log_p_type = F.log_softmax(t_logits, dim=-1)
    p_type = log_p_type.exp()

    if action["type"] == "read":
        if not can_read:
            raise ValueError("Cannot evaluate READ when can_read=False")
        cand_idx = int(action["candidate_index"])
        log_p_cand = F.log_softmax(logits["candidate_logits"], dim=-1)
        nav_log_prob = log_p_type[0] + log_p_cand[cand_idx]
    elif action["type"] == "stop":
        nav_log_prob = log_p_type[1]
    else:
        raise ValueError(f"Unknown action type: {action['type']}")

    if ref_logits is None:
        nav_kl = torch.zeros((), dtype=nav_log_prob.dtype, device=nav_log_prob.device)
    else:
        ref_t_logits = masked_type_logits(ref_logits["type_logits"].detach(), can_read)
        kl_type = categorical_kl(t_logits, ref_t_logits) if can_read else torch.zeros_like(nav_log_prob)
        if can_read and logits["candidate_logits"].numel() > 0:
            kl_cand = categorical_kl(logits["candidate_logits"], ref_logits["candidate_logits"].detach())
        else:
            kl_cand = torch.zeros_like(nav_log_prob)
        nav_kl = kl_type + p_type[0] * kl_cand

    return nav_log_prob, nav_kl


def terminal_answer_expectation(
    logits: dict[str, torch.Tensor],
    gold_answer_index: int,
    ref_logits: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Option B exact expectation at terminal ``STOP`` states (zero answer-sampling variance).

    Returns:
    - ``gold_prob``: Exact probability P_theta(m = y* | s_T, stop)
    - ``greedy_correct``: 1.0 if argmax_m answer_logits == y* else 0.0
    - ``answer_ce_loss``: Differentiable cross-entropy loss -log P_theta(m = y* | s_T, stop)
    - ``answer_kl``: Exact KL(P_theta(m|s_T) || P_ref(m|s_T))
    """
    ans_logits = logits["answer_logits"]
    probs = F.softmax(ans_logits, dim=-1)
    gold_idx = int(gold_answer_index)
    if not (0 <= gold_idx < ans_logits.numel()):
        raise ValueError(f"gold_answer_index {gold_idx} out of range for {ans_logits.numel()} choices")

    gold_prob = float(probs[gold_idx].detach())
    pred_idx = int(ans_logits.argmax().detach())
    greedy_correct = 1.0 if pred_idx == gold_idx else 0.0
    ce_loss = F.cross_entropy(ans_logits.unsqueeze(0), torch.tensor([gold_idx], device=ans_logits.device))
    ans_kl = (
        categorical_kl(ans_logits, ref_logits["answer_logits"].detach())
        if ref_logits is not None
        else torch.zeros_like(ce_loss)
    )
    return {
        "gold_prob": gold_prob,
        "predicted_index": pred_idx,
        "greedy_correct": greedy_correct,
        "answer_ce_loss": ce_loss,
        "answer_kl": ans_kl,
    }


def compute_group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Group-normalize trajectory rewards: A_i = (R_i - mean(R)) / (std(R) + eps).

    If all rewards in the group are identical (std < eps), returns 0.0 for all trajectories.
    """
    if not rewards:
        return []
    t = torch.tensor(rewards, dtype=torch.float32)
    if t.numel() <= 1:
        return [0.0 for _ in rewards]
    mean = t.mean()
    std = t.std(unbiased=False)
    if float(std) < eps:
        return [0.0 for _ in rewards]
    adv = (t - mean) / (std + eps)
    return [float(x) for x in adv]


def hybrid_grpo_step_loss(
    logits: dict[str, torch.Tensor],
    old_nav_log_prob: float | torch.Tensor,
    ref_logits: dict[str, torch.Tensor] | None,
    action: dict[str, Any],
    advantage: float,
    can_read: bool = True,
    is_terminal_stop: bool = False,
    gold_answer_index: int | None = None,
    answer_loss_weight: float = 0.5,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the A+B Hybrid GRPO loss for a single decision step.

    1. Navigation term (Option A): Clipped policy-gradient surrogate on ``nav_log_prob``
       using the trajectory's group-normalized advantage ``A_i``.
    2. Terminal Answer term (Option B): When ``is_terminal_stop`` and ``gold_answer_index``
       is provided, applies exact cross-entropy supervision on ``answer_logits`` + exact answer KL.
    3. Exact analytic KL penalty ``kl_beta * KL(pi_theta || pi_ref)``.
    """
    nav_log_prob, nav_kl = nav_action_log_prob_and_kl(logits, ref_logits, action, can_read=can_read)
    old_lp = torch.as_tensor(old_nav_log_prob, dtype=nav_log_prob.dtype, device=nav_log_prob.device).detach()
    ratio = torch.exp((nav_log_prob - old_lp).clamp(-10.0, 10.0))
    adv_t = torch.as_tensor(float(advantage), dtype=nav_log_prob.dtype, device=nav_log_prob.device)

    # When can_read is False (forced stop at budget=0), type_logits has no degree of freedom
    if can_read:
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        pg_loss = -torch.min(surr1, surr2)
    else:
        pg_loss = torch.zeros_like(nav_log_prob)

    total_kl = nav_kl
    ans_ce_val = 0.0
    ans_loss_term = torch.zeros_like(nav_log_prob)

    if is_terminal_stop and gold_answer_index is not None:
        term_info = terminal_answer_expectation(logits, gold_answer_index, ref_logits=ref_logits)
        ans_loss_term = float(answer_loss_weight) * term_info["answer_ce_loss"]
        total_kl = total_kl + term_info["answer_kl"]
        ans_ce_val = float(term_info["answer_ce_loss"].detach())

    loss = pg_loss + ans_loss_term + float(kl_beta) * total_kl
    metrics = {
        "loss": float(loss.detach()),
        "pg_loss": float(pg_loss.detach()),
        "answer_ce_loss": ans_ce_val,
        "kl": float(total_kl.detach()),
        "ratio": float(ratio.detach()),
        "advantage": float(advantage),
    }
    return loss, metrics


@torch.no_grad()
def sample_navigation_action(
    logits: dict[str, torch.Tensor],
    state: dict[str, Any],
    temperature: float = 1.0,
    greedy: bool = False,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Sample a navigation action (READ or STOP) and compute Option B terminal answer stats."""
    can_read = bool(state.get("candidates")) and float(state.get("remaining_budget", 0)) > 0
    temp = max(float(temperature), 1e-3)
    t_logits = masked_type_logits(logits["type_logits"], can_read) / temp

    if greedy or not can_read:
        type_idx = 0 if (can_read and int(t_logits.argmax()) == 0) else 1
    else:
        probs_type = F.softmax(t_logits, dim=-1)
        views = state.get("views", [])
        last_ds = float(views[-1].get("downsample", views[-1].get("requested_downsample", 64))) if views else 64.0
        if last_ds <= 4.0:
            # Ensure group rollouts explore stopping on high-resolution fields (DS <= 4)
            stop_p = float(probs_type[1].clamp(min=0.30, max=0.85))
            probs_type = torch.tensor([1.0 - stop_p, stop_p], dtype=probs_type.dtype, device=probs_type.device)
        elif not views or last_ds >= 16.0:
            # Prevent wasted step-0/coarse premature stops during exploration
            stop_p = float(probs_type[1].clamp(max=0.10))
            probs_type = torch.tensor([1.0 - stop_p, stop_p], dtype=probs_type.dtype, device=probs_type.device)
        type_idx = int(torch.multinomial(probs_type, 1).item())

    if type_idx == 0 and can_read:
        c_logits = logits["candidate_logits"] / temp
        if greedy:
            cand_idx = int(c_logits.argmax().item())
        else:
            probs_cand = F.softmax(c_logits, dim=-1)
            cand_idx = int(torch.multinomial(probs_cand, 1).item())
        action = {"type": "read", "candidate_index": cand_idx}
    else:
        # At STOP, choose the greedy answer for environment termination;
        # reward & gradient use the full Option B answer distribution!
        ans_idx = int(logits["answer_logits"].argmax().item())
        action = {"type": "stop", "answer_index": ans_idx}

    nav_lp, _ = nav_action_log_prob_and_kl(logits, None, action, can_read=can_read)
    ans_probs = F.softmax(logits["answer_logits"], dim=-1).detach().cpu().tolist()
    info = {
        "can_read": can_read,
        "stop_probability": float(F.softmax(masked_type_logits(logits["type_logits"], can_read), dim=-1)[1]),
        "answer_probabilities": ans_probs,
    }
    return action, float(nav_lp.detach()), info


def compute_episode_hybrid_reward(
    task_kind: str,
    terminal_answer_probs: list[float],
    gold_answer_index: int,
    views: list[dict[str, Any]],
    reads_used: int,
    cam_evidence: dict[str, Any] | None = None,
    read_penalty: float = 0.02,
) -> dict[str, float]:
    """Compute the A+B Hybrid trajectory reward R(tau_i).

    - Uses exact terminal answer probability ``p_gold = P(m = y* | s_T, stop)`` (Option B)
      instead of a noisy single-sample binary draw.
    - On CAMELYON16: couples ``p_gold`` with pixel-level ground-truth field qualification
      so the agent is rewarded for stopping ON a high-resolution tumor region AND knowing it is tumor.
    - On WSI-VQA: rewards ``p_gold`` + greedy accuracy bonus + high-resolution foreground tissue support.
    """
    p_gold = float(terminal_answer_probs[int(gold_answer_index)])
    pred_idx = int(max(range(len(terminal_answer_probs)), key=lambda i: terminal_answer_probs[i]))
    greedy_hit = 1.0 if pred_idx == int(gold_answer_index) else 0.0

    last_view = views[-1] if views else {}
    last_ds = float(last_view.get("downsample", last_view.get("requested_downsample", 64)))
    high_res = 1.0 if (views and last_ds <= 4.0) else 0.0

    if task_kind == "cam":
        ev = cam_evidence or {}
        last_field = ev.get("last_field", {})
        qualified = 1.0 if bool(ev.get("qualified_field_success", ev.get("success", False))) else 0.0
        tumor_frac = float(last_field.get("field_tumor_fraction", 0.0))
        oracle_best = 1.0 if bool(ev.get("oracle_best_visited_success", False)) else 0.0
        coarse_stop_penalty = -0.50 if high_res < 0.5 else 0.15
        efficiency_bonus = float(read_penalty) * max(0.0, 8.0 - float(reads_used)) * qualified
        # Target answer on CAM localization is choice representing "Metastatic tumor is supported"
        # ONLY when the field itself is qualified; otherwise the honest field answer is non-tumor.
        reward = (
            1.0 * qualified * p_gold
            + 0.5 * qualified
            + 0.35 * high_res * tumor_frac
            + 0.10 * oracle_best
            + coarse_stop_penalty
            + efficiency_bonus
        )
        return {
            "reward": float(reward),
            "p_gold": p_gold,
            "greedy_hit": greedy_hit,
            "qualified_last_field": qualified,
            "field_tumor_fraction": tumor_frac,
            "grounded_success": 1.0 if (qualified > 0.5 and greedy_hit > 0.5) else 0.0,
        }
    else:
        tissue_frac = float(last_view.get("tissue_fraction", 0.25 if views else 0.0))
        valid_high_res = 1.0 if (high_res > 0.5 and tissue_frac >= 0.20) else 0.0
        coarse_stop_penalty = -0.50 if valid_high_res < 0.5 else 0.20
        efficiency_bonus = float(read_penalty) * max(0.0, 8.0 - float(reads_used)) * valid_high_res * greedy_hit
        reward = (
            1.0 * p_gold * valid_high_res
            + 0.30 * greedy_hit * valid_high_res
            + coarse_stop_penalty
            + efficiency_bonus
        )
        return {
            "reward": float(reward),
            "p_gold": p_gold,
            "greedy_hit": greedy_hit,
            "valid_high_res": valid_high_res,
            "grounded_success": 1.0 if (valid_high_res > 0.5 and greedy_hit > 0.5) else 0.0,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train JEV Agent with A+B Hybrid GRPO.")
    p.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--sft-checkpoint", required=True, help="Initialized SFT checkpoint (also serves as pi_ref)")
    p.add_argument("--output", required=True, help="Output directory for GRPO checkpoints and logs")
    p.add_argument("--group-size", type=int, default=4, help="Number of rollout trajectories G per question/slide")
    p.add_argument("--budget", type=int, default=8, help="Read budget B during online rollouts")
    p.add_argument("--steps", type=int, default=300, help="Number of GRPO optimization steps")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate for GRPO fine-tuning")
    p.add_argument("--kl-beta", type=float, default=0.04, help="Analytic KL regularization weight")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO/GRPO probability ratio clipping epsilon")
    p.add_argument("--answer-loss-weight", type=float, default=0.5, help="Option B terminal answer CE weight")
    p.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature during group rollouts")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    return p


__all__ = [
    "action_log_prob_and_kl",
    "build_arg_parser",
    "categorical_kl",
    "compute_episode_hybrid_reward",
    "compute_group_advantages",
    "hybrid_grpo_step_loss",
    "masked_type_logits",
    "nav_action_log_prob_and_kl",
    "sample_navigation_action",
    "terminal_answer_expectation",
]
