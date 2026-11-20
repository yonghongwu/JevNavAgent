# JevNavAgent

**Budget-Decoupled Single-Forward Visual Navigation and Counterfactual Trajectory Alignment for Gigapixel Whole-Slide Pathology**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-renyimen%2FJevNavAgent-blue)](https://huggingface.co/datasets/renyimen/JevNavAgent)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20Models-renyimen%2FJevNavAgent-yellow)](https://huggingface.co/renyimen/JevNavAgent)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

`JevNavAgent` is both:
1. A **single-forward autonomous visual navigation agent** (built on `Qwen3-VL-2B-Instruct` + LoRA + three factorized decision heads, generating **0 autoregressive tokens** per step) trained with **Stage-1 Counterfactual Trajectory Alignment** and **Stage-2 Online Trajectory-Level GRPO**.
2. An **open, pluggable gigapixel WSI active navigation benchmark & evaluation SDK** (`v1.0-Breast`) covering **Clean Visual WSI-VQA** (`1,353` audited questions / `485` TCGA-BRCA WSIs) and **Exhaustively Verified CAMELYON16** (`311` verified WSIs, excluding the 20 partially-annotated training slides).

---

## 1. Benchmark Leaderboard (`v1.0-Breast`)

### Clean Visual WSI-VQA (`351` Held-Out Questions: `179` Val + `172` Test, `153` Unseen WSIs)

| Method | Params | Val | Test | **Total** | Macro-Global | **Micro-Focal** | Clin. A–D | Avg Reads | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-6.1-Terra | Frontier | 41.90 | 51.16 | 46.44 | 52.17 | 38.19 | 45.61 | 3.33 | 70.04 |
| GPT-6.1-Sol | Frontier | 48.60 | 51.16 | 49.86 | 53.14 | 45.14 | 50.00 | 2.69 | 66.64 |
| Patho-R1-7B (ZS) | 7B | 32.96 | 38.95 | 35.90 | 36.71 | 34.72 | 33.78 | 0.00 | 0.78 |
| Quilt-LLaVA-7B (ZS) | 7B | 37.99 | 39.53 | 38.75 | 34.78 | 44.44 | 36.15 | 0.00 | 0.68 |
| Qwen3-VL-8B + PLIP | 8B | 44.13 | 49.42 | 46.72 | 51.69 | 39.58 | 44.93 | 2.98 | 1.88 |
| MedGemma-4B + PLIP | 4B | 50.28 | 50.58 | 50.43 | **53.14** | 46.53 | 51.35 | 2.98 | 2.53 |
| **JevNavAgent (SFT, 2B)** | **2B** | 50.84 | 48.84 | 49.86 | 50.24 | 49.31 | 53.38 | 3.09 | 5.25 |
| **JevNavAgent + Traj-GRPO (Step-T)** | **2B** | 50.84 | 50.00 | 50.43 | 49.76 | **51.39** | 53.38 | 3.05 | 5.63 |
| **JevNavAgent + Traj-GRPO (Multi-Scale)** | **2B** | **51.40** | **51.74** | **51.57** | 51.69 | **51.39** | **54.73** | 3.05 | 5.63 |

### Exhaustively Verified CAMELYON16 Official Test (`129` WSIs: `49` Pos / `80` Neg, Budget `B=12`)

| Method | Pos Loc Hit | Visited Hit | Neg Spec | **Strict Loc+Spec** | **Slide Acc** | Pos / Neg Reads |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| JevNavAgent (SFT, 2B) | 10.20 | 22.45 | 88.75 | 58.91 | 66.67 | 10.16 / 11.45 |
| **JevNavAgent + Traj-GRPO (Online)** | 16.33 | 24.49 | **98.75** | **67.44** | 68.99 | 10.29 / 11.75 |
| **JevNavAgent + Traj-GRPO (Calib.)** | **22.45** | **24.49** | 93.75 | 66.67 | **69.77** | 10.29 / 11.75 |

---

## 2. Installation

Requires Python 3.10+ and system `openslide` (`sudo apt-get install -y openslide-tools libopenslide0` on Ubuntu/Debian):

```bash
git clone https://github.com/yonghongwu/JevNavAgent.git
cd JevNavAgent
pip install -e ".[dev]"
```

---

## 3. Download Benchmark Metadata & Whole-Slide Images

All curated manifests, visual-grounding audit ledgers, CAMELYON16 XML polygon masks, counterfactual trajectories (`formal_terra_v5_train.parquet`), and official baseline predictions are hosted on HuggingFace at [`renyimen/JevNavAgent`](https://huggingface.co/datasets/renyimen/JevNavAgent).

```bash
# 1. Download metadata, XML annotations, counterfactual trajectories & baseline predictions (~107 MB)
jevnav-download --mode metadata --out ./data

# 2. Download evaluation WSIs directly from NIH GDC & CAMELYON16 S3 (with resume + MD5 check)
jevnav-download --mode eval-slides --out ./data --workers 4

# (Optional) Download full training + evaluation WSIs
jevnav-download --mode full --out ./data --workers 4
```

---

## 4. Evaluate Your Own Model or API (Pluggable Interface)

### Option A: Score Official Baseline Predictions Immediately (Zero WSI Download Needed)

```bash
jevnav-score \
  --benchmark wsivqa_351 \
  --manifest ./data/manifests/wsivqa_clean_visual_val_179.json ./data/manifests/wsivqa_clean_visual_test_172.json \
  --predictions ./data/baseline_predictions/wsivqa_351/jevnav_traj_grpo_vqa_kl_B4.jsonl
```

### Option B: Evaluate Any OpenAI-Compatible Vision API (Zero Code)

Plug in GPT-4o, Claude/Gemini proxies, or any local vLLM/SGLang VLM server:

```bash
jevnav-eval \
  --benchmark wsivqa_351 \
  --policy openai_api \
  --api-base https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --api-model gpt-4o \
  --data-dir ./data \
  --budget 4 \
  --output results/gpt4o_wsivqa351.jsonl
```

### Option C: Plug In Your Own Local Python Policy

Subclass `BaseWSINavPolicy` and implement `act(obs: NavigationObservation) -> NavigationAction` (see [`examples/custom_policy_template.py`](examples/custom_policy_template.py)):

```bash
jevnav-eval \
  --benchmark wsivqa_351 \
  --policy examples/custom_policy_template.py:MyCustomPathologyPolicy \
  --data-dir ./data \
  --output results/my_policy.jsonl
```

### Option D: Evaluate Pretrained `JevNavAgent` Checkpoints from HuggingFace

```bash
# Stage-2 Traj-GRPO (VQA+KL) on Clean Visual WSI-VQA
jevnav-eval \
  --benchmark wsivqa_351 \
  --policy jevnav \
  --checkpoint renyimen/JevNavAgent:stage2_traj_grpo_vqa_kl \
  --backbone Qwen/Qwen3-VL-2B-Instruct \
  --data-dir ./data \
  --output results/jevnav_vqa.jsonl

# Stage-2 Traj-GRPO (CAM16) on Verified CAMELYON16 Test-129
jevnav-eval \
  --benchmark cam16_test \
  --policy jevnav \
  --checkpoint renyimen/JevNavAgent:stage2_traj_grpo_cam16 \
  --backbone Qwen/Qwen3-VL-2B-Instruct \
  --data-dir ./data \
  --budget 12 \
  --output results/jevnav_cam16_test129.jsonl
```

---

## 5. Training on the Counterfactual Trajectory Corpus

Use our `CounterfactualTrajectoryDataset` (`16,635` transitions across `449` active training WSIs) directly in PyTorch or run Stage-1 SFT:

```python
from jevnav.data import CounterfactualTrajectoryDataset

ds = CounterfactualTrajectoryDataset("./data/trajectories/formal_terra_v5_train.parquet")
print(len(ds), ds[0]["slide_id"], ds[0]["target"])
```

```bash
jevnav-train-sft \
  --train-data ./data/trajectories/formal_terra_v5_train.parquet \
  --backbone Qwen/Qwen3-VL-2B-Instruct \
  --output-dir ./runs/jevnav_sft_repro
```

---

## 6. Roadmap (`v2.0` Multi-Organ Benchmark Expansion)

`v1.0-Breast` establishes the unified active navigation environment and evaluation protocol across breast pathology (`TCGA-BRCA` + `CAMELYON16`). Upcoming releases will expand to multi-organ cohorts (prostate needle biopsies, colorectal resections, and non-small-cell lung cancer) under the same `jevnav` interface.

## 7. License

Code, curated manifests, audit ledgers, and counterfactual trajectory annotations are released under the **Apache-2.0 License**. Underlying whole-slide images remain governed by their original public licenses (NIH GDC TCGA-BRCA Open Access and CAMELYON16 CC0).
