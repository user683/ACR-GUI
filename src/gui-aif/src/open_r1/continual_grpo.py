# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import yaml
import math
import random
import logging
import copy
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset
from filelock import FileLock

from transformers import (
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
)
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments

from vlm_modules.qwen_module import Qwen2VLModule
from trainer import VLMGRPOTrainer, GRPOConfig

# ----------------------- Monkey-patch flash attention bug -----------------------
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler

torch.serialization.add_safe_globals([ZeroStageEnum, LossScaler])

from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLVisionFlashAttention2,
        apply_rotary_pos_emb_flashatt,
        flash_attn_varlen_func,
    )

    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

    Qwen2_5_VLVisionFlashAttention2.forward = custom_forward
    logger.info("Flash attention monkey-patch applied.")
except ImportError:
    logger.warning("flash_attn not available, skipping flash attention monkey-patch. Using sdpa instead.")


# ----------------------- Config Dataclasses -----------------------

@dataclass
class ContinualRewardConfig:
    """Configuration for the three continual reward functions."""

    # ---- Sampling ----
    K: int = field(default=16, metadata={"help": "Number of samples per prompt."})

    # ---- R1: Entropy-Weighted Spatial Consensus ----
    grid_H: int = field(default=10, metadata={"help": "Voting grid height."})
    grid_W: int = field(default=10, metadata={"help": "Voting grid width."})

    # ---- R2: OCR Verification ----
    ocr_enabled: bool = field(default=True, metadata={"help": "Enable OCR reward."})
    ocr_lang_list: List[str] = field(
        default_factory=lambda: ["en"],
        metadata={"help": "Language list for EasyOCR."},
    )
    ocr_gpu: bool = field(default=True, metadata={"help": "Use GPU for EasyOCR."})
    m_min: int = field(default=50, metadata={"help": "Minimum crop side length (pixels)."})

    # ---- R3: Visual-Semantic Similarity ----
    siglip_enabled: bool = field(default=True, metadata={"help": "Enable SigLIP reward."})
    siglip_model_name: str = field(
        default="google/siglip-base-patch16-224",
        metadata={"help": "SigLIP model ID on HuggingFace."},
    )

    # ---- Adaptive Weights ----
    # Text-instruction weights
    alpha1: float = field(default=0.3, metadata={"help": "R1 weight for text instructions."})
    alpha2: float = field(default=0.5, metadata={"help": "R2 weight for text instructions."})
    alpha3: float = field(default=0.2, metadata={"help": "R3 weight for text instructions."})
    # Icon-instruction weights
    alpha1_prime: float = field(default=0.4, metadata={"help": "R1 weight for icon instructions."})
    alpha3_prime: float = field(default=0.6, metadata={"help": "R3 weight for icon instructions."})

    # ---- Domain-Aware KL Scheduling ----
    kl_lambda: float = field(default=5.0, metadata={"help": "KL scheduling sensitivity λ."})
    kl_N: int = field(default=50, metadata={"help": "KL eval interval in steps."})
    kl_n_eval: int = field(
        default=50, metadata={"help": "Number of eval samples per old domain."}
    )

    # ---- Continual Learning ----
    domain_sequence: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of domain names, e.g. 'mobile,web,desktop'."},
    )
    old_domain_eval_data: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML file containing eval data paths for old domains."},
    )

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the continual GRPO training script."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["continual"],
        metadata={
            "help": "List of reward functions. Default: ['continual'] for the combined three-way reward."
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056, metadata={"help": "Maximum number of pixels for the image."}
    )
    min_pixels: Optional[int] = field(
        default=3136, metadata={"help": "Minimum number of pixels for the image."}
    )
    image_root: Optional[str] = field(
        default=None, metadata={"help": "Root directory of the image."}
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)."},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


# ----------------------- Dataset -----------------------

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


class LazyUnsupervisedDataset(Dataset):
    """Unsupervised dataset: only (image, instruction) pairs — no ground-truth bounding boxes."""

    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super().__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets", [])

            for data in datasets:
                json_path = data.get("json_path")
                sampling_strategy = data.get("sampling_strategy", "all")
                sampling_number = None

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(
                            int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100
                        )
                    else:
                        sampling_number = int(sampling_number)

                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]

                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": "Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].".format(
                                    example["instruction"]
                                ),
                            },
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root

        if "image_path" in example:
            image_path = example["image_path"]
            attempts = 0
            while not os.path.exists(image_path) and attempts < 5:
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example.get("image", ""))
                attempts += 1
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                print("--------------- No image found....... ---------------")
                image = Image.new("RGB", (224, 224))
        elif "image" in example:
            image_path = example.get("image", "")
            try:
                image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else Image.new("RGB", (224, 224))
            except Exception:
                image = Image.new("RGB", (224, 224))
        else:
            image = Image.new("RGB", (224, 224))
            image_path = ""

        # abs_box is optional (used only for debugging/logging)
        solution = example.get("abs_box", [0, 0, 0, 0])

        return {
            "image": image,
            "image_path": image_path,
            "problem": example["instruction"],
            "solution": solution,
            "prompt": make_conversation_image(example)["prompt"],
        }


# ----------------------- Helper: BBox Parsing -----------------------

_BBOX_PATTERN = re.compile(
    r"\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]"
)


def parse_bbox_from_text(text: str) -> Optional[List[float]]:
    """Extract bounding box [x1, y1, x2, y2] from a completion string."""
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    m = _BBOX_PATTERN.search(text.strip())
    if m:
        try:
            return [float(m.group(i)) for i in range(1, 5)]
        except (ValueError, IndexError):
            return None
    return None


def get_center_from_bbox(bbox: List[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


# ----------------------- R1: Entropy-Weighted Spatial Consensus -----------------------

def compute_spatial_consensus_rewards(
    bboxes: List[Optional[List[float]]],
    img_width: int,
    img_height: int,
    grid_H: int = 10,
    grid_W: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    R₁: Entropy-Weighted Spatial Consensus.

    Args:
        bboxes: List of K predicted bboxes (or None for invalid).
        img_width, img_height: Image dimensions in pixels.
        grid_H, grid_W: Voting grid size.

    Returns:
        rewards: shape (K,) — R₁ values in [0, 1].
        votes: shape (H, W) — voting matrix.
    """
    K = len(bboxes)
    votes = np.zeros((grid_H, grid_W), dtype=np.float32)

    # Build voting grid
    cell_indices = []
    for bbox in bboxes:
        if bbox is None:
            cell_indices.append(None)
            continue
        cx, cy = get_center_from_bbox(bbox)
        # Map to grid cell
        col = min(int(cx / img_width * grid_W), grid_W - 1)
        row = min(int(cy / img_height * grid_H), grid_H - 1)
        votes[row, col] += 1.0
        cell_indices.append((row, col))

    max_votes = votes.max()
    if max_votes == 0:
        return np.zeros(K, dtype=np.float32), votes

    # Per-prediction raw score: normalized vote count of its cell
    raw_scores = np.zeros(K, dtype=np.float32)
    for k, (bbox, cell) in enumerate(zip(bboxes, cell_indices)):
        if bbox is not None and cell is not None:
            raw_scores[k] = votes[cell[0], cell[1]] / max_votes

    # Entropy of the vote distribution
    p = votes.flatten() / K
    p = p[p > 0]  # avoid log(0)
    H = -np.sum(p * np.log(p))
    H_max = np.log(grid_H * grid_W)
    entropy_factor = 1.0 - (H / H_max) if H_max > 0 else 0.0

    rewards = raw_scores * entropy_factor
    return rewards, votes


# ----------------------- R2: OCR Verification -----------------------

# Keywords that suggest a text-type instruction
_TEXT_INDICATOR_PATTERNS = [
    r'"([^"]+)"',            # quoted text
    r"'([^']+)'",            # single-quoted text
    r"text\s+['\"]([^'\"]+)",  # "text '...'"
    r"label\s+['\"]([^'\"]+)",
    r"button\s+['\"]([^'\"]+)",
    r"tab\s+['\"]([^'\"]+)",
    r"menu\s+['\"]([^'\"]+)",
    r"link\s+['\"]([^'\"]+)",
]

# Keywords for text-type without quotes
_TEXT_NOUN_KEYWORDS = [
    "button", "tab", "menu", "link", "label", "text", "input",
    "field", "header", "title", "heading", "paragraph", "icon",
]


def extract_keywords_from_instruction(instruction: str) -> List[str]:
    """Extract target keyword set W_q from the instruction."""
    keywords = []
    for pattern in _TEXT_INDICATOR_PATTERNS:
        matches = re.findall(pattern, instruction, re.IGNORECASE)
        keywords.extend(matches)
    return [w.strip().lower() for w in keywords if w.strip()]


def classify_instruction_type(instruction: str) -> bool:
    """
    Returns True if the instruction targets a text element, False for icon.

    I_text(q) = 1 if instruction contains quoted text OR
    contains a text-noun + specific name.
    """
    # Check for quoted content
    for pattern in _TEXT_INDICATOR_PATTERNS:
        if re.search(pattern, instruction, re.IGNORECASE):
            return True

    # Check for text-noun keywords + specific name nearby
    instruction_lower = instruction.lower()
    for kw in _TEXT_NOUN_KEYWORDS:
        if kw in instruction_lower:
            # Heuristic: if keyword is followed by a capitalized or specific word
            idx = instruction_lower.find(kw)
            after = instruction[idx + len(kw):].strip()
            if after and (after[0].isupper() or after.startswith('"') or after.startswith("'")):
                return True
    return False


def levenshtein_sim(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len_s1, len_s2 = len(s1), len(s2)
    # DP with two rows
    prev = list(range(len_s2 + 1))
    curr = [0] * (len_s2 + 1)
    for i in range(1, len_s1 + 1):
        curr[0] = i
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    dist = prev[len_s2]
    max_len = max(len_s1, len_s2)
    return 1.0 - dist / max_len


def compute_ocr_rewards(
    image: Image.Image,
    center_points: List[Optional[Tuple[float, float]]],
    instruction: str,
    ocr_engine,
    crop_size: int,
) -> np.ndarray:
    """
    R₂: OCR Verification.

    Crops a region around each predicted center, runs OCR, fuzzy-matches
    with extracted instruction keywords.
    """
    K = len(center_points)
    keywords = extract_keywords_from_instruction(instruction)

    rewards = np.zeros(K, dtype=np.float32)
    if not keywords or ocr_engine is None:
        return rewards

    for k, cp in enumerate(center_points):
        if cp is None:
            continue
        cx, cy = cp
        half = crop_size / 2.0
        left = max(0, int(cx - half))
        upper = max(0, int(cy - half))
        right = min(image.width, int(cx + half))
        lower = min(image.height, int(cy + half))

        if right <= left or lower <= upper:
            continue

        crop = image.crop((left, upper, right, lower))
        try:
            ocr_result = ocr_engine.readtext(np.array(crop))
            recognized_texts = [item[1].strip().lower() for item in ocr_result if item[1].strip()]
        except Exception:
            continue

        if not recognized_texts:
            continue

        # Best fuzzy match between any recognized text and any keyword
        best_sim = 0.0
        for rt in recognized_texts:
            for kw in keywords:
                sim = levenshtein_sim(rt, kw)
                if sim > best_sim:
                    best_sim = sim
        rewards[k] = best_sim

    return rewards


# ----------------------- R3: Visual-Semantic Similarity (SigLIP) -----------------------

def compute_siglip_rewards(
    image: Image.Image,
    center_points: List[Optional[Tuple[float, float]]],
    instruction: str,
    siglip_model,
    siglip_processor,
    crop_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    R₃: Visual-Semantic Similarity via frozen SigLIP.

    Returns raw cosine similarities (to be batch-normalized later).
    """
    K = len(center_points)
    similarities = np.zeros(K, dtype=np.float32)

    if siglip_model is None:
        return similarities

    # Encode instruction text once
    try:
        text_inputs = siglip_processor(
            text=[instruction], return_tensors="pt", padding=True, truncation=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            t_emb = siglip_model.get_text_features(**text_inputs)
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    except Exception:
        return similarities

    for k, cp in enumerate(center_points):
        if cp is None:
            continue
        cx, cy = cp
        half = crop_size / 2.0
        left = max(0, int(cx - half))
        upper = max(0, int(cy - half))
        right = min(image.width, int(cx + half))
        lower = min(image.height, int(cy + half))

        if right <= left or lower <= upper:
            continue

        crop = image.crop((left, upper, right, lower))
        try:
            img_inputs = siglip_processor(images=crop, return_tensors="pt")
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            with torch.no_grad():
                v_emb = siglip_model.get_image_features(**img_inputs)
                v_emb = v_emb / v_emb.norm(dim=-1, keepdim=True)
            similarities[k] = float((v_emb * t_emb).sum(dim=-1).item())
        except Exception:
            continue

    return similarities


# ----------------------- Combined Reward Function Factory -----------------------

def make_continual_reward_function(
    reward_cfg: ContinualRewardConfig,
    ocr_engine,
    siglip_model,
    siglip_processor,
    device: torch.device,
):
    """
    Factory that returns the combined continual reward function.

    The returned callable has signature:
        reward_func(completions, **kwargs) -> List[float]

    where kwargs includes 'image', 'problem', 'image_path', etc.

    R₃ uses batch-level min-max normalization across all completions.
    """

    def continual_reward(completions, **kwargs) -> List[float]:
        configured_K = reward_cfg.K
        images = kwargs.get("image", [])
        instructions = kwargs.get("problem", [])

        total_completions = len(completions)
        prompt_count = len(images)
        if prompt_count > 0 and total_completions % prompt_count == 0:
            K = total_completions // prompt_count
        else:
            K = configured_K
        num_groups = total_completions // K if K > 0 else 0

        if os.getenv("DEBUG_MODE") == "true" and (configured_K != K or num_groups == 0):
            log_path = os.getenv("LOG_PATH")
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if log_path and rank == 0:
                with open(log_path, "a") as f:
                    f.write(
                        f"\n[continual_reward] configured_K={configured_K}, inferred_K={K}, "
                        f"prompt_count={prompt_count}, total_completions={total_completions}, "
                        f"num_groups={num_groups}\n"
                    )

        # Stage 1: compute per-group rewards, collecting raw R₃ values
        group_r1 = []       # list of np.ndarray (K,)
        group_r2 = []       # list of np.ndarray (K,)
        group_r3_raw = []   # list of np.ndarray (K,)
        group_is_text = []  # list of bool

        siglip_device = device
        if siglip_model is not None:
            try:
                siglip_device = next(siglip_model.parameters()).device
            except StopIteration:
                pass

        for g in range(num_groups):
            start = g * K
            end = start + K
            group_completions = completions[start:end]
            group_image = images[g]
            group_instruction = instructions[g]

            contents = [c[0]["content"] for c in group_completions]
            bboxes = [parse_bbox_from_text(ct) for ct in contents]
            centers = [get_center_from_bbox(b) if b else None for b in bboxes]

            img_width, img_height = group_image.size
            if img_width < 1 or img_height < 1:
                group_r1.append(np.zeros(K, dtype=np.float32))
                group_r2.append(np.zeros(K, dtype=np.float32))
                group_r3_raw.append(np.zeros(K, dtype=np.float32))
                group_is_text.append(False)
                continue

            # Adaptive crop size
            valid_centers_list = [c for c in centers if c is not None]
            if len(valid_centers_list) >= 2:
                centers_arr = np.array(valid_centers_list)
                cluster_std = np.mean(np.std(centers_arr, axis=0))
                crop_size = int(max(3.0 * cluster_std, float(reward_cfg.m_min)))
            else:
                crop_size = reward_cfg.m_min

            is_text = classify_instruction_type(group_instruction)

            # R₁
            r1, _ = compute_spatial_consensus_rewards(
                bboxes, img_width, img_height, reward_cfg.grid_H, reward_cfg.grid_W
            )

            # R₂
            if reward_cfg.ocr_enabled and is_text and ocr_engine is not None:
                r2 = compute_ocr_rewards(
                    group_image, centers, group_instruction, ocr_engine, crop_size
                )
            else:
                r2 = np.zeros(K, dtype=np.float32)

            # R₃ (raw, before normalization)
            if reward_cfg.siglip_enabled and siglip_model is not None:
                r3_raw = compute_siglip_rewards(
                    group_image, centers, group_instruction,
                    siglip_model, siglip_processor, crop_size, siglip_device,
                )
            else:
                r3_raw = np.zeros(K, dtype=np.float32)

            group_r1.append(r1)
            group_r2.append(r2)
            group_r3_raw.append(r3_raw)
            group_is_text.append(is_text)

        # Stage 2: batch-level min-max normalization of R₃
        all_r3_raw = np.concatenate(group_r3_raw) if group_r3_raw else np.array([0.0])
        r3_min = all_r3_raw.min()
        r3_max = all_r3_raw.max()
        if r3_max > r3_min:
            all_r3_norm = (all_r3_raw - r3_min) / (r3_max - r3_min)
        else:
            all_r3_norm = np.zeros_like(all_r3_raw)

        # Stage 3: combine with adaptive weights
        all_rewards = [0.0] * total_completions
        r3_offset = 0
        for g in range(num_groups):
            K_g = len(group_r1[g])
            r1 = group_r1[g]
            r2 = group_r2[g]
            r3_norm = all_r3_norm[r3_offset : r3_offset + K_g]
            r3_offset += K_g

            is_text = group_is_text[g]
            if is_text:
                w1, w2, w3 = reward_cfg.alpha1, reward_cfg.alpha2, reward_cfg.alpha3
            else:
                w1, w2, w3 = reward_cfg.alpha1_prime, 0.0, reward_cfg.alpha3_prime

            rk = w1 * r1 + w2 * r2 + w3 * r3_norm

            start = g * K_g
            for k in range(K_g):
                all_rewards[start + k] = float(rk[k])

        return all_rewards

    return continual_reward


# ----------------------- Format Reward (auxiliary) -----------------------

def format_reward(completions, **kwargs):
    """Check if the completion contains a valid [x1,y1,x2,y2] format."""
    pattern = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.fullmatch(pattern, content.split("assistant\n")[-1].strip(), re.DOTALL)
        for content in completion_contents
    ]
    image_paths = kwargs.get("image_path", [])
    problems = kwargs.get("problem", [])
    prompt_count = len(problems)
    generations_per_prompt = max(1, len(completions) // prompt_count) if prompt_count else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    for i, num in enumerate([1.0 if match else 0.0 for match in matches]):
        if num < 1 and os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            prompt_idx = min(i // generations_per_prompt, prompt_count - 1) if prompt_count else 0
            image_path = image_paths[prompt_idx] if prompt_idx < len(image_paths) else "N/A"
            problem = problems[prompt_idx] if prompt_idx < len(problems) else "N/A"
            if log_path:
                with open(log_path, "a") as f:
                    f.write(f"\n|||||||| RANK: {rank}, format match: {num} ||||||||\n")
                    f.write(f"Image Path: \n{image_path}\n")
                    f.write(f"Instruction: \n{problem}\n")
                    f.write(f"Content: \n{completion_contents[i]}\n")
    return [1.0 if match else 0.0 for match in matches]


# ----------------------- Domain-Aware KL Scheduler -----------------------

class DomainKLScheduler:
    """
    Manages Domain-Aware KL penalty scheduling.

    Tracks R₁ baselines per domain and adjusts β based on forgetting rate.
    """

    def __init__(
        self,
        beta0: float,
        kl_lambda: float = 5.0,
        N: int = 50,
        n_eval: int = 50,
    ):
        self.beta0 = beta0
        self.beta = beta0
        self.kl_lambda = kl_lambda
        self.N = N  # evaluation interval in steps
        self.n_eval = n_eval

        # Per-domain R₁ baselines: {domain_name: baseline_value}
        self.domain_baselines: Dict[str, float] = {}
        # Current domain index
        self.current_domain_idx = -1

    def record_domain_baseline(self, domain_name: str, baseline_r1: float):
        """Record the R₁ baseline at the end of training a domain."""
        self.domain_baselines[domain_name] = baseline_r1

    def update_beta(
        self,
        current_domain: str,
        old_domain_r1_values: Dict[str, float],
    ) -> float:
        """
        Compute new β based on average forgetting across old domains.

        Args:
            current_domain: Name of the domain currently being trained.
            old_domain_r1_values: Dict {domain_name: current_R1_mean} for all old domains.

        Returns:
            Updated β value.
        """
        if not self.domain_baselines:
            return self.beta0

        deltas = []
        for domain_name, baseline in self.domain_baselines.items():
            if domain_name == current_domain:
                continue
            current_r1 = old_domain_r1_values.get(domain_name, baseline)
            delta = max(0.0, baseline - current_r1)
            deltas.append(delta)

        if not deltas:
            return self.beta0

        avg_delta = sum(deltas) / len(deltas)
        self.beta = self.beta0 * (1.0 + self.kl_lambda * avg_delta)
        return self.beta


# ----------------------- Utility -----------------------

def object_to_dict(obj):
    return {key: value for key, value in obj.__dict__.items()}


def write_configs_to_txt(filename, *configs):
    with open(filename, "a", encoding="utf-8") as f:
        names = [
            "GRPOScriptArguments",
            "GRPOConfig",
            "GRPOModelConfig",
            "ContinualRewardConfig",
        ]
        for i, config in enumerate(configs):
            f.write(f"\n=== {names[i]} ===\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")


# ----------------------- Reward Registry -----------------------

reward_funcs_registry = {
    "continual": None,  # placeholder — will be replaced by factory output
    "format": format_reward,
}


# ----------------------- Model Router -----------------------

def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "checkpoint" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "reverse" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        raise NotImplementedError("InternVL support not implemented in this version.")
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")


# ----------------------- Main -----------------------

def main(script_args, training_args, model_args, reward_cfg):
    # Load VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("Using VLM module:", vlm_module_cls.__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load OCR engine (lazy) ----
    ocr_engine = None
    if reward_cfg.ocr_enabled:
        try:
            import easyocr

            ocr_engine = easyocr.Reader(
                reward_cfg.ocr_lang_list, gpu=reward_cfg.ocr_gpu
            )
            print("EasyOCR engine loaded.")
        except ImportError:
            print("WARNING: easyocr not installed. OCR reward disabled.")
        except Exception as e:
            print(f"WARNING: Failed to load EasyOCR: {e}. OCR reward disabled.")

    # ---- Load SigLIP model ----
    siglip_model = None
    siglip_processor = None
    if reward_cfg.siglip_enabled:
        try:
            siglip_processor = AutoProcessor.from_pretrained(
                reward_cfg.siglip_model_name, trust_remote_code=True
            )
            siglip_model = AutoModel.from_pretrained(
                reward_cfg.siglip_model_name, trust_remote_code=True
            )
            siglip_model = siglip_model.to(device)
            siglip_model.eval()
            for p in siglip_model.parameters():
                p.requires_grad = False
            print(f"SigLIP model loaded: {reward_cfg.siglip_model_name}")
        except Exception as e:
            print(f"WARNING: Failed to load SigLIP: {e}. Visual-semantic reward disabled.")

    # ---- Build reward functions ----
    continual_reward_fn = make_continual_reward_function(
        reward_cfg, ocr_engine, siglip_model, siglip_processor, device
    )
    reward_funcs_registry["continual"] = continual_reward_fn

    reward_func_names = script_args.reward_funcs
    reward_funcs = [reward_funcs_registry[func] for func in reward_func_names]
    print("Reward functions:", reward_func_names)
    print("Grid size:", reward_cfg.grid_H, "x", reward_cfg.grid_W)
    print("K:", reward_cfg.K, "temperature:", training_args.temperature)

    # ---- Load dataset ----
    dataset = LazyUnsupervisedDataset(script_args.dataset_name, script_args)

    # ---- Sync reward_cfg.K with training_args.num_generations ----
    reward_cfg.K = training_args.num_generations

    # ---- Initialize Domain KL Scheduler ----
    kl_scheduler = None
    domain_names = None
    if reward_cfg.domain_sequence:
        domain_names = [d.strip() for d in reward_cfg.domain_sequence.split(",")]
        kl_scheduler = DomainKLScheduler(
            beta0=training_args.beta,
            kl_lambda=reward_cfg.kl_lambda,
            N=reward_cfg.kl_N,
            n_eval=reward_cfg.kl_n_eval,
        )
        print(f"Domain sequence: {domain_names}")
        print(f"KL scheduler: lambda={reward_cfg.kl_lambda}, N={reward_cfg.kl_N}")

    # ---- Initialize GRPO trainer ----
    trainer_cls = VLMGRPOTrainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        cl_args=reward_cfg,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.dtype,
        # Pass KL scheduler and domain info
        kl_scheduler=kl_scheduler,
        domain_names=domain_names,
    )

    # ---- Train ----
    if training_args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        print("Starting training from scratch.")
        trainer.train()

    # ---- Save ----
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser(
        (GRPOScriptArguments, GRPOConfig, GRPOModelConfig, ContinualRewardConfig)
    )
    script_args, training_args, model_args, reward_cfg = parser.parse_args_and_config()

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        if dist.get_rank() == 0:
            write_configs_to_txt(
                log_path,
                object_to_dict(script_args),
                object_to_dict(training_args),
                object_to_dict(model_args),
                object_to_dict(reward_cfg),
            )

    main(script_args, training_args, model_args, reward_cfg)
