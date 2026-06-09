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
import hashlib
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
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
)
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments

from vlm_modules.qwen_module import Qwen2VLModule
from trainer import VLMGRPOTrainer, GRPOConfig
from memory import GroundingMemoryBank, GroundingMemoryItem, bbox_to_anchor, compute_anchor_reward, normalize_bbox

# ----------------------- Monkey-patch flash attention bug -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler

torch.serialization.add_safe_globals([ZeroStageEnum, LossScaler])

from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


# ----------------------- Config Dataclasses -----------------------

@dataclass
class ContinualRewardConfig:
    """Configuration for anchor-collapse regularized continual reward."""

    # ---- Sampling ----
    K: int = field(default=16, metadata={"help": "Number of samples per prompt."})
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature."})

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

    # ---- Anchor-Collapse Regularization (ACR) ----
    acr_enabled: bool = field(default=True, metadata={"help": "Enable anchor-collapse regularized reward."})
    memory_jsonl: Optional[str] = field(default=None, metadata={"help": "Path to grounding memory metadata JSONL."})
    memory_npy: Optional[str] = field(default=None, metadata={"help": "Path to grounding memory embedding matrix NPY."})
    memory_write_enabled: bool = field(default=True, metadata={"help": "Write current-domain anchors to memory after training."})
    memory_write_jsonl: Optional[str] = field(default=None, metadata={"help": "Output JSONL path for updated grounding memory."})
    memory_write_npy: Optional[str] = field(default=None, metadata={"help": "Output NPY path for updated grounding memory embeddings."})
    # ---- Step 2: confidence-filtered memory extraction ----
    memory_extract_mode: str = field(default="predict", metadata={"help": "How to build current-domain memory after training: 'predict' (re-infer with the trained model and keep r_i > tau_r) or 'gt' (write ground-truth boxes directly)."})
    tau_r: float = field(default=0.5, metadata={"help": "Confidence threshold for keeping a re-predicted anchor in memory (Step 2)."})
    extract_max_new_tokens: int = field(default=128, metadata={"help": "Max new tokens when re-predicting for memory extraction."})
    # ---- Step 3: memory capacity ----
    memory_capacity: int = field(default=0, metadata={"help": "Max anchors retained in memory (0 = unbounded). SelectTopN applied after merge."})
    memory_per_domain: bool = field(default=True, metadata={"help": "Balance SelectTopN across domains so no domain is fully evicted."})
    acr_top_k: int = field(default=5, metadata={"help": "Number of compatible historical anchors to retrieve."})
    lambda_acr: float = field(default=0.2, metadata={"help": "Weight for anchor-collapse regularization."})
    tau_sim: float = field(default=0.35, metadata={"help": "Semantic retrieval gate threshold."})
    tau_conf: float = field(default=0.8, metadata={"help": "Memory confidence gate threshold."})
    eta_point: float = field(default=1.0, metadata={"help": "Historical point reward weight."})
    eta_size: float = field(default=0.5, metadata={"help": "Historical size reward weight."})
    eta_zone: float = field(default=0.25, metadata={"help": "Historical layout-zone reward weight."})
    beta_text: float = field(default=0.2, metadata={"help": "Text context compatibility weight."})
    beta_element: float = field(default=0.2, metadata={"help": "Element-type context compatibility weight."})
    beta_layout: float = field(default=0.15, metadata={"help": "Layout context compatibility weight."})
    beta_domain: float = field(default=0.2, metadata={"help": "Domain context compatibility weight."})
    sigma_point: float = field(default=0.15, metadata={"help": "Minimum normalized std for historical point reward."})
    sigma_size: float = field(default=0.15, metadata={"help": "Minimum normalized std for historical size reward."})

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
                dataset_domain = data.get("domain") or data.get("name") or self._infer_domain_from_path(json_path)

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
                for example in cur_data_dict:
                    example.setdefault("domain", dataset_domain)
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
            "context_text": self._first_non_empty(
                example,
                ("context_text", "target_text", "text", "label", "name", "title", "instruction"),
            ),
            "element_type": self._first_non_empty(
                example,
                ("element_type", "ui_type", "widget_type", "component_type", "control_type", "type", "category"),
            ),
            "domain": self._first_non_empty(example, ("domain", "source", "dataset")),
            "prompt": make_conversation_image(example)["prompt"],
        }

    @staticmethod
    def _first_non_empty(example: dict, keys: tuple[str, ...]) -> Optional[str]:
        for key in keys:
            value = example.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
            elif isinstance(value, (int, float, bool)):
                return str(value)
        return None

    @staticmethod
    def _infer_domain_from_path(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path or ""))[0]
        parent = os.path.basename(os.path.dirname(path or ""))
        return stem or parent or "unknown"


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


# ----------------------- Anchor-Collapse Reward Function Factory -----------------------

def make_continual_reward_function(
    reward_cfg: ContinualRewardConfig,
    memory_bank: Optional[GroundingMemoryBank] = None,
):
    """
    Factory that returns the ACR continual reward function.

    The returned callable has signature:
        reward_func(completions, **kwargs) -> List[float]

    where kwargs includes 'image', 'problem', 'image_path', etc.
    """

    def continual_reward(completions, **kwargs) -> List[float]:
        K = reward_cfg.K
        images = kwargs.get("image", [])
        instructions = kwargs.get("problem", [])
        context_texts = kwargs.get("context_text", [])
        element_types = kwargs.get("element_type", [])
        domains = kwargs.get("domain", [])

        total_completions = len(completions)
        num_groups = total_completions // K

        all_rewards = [0.0] * total_completions
        for g in range(num_groups):
            start = g * K
            end = start + K
            group_completions = completions[start:end]
            group_image = _group_value(images, g, start, num_groups, total_completions)
            group_instruction = _group_value(instructions, g, start, num_groups, total_completions)
            group_context_text = _group_value(context_texts, g, start, num_groups, total_completions)
            group_element_type = _group_value(element_types, g, start, num_groups, total_completions)
            group_domain = _group_value(domains, g, start, num_groups, total_completions)
            if group_image is None or group_instruction is None:
                continue

            contents = [c[0]["content"] for c in group_completions]
            bboxes = [parse_bbox_from_text(ct) for ct in contents]

            img_width, img_height = group_image.size
            if img_width < 1 or img_height < 1:
                continue

            if reward_cfg.acr_enabled and memory_bank is not None:
                for k, bbox in enumerate(bboxes):
                    if bbox is None:
                        continue
                    all_rewards[start + k] = reward_cfg.lambda_acr * compute_anchor_reward(
                        group_instruction,
                        tuple(bbox),
                        memory_bank,
                        image_size=(img_width, img_height),
                        context_text=group_context_text,
                        element_type=group_element_type,
                        domain=group_domain,
                        top_k=reward_cfg.acr_top_k,
                        tau_sim=reward_cfg.tau_sim,
                        tau_conf=reward_cfg.tau_conf,
                        eta_point=reward_cfg.eta_point,
                        eta_size=reward_cfg.eta_size,
                        eta_zone=reward_cfg.eta_zone,
                        beta_text=reward_cfg.beta_text,
                        beta_element=reward_cfg.beta_element,
                        beta_layout=reward_cfg.beta_layout,
                        beta_domain=reward_cfg.beta_domain,
                        sigma_point=reward_cfg.sigma_point,
                        sigma_size=reward_cfg.sigma_size,
                    )

        return all_rewards

    return continual_reward


def _group_value(values: list, group_idx: int, start_idx: int, num_groups: int, total_completions: int):
    if len(values) == num_groups and group_idx < len(values):
        return values[group_idx]
    if len(values) == total_completions and start_idx < len(values):
        return values[start_idx]
    return None


# ----------------------- Format Reward (auxiliary) -----------------------

def format_reward(completions, **kwargs):
    """Check if the completion contains a valid [x1,y1,x2,y2] format."""
    pattern = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.fullmatch(pattern, content.split("assistant\n")[-1], re.DOTALL)
        for content in completion_contents
    ]
    for i, num in enumerate([1.0 if match else 0.0 for match in matches]):
        if num < 1 and os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(
                    f"\n|||||||| RANK: {dist.get_rank()}, format match: {num} ||||||||\n"
                )
                f.write(f"Image Path: \n{kwargs.get('image_path', ['N/A'])[i]}\n")
                f.write(f"Instruction: \n{kwargs.get('problem', ['N/A'])[i]}\n")
                f.write(f"Content: \n{completion_contents[i]}\n")
    return [1.0 if match else 0.0 for match in matches]


# ----------------------- Domain-Aware KL Scheduler -----------------------

class DomainKLScheduler:
    """
    Manages Domain-Aware KL penalty scheduling.

    Tracks reward baselines per domain and adjusts β based on forgetting rate.
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

        # Per-domain reward baselines: {domain_name: baseline_value}
        self.domain_baselines: Dict[str, float] = {}
        # Current domain index
        self.current_domain_idx = -1
        self.last_observed_domains: set[str] = set()

    def record_domain_baseline(self, domain_name: str, baseline_reward: float):
        """Record the reward baseline at the end of training a domain."""
        self.domain_baselines[domain_name] = baseline_reward

    def observe_domain_rewards(self, domains: List[Optional[str]], rewards: List[float]) -> float:
        """Update β from observed per-domain rewards in the current training stream."""
        domain_values: Dict[str, List[float]] = {}
        for domain, reward in zip(domains, rewards):
            if not domain:
                continue
            domain_values.setdefault(str(domain), []).append(float(reward))

        if not domain_values:
            return self.beta

        domain_means = {
            domain: sum(values) / len(values)
            for domain, values in domain_values.items()
            if values
        }
        for domain, mean_reward in domain_means.items():
            self.domain_baselines.setdefault(domain, mean_reward)

        self.last_observed_domains = set(domain_means)
        current_domain = next(reversed(domain_means))
        return self.update_beta(current_domain, domain_means)

    def update_beta(
        self,
        current_domain: str,
        old_domain_reward_values: Dict[str, float],
    ) -> float:
        """
        Compute new β based on average forgetting across old domains.

        Args:
            current_domain: Name of the domain currently being trained.
            old_domain_reward_values: Dict {domain_name: current reward mean} for all old domains.

        Returns:
            Updated β value.
        """
        if not self.domain_baselines:
            return self.beta0

        deltas = []
        for domain_name, baseline in self.domain_baselines.items():
            if domain_name == current_domain:
                continue
            current_reward = old_domain_reward_values.get(domain_name, baseline)
            delta = max(0.0, baseline - current_reward)
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


def update_memory_from_dataset(
    memory_bank: GroundingMemoryBank,
    dataset: LazyUnsupervisedDataset,
) -> int:
    """Step 2 ('gt' mode): write ground-truth anchors directly (confidence 1.0).

    Faithful but requires GT boxes; performs no model re-prediction or filtering.
    Saving and SelectTopN are handled by the caller.
    """
    existing_ids = {item.id for item in memory_bank.items}
    added = 0

    for example in dataset.list_data_dict:
        instruction = example.get("instruction")
        if not instruction:
            continue

        bbox = _extract_memory_bbox(example)
        if bbox is None:
            continue

        point, size, layout_role = bbox_to_anchor(bbox)
        domain = LazyUnsupervisedDataset._first_non_empty(example, ("domain", "source", "dataset")) or "unknown"
        item_id = _memory_item_id(domain, instruction, bbox)
        if item_id in existing_ids:
            continue

        item = GroundingMemoryItem(
            id=item_id,
            domain=domain,
            instruction=instruction,
            embedding=None,
            bbox=bbox,
            point=point,
            size=size,
            context_text=LazyUnsupervisedDataset._first_non_empty(
                example,
                ("context_text", "target_text", "text", "label", "name", "title", "instruction"),
            ),
            element_type=LazyUnsupervisedDataset._first_non_empty(
                example,
                ("element_type", "ui_type", "widget_type", "component_type", "control_type", "type", "category"),
            ),
            layout_role=layout_role,
            confidence=1.0,
            success=True,
        )
        memory_bank.add(item)
        existing_ids.add(item_id)
        added += 1

    return added


def _point_confidence(pred_bbox, gt_bbox, alpha: float = 0.5) -> float:
    """Gaussian point-distance score in [0, 1] between two boxes (same coord space).

    Used as the per-sample confidence r_i for Step 2 memory filtering: how close the
    model's re-predicted center is to the ground-truth center, scaled by GT size.
    """
    gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
    gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
    gt_w = gt_bbox[2] - gt_bbox[0]
    gt_h = gt_bbox[3] - gt_bbox[1]
    sigma_x = alpha * gt_w
    sigma_y = alpha * gt_h
    if sigma_x <= 0 or sigma_y <= 0:
        return 0.0
    pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
    pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
    exponent = -0.5 * (
        (pred_cx - gt_cx) ** 2 / (sigma_x ** 2) + (pred_cy - gt_cy) ** 2 / (sigma_y ** 2)
    )
    return round(math.exp(exponent), 3)


def extract_domain_memory(
    trainer: VLMGRPOTrainer,
    dataset: LazyUnsupervisedDataset,
    memory_bank: GroundingMemoryBank,
    reward_cfg: ContinualRewardConfig,
) -> int:
    """Step 2 ('predict' mode): re-predict the current domain with the trained model,
    keep high-confidence anchors (r_i > tau_r), and store the GT box as the anchor
    with confidence = r_i. Saving and SelectTopN are handled by the caller.
    """
    existing_ids = {item.id for item in memory_bank.items}
    batch_size = max(1, int(trainer.args.per_device_train_batch_size))
    added = 0

    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        preds = trainer.generate_predictions(
            batch, max_new_tokens=reward_cfg.extract_max_new_tokens
        )
        for ex, pred_text in zip(batch, preds):
            instruction = ex.get("problem")
            image = ex.get("image")
            solution = ex.get("solution")
            if not instruction or image is None or not solution:
                continue
            width, height = image.size
            if width < 1 or height < 1:
                continue

            try:
                gt_norm = normalize_bbox(
                    tuple(float(v) for v in solution), image_size=(width, height)
                )
            except (TypeError, ValueError):
                continue
            if not _valid_bbox(gt_norm):
                continue

            pred_bbox = parse_bbox_from_text(pred_text)
            if pred_bbox is None:
                continue
            pred_norm = normalize_bbox(tuple(pred_bbox), image_size=(width, height))

            r_i = _point_confidence(pred_norm, gt_norm)
            if r_i <= reward_cfg.tau_r:
                continue

            point, size, layout_role = bbox_to_anchor(gt_norm)
            domain = ex.get("domain") or "unknown"
            item_id = _memory_item_id(domain, instruction, gt_norm)
            if item_id in existing_ids:
                continue

            item = GroundingMemoryItem(
                id=item_id,
                domain=domain,
                instruction=instruction,
                embedding=None,
                bbox=gt_norm,
                point=point,
                size=size,
                context_text=ex.get("context_text"),
                element_type=ex.get("element_type"),
                layout_role=layout_role,
                confidence=float(r_i),
                success=True,
            )
            memory_bank.add(item)
            existing_ids.add(item_id)
            added += 1

    return added


def resolve_memory_write_paths(reward_cfg: ContinualRewardConfig, output_dir: str) -> tuple[str, str]:
    jsonl_path = reward_cfg.memory_write_jsonl or reward_cfg.memory_jsonl
    npy_path = reward_cfg.memory_write_npy or reward_cfg.memory_npy
    if not jsonl_path:
        jsonl_path = os.path.join(output_dir, "acr_memory.jsonl")
    if not npy_path:
        npy_path = os.path.join(output_dir, "acr_memory.npy")
    return jsonl_path, npy_path


def _extract_memory_bbox(example: dict) -> Optional[tuple[float, float, float, float]]:
    for key in ("rela_box", "rel_box", "normalized_box"):
        bbox = example.get(key)
        if bbox is not None:
            bbox = tuple(float(v) for v in bbox)
            return bbox if _valid_bbox(bbox) else None

    bbox = example.get("bbox")
    if bbox is not None:
        bbox = tuple(float(v) for v in bbox)
        if max(abs(v) for v in bbox) <= 1.5:
            return bbox if _valid_bbox(bbox) else None

    abs_box = example.get("abs_box")
    if abs_box is None:
        return None
    width = example.get("width")
    height = example.get("height")
    if not width or not height:
        return None
    bbox = normalize_bbox(tuple(float(v) for v in abs_box), image_size=(int(width), int(height)))
    return bbox if _valid_bbox(bbox) else None


def _valid_bbox(bbox: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def _memory_item_id(domain: str, instruction: str, bbox: tuple[float, float, float, float]) -> str:
    payload = json.dumps(
        {
            "domain": domain,
            "instruction": instruction,
            "bbox": [round(v, 6) for v in bbox],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"mem_{digest}"


# ----------------------- R_base: Supervised Gaussian Grounding Rewards -----------------------
# Ported from gaussian_grpo.py (GUI-G²). Serves as the base task reward R_base when
# ground-truth boxes are available (`solution` = abs_box, in pixel coordinates).
# The trainer sums all reward functions, realizing R = R_base + lambda_acr * R_anchor,
# where the ACR ("continual") reward is already scaled by lambda_acr internally.

def gaussian_point_reward(completions, solution, **kwargs):
    """R_base (point term): Gaussian on predicted vs GT center-point distance."""

    def g_point_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
        pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
        gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
        gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
        gt_w = gt_bbox[2] - gt_bbox[0]
        gt_h = gt_bbox[3] - gt_bbox[1]

        sigma_x = alpha * gt_w
        sigma_y = alpha * gt_h
        if sigma_x <= 0 or sigma_y <= 0:
            return 0.0

        exponent = -0.5 * (
            (pred_cx - gt_cx) ** 2 / (sigma_x ** 2)
            + (pred_cy - gt_cy) ** 2 / (sigma_y ** 2)
        )
        return round(math.exp(exponent), 3)

    contents = [c[0]["content"] for c in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        bbox = parse_bbox_from_text(content)
        try:
            if bbox is not None:
                sol = [float(v) for v in sol]
                reward = g_point_reward(bbox, sol)
        except Exception:
            pass
        rewards.append(reward)
    return rewards


def gaussian_plane_reward(completions, solution, **kwargs):
    """R_base (plane term): Bhattacharyya-distance Gaussian over box position + scale."""

    def g_plane_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        eps = 1e-8
        pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
        pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
        pred_w = pred_bbox[2] - pred_bbox[0]
        pred_h = pred_bbox[3] - pred_bbox[1]
        gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
        gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
        gt_w = gt_bbox[2] - gt_bbox[0]
        gt_h = gt_bbox[3] - gt_bbox[1]

        pred_mu = np.array([pred_cx, pred_cy])
        gt_mu = np.array([gt_cx, gt_cy])
        pred_cov = np.array([[(pred_w * alpha) ** 2, 0.0], [0.0, (pred_h * alpha) ** 2]])
        gt_cov = np.array([[(gt_w * alpha) ** 2, 0.0], [0.0, (gt_h * alpha) ** 2]])

        sigma_avg = (pred_cov + gt_cov) / 2
        mu_diff = pred_mu - gt_mu
        sigma_avg_inv = np.linalg.inv(sigma_avg + eps * np.eye(2))
        term1 = (1 / 8) * np.dot(mu_diff.T, np.dot(sigma_avg_inv, mu_diff))

        det_avg = np.linalg.det(sigma_avg)
        det_pred = np.linalg.det(pred_cov)
        det_gt = np.linalg.det(gt_cov)
        try:
            term2 = 0.5 * np.log(det_avg / np.sqrt(det_pred * det_gt + eps))
        except Exception:
            return 0.0
        return round(float(np.exp(-(term1 + term2))), 3)

    contents = [c[0]["content"] for c in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        bbox = parse_bbox_from_text(content)
        try:
            if bbox is not None:
                sol = [float(v) for v in sol]
                reward = g_plane_reward(bbox, sol)
        except Exception:
            pass
        rewards.append(reward)
    return rewards


# ----------------------- Reward Registry -----------------------

reward_funcs_registry = {
    "continual": None,  # placeholder — will be replaced by factory output
    "gaussian_point": gaussian_point_reward,
    "gaussian_plane": gaussian_plane_reward,
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

    # ---- Load continual grounding memory ----
    memory_bank = None
    if reward_cfg.acr_enabled:
        if reward_cfg.memory_jsonl and reward_cfg.memory_npy:
            try:
                memory_bank = GroundingMemoryBank()
                memory_bank.load(reward_cfg.memory_jsonl, reward_cfg.memory_npy)
                print(f"ACR memory loaded: {len(memory_bank.items)} anchors.")
            except Exception as e:
                print(f"WARNING: Failed to load ACR memory: {e}. ACR reward disabled.")
                memory_bank = None
        else:
            print("WARNING: ACR enabled but memory_jsonl/memory_npy not set. ACR reward disabled.")

    # ---- Build reward functions ----
    continual_reward_fn = make_continual_reward_function(
        reward_cfg, memory_bank
    )
    reward_funcs_registry["continual"] = continual_reward_fn

    reward_func_names = script_args.reward_funcs
    reward_funcs = [reward_funcs_registry[func] for func in reward_func_names]
    print("Reward functions:", reward_func_names)
    print("K:", reward_cfg.K, "temperature:", reward_cfg.temperature)

    # ---- Load dataset ----
    dataset = LazyUnsupervisedDataset(script_args.dataset_name, script_args)

    # ---- Override training_args with reward_cfg values ----
    training_args.num_generations = reward_cfg.K
    training_args.temperature = reward_cfg.temperature

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
        torch_dtype=model_args.torch_dtype,
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

    # ---- Update continual grounding memory after the current training stream ----
    # Step 2: extract current-domain anchors M_t (predict + filter, or GT).
    # Step 3: merge into M (already in memory_bank) and SelectTopN under capacity.
    if reward_cfg.memory_write_enabled:
        if memory_bank is None:
            memory_bank = GroundingMemoryBank()

        if reward_cfg.memory_extract_mode == "predict":
            # Runs collectively on all ranks (generation gathers sharded params).
            added = extract_domain_memory(trainer, dataset, memory_bank, reward_cfg)
        else:
            added = update_memory_from_dataset(memory_bank, dataset)

        removed = memory_bank.prune(
            reward_cfg.memory_capacity, per_domain=reward_cfg.memory_per_domain
        )

        # Only rank 0 writes the shared memory files to avoid races.
        is_main = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_main:
            memory_jsonl_path, memory_npy_path = resolve_memory_write_paths(
                reward_cfg, training_args.output_dir
            )
            os.makedirs(os.path.dirname(memory_jsonl_path) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(memory_npy_path) or ".", exist_ok=True)
            memory_bank.save(memory_jsonl_path, memory_npy_path)
            print(
                f"ACR memory updated [{reward_cfg.memory_extract_mode}]: added {added}, "
                f"pruned {removed}; total {len(memory_bank.items)}. "
                f"Saved to {memory_jsonl_path} and {memory_npy_path}."
            )
        if dist.is_initialized():
            dist.barrier()

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
