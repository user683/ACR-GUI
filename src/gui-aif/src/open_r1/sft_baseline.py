import os
import json
import yaml
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import Dict, List, Optional, Any
from PIL import Image
import torch.distributed as dist
import os
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
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
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)
            

        q_float, k_float = apply_rotary_pos_emb_flashatt(q.unsqueeze(0).float(), k.unsqueeze(0).float(), cos, sin)
        
        q = q_float.squeeze(0).type_as(q) 
        k = k_float.squeeze(0).type_as(k)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "Override the default `torch.dtype` and load the model under this dtype."})
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"help": "The attention implementation to use in the model."})

@dataclass
class ScriptArguments:
    dataset_name: str = field(metadata={"help": "Path to the dataset config yaml file."})
    image_root: str = field(default=".", metadata={"help": "Root directory of the images."})


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, image_root: str):
        super().__init__()
        self.image_root = image_root
        self.list_data_dict = []

        print(f"Loading SFT configuration from: {data_path}")
        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
            
            active_datasets = yaml_data.get("datasets", [])
            if not active_datasets:
                raise ValueError(f"No active datasets found in {data_path}. Please check the yaml file.")

            print("Found active datasets to load:")
            for data_info in active_datasets:
                json_path = data_info.get("json_path")
                if not json_path:
                    continue
                
                print(f"  - Loading from: {json_path}")
                try:
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            self.list_data_dict.append(json.loads(line.strip()))
                except Exception as e:
                    print(f"    ERROR: Could not load or parse {json_path}. Skipping. Error: {e}")

            print(f"Total samples loaded for SFT: {len(self.list_data_dict)}")
        else:
            raise ValueError(f"Unsupported config file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        example = self.list_data_dict[i]
        
        instruction = example["instruction"]
        prompt_text = f"Outline the position corresponding to the instruction: {instruction}. The output should be only [x1,y1,x2,y2]."
        completion_text = str(example['abs_box'])

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": completion_text}
        ]
        
        image_path = os.path.join(self.image_root, example.get('image_path', ''))
        messages[0]["content"][0]["image_path"] = image_path
        
        output_dict = {"messages": messages}

        return output_dict


@dataclass
class DataCollatorForSupervisedDataset:
    processor: Any

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        images = []
        all_messages = []
        for instance in instances:
            messages = instance["messages"]
            image_path = messages[0]["content"][0].pop("image_path")
            try:
                images.append(Image.open(image_path).convert("RGB"))
                all_messages.append(messages)
            except FileNotFoundError:
                print(f"Warning: Skipping sample because image not found at {image_path}")
                continue

        if not images:
            return {}

        full_texts = [
            self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ) for messages in all_messages
        ]

        inputs = self.processor(text=full_texts, images=images, return_tensors="pt", padding="longest")

        labels = inputs.input_ids.clone()

        prompts_only_text = [
            self.processor.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            ) for messages in all_messages
        ]
        
        prompt_token_lengths = [len(self.processor.tokenizer(p, add_special_tokens=False).input_ids) for p in prompts_only_text]

        for i in range(len(labels)):
            prompt_len = prompt_token_lengths[i]
            labels[i, :prompt_len] = -100 
        
        inputs["labels"] = labels
        
        return inputs

def main():
    parser = transformers.HfArgumentParser((ModelArguments, ScriptArguments, TrainingArguments))

    model_args, script_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Loading model from {model_args.model_name_or_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = 'left'

    if training_args.local_rank == 0: 
        gpu_count = torch.cuda.device_count()
        print("="*50)
        print(f"Training will run on {gpu_count} GPUs.")
        print("="*50)

    train_dataset = LazySupervisedDataset(
        data_path=script_args.dataset_name, 
        image_root=script_args.image_root
    )
    data_collator = DataCollatorForSupervisedDataset(processor=processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    if training_args.resume_from_checkpoint:
        print(f"INFO: Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        print("INFO: No checkpoint specified, starting training from scratch.")
        trainer.train()
    
    trainer.save_model(training_args.output_dir)
    print(f"SFT baseline model saved to {training_args.output_dir}")
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)
        print(f"Processor (tokenizer & image_processor) saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()