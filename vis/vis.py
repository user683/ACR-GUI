
import argparse
import math
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import cv2
import os
import re 

import torch.nn.functional as F

def load_model_and_processor(qwen_path, lora_path=None, device="cuda"):
    print(f"Loading processor from {qwen_path}...")
    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
    
    print(f"Loading model from {qwen_path} with device_map='auto'...")

    max_memory = {
        0: "40GiB",
    }

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    ).eval()

    print("\n" + "="*40)
    print("Model Loaded. Device Map:")
    print(f"PyTorch can see {torch.cuda.device_count()} CUDA devices.")
    
    if hasattr(model, 'hf_device_map'):
        print("Model layer distribution (`model.hf_device_map`):")
        print(model.hf_device_map) 
    else:
        print("Warning: `model.hf_device_map` not found. Model might be on a single device.")
    print("="*40 + "\n")

    if lora_path:
        print(f"Applying LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path).eval()

    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor, model.device

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    rgb = np.float32(img) / 255.0
    return img, rgb

def parse_bbox(bbox_string):

    match = re.search(r'BBOX\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', bbox_string)
    if match:
        coords = [int(g) for g in match.groups()]
        return coords # [x1, y1, x2, y2]
    

    match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', bbox_string)
    if match:
        coords = [int(g) for g in match.groups()]
        return coords # [x1, y1, x2, y2]
        
    raise ValueError(f"Can not parseBBox: {bbox_string}")

def draw_gaussian_heatmap(image_shape, bbox):

    h, w = image_shape[:2]
    

    heatmap = np.zeros((h, w), dtype=np.float32)
    

    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(w, bbox[2]), min(h, bbox[3])
    

    cv2.rectangle(heatmap, (x1, y1), (x2, y2), 1.0, -1) 
    

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    

    sigma = (bbox_w + bbox_h) / 10 
    

    ksize = int(sigma * 4) + 1
    if ksize % 2 == 0:
        ksize += 1
    
    heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), 0)
    

    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn > 1e-8:
        heatmap = (heatmap - mn) / (mx - mn)
    
    return heatmap


def upsample_to_image(grid, image_shape):

    img_h, img_w = image_shape[0], image_shape[1]
    grid_resized = cv2.resize(grid.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    
    mn, mx = grid_resized.min(), grid_resized.max()
    if mx - mn > 1e-8:
        grid_resized = (grid_resized - mn) / (mx - mn)
    else:
        grid_resized = np.zeros_like(grid_resized)
    return grid_resized

def save_overlay(image_rgb, heatmap, out_path, alpha=0.5):
    """image_rgb: float32 HxWx3 0..1, heatmap: HxW 0..1"""

    if heatmap.shape[:2] != image_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    heat_uint8 = np.uint8(255 * heatmap)
    cmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    cmap = cmap[..., ::-1] # BGR -> RGB
    cmap = cmap.astype(np.float32) / 255.0
    overlay = (1 - alpha) * image_rgb + alpha * cmap
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, device = load_model_and_processor(args.qwen_path, args.lora_path, device=device)


    original_image, vis_image = load_image(args.image_path)


    prompt_origin = "Find the UI element for the command \"{}\". Respond ONLY with its bounding box in the format BBOX(x1, y1, x2, y2). Do not provide any other text."
    prompt = prompt_origin.format(args.instruction)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": "placeholder.jpg"}, 
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text_with_placeholder = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=text_with_placeholder,
        images=original_image,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():

        gen_out = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False,
            output_attentions=False 
        )
    
    generated_ids = gen_out
    generated_text = processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    



    try:
        bbox_coords = parse_bbox(generated_text) # [x1, y1, x2, y2]
        print(f"[info] parsed BBox coordinates: {bbox_coords}")
    except ValueError as e:
        print(f"!!! Error: Cannot parse BBox from model output. {e}")
        return


    heatmap = draw_gaussian_heatmap(vis_image.shape, bbox_coords)


    out_dir = os.path.dirname(args.output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    save_overlay(vis_image, heatmap, args.output_path, alpha=0.5)
    print(f"BBox heatmap saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--instruction', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="semantic_heatmap.png")
    args = parser.parse_args()
    main(args)