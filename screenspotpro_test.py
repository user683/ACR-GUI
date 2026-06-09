# screenspotpro_test.py

import os
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import json
import argparse
from PIL import Image
import logging
from tqdm import tqdm
from collections import defaultdict
from process_utils import pred_2_point, extract_bbox

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.manual_seed(1234)
if torch.cuda.is_available():
    logging.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    logging.info(f"Current device: {torch.cuda.current_device()}")
    logging.info(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    logging.error("!!! CUDA is NOT available. Running on CPU !!!")

parser = argparse.ArgumentParser(description="Evaluate a model on the ScreenSpot-Pro benchmark.")
parser.add_argument('--qwen_path', type=str, required=True, help="The mode path")
parser.add_argument('--screenspot_imgs', type=str, required=True, help="ScreenSpot-Pro mages path")
parser.add_argument('--screenspot_test', type=str, required=True, help="ScreenSpot-Pro annotations path")


args = parser.parse_args()

tokenizer_load_path = args.tokenizer_path if args.tokenizer_path else args.qwen_path
logging.info(f"Loading Tokenizer/Processor from {tokenizer_load_path}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(tokenizer_load_path, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
logging.info("Tokenizer/Processor loaded successfully.")

logging.info(f"Loading model weights from {args.qwen_path}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.qwen_path,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
).eval()
logging.info("Base model loaded successfully.")

if args.lora_path:
    logging.info(f"Loading and applying LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path).eval()
    logging.info("LoRA adapter applied successfully.")
else:
    logging.info("No LoRA path provided, using the model from --qwen_path directly.")

test_files = []
logging.info(f"Discovering test files in {args.screenspot_test}...")
try:
    for f_name in os.listdir(args.screenspot_test):
        if f_name.endswith('.json'):
            test_files.append(f_name)
except FileNotFoundError:
    logging.error(f"Error: Annotation directory not found at {args.screenspot_test}")
    exit()
logging.info(f"Found {len(test_files)} test files: {test_files}")


category_results = defaultdict(lambda: {"text_correct": 0, "text_total": 0, "icon_correct": 0, "icon_total": 0})

categories_order = ["CAD", "Dev", "Creative", "Scientific", "Office", "OS"]


for dataset_filename in test_files:
    dataset_path = os.path.join(args.screenspot_test, dataset_filename)
    logging.info(f"Processing test file: {dataset_path}")
    
    try:
        screenspot_data = json.load(open(dataset_path, 'r'))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading {dataset_path}: {e}")
        continue

    category = "Unknown" 
    if screenspot_data: 
        category = screenspot_data[0].get("group", "Unknown")

    task_name = dataset_filename.replace('.json', '')
    logging.info(f"Num of samples for task '{task_name}': {len(screenspot_data)} | Category: {category}")
    
    prompt_origin = "Find the UI element for the command \"{}\". Respond ONLY with its bounding box in the format BBOX(x1, y1, x2, y2). Do not provide any other text."
    
    num_action = 0
    corr_action = 0
    text_correct_list = []
    icon_correct_list = []
    num_wrong_format = 0

    for j, item in tqdm(enumerate(screenspot_data), desc=f"Testing {task_name}"):
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            logging.warning(f"Image not found, skipping: {img_path}")
            num_action -= 1; continue
        image = Image.open(img_path)
        instruction = item["instruction"]
        bbox_xyxy = item["bbox"]
        img_size = image.size
        bbox_normalized = [bbox_xyxy[0]/img_size[0], bbox_xyxy[1]/img_size[1], bbox_xyxy[2]/img_size[0], bbox_xyxy[3]/img_size[1]]
        prompt = prompt_origin.format(instruction)
        with torch.no_grad():
            messages = [{"role": "user", "content": [{"type": "image", "image_url": "placeholder.jpg"}, {"type": "text", "text": prompt}]}]
            text_with_placeholder = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_with_placeholder, images=image, return_tensors="pt").to(model.device)
            gen_kwargs = {"max_new_tokens": 1024, "do_sample": False}
            generate_ids = model.generate(**inputs, **gen_kwargs)
            response_with_prompt = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            try: response = response_with_prompt.split(prompt)[-1].strip()
            except: response = response_with_prompt
        try:
            if 'box' in response:
                pred_bbox = extract_bbox(response)
                click_point_pixels = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                click_point = [click_point_pixels[0] / img_size[0], click_point_pixels[1] / img_size[1]]
            else:
                click_point_pixels = pred_2_point(response)
                click_point = [click_point_pixels[0] / img_size[0], click_point_pixels[1] / img_size[1]]
            ui_type = item.get("ui_type", "icon")
            if (bbox_normalized[0] <= click_point[0] <= bbox_normalized[2]) and (bbox_normalized[1] <= click_point[1] <= bbox_normalized[3]):
                corr_action += 1
                if ui_type == 'text': text_correct_list.append(1)
                else: icon_correct_list.append(1)
                logging.info(f"Step {j}: MATCH - Running Acc: {corr_action / num_action:.4f}")
            else:
                if ui_type == 'text': text_correct_list.append(0)
                else: icon_correct_list.append(0)
                logging.info(f"Step {j}: MISMATCH - Running Acc: {corr_action / num_action:.4f}")
        except Exception as e:
            num_wrong_format += 1
            ui_type = item.get("ui_type", "icon")
            if ui_type == 'text': text_correct_list.append(0)
            else: icon_correct_list.append(0)
            logging.warning(f"Step {j}: WRONG FORMAT - Response: '{response}'. Error: {e}")

    action_acc = corr_action / num_action if num_action > 0 else 0
    text_acc = sum(text_correct_list) / len(text_correct_list) if text_correct_list else 0
    icon_acc = sum(icon_correct_list) / len(icon_correct_list) if icon_correct_list else 0
    
    logging.info(f"--- Results for Task: {task_name} ---")
    logging.info(f"Action Acc: {action_acc:.4f} | Text Acc: {text_acc:.4f} | Icon Acc: {icon_acc:.4f}")
    logging.info("-------------------------------------------")

    category_results[category]["text_total"] += len(text_correct_list)
    category_results[category]["text_correct"] += sum(text_correct_list)
    category_results[category]["icon_total"] += len(icon_correct_list)
    category_results[category]["icon_correct"] += sum(icon_correct_list)


logging.info("\n\n=================================================")
logging.info("=== Final Aggregated Results by Category ===")
logging.info("=================================================")

for category in categories_order:
    if category in category_results:
        stats = category_results[category]
        
        cat_text_acc = stats["text_correct"] / stats["text_total"] if stats["text_total"] > 0 else 0
        cat_icon_acc = stats["icon_correct"] / stats["icon_total"] if stats["icon_total"] > 0 else 0
        
        logging.info(f"\n--- Category: {category} ---")
        logging.info(f"  Text Acc: {cat_text_acc:.4f} ({stats['text_correct']}/{stats['text_total']})")
        logging.info(f"  Icon Acc: {cat_icon_acc:.4f} ({stats['icon_correct']}/{stats['icon_total']})")

other_categories = set(category_results.keys()) - set(categories_order)
if other_categories:
    logging.info("\n--- Category: Other ---")
    for category in sorted(list(other_categories)):
        stats = category_results[category]
        cat_text_acc = stats["text_correct"] / stats["text_total"] if stats["text_total"] > 0 else 0
        cat_icon_acc = stats["icon_correct"] / stats["icon_total"] if stats["icon_total"] > 0 else 0
        logging.info(f"  - {category}: Text Acc: {cat_text_acc:.4f}, Icon Acc: {cat_icon_acc:.4f}")

logging.info("=================================================")