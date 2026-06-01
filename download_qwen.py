import os
from modelscope import snapshot_download

model_id = "qwen/Qwen2.5-VL-3B-Instruct"
cache_dir = "/XYFS01/HDD_POOL/neu_chzhao/neu_chzhao_1/chuzhao/models"

print(f"Downloading {model_id} to {cache_dir} ...")
os.makedirs(cache_dir, exist_ok=True)

model_dir = snapshot_download(
    model_id=model_id,
    cache_dir=cache_dir,
)

print(f"Download complete! Model saved to: {model_dir}")
