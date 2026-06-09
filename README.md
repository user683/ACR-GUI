# ACR-GUI

Anchor-Collapse Regularized Continual GUI Grounding.

ACR-GUI targets continual GUI grounding, where a model receives a stream of GUI domains and must adapt to the current domain while preserving grounding ability on previous domains. The core failure mode addressed here is anchor collapse: predictions on old domains drift toward a few frequent centers, scales, or layout zones after training on new domains.

## Method

ACR-GUI represents each predicted box as an anchor tuple:

```text
anchor = (center, size, layout_zone)
```

The training reward uses a continual grounding memory. The memory stores compact instruction-anchor records rather than raw historical images:

```text
instruction, bbox, center, size, layout_zone, context_text, element_type, domain, confidence
```

During training, ACR-GUI loads old memory as a read-only anchor bank. For each generated candidate box, it retrieves context-compatible historical anchors and computes:

```text
R_anchor = eta_point * R_hist_point
         + eta_size  * R_hist_size
         + eta_zone  * R_hist_zone
```

The final reward used by continual GRPO is:

```text
R = lambda_acr * R_anchor
```

Retrieval is gated by semantic similarity and memory confidence. Context reweighting uses instruction text, optional target/context text, optional element type, predicted layout zone, and domain.

After the current training stream finishes, ACR-GUI writes the current domain anchors back to the memory bank. The next training run can then use that updated memory as historical grounding experience.

## Repository Layout

```text
run_continual.sh                         continual ACR-GUI training entry
src/gui-aif/src/open_r1/continual_grpo.py continual GRPO reward, memory update, KL scheduler
src/gui-aif/src/open_r1/memory/           grounding memory schema, retrieval, reward
src/gui-aif/src/open_r1/trainer/          GRPO trainer
src/gui-aif/tests/test_memory.py          memory and anchor reward tests
```

## Installation

```bash
conda create -n acr-gui python=3.12
conda activate acr-gui
bash setup.sh
pip install deepspeed==0.15.4 filelock qwen_vl_utils
```

Install the model/runtime dependencies required by your Qwen2.5-VL and DeepSpeed environment as needed.

## Data Format

Training samples should contain at least:

```json
{
  "image_path": "/path/to/screenshot.png",
  "instruction": "tap the search button",
  "abs_box": [1122, 18, 1206, 48],
  "width": 1920,
  "height": 1080,
  "domain": "mobile"
}
```

ACR-GUI can also consume normalized boxes through `rela_box`, `rel_box`, `normalized_box`, or normalized `bbox`.

Optional context fields:

```text
context_text, target_text, text, label, name, title
element_type, ui_type, widget_type, component_type, control_type, type, category
domain, source, dataset
```

Dataset YAML example:

```yaml
datasets:
  - name: mobile
    domain: mobile
    json_path: /path/to/mobile_train.json
    sampling_strategy: all
```

## Training

Train with old memory and write updated memory after training:

```bash
CKPT_PATH=/path/to/Qwen2.5-VL \
DATA_PATH=/path/to/domain_stream.yaml \
MEMORY_JSONL=/path/to/old_memory.jsonl \
MEMORY_NPY=/path/to/old_memory.npy \
MEMORY_WRITE_JSONL=/path/to/updated_memory.jsonl \
MEMORY_WRITE_NPY=/path/to/updated_memory.npy \
bash run_continual.sh
```

If `MEMORY_JSONL` and `MEMORY_NPY` are not provided, training still runs, but the ACR reward has no old anchors to retrieve. If write paths are not provided, updated memory is saved under the training output directory.

Useful environment variables:

```text
CKPT_PATH              base model path
DATA_PATH              dataset YAML
SAVE_PATH              training output directory
ACR_ENABLED            enable ACR reward, default true
MEMORY_JSONL           old memory metadata
MEMORY_NPY             old memory embeddings
MEMORY_WRITE_ENABLED   write current anchors after training, default true
MEMORY_WRITE_JSONL     updated memory metadata output
MEMORY_WRITE_NPY       updated memory embeddings output
LAMBDA_ACR             anchor reward weight
```

## Verification

Syntax check:

```bash
cd src/gui-aif
python -m py_compile \
  src/open_r1/continual_grpo.py \
  src/open_r1/trainer/grpo_trainer.py \
  src/open_r1/memory/schema.py \
  src/open_r1/memory/grounding_memory.py \
  src/open_r1/memory/reward.py
```

Memory tests:

```bash
cd src/gui-aif
PYTHONPATH=src python -m pytest tests/test_memory.py
```
