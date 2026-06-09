<h1 align="center">ACR-GUI</h1>

<p align="center">
  <em>Anchor-collapse regularized continual GUI grounding</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.20732"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
  <a href="https://github.com/xavierliu34/GUI-AiF"><img src="https://img.shields.io/badge/Code-GitHub-black"></a>
</p>

<hr>

<p align="center">
  <img src="assets/arch_01.png"  width="80%">
</p>

<p align="center">
  <em>ACR-GUI augments GUI-G<sup>2</sup> style GRPO grounding with a continual grounding memory. It retrieves context-compatible historical instruction-anchor mappings and regularizes candidate boxes with point, size, and layout-zone rewards to reduce anchor collapse across domain streams.</em>
</p>

## Motivation

<p align="center">
  <img src="assets/guicl_01.png" alt="Motivation Chart" width="65%">
</p>

Continual GUI Agents operate under evolving scenarios: domain-in-flux (e.g., from Mobile OS to Web OS) and resolution-in-flux (e.g., scaling from 1080p to 4K).

ACR-GUI targets a spatial failure mode in continual GUI grounding: after adapting to new domains, old-domain predictions can collapse toward a few frequent centers, scales, or layout zones even when the instruction semantics remain understandable. The method stores high-confidence historical anchors rather than raw images, then gates the anchor reward by retrieval similarity and memory confidence to avoid negative transfer.

## Installation

```bash
conda create -n gui-aif python=3.12
conda activate gui-aif
bash setup.sh
```

then install the dependencies:

```bash
pip install deepspeed==0.15.4
pip install filelock
pip install qwen_vl_utils
```

## Start

Train GUI-G<sup>2</sup>-style GRPO on your own data:

```bash
cd gui-aif
bash run_grpo.sh
```

Train the continual ACR-GUI reward:

```bash
CKPT_PATH=/path/to/Qwen2.5-VL \
DATA_PATH=/path/to/domain_stream.yaml \
ACR_ENABLED=true \
MEMORY_JSONL=/path/to/memory.jsonl \
MEMORY_NPY=/path/to/memory.npy \
bash run_continual.sh
```

You should configure:

* `DATA_PATH` : Path to your dataset YAML config, where sequentially set the GUI dataset required to train
* `CKPT_PATH` : Model checkpoint path
* `LOG_DIR` , `SAVE_PATH` : Output folders
* `ACR_ENABLED` : Enable anchor-collapse regularization in continual GRPO
* `MEMORY_JSONL`, `MEMORY_NPY` : Continual grounding memory metadata and embedding matrix
* `LAMBDA_ACR` : Weight of the gated historical anchor reward

Training data should follow the JSONL format demonstrated in:

```text
example_training_json.json
```

To evaluate:

```bash
run screenspotpro_test.py
```

You should configure:
* `Qwen_path` : Your trained model path
* `Screenspot_imgs` : ScreenSpot-Pro images path
* `Screenspot_test` : ScreenSpot-Pro annotations path

For other benchmarks evaluation, you can modedify this code.

## Acknowledgement

The code is built from [GUI-G<sup>2</sup>](https://github.com/zju-real/GUI-G2).

## Citation

If you use GUI-AiF, please cite our work:

```bibtex
@misc{liu2026continualguiagents,
      title={Continual GUI Agents}, 
      author={Ziwei Liu and Borui Kang and Hangjie Yuan and Zixiang Zhao and Wei Li and Yifan Zhu and Tao Feng},
      year={2026},
      eprint={2601.20732},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.20732}, 
}
```

If you like our project, please give us a star ⭐ on GitHub.
