# SmolLM-135M from Scratch – PyTorch Implementation  
**A clean, educational, Llama-style re-implementation of HuggingFaceTB/SmolLM-135M**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

---

### Overview
This repository contains a **minimal, readable, nanoGPT-style** implementation of the **SmolLM-135M** model – a 135 million parameter decoder-only transformer released by Hugging Face in 2024.

It is built from scratch in pure PyTorch and closely follows the official architecture used in Llama 3 / SmolLM, including:

- RMSNorm (instead of LayerNorm)
- SwiGLU activation (instead of GELU)
- Rotary Positional Embeddings (RoPE)
- Grouped-Query Attention (GQA) – 9 query heads, 3 KV heads
- No biases in linear layers
- Embedding weight tying
- Hybrid RoPE/NoPE (RoPE skipped every 4th layer – as in the original)

The code is designed to be **easy to understand, modify, and run on modest hardware** (tested on a 4GB GPU).

---

### Model Specifications (SmolLM-135M)

| Parameter                     | Value       | Notes |
|------------------------------|-------------|-------|
| `hidden_size` (n_embd)       | 576         | Embedding dimension |
| `num_hidden_layers`          | 30          | Transformer blocks |
| `num_attention_heads`        | 9           | Query heads |
| `num_key_value_heads`        | 3           | KV heads → GQA (3:1) |
| `intermediate_size`          | 1536        | SwiGLU hidden dim |
| `vocab_size`                 | 49,152      | Official (here: 50,304 for GPT-2 tokenizer compatibility) |
| `max_position_embeddings`    | 2048        | Context length |
| `rope_theta`                 | 100,000.0   | RoPE base |
| `rms_norm_eps`               | 1e-5        | |
| **Total Parameters**         | ~135M       | Confirmed |

---

### Key Architectural Differences from GPT-2 (nanoGPT)

| Feature               | GPT-2 (original)         | SmolLM-135M (this repo)         |
|-----------------------|--------------------------|----------------------------------|
| Normalization         | LayerNorm                | **RMSNorm**                      |
| MLP Activation        | GELU                     | **SwiGLU** (SiLU-gated)          |
| Positional Encoding   | Absolute learned         | **RoPE** (rotary)                |
| Attention Type        | Multi-Head (MHA)         | **Grouped-Query (GQA)**          |
| Biases in Linear      | Yes                      | **No** (except norms)            |
| Positional Embeddings | Added to input           | **Applied inside attention**     |

---

### Training on Your Dataset (`input.txt`)

This implementation is configured to train directly on your `input.txt` file using the **GPT-2 tokenizer** (for simplicity and compatibility).

#### Current Training Settings (Safe for 4GB GPU)
```python
block_size = 512      # Context length
batch_size = 4
vocab_size = 50304    # GPT-2 tokenizer
```

#### Example Training Log
Below is a snapshot of the training log. You can view the [full training log here](training.log).

```text
using device: cuda
Model parameters: 135.18M
loaded 338025 tokens
1 epoch = 165 batches
Starting training...
step 0 | loss: 10.8725 | dt: 1274.74ms | tok/sec: 1606.61
...
step 500 | loss: 5.2739 | dt: 2679.78ms | tok/sec: 764.24

--- Generating text at step 500 ---
!ua any cons mocking
I they have we be passion so his all
...
step 4990 | loss: 0.0244 | dt: 2699.65ms | tok/sec: 758.62
Saving model to smollm_135_checkpoint.pth
```

---

### Features Included

- Full training loop (5000 steps by default)
- Sample text generation every 500 steps
- Model checkpointing (`smollm_135_checkpoint.pth`)
- Resume training from checkpoint
- Mixed precision training (`bfloat16` via `torch.autocast`)
- Clean, well-commented code

---

### How to Run

```bash
# 1. Clone and enter
git clone https://github.com/yourname/smollm-135m-from-scratch.git
cd smollm-135m-from-scratch

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tiktoken tqdm

# 3. Place your text file
cp your_dataset.txt input.txt

# 4. Run training
python train.py
# or open and run Smollm_135.ipynb in Jupyter/Colab
```

---

### Next Steps & Ideas

| Goal                          | How to Do It |
|-----------------------------|------------|
| Train longer / better data  | Increase `max_iters`, use larger datasets |
| Use official tokenizer      | Load `HuggingFaceTB/SmolLM-135M` tokenizer from HF |
| Increase context length     | Set `block_size = 2048` (needs more VRAM) |
| Evaluate on benchmarks      | Add Hellaswag, ARC, etc. evaluation scripts |
| Export to GGUF / llama.cpp  | Use `convert_hf_to_gguf.py` after training |

---

### References & Credits

- Official Model: [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- Architecture Paper: Llama 3 (Meta AI, 2024)
- RMSNorm: [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)
- SwiGLU: [Shazeer, 2020](https://arxiv.org/abs/2002.05202)
- RoPE: [Su et al., 2021](https://arxiv.org/abs/2104.09864)
- GQA: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)
- Inspiration: Andrej Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT)

---

**License**: MIT