# SmolLM-135M from Scratch – PyTorch Implementation  
**A clean, educational, Llama-style re-implementation with DeepSeek V3 innovations**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

---

### Overview
This repository contains a **minimal, readable, nanoGPT-style** implementation of the **SmolLM-135M** model – a 135 million parameter decoder-only transformer released by Hugging Face in 2024.

**NEW: Now includes DeepSeek V3 innovations!** The notebook (`smollm135_deepseek.ipynb`) implements cutting-edge techniques from DeepSeek V3:
- **Multi-Head Latent Attention (MLA)** - 75-90% KV cache reduction
- **DeepSeekMoE** - Mixture of Experts with auxiliary-loss-free load balancing
- **Complete integration** - SmolLM-DeepSeek model combining both innovations

The base implementation is built from scratch in pure PyTorch and closely follows the official architecture used in Llama 3 / SmolLM, including:

- RMSNorm (instead of LayerNorm)
- SwiGLU activation (instead of GELU)
- Rotary Positional Embeddings (RoPE)
- Grouped-Query Attention (GQA) – 9 query heads, 3 KV heads
- No biases in linear layers
- Embedding weight tying
- Hybrid RoPE/NoPE (RoPE skipped every 4th layer – as in the original)

The code is designed to be **easy to understand, modify, and run on modest hardware** (tested on a 4GB GPU).

#### GPU Memory Considerations

For **4GB GPU** (e.g., GTX 1650), use these settings to avoid OOM:

**Base SmolLM:**
- `batch_size = 4`, `block_size = 512` ✅ Safe

**SmolLM-DeepSeek:**
- `batch_size = 2`, `block_size = 256` ✅ Recommended
- Or reduce MoE: `n_experts = 4`, `moe_layer_freq = 4`

**Memory Breakdown:**
- Model parameters: ~15-20% of VRAM
- Optimizer states: ~30-40% (AdamW stores 2x params)
- Activations: ~30-40% (scales with batch × sequence length)
- Gradients: ~15-20%

**Note:** Dedicated VRAM (not shared memory) is the bottleneck. PyTorch/CUDA only uses dedicated VRAM.

---

### Model Specifications

#### SmolLM-135M (Base)

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
| **Total Parameters**         | **135.18M** | Confirmed |

#### SmolLM-DeepSeek (Full Integration)

| Parameter                     | Value       | Notes |
|------------------------------|-------------|-------|
| `hidden_size` (n_embd)       | 576         | Embedding dimension |
| `num_hidden_layers`          | 30          | Transformer blocks (15 MoE, 15 dense) |
| `num_attention_heads`        | 9           | Query heads |
| `attention_type`             | MLA         | Multi-Head Latent Attention |
| `n_experts`                  | 8           | Routed experts per MoE layer |
| `top_k`                      | 2           | Experts selected per token |
| `shared_expert`              | 1           | Always active expert |
| `moe_layer_freq`             | 2           | MoE every 2nd layer |
| `intermediate_size`          | 1536        | Base FFN size (experts scaled down) |
| `vocab_size`                 | 50,304      | GPT-2 tokenizer compatibility |
| `max_position_embeddings`    | 512         | Context length (configurable) |
| `rope_theta`                 | 10,000.0    | RoPE base |
| `rms_norm_eps`               | 1e-5        | |
| **Total Parameters**         | **205.75M** | Confirmed |
| **Active Parameters/Token**  | ~37M        | Only top-k experts + shared expert active |

---

### Key Architectural Differences from GPT-2 (nanoGPT)

| Feature               | GPT-2 (original)         | SmolLM-135M (this repo)         | SmolLM-DeepSeek (DeepSeek V3)   |
|-----------------------|--------------------------|----------------------------------|----------------------------------|
| Normalization         | LayerNorm                | **RMSNorm**                      | **RMSNorm**                      |
| MLP Activation        | GELU                     | **SwiGLU** (SiLU-gated)          | **DeepSeekMoE** (8 experts + shared) |
| Positional Encoding   | Absolute learned         | **RoPE** (rotary)                | **RoPE** (decoupled in MLA)     |
| Attention Type        | Multi-Head (MHA)         | **Grouped-Query (GQA)**          | **Multi-Head Latent (MLA)**     |
| KV Cache              | Full K & V stored        | Full K & V stored                | **Compressed latent** (75-90% reduction) |
| FFN Architecture      | Dense (all params active)| Dense (all params active)        | **Sparse MoE** (top-k experts active) |
| Expert Routing        | N/A                      | N/A                              | **Auxiliary-loss-free** (dynamic bias) |
| Biases in Linear      | Yes                      | **No** (except norms)            | **No** (except norms)           |
| Positional Embeddings | Added to input           | **Applied inside attention**     | **Decoupled in MLA**            |
| Load Balancing        | N/A                      | N/A                              | **Dynamic bias adjustment** (no aux loss) |
| Parameters            | ~117M (GPT-2 small)      | 135.18M                          | **205.75M** (sparse activation) |
| Active Params/Token    | 100%                     | 100%                             | **~18%** (top-2 experts + shared) |

#### DeepSeek V3 Innovation Highlights

**Multi-Head Latent Attention (MLA):**
- **Revolutionary KV compression**: Instead of storing full K and V tensors, compresses them into a low-rank latent space
- **Memory efficiency**: 75-90% reduction in KV cache size enables much longer context windows
- **Decoupled RoPE**: Position information handled separately, maintaining position awareness despite compression
- **Performance**: Maintains or improves model quality while dramatically reducing memory

**DeepSeekMoE:**
- **Sparse activation**: Only top-k experts (default: 2) are activated per token, plus 1 shared expert
- **Efficiency**: 205.75M total parameters, but only ~37M active per token (~18% activation rate)
- **Auxiliary-loss-free balancing**: Dynamic bias adjustment achieves expert load balancing without adding loss terms
- **Better specialization**: Experts can truly specialize without performance trade-offs from auxiliary losses

**Combined Benefits:**
- **Memory**: MLA reduces KV cache, MoE reduces active parameters
- **Capacity**: 205.75M parameters with only 18% active per token
- **Quality**: No performance degradation from auxiliary losses
- **Scalability**: Architecture scales efficiently to larger models

---

### Training on Your Dataset (`input.txt`)

This implementation is configured to train directly on your `input.txt` file using the **GPT-2 tokenizer** (for simplicity and compatibility).

**Default Dataset:**
- File: `input.txt`
- Token count: **338,025 tokens** (when using default dataset)
- Tokenizer: GPT-2 encoding (50,304 vocab size)
- Epochs: Automatically calculated based on batch size and sequence length
  - With `batch_size=4, block_size=512`: 1 epoch = 165 batches

#### Current Training Settings (Safe for 4GB GPU)

**Base SmolLM:**
```python
block_size = 512      # Context length
batch_size = 4
vocab_size = 50304    # GPT-2 tokenizer
max_steps = 5000      # Training steps
```

**SmolLM-DeepSeek (Enhanced):**
```python
block_size = 512      # Context length
batch_size = 4
vocab_size = 50304    # GPT-2 tokenizer
max_steps = 10000     # Training steps
checkpoint_interval = 1000  # Save checkpoint every 1000 steps
inference_interval = 100    # Run inference every 100 steps
```

**Model Configurations:**

| Model | Parameters | Attention | FFN | Use Case |
|-------|-----------|-----------|-----|----------|
| **SmolLM-135M** | 135.18M | GQA | SwiGLU | Base implementation |
| **SmolLM-DeepSeek (MLA only)** | ~135M | MLA | SwiGLU | Memory-efficient long context |
| **SmolLM-DeepSeek (MoE only)** | ~180M | GQA | MoE | Increased capacity |
| **SmolLM-DeepSeek (Full)** | **205.75M** | MLA | MoE | Best of both worlds |

**Note:** The full SmolLM-DeepSeek model (MLA + MoE) has been confirmed at **205.75M parameters** with:
- 30 layers total (15 MoE layers, 15 dense layers)
- 8 routed experts + 1 shared expert per MoE layer
- Top-2 expert selection
- MLA attention in all layers

#### Example Training Log

**Base SmolLM-135M:**
```text
using device: cuda
Model parameters: 135.18M
loaded 338025 tokens
1 epoch = 165 batches
Starting training...
step 0 | loss: 10.8725 | dt: 1274.74ms | tok/sec: 1606.61
...
step 500 | loss: 5.2739 | dt: 2679.78ms | tok/sec: 764.24
step 4990 | loss: 0.0244 | dt: 2699.65ms | tok/sec: 758.62
Saving model to smollm_135_checkpoint.pth
```

**SmolLM-DeepSeek (Enhanced):**
```text
SmolLM-DeepSeek Parameters: 205.75M
Creating standalone DataLoader...
Loaded 338025 tokens

Training SmolLM-DeepSeek for 10000 steps...
Checkpointing every 1000 steps
Inference every 100 steps
==================================================
step     0 | loss: 10.9393
step    20 | loss: 6.6686
...
step   100 | loss: 5.0293

--- Generating text at step 100 ---
! Is a the day; my lord,
The me with in the king, that the wind.
...
step  1000 | loss: 3.3088

--- Saving checkpoint at step 1000 ---
Checkpoint saved to: checkpoints/checkpoint_step_1000.pth
Latest checkpoint saved to: checkpoints/checkpoint_latest.pth
...
step  5000 | loss: 0.0863
```

**Dataset Information:**
- Default dataset: `input.txt` with **338,025 tokens** (using GPT-2 tokenizer)
- 1 epoch = 165 batches (with batch_size=4, block_size=512)
- DataLoader automatically handles dataset wrapping for continuous training

---

### DeepSeek V3 Innovations (NEW!)

The notebook includes implementations of three key innovations from DeepSeek V3:

#### 1. Multi-Head Latent Attention (MLA)
- **75-90% KV cache reduction** compared to standard attention
- Compresses K and V into low-rank latent space
- Decoupled RoPE for position awareness
- Enables longer context with manageable memory

#### 2. DeepSeekMoE - Mixture of Experts
- **8 routed experts + 1 shared expert** per MoE layer
- Top-K expert selection (configurable, default top-2)
- **Auxiliary-loss-free load balancing** via dynamic bias adjustment
- Better expert specialization without performance degradation

#### 3. SmolLM-DeepSeek Integrated Model
- Combines MLA and MoE in a unified architecture
- Configurable: toggle MLA/MoE independently
- MoE can be applied to specific layers (e.g., every 2nd layer)
- **205.75M parameters** with sparse activation (only top-k experts active per token)

**Model Summary Output:**
```
==================================================
SmolLM-DeepSeek Configuration
==================================================
Total Parameters: 205.75M
Layers: 30 (15 MoE, 15 dense)
Attention: MLA
FFN: MoE
  Experts: 8, Top-K: 2
Context: 512
==================================================
```

**See `smollm135_deepseek.ipynb` for complete implementations and usage examples.**

---

### Features Included

#### Base SmolLM Implementation
- Full training loop (5000 steps by default)
- Sample text generation every 500 steps
- Model checkpointing (`smollm_135_checkpoint.pth`)
- Resume training from checkpoint
- Mixed precision training (`bfloat16` via `torch.autocast`)
- Clean, well-commented code

#### Enhanced Training (DeepSeek Section)
- **Automatic checkpointing every 1000 steps** - Saves to `checkpoints/checkpoint_step_{N}.pth`
- **Inference every 100 steps** - Monitor model quality during training
- **Latest checkpoint tracking** - Always saves `checkpoints/checkpoint_latest.pth` for easy resume
- **Final checkpoint** - Saves at end of training
- **MoE load balancing statistics** - Monitor expert utilization

---

### Checkpointing & Resume Training

This implementation includes a robust checkpointing system that allows you to save and resume training at any point.

#### What's Saved in the Checkpoint

The checkpoint file contains:

| Component | Description |
|-----------|-------------|
| `step` | Current training step number |
| `model_state_dict` | Complete model weights and parameters |
| `optimizer_state_dict` | Optimizer state (learning rates, momentum, etc.) |
| `data_loader_position` | Current position in the dataset |
| `config` | Model configuration (architecture settings) |
| `loss` | Current loss value (in enhanced version) |

#### Checkpoint Locations

**Base SmolLM:**
- `smollm_135_checkpoint.pth` - Saved at end of training

**Enhanced Training (DeepSeek section):**
- `checkpoints/checkpoint_step_{N}.pth` - Saved every 1000 steps
- `checkpoints/checkpoint_latest.pth` - Always points to most recent checkpoint
- `checkpoints/checkpoint_final_step_{N}.pth` - Final checkpoint at end of training

#### Enhanced Checkpointing Features

The DeepSeek training section includes:
- **Automatic checkpointing every 1000 steps** - Never lose progress
- **Latest checkpoint tracking** - Easy resume with `checkpoint_latest.pth`
- **Inference monitoring** - Text generation every 100 steps to track quality
- **Loss tracking** - Current loss saved in each checkpoint

#### Resuming Training from Checkpoint

To resume training from a saved checkpoint:

```python
# Load checkpoint (works for both base and enhanced versions)
checkpoint = torch.load('checkpoints/checkpoint_latest.pth', weights_only=False)
start_step = checkpoint['step']

# Restore model (use appropriate model class)
# For base SmolLM:
model = SmolLM(checkpoint['config'])
# For SmolLM-DeepSeek:
# model = SmolLMDeepSeek(checkpoint['config'])

model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore data loader position
train_loader.current_position = checkpoint['data_loader_position']

# Continue training from start_step + 1
for i in range(start_step + 1, max_steps):
    # ... training code ...
```

#### Resume Training Example

The notebook demonstrates resuming training by:
1. Loading the checkpoint (from step 5000 or latest)
2. Restoring all training state (model, optimizer, data position)
3. Continuing training seamlessly without losing progress

---

### How to Run

#### Option 1: Base SmolLM-135M (Original Implementation)
```bash
# 1. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tiktoken tqdm

# 2. Place your text file
cp your_dataset.txt input.txt

# 3. Open notebook and run Cell 0 (original SmolLM implementation)
# The notebook contains the complete base implementation
```

#### Option 2: SmolLM with DeepSeek V3 Innovations (Recommended)
```bash
# 1. Install dependencies (same as above)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tiktoken tqdm

# 2. Place your text file
cp your_dataset.txt input.txt

# 3. Open smollm135_deepseek.ipynb and run cells in order:
#    - Cell 1: Setup (imports and config)
#    - Cell 2: MLA implementation
#    - Cell 3: MoE implementation  
#    - Cell 4: Complete integration (SmolLM-DeepSeek)
#    - Cell 5: Enhanced training with checkpointing & inference
```

#### Notebook Structure

The `smollm135_deepseek.ipynb` notebook is organized as follows:

1. **Cell 0**: Empty (for original SmolLM code if you want to add it)
2. **Cell 1**: Setup - Standalone execution support with all dependencies
3. **Cell 2**: Multi-Head Latent Attention (MLA) implementation
4. **Cell 3**: DeepSeekMoE with auxiliary-loss-free load balancing
5. **Cell 4**: Complete SmolLM-DeepSeek integration model
6. **Cell 5**: Enhanced training loop with:
   - Checkpointing every 1000 steps
   - Inference every 100 steps
   - MoE statistics monitoring
7. **Cell 6**: Summary and usage notes

**Note:** Cells are designed to work standalone - you can run the DeepSeek section independently!

---

### Next Steps & Ideas

| Goal                          | How to Do It |
|-----------------------------|------------|
| Train longer / better data  | Increase `max_steps`, use larger datasets |
| Use official tokenizer      | Load `HuggingFaceTB/SmolLM-135M` tokenizer from HF |
| Increase context length     | Set `block_size = 2048` (needs more VRAM) |
| Evaluate on benchmarks      | Add Hellaswag, ARC, etc. evaluation scripts |
| Export to GGUF / llama.cpp  | Use `convert_hf_to_gguf.py` after training |
| **Experiment with MLA**     | Compare memory usage vs standard attention |
| **Tune MoE configuration**  | Adjust `n_experts`, `top_k`, `moe_layer_freq` |
| **Monitor expert utilization** | Use `get_moe_statistics()` to analyze load balancing |
| **Resume from checkpoints** | Use `checkpoints/checkpoint_latest.pth` for seamless continuation |

---

### References & Credits

#### Base Architecture
- Official Model: [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- Architecture Paper: Llama 3 (Meta AI, 2024)
- RMSNorm: [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)
- SwiGLU: [Shazeer, 2020](https://arxiv.org/abs/2002.05202)
- RoPE: [Su et al., 2021](https://arxiv.org/abs/2104.09864)
- GQA: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)
- Inspiration: Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)

#### DeepSeek V3 Innovations
- **DeepSeek V3 Paper**: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) - Complete technical report
- **DeepSeek V2 Paper (MLA)**: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) - Multi-Head Latent Attention introduction
- **DeepSeek-MoE Paper**: [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) - DeepSeekMoE architecture details
- **DeepSeek GitHub**: [github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- **DeepSeek Website**: [deepseek.com](https://www.deepseek.com)

#### Additional Resources
- For comprehensive DeepSeek architecture analysis, see `DeepSeek_Architecture_Analysis.md`
- For quick reference guide, see `DeepSeek_QuickReference.md`

---

**License**: MIT