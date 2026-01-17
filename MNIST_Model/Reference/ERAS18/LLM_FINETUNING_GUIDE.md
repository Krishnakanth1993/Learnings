# Complete Guide to LLM Fine-Tuning with GRPO, QLoRA, and Reward Modeling

> A comprehensive guide for understanding and implementing LLM fine-tuning, from high-level concepts to mathematical details.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Core Concepts Explained](#2-core-concepts-explained)
   - [Fine-Tuning Fundamentals](#21-fine-tuning-fundamentals)
   - [QLoRA: Memory-Efficient Training](#22-qlora-memory-efficient-training)
   - [GRPO: Preference Learning](#23-grpo-group-relative-policy-optimization)
   - [Reward Functions](#24-reward-functions-shaping-model-behavior)
3. [Deep Dive: The Mathematics](#3-deep-dive-the-mathematics)
   - [LoRA Math: Low-Rank Decomposition](#31-lora-mathematics)
   - [GRPO Algorithm Details](#32-grpo-algorithm-in-depth)
   - [Quantization Theory](#33-quantization-preserving-quality)
   - [Reward Engineering](#34-reward-function-design)
4. [Our Implementation](#4-our-implementation)
5. [References & Learning Resources](#5-references--learning-resources)

---

## 1. High-Level Overview

### What Are We Building?

```
┌─────────────────┐     Fine-tuning      ┌─────────────────────┐
│   Phi-2 Model   │ ──────────────────▶  │  Your Custom Model  │
│  (General AI)   │   + OASST1 data      │ (Better Assistant)  │
└─────────────────┘                      └─────────────────────┘
```

We're taking Microsoft's **Phi-2** (a 2.7 billion parameter language model) and training it to be a better conversational assistant using:

| Component | Purpose |
|-----------|---------|
| **OASST1 Dataset** | Real human-AI conversations with quality ratings |
| **QLoRA** | Memory-efficient training (fits on consumer GPUs) |
| **GRPO** | Learning from preferences (not just copying answers) |
| **Reward Function** | Defines what "good response" means |

### The Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────┘

     OASST1 Dataset                    Phi-2 Model
          │                                 │
          ▼                                 ▼
   ┌──────────────┐                ┌────────────────┐
   │ Extract      │                │ Load in 4-bit  │
   │ English Q&A  │                │ (Quantization) │
   │ pairs with   │                │                │
   │ quality > 0.5│                │ Add LoRA       │
   └──────┬───────┘                │ adapters       │
          │                        └───────┬────────┘
          │                                │
          ▼                                ▼
   ┌─────────────────────────────────────────────────┐
   │              GRPO TRAINING LOOP                 │
   │                                                 │
   │  For each prompt:                               │
   │    1. Model generates 2 responses               │
   │    2. Reward function scores both               │
   │    3. Model learns to prefer better response    │
   │                                                 │
   │  Repeat for ~10,000 prompts                     │
   └─────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Fine-tuned      │
              │  LoRA Adapter    │
              │  (~500MB)        │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Deploy to       │
              │  HuggingFace     │
              │  Space           │
              └──────────────────┘
```

### Why Each Component?

| Component | The Problem It Solves |
|-----------|----------------------|
| **Quantization** | Phi-2 needs 10GB+ in full precision; we only have 15GB on T4 |
| **LoRA** | Training 2.7B parameters is slow and memory-intensive |
| **GRPO** | Simple fine-tuning just copies answers; GRPO learns *why* answers are good |
| **Quality Labels** | Need to know which responses to prefer |

---

## 2. Core Concepts Explained

### 2.1 Fine-Tuning Fundamentals

**Pre-training** vs **Fine-tuning**:

| Pre-training | Fine-tuning |
|--------------|-------------|
| Train from scratch on massive data | Adapt existing model to specific task |
| Learn general language patterns | Learn task-specific patterns |
| Requires enormous compute | Can be done on consumer hardware |
| Months of training | Hours to days |

**Analogy**: Pre-training is like medical school (general knowledge). Fine-tuning is like residency (specialization).

**Types of Fine-tuning**:

```
1. FULL FINE-TUNING
   - Update all parameters
   - Best quality, highest cost
   - Risk of "catastrophic forgetting"

2. SUPERVISED FINE-TUNING (SFT)
   - Show model: input → desired output
   - Model learns to copy good examples
   - Simple but limited

3. REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF)
   - Train reward model on preferences
   - Use RL to optimize for reward
   - Complex but powerful (used in ChatGPT)

4. PARAMETER-EFFICIENT FINE-TUNING (PEFT)
   - Only train small adapter layers
   - Freeze main model weights
   - LoRA, Prefix Tuning, Adapters
```

### 2.2 QLoRA: Memory-Efficient Training

**QLoRA** = **Q**uantization + **Lo**w-**R**ank **A**daptation

#### Part 1: Quantization (The "Q")

Reduces model memory by using fewer bits per number:

```
FP32 (standard):    32 bits per parameter → 10.8 GB for Phi-2
FP16 (half):        16 bits per parameter → 5.4 GB
INT8 (quantized):    8 bits per parameter → 2.7 GB
INT4 (aggressive):   4 bits per parameter → 1.35 GB
```

**NF4** (Normal Float 4-bit): Special 4-bit format optimized for neural network weight distributions.

#### Part 2: LoRA (Low-Rank Adaptation)

Instead of updating all 2.7B parameters, add small trainable "adapters":

```
Original: 2,700,000,000 parameters (frozen)
LoRA:         8,000,000 parameters (trainable)
              ────────────────────────────────
              Only 0.3% of parameters trained!
```

**Combined Effect**:
- Model fits in ~3GB instead of 10GB
- Training is 10x faster
- Can fine-tune on free Google Colab (T4 GPU)

### 2.3 GRPO: Group Relative Policy Optimization

**The Evolution of Training Methods**:

```
SFT (Supervised Fine-Tuning)
│   "Here's the right answer, copy it"
│   ✓ Simple
│   ✗ Only learns to mimic
▼
RLHF (Reinforcement Learning from Human Feedback)  
│   "Learn from human preferences using a reward model"
│   ✓ Learns WHY responses are good
│   ✗ Complex: needs 4 models in memory
▼
DPO (Direct Preference Optimization)
│   "Skip the reward model, learn preferences directly"
│   ✓ Simpler than RLHF
│   ✗ Still needs paired preference data
▼
GRPO (Group Relative Policy Optimization)
    "Generate multiple responses, prefer the better ones"
    ✓ Simple as DPO
    ✓ Works with any reward function
    ✓ Only needs prompts, not paired data
```

**How GRPO Works**:

```
For each prompt:

1. GENERATE multiple responses
   Prompt: "What is AI?"
   Response A: "AI is artificial intelligence, a field..."
   Response B: "idk its like robots or something"

2. SCORE each response
   Score(A) = 0.85 (good)
   Score(B) = 0.25 (bad)

3. NORMALIZE within group
   Â = +1.0 (above average)
   B̂ = -1.0 (below average)

4. UPDATE model
   Increase probability of A-like responses
   Decrease probability of B-like responses
```

### 2.4 Reward Functions: Shaping Model Behavior

**The reward function defines what "good" means.**

Our reward combines three signals:

```
REWARD = 0.4 × length_score + 0.4 × coherence_score + 0.2 × format_score

length_score:    Prefer 50-300 words (not too short, not too long)
coherence_score: Penalize repetition (unique words / total words)
format_score:    Reward complete sentences (ends with . ! ? ")
```

**Why multiple signals?** Prevents "reward hacking":
- Length only → "AAAAAAAAAA..." (infinite gibberish)
- Keywords only → Unnatural keyword stuffing
- Balanced metrics → Genuine quality

---

## 3. Deep Dive: The Mathematics

### 3.1 LoRA Mathematics

#### The Core Idea: Low-Rank Decomposition

A weight matrix W of size (d × d) has d² parameters. But for fine-tuning, the *changes* to W often lie in a low-dimensional subspace.

**LoRA Decomposition**:

```
W_new = W_original + ΔW
      = W_original + A × B

Where:
- W_original: (d × d) matrix, FROZEN
- A: (d × r) matrix, trainable
- B: (r × d) matrix, trainable
- r: rank (typically 8-64, we use 16)
```

**Parameter Savings**:

```
Full fine-tuning:  d × d parameters
LoRA:              d × r + r × d = 2 × d × r parameters

Example with d=2048, r=16:
Full:  2048 × 2048 = 4,194,304 parameters
LoRA:  2 × 2048 × 16 = 65,536 parameters

Reduction: 64× fewer parameters!
```

#### Forward Pass with LoRA

```python
# Original forward pass
output = input @ W

# With LoRA
output = input @ W + (input @ A @ B) × (α/r)

# Where α (alpha) is a scaling factor
# α/r controls how much LoRA affects output
```

#### Why It Works

The **rank** of a matrix represents its "degrees of freedom." Fine-tuning for a specific task only needs a few degrees of freedom—the model already knows language, we're just steering it.

```
Intuition:
- Full weight space: millions of dimensions
- Task-specific changes: live in ~16-64 dimensions
- LoRA captures this subspace efficiently
```

### 3.2 GRPO Algorithm In-Depth

#### Mathematical Formulation

Given:
- Policy πθ (the model with parameters θ)
- Prompt p
- Reward function R

GRPO generates G responses and optimizes:

```
L_GRPO = -E[Σᵢ r̂ᵢ × log πθ(rᵢ|p)]

Where:
- rᵢ = i-th generated response
- r̂ᵢ = normalized reward = (R(rᵢ) - μ) / σ
- μ = mean reward in group
- σ = std of rewards in group
```

#### Step-by-Step Algorithm

```
ALGORITHM: GRPO Training
────────────────────────────────────────────────────────────
Input: Dataset D of prompts, model πθ, reward function R
Output: Fine-tuned model πθ*

for epoch in 1..num_epochs:
    for batch of prompts P in D:
        
        # Step 1: Generate G responses per prompt
        for each prompt p in P:
            responses[p] = [πθ.generate(p) for _ in range(G)]
        
        # Step 2: Compute rewards
        for each prompt p in P:
            rewards[p] = [R(r) for r in responses[p]]
        
        # Step 3: Normalize rewards within each group
        for each prompt p in P:
            μ = mean(rewards[p])
            σ = std(rewards[p])
            normalized[p] = [(r - μ) / σ for r in rewards[p]]
        
        # Step 4: Compute loss
        loss = 0
        for each prompt p in P:
            for i in range(G):
                log_prob = πθ.log_probability(responses[p][i] | p)
                loss -= normalized[p][i] * log_prob
        
        # Step 5: Update model
        θ = θ - learning_rate * ∇loss
────────────────────────────────────────────────────────────
```

#### Why Normalization Matters

**Without normalization**:
- If all responses score high (0.8, 0.9), gradients are all positive
- Model just increases probability of everything
- No learning signal about *which* is better

**With normalization**:
- Scores become (+1, -1) after normalization
- Clear signal: increase prob of first, decrease prob of second
- Always learning, regardless of absolute reward scale

```
Before:  [0.9, 0.8]  →  Both positive, unclear preference
After:   [+1, -1]    →  Clear: prefer first over second
```

### 3.3 Quantization: Preserving Quality

#### The Quantization Problem

We need to map continuous values to discrete levels:

```
Continuous weight: 0.0523847...
4-bit can only represent 16 values: {v₀, v₁, ..., v₁₅}
We must pick the closest: vᵢ = 0.0796

Quantization error: |0.0523 - 0.0796| = 0.0273
```

#### Uniform vs Non-Uniform Quantization

**Uniform** (standard INT4):
```
├───┼───┼───┼───┼───┼───┼───┼───┤
-1                 0                 +1
Equal spacing, wastes precision where weights are sparse
```

**Non-Uniform** (NF4):
```
├┼┼─┼──┼────┼────────┼────┼──┼─┼┼┤
-1       -0.3    0    0.3        +1
More precision near 0 where most weights cluster
```

#### NF4: Normal Float 4-bit

NF4 levels are computed to minimize quantization error for normally-distributed weights:

```python
# The 16 NF4 quantization levels
nf4_levels = [
    -1.0, -0.6962, -0.5251, -0.3949, 
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 
    0.4407, 0.5626, 0.7230, 1.0
]

# Note: More levels clustered near 0
# Matches the distribution of neural network weights
```

#### Double Quantization

Additional memory savings by quantizing the scale factors:

```
Standard: Each 64-weight block has FP32 scale = 4 bytes overhead
Double:   Each 64-weight block has INT8 scale = 1 byte overhead
          Plus one FP32 "scale of scales" per 256 blocks

Memory savings: ~0.4 bits per parameter
```

#### Why Quality is Preserved

1. **Distribution matching**: NF4 matches weight distribution
2. **Block-wise scaling**: Local adaptation to weight ranges
3. **LoRA compensation**: Trainable adapters correct errors

```
Effective weight = Quantized(W) + A × B
                   └─────┬─────┘   └──┬──┘
                   Has errors    Learns corrections
```

### 3.4 Reward Function Design

#### The Reward Shaping Problem

The model optimizes **exactly what you reward**:

```
Reward = length           → Infinite rambling
Reward = "helpful" count  → "helpful helpful helpful..."
Reward = user happiness   → Sycophantic agreement
```

#### Multi-Objective Reward Design

Our reward balances three objectives:

```python
def reward_function(response):
    # Objective 1: Appropriate length
    words = len(response.split())
    if words < 10:
        length_score = 0.1
    elif words < 50:
        length_score = words / 50
    elif words <= 300:
        length_score = 1.0
    else:
        length_score = max(0.5, 1.0 - (words - 300) / 500)
    
    # Objective 2: Coherence (anti-repetition)
    unique_ratio = len(set(words)) / len(words)
    coherence_score = min(1.0, unique_ratio * 1.2)
    
    # Objective 3: Completeness
    format_score = 1.0 if response.endswith(('.','!','?')) else 0.7
    
    # Weighted combination
    return 0.4 * length_score + 0.4 * coherence_score + 0.2 * format_score
```

#### Visualizing the Reward Landscape

```
Length Component:
Reward
1.0 │              ┌────────────────┐
    │             ╱                  ╲
0.5 │            ╱                    ╲────
    │           ╱
0.0 │──────────╱
    └───────────┬─────┬─────┬─────┬───────▶ Words
               10    50   200   300   500

Coherence Component:
Reward
1.0 │                          ┌──────────
    │                         ╱
0.5 │                        ╱
    │                       ╱
0.0 │──────────────────────╱
    └───────────┬─────┬─────┬─────────────▶ Unique Ratio
               0.3   0.5   0.8

Combined: Model must satisfy ALL objectives for high reward
```

---

## 4. Our Implementation

### Configuration Summary

```python
# Model
MODEL_ID = "microsoft/phi-2"  # 2.7B parameter base model

# QLoRA Settings
load_in_4bit = True           # 4-bit quantization
bnb_4bit_quant_type = "nf4"   # Normal Float 4-bit
bnb_4bit_use_double_quant = True

# LoRA Settings
r = 16                        # Rank
lora_alpha = 32               # Scaling (α/r = 2)
target_modules = ["q_proj", "k_proj", "v_proj", "dense"]

# GRPO Settings
num_generations = 2           # Responses per prompt
max_length = 512              # Sequence length
batch_size = 2
gradient_accumulation = 8     # Effective batch = 16

# Dataset
max_samples = 10000           # Training examples
min_quality = 0.5             # Quality threshold
```

### Memory Breakdown (T4 GPU - 15GB)

```
Component                    Memory
─────────────────────────────────────
Base model (4-bit)           ~3.0 GB
LoRA adapters                ~0.2 GB
Optimizer states             ~0.4 GB
Gradients                    ~0.3 GB
Activations (batch=2)        ~6.0 GB
GRPO generations             ~3.0 GB
─────────────────────────────────────
Total                       ~13.0 GB  ✓ Fits!
```

---

## 5. References & Learning Resources

### 📄 Foundational Papers

#### LoRA & Parameter-Efficient Fine-Tuning

| Paper | Description | Link |
|-------|-------------|------|
| **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) | Original LoRA paper | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023) | Combines quantization with LoRA | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| **LLM.int8(): 8-bit Matrix Multiplication** (Dettmers et al., 2022) | Foundation for quantization | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |

#### RLHF & Preference Learning

| Paper | Description | Link |
|-------|-------------|------|
| **Training Language Models to Follow Instructions with Human Feedback** (Ouyang et al., 2022) | InstructGPT/RLHF | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| **Direct Preference Optimization** (Rafailov et al., 2023) | DPO - simpler alternative to RLHF | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| **GRPO: Group Relative Policy Optimization** | TRL implementation | [HuggingFace TRL Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) |
| **DeepSeekMath: GRPO** (Shao et al., 2024) | GRPO in math reasoning | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |

#### Quantization

| Paper | Description | Link |
|-------|-------------|------|
| **GPTQ: Accurate Post-Training Quantization** (Frantar et al., 2022) | Weight quantization | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| **AWQ: Activation-aware Weight Quantization** (Lin et al., 2023) | Improved quantization | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |

### 📚 Articles & Blog Posts

#### Conceptual Understanding

| Article | Description | Link |
|---------|-------------|------|
| **Hugging Face PEFT Documentation** | Official LoRA/PEFT guide | [HuggingFace PEFT](https://huggingface.co/docs/peft) |
| **Hugging Face TRL Documentation** | GRPO Trainer guide | [HuggingFace TRL](https://huggingface.co/docs/trl) |
| **The Illustrated Transformer** | Visual transformer explanation | [Jay Alammar](https://jalammar.github.io/illustrated-transformer/) |
| **RLHF: Reinforcement Learning from Human Feedback** | Chip Huyen's guide | [Chip Huyen Blog](https://huyenchip.com/2023/05/02/rlhf.html) |
| **A Visual Guide to Quantization** | Understanding quantization | [Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) |

#### Practical Tutorials

| Article | Description | Link |
|---------|-------------|------|
| **Fine-tuning LLMs with QLoRA** | Step-by-step guide | [Hugging Face Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes) |
| **GRPO Training Tutorial** | Official TRL example | [TRL Examples](https://github.com/huggingface/trl/tree/main/examples/scripts) |
| **OpenAssistant Dataset** | Dataset documentation | [HuggingFace Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) |

### 🎥 YouTube Courses & Videos

#### Comprehensive Courses

| Course | Description | Link |
|--------|-------------|------|
| **Andrej Karpathy - Let's build GPT** | Building transformers from scratch | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| **Andrej Karpathy - State of GPT** | Understanding modern LLM training | [YouTube](https://www.youtube.com/watch?v=bZQun8Y4L2A) |
| **Umar Jamil - Attention is All You Need** | Transformer deep dive | [YouTube](https://www.youtube.com/watch?v=bCz4OMemCcA) |
| **DeepLearning.AI - Finetuning Large Language Models** | Short course on fine-tuning | [DeepLearning.AI](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/) |

#### Specific Topics

| Video | Topic | Link |
|-------|-------|------|
| **Yannic Kilcher - LoRA Explained** | LoRA paper walkthrough | [YouTube](https://www.youtube.com/watch?v=PXWYUTMt-AU) |
| **Yannic Kilcher - QLoRA** | QLoRA paper explanation | [YouTube](https://www.youtube.com/watch?v=y9PHWGOa8HA) |
| **Umar Jamil - RLHF Explained** | Complete RLHF breakdown | [YouTube](https://www.youtube.com/watch?v=2MBJOuVq380) |
| **Umar Jamil - DPO Explained** | Direct Preference Optimization | [YouTube](https://www.youtube.com/watch?v=XZLc09hkMwA) |
| **AI Coffee Break - Quantization** | LLM quantization explained | [YouTube](https://www.youtube.com/watch?v=mJRMk2EBdmY) |

#### Hands-On Tutorials

| Video | Description | Link |
|-------|-------------|------|
| **Trelis Research - QLoRA Fine-tuning** | Practical QLoRA tutorial | [YouTube](https://www.youtube.com/watch?v=XpoKB3usmKc) |
| **Sam Witteveen - TRL Training** | Using TRL library | [YouTube](https://www.youtube.com/watch?v=l5bg8KQLaT0) |
| **1littlecoder - Fine-tune LLMs** | Step-by-step fine-tuning | [YouTube](https://www.youtube.com/watch?v=eC6Hd1hFvos) |

### 🛠️ Libraries & Tools

| Tool | Purpose | Link |
|------|---------|------|
| **Transformers** | Base model loading | [GitHub](https://github.com/huggingface/transformers) |
| **PEFT** | LoRA implementation | [GitHub](https://github.com/huggingface/peft) |
| **TRL** | GRPO/RLHF training | [GitHub](https://github.com/huggingface/trl) |
| **bitsandbytes** | Quantization | [GitHub](https://github.com/TimDettmers/bitsandbytes) |
| **Accelerate** | Distributed training | [GitHub](https://github.com/huggingface/accelerate) |

### 📊 Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **OpenAssistant OASST1** | Human-AI conversations with ratings | [HuggingFace](https://huggingface.co/datasets/OpenAssistant/oasst1) |
| **Anthropic HH-RLHF** | Helpful/Harmless preference data | [HuggingFace](https://huggingface.co/datasets/Anthropic/hh-rlhf) |
| **UltraFeedback** | Large preference dataset | [HuggingFace](https://huggingface.co/datasets/openbmb/UltraFeedback) |

### 🎓 Recommended Learning Path

```
Week 1: Foundations
├── Watch: "Let's build GPT" by Karpathy
├── Read: "The Illustrated Transformer"
└── Practice: Run a base model inference

Week 2: Fine-tuning Basics
├── Watch: "State of GPT" by Karpathy
├── Read: HuggingFace PEFT documentation
├── Watch: "LoRA Explained" by Yannic Kilcher
└── Practice: Simple LoRA fine-tuning

Week 3: Quantization & Efficiency
├── Read: QLoRA paper
├── Watch: "QLoRA" by Yannic Kilcher
├── Read: "Visual Guide to Quantization"
└── Practice: Run QLoRA training

Week 4: RLHF & Preferences
├── Watch: "RLHF Explained" by Umar Jamil
├── Read: DPO paper
├── Read: TRL GRPO documentation
└── Practice: GRPO training (this notebook!)

Week 5: Advanced Topics
├── Study: Reward function design
├── Read: DeepSeekMath GRPO paper
├── Experiment: Custom reward functions
└── Deploy: HuggingFace Space
```

---

## Summary

This guide covered the complete stack for modern LLM fine-tuning:

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Memory** | Quantization (NF4) | Fit model in GPU memory |
| **Compute** | LoRA | Train only ~0.3% of parameters |
| **Learning** | GRPO | Learn from preferences, not just examples |
| **Signal** | Reward Function | Define what "good" means mathematically |

The combination enables training on consumer hardware (Google Colab T4) while achieving results comparable to full fine-tuning.

---

*Last updated: January 2026*
