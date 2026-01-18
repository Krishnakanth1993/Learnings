# GRPO Fine-tuning Phi-2 with QLoRA on OASST1

This project fine-tunes Microsoft's **Phi-2** model using **GRPO (Group Relative Policy Optimization)** with **QLoRA** on the **OpenAssistant/oasst1** dataset.

## Overview

- **Base Model**: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- **Dataset**: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- **Training Method**: GRPO (Group Relative Policy Optimization) from TRL
- **Optimization**: QLoRA (4-bit quantization + LoRA adapters)
- **Hardware**: Optimized for Google Colab Free Tier (T4 GPU)

---

## Architecture

### High-Level Training Pipeline

```mermaid
flowchart LR
    subgraph DataPipeline [Data Pipeline]
        A[OASST1 Dataset] --> B[Filter English]
        B --> C[Extract QA Pairs]
        C --> D[Compute Quality Rewards]
    end
    
    subgraph ModelPipeline [Model Pipeline]
        E[Phi-2 Base] --> F[4-bit Quantization]
        F --> G[Add LoRA Adapters]
    end
    
    subgraph Training [GRPO Training]
        D --> H[GRPO Trainer]
        G --> H
        H --> I[Fine-tuned Adapter]
    end
    
    subgraph Deployment [Deployment]
        I --> J[HuggingFace Hub]
        J --> K[Gradio Space]
    end
```

### QLoRA Architecture

```mermaid
flowchart TB
    subgraph Original [Original Phi-2 Layer]
        W[Weight Matrix W<br>2048 x 2048<br>4M params]
    end
    
    subgraph QLoRA [QLoRA Optimization]
        Q[4-bit Quantized W<br>~1MB instead of 16MB]
        A[LoRA Matrix A<br>2048 x 16<br>32K params]
        B[LoRA Matrix B<br>16 x 2048<br>32K params]
    end
    
    subgraph Output [Forward Pass]
        Q --> |Frozen| O[Output]
        A --> |Trainable| AB[A x B]
        B --> AB
        AB --> |Add| O
    end
    
    Original -.-> |Compress| Q
    Original -.-> |Decompose| A
    Original -.-> |Decompose| B
```

### GRPO Training Flow

```mermaid
flowchart TD
    P[Prompt] --> G1[Generate Response 1]
    P --> G2[Generate Response 2]
    
    G1 --> R1[Reward: 0.8]
    G2 --> R2[Reward: 0.3]
    
    R1 --> N[Normalize Rewards]
    R2 --> N
    
    N --> N1["Normalized: +1.0 ✓"]
    N --> N2["Normalized: -1.0 ✗"]
    
    N1 --> U[Update Policy]
    N2 --> U
    
    U --> |Increase prob| Better[Better Responses]
    U --> |Decrease prob| Worse[Worse Responses]
```

### Reward Function Components

```mermaid
pie title Reward Weights
    "Length Score" : 40
    "Coherence Score" : 40
    "Format Score" : 20
```

### Deployment Architecture

```mermaid
flowchart LR
    subgraph Training [Google Colab]
        T1[Load Phi-2 + QLoRA]
        T2[GRPO Training]
        T3[Save LoRA Adapter]
        T1 --> T2 --> T3
    end
    
    subgraph Hub [HuggingFace Hub]
        H1[Model Repo<br>LoRA Weights]
        H2[Space Repo<br>Gradio App]
    end
    
    subgraph Inference [HF Space Runtime]
        I1[Load Phi-2]
        I2[Load LoRA Adapter]
        I3[Gradio Interface]
        I1 --> I2 --> I3
    end
    
    T3 --> |Push| H1
    T3 --> |Deploy| H2
    H1 --> |Pull| I2
    H2 --> |Run| I3
```

---

## Files

| File | Description |
|------|-------------|
| `train_grpo.ipynb` | Main training notebook for Google Colab |
| `data_utils.py` | Dataset preprocessing utilities |
| `app.py` | Gradio app for Hugging Face Space deployment |
| `requirements.txt` | Dependencies for the HF Space |
| `LLM_FINETUNING_GUIDE.md` | Comprehensive learning guide |

---

## Quick Start

### Training (Google Colab)

1. Open `train_grpo.ipynb` in Google Colab
2. Ensure you have a T4 GPU runtime selected
3. Add your Hugging Face token to Colab Secrets (key: `HF_TOKEN`)
4. Run all cells sequentially

### Training Configuration

```python
MAX_LENGTH = 512        # Reduced sequence length
BATCH_SIZE = 2          # Small batch size
GRADIENT_ACCUMULATION = 8
NUM_GENERATIONS = 2     # GRPO generations per prompt
```

### QLoRA Configuration

```python
# 4-bit quantization
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
bnb_4bit_use_double_quant=True

# LoRA
r=16
lora_alpha=32
target_modules=["q_proj", "k_proj", "v_proj", "dense"]
```

---

## How It Works

### Data Flow

```mermaid
flowchart LR
    A[Raw OASST1<br>84K messages] --> B{Filter}
    B --> |English only| C[~40K messages]
    C --> D{Extract Pairs}
    D --> |Prompter → Assistant| E[~15K pairs]
    E --> F{Quality Filter}
    F --> |score > 0.5| G[~10K high-quality pairs]
    G --> H[Training Dataset]
```

### Memory Optimization

```mermaid
flowchart TD
    subgraph Before [Without Optimization]
        B1[Phi-2 FP32<br>10.8 GB]
        B2[Gradients<br>10.8 GB]
        B3[Optimizer<br>21.6 GB]
        B1 --> BT[Total: 43+ GB ❌]
        B2 --> BT
        B3 --> BT
    end
    
    subgraph After [With QLoRA]
        A1[Phi-2 4-bit<br>~3 GB]
        A2[LoRA Grads<br>~0.2 GB]
        A3[Optimizer<br>~0.4 GB]
        A1 --> AT[Total: ~12 GB ✓]
        A2 --> AT
        A3 --> AT
    end
```

---

## Deployment

After training, the notebook will:
1. Push the LoRA adapter to your Hugging Face Hub
2. Create and deploy a Gradio Space

### Manual Space Deployment

1. Create a new Space on Hugging Face (Gradio SDK)
2. Update `ADAPTER_MODEL` in `app.py` with your model repo
3. Upload `app.py` and `requirements.txt` to the Space

### App Features

```mermaid
flowchart TB
    subgraph App [Gradio App]
        T1[Generate Response Tab]
        T2[Compare Models Tab]
    end
    
    subgraph Generate [Single Generation]
        T1 --> G1[Enable Adapter]
        G1 --> G2[Generate]
        G2 --> G3[Fine-tuned Response]
    end
    
    subgraph Compare [Comparison Mode]
        T2 --> C1[Disable Adapter]
        C1 --> C2[Base Response]
        C2 --> C3[Enable Adapter]
        C3 --> C4[Fine-tuned Response]
        C4 --> C5[Side-by-Side Display]
    end
```

---

## Reward Function

The GRPO training uses a custom reward function:

```mermaid
flowchart LR
    R[Response] --> L[Length Check<br>50-300 words]
    R --> C[Coherence Check<br>unique/total ratio]
    R --> F[Format Check<br>ends with . ! ?]
    
    L --> |40%| S[Combined Score]
    C --> |40%| S
    F --> |20%| S
    
    S --> N[Normalize to -1,1]
    N --> G[GRPO Update]
```

---

## Expected Results

| Metric | Value |
|--------|-------|
| Training Time | ~4-6 hours on T4 GPU |
| GPU Memory Usage | ~12-14 GB |
| Adapter Size | ~500 MB |
| Trainable Parameters | ~0.3% of total |

---

## Requirements

```
transformers>=4.36.0
trl>=0.7.0
peft>=0.7.0
bitsandbytes>=0.41.0
datasets
accelerate
gradio>=4.0.0
huggingface_hub
```

---

## References

- [TRL GRPO Trainer Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [PEFT/LoRA Documentation](https://huggingface.co/docs/peft)
- [Phi-2 Model Card](https://huggingface.co/microsoft/phi-2)
- [OASST1 Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)

---

## License

This project is for educational purposes. Please refer to the licenses of the underlying models and datasets.
