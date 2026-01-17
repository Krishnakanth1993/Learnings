"""
Phi-2 GRPO Fine-tuned Model - With Comparison
Robust error handling version.
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import time
import traceback

# Configuration
BASE_MODEL = "microsoft/phi-2"
ADAPTER_MODEL = "Krishnakanth1993/phi2-grpo-oasst1"  # Your HF repo

print("=" * 60)
print("Phi-2 GRPO Fine-tuned Assistant")
print("=" * 60)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer ready!")

# Load model
print("Loading Phi-2 model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print("✓ Base model loaded!")

# Store reference to base model for comparison
base_model_state = None

# Load LoRA adapter
adapter_loaded = False
try:
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
    adapter_loaded = True
    print("✓ LoRA adapter loaded!")
except Exception as e:
    print(f"⚠ Could not load adapter: {e}")
    print("Running in base model only mode.")

model.eval()
print("=" * 60)
print("Model ready!")
print("=" * 60)


def generate_text(prompt: str, max_tokens: int, temperature: float, disable_adapter: bool = False) -> tuple:
    """Generate text with optional adapter control."""
    adapter_was_disabled = False
    
    try:
        formatted = f"Instruct: {prompt}\nOutput:"
        inputs = tokenizer(formatted, return_tensors="pt")
        
        # Toggle adapter if requested
        if adapter_loaded and disable_adapter:
            try:
                if hasattr(model, 'disable_adapters'):
                    model.disable_adapters()
                    adapter_was_disabled = True
                elif hasattr(model, 'disable_adapter_layers'):
                    model.disable_adapter_layers()
                    adapter_was_disabled = True
            except Exception as e:
                print(f"Warning: Could not disable adapter: {e}")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        elapsed = time.time() - start_time
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Output:" in response:
            response = response.split("Output:")[-1].strip()
        
        return response, elapsed, None
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Generation error: {traceback.format_exc()}")
        return None, 0, error_msg
    
    finally:
        # ALWAYS re-enable adapter after generation
        if adapter_loaded and adapter_was_disabled:
            try:
                if hasattr(model, 'enable_adapters'):
                    model.enable_adapters()
                elif hasattr(model, 'enable_adapter_layers'):
                    model.enable_adapter_layers()
            except Exception as e:
                print(f"Warning: Could not re-enable adapter: {e}")


def generate_finetuned(prompt: str, max_tokens: int = 200, temperature: float = 0.7):
    """Generate from fine-tuned model."""
    if not prompt.strip():
        return "Please enter a prompt."
    
    response, elapsed, error = generate_text(prompt, max_tokens, temperature, disable_adapter=False)
    
    if error:
        return f"❌ {error}"
    
    return f"{response}\n\n---\n⏱️ {elapsed:.1f}s | 📝 {len(response.split())} words"


def compare_models(prompt: str, max_tokens: int = 150, temperature: float = 0.7, progress=gr.Progress()):
    """Compare base vs fine-tuned responses."""
    if not prompt.strip():
        return "Please enter a prompt.", "Please enter a prompt.", "", ""
    
    # Generate FINE-TUNED response first (adapter enabled - default state)
    progress(0.2, desc="Generating fine-tuned response...")
    ft_response, ft_time, ft_error = generate_text(prompt, max_tokens, temperature, disable_adapter=False)
    
    if ft_error:
        ft_output = f"❌ {ft_error}"
        ft_stats = "Error"
    else:
        ft_output = ft_response
        ft_stats = f"⏱️ {ft_time:.1f}s | 📝 {len(ft_response.split())} words"
    
    # Generate BASE response (try to disable adapter)
    progress(0.6, desc="Generating base model response...")
    base_response, base_time, base_error = generate_text(prompt, max_tokens, temperature, disable_adapter=True)
    
    if base_error:
        base_output = f"❌ {base_error}"
        base_stats = "Error"
    else:
        base_output = base_response
        base_stats = f"⏱️ {base_time:.1f}s | 📝 {len(base_response.split())} words"
    
    progress(1.0, desc="Done!")
    
    return base_output, ft_output, base_stats, ft_stats


def generate_single(prompt: str, max_tokens: int = 200, temperature: float = 0.7):
    """Simple single generation - ALWAYS uses fine-tuned model."""
    if not prompt.strip():
        return "Please enter a prompt."
    
    try:
        # ENSURE adapter is enabled (fine-tuned mode)
        if adapter_loaded:
            try:
                if hasattr(model, 'enable_adapters'):
                    model.enable_adapters()
                elif hasattr(model, 'enable_adapter_layers'):
                    model.enable_adapter_layers()
            except Exception:
                pass  # Adapter might already be enabled
        
        formatted = f"Instruct: {prompt}\nOutput:"
        inputs = tokenizer(formatted, return_tensors="pt")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        elapsed = time.time() - start_time
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Output:" in response:
            response = response.split("Output:")[-1].strip()
        
        model_type = "Fine-tuned" if adapter_loaded else "Base"
        return f"{response}\n\n---\n🤖 {model_type} | ⏱️ {elapsed:.1f}s | 📝 {len(response.split())} words"
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease try again or use a shorter prompt."


# Build the Gradio interface
with gr.Blocks(title="Phi-2 GRPO Assistant") as demo:
    
    status = "✅ Fine-tuned model loaded" if adapter_loaded else "⚠️ Base model only"
    
    gr.Markdown(f"""
    # 🤖 Phi-2 GRPO Fine-tuned Assistant
    
    Fine-tuned using **GRPO + QLoRA** on the OpenAssistant dataset.
    
    **Status:** {status}
    
    > *Running on CPU - responses take 30-60 seconds*
    """)
    
    with gr.Tabs():
        # Tab 1: Single Generation (Most Reliable)
        with gr.TabItem("⚡ Generate Response"):
            gr.Markdown("Get a response from the fine-tuned model.")
            
            prompt_single = gr.Textbox(
                label="Your Question",
                placeholder="Ask me anything...",
                lines=3,
            )
            
            with gr.Row():
                max_tokens_single = gr.Slider(50, 400, 200, step=25, label="Max Tokens")
                temp_single = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="Temperature")
            
            generate_btn = gr.Button("⚡ Generate", variant="primary", size="lg")
            
            single_output = gr.Textbox(label="Response", lines=12)
            
            generate_btn.click(
                fn=generate_single,
                inputs=[prompt_single, max_tokens_single, temp_single],
                outputs=single_output,
            )
            
            prompt_single.submit(
                fn=generate_single,
                inputs=[prompt_single, max_tokens_single, temp_single],
                outputs=single_output,
            )
        
        # Tab 2: Comparison (Experimental)
        with gr.TabItem("🔄 Compare (Experimental)"):
            gr.Markdown("""
            **Experimental:** Compare base vs fine-tuned responses.  
            *Note: Adapter toggling may not work on all PEFT versions.*
            """)
            
            prompt_compare = gr.Textbox(
                label="Enter your prompt",
                placeholder="Ask a question...",
                lines=3,
            )
            
            with gr.Row():
                max_tokens_cmp = gr.Slider(50, 200, 150, step=25, label="Max Tokens")
                temp_cmp = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="Temperature")
            
            compare_btn = gr.Button("🔄 Compare", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🟠 Base Phi-2")
                    base_output = gr.Textbox(label="Base Response", lines=8)
                    base_stats = gr.Markdown()
                
                with gr.Column():
                    gr.Markdown("### 🟢 GRPO Fine-tuned")
                    ft_output = gr.Textbox(label="Fine-tuned Response", lines=8)
                    ft_stats = gr.Markdown()
            
            compare_btn.click(
                fn=compare_models,
                inputs=[prompt_compare, max_tokens_cmp, temp_cmp],
                outputs=[base_output, ft_output, base_stats, ft_stats],
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["What is machine learning?"],
            ["Explain recursion with an example."],
            ["Write a poem about AI."],
            ["What are the benefits of renewable energy?"],
        ],
        inputs=prompt_single,
        label="📝 Examples"
    )
    
    gr.Markdown("""
    ---
    **Model:** microsoft/phi-2 | **Training:** GRPO + QLoRA on OASST1
    """)

if __name__ == "__main__":
    demo.launch()
