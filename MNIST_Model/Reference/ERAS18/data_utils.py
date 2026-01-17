"""
Data utilities for preprocessing the OpenAssistant/oasst1 dataset
for GRPO training with quality-based rewards.
"""

from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import numpy as np


def load_oasst1_dataset(split: str = "train") -> Dict:
    """Load the OpenAssistant oasst1 dataset."""
    dataset = load_dataset("OpenAssistant/oasst1", split=split)
    return dataset


def extract_quality_reward(labels: Dict) -> float:
    """
    Extract reward from OASST1 quality labels.
    
    The labels dict contains:
    - name: list of label names
    - value: list of label values (0-1 scale)
    - count: list of annotation counts
    
    We focus on 'quality' and 'helpfulness' labels.
    """
    if labels is None:
        return 0.5  # Default neutral reward
    
    names = labels.get("name", [])
    values = labels.get("value", [])
    
    quality_score = 0.5
    helpfulness_score = 0.5
    
    for name, value in zip(names, values):
        if name == "quality":
            quality_score = value
        elif name == "helpfulness":
            helpfulness_score = value
    
    # Combine scores: 60% quality, 40% helpfulness
    base_reward = 0.6 * quality_score + 0.4 * helpfulness_score
    
    return base_reward


def build_conversation_trees(dataset) -> Dict[str, Dict]:
    """
    Build conversation trees from the dataset.
    Returns a dict mapping message_id to message data.
    """
    messages = {}
    for item in dataset:
        messages[item["message_id"]] = {
            "text": item["text"],
            "role": item["role"],
            "parent_id": item["parent_id"],
            "labels": item["labels"],
            "lang": item["lang"],
            "rank": item["rank"],
            "message_tree_id": item["message_tree_id"],
        }
    return messages


def extract_prompt_response_pairs(
    dataset,
    lang: str = "en",
    min_quality: float = 0.5,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """
    Extract prompt-response pairs with rewards from the OASST1 dataset.
    
    Args:
        dataset: The loaded OASST1 dataset
        lang: Language filter (default: English)
        min_quality: Minimum quality threshold for filtering
        max_samples: Maximum number of samples to return
    
    Returns:
        List of dicts with keys: prompt, response, reward
    """
    # Build message lookup
    messages = build_conversation_trees(dataset)
    
    pairs = []
    
    for msg_id, msg_data in messages.items():
        # We want assistant responses
        if msg_data["role"] != "assistant":
            continue
        
        # Language filter
        if msg_data["lang"] != lang:
            continue
        
        # Get parent (should be prompter)
        parent_id = msg_data["parent_id"]
        if parent_id is None or parent_id not in messages:
            continue
        
        parent = messages[parent_id]
        
        # Parent should be prompter
        if parent["role"] != "prompter":
            continue
        
        # Language filter on parent too
        if parent["lang"] != lang:
            continue
        
        # Calculate reward from quality labels
        reward = extract_quality_reward(msg_data["labels"])
        
        # Filter by minimum quality
        if reward < min_quality:
            continue
        
        # Extract the pair
        pair = {
            "prompt": parent["text"],
            "response": msg_data["text"],
            "reward": reward,
        }
        
        pairs.append(pair)
        
        if max_samples and len(pairs) >= max_samples:
            break
    
    return pairs


def format_prompt_for_phi2(prompt: str) -> str:
    """
    Format a prompt for Phi-2 model.
    Phi-2 uses a simple instruction format.
    """
    return f"Instruct: {prompt}\nOutput:"


def prepare_grpo_dataset(
    pairs: List[Dict],
    tokenizer,
    max_length: int = 512
) -> List[Dict]:
    """
    Prepare dataset for GRPO training.
    
    Args:
        pairs: List of prompt-response-reward dicts
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        List of formatted training examples
    """
    formatted_data = []
    
    for pair in pairs:
        formatted_prompt = format_prompt_for_phi2(pair["prompt"])
        
        # Check if the formatted example fits within max_length
        full_text = formatted_prompt + " " + pair["response"]
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        
        if len(tokens) <= max_length:
            formatted_data.append({
                "prompt": formatted_prompt,
                "completion": pair["response"],
                "reward": pair["reward"],
            })
    
    return formatted_data


def create_reward_function(quality_weight: float = 0.6, length_penalty_threshold: int = 50):
    """
    Create a reward function for GRPO training.
    
    Args:
        quality_weight: Weight for quality score (1 - quality_weight for helpfulness)
        length_penalty_threshold: Minimum words before length penalty kicks in
    
    Returns:
        A reward function that takes response text and returns a float
    """
    def reward_fn(responses: List[str], prompts: List[str] = None) -> List[float]:
        """
        Compute rewards for a batch of responses.
        
        For GRPO, we compute relative rewards within the group.
        Since we don't have labels at inference time, we use heuristics:
        - Response length (not too short, not too long)
        - Coherence signals (ends properly, no repetition)
        """
        rewards = []
        
        for response in responses:
            # Base reward
            reward = 0.5
            
            # Length component: prefer responses between 50-500 words
            word_count = len(response.split())
            if word_count < 10:
                length_score = 0.2
            elif word_count < length_penalty_threshold:
                length_score = word_count / length_penalty_threshold
            elif word_count <= 500:
                length_score = 1.0
            else:
                # Slight penalty for very long responses
                length_score = max(0.7, 1.0 - (word_count - 500) / 1000)
            
            # Coherence: penalize repetition
            words = response.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                coherence_score = min(1.0, unique_ratio * 1.5)
            else:
                coherence_score = 0.0
            
            # Combine scores
            reward = 0.5 * length_score + 0.5 * coherence_score
            
            rewards.append(reward)
        
        return rewards
    
    return reward_fn


def get_dataset_statistics(pairs: List[Dict]) -> Dict:
    """Get statistics about the extracted dataset."""
    rewards = [p["reward"] for p in pairs]
    prompt_lengths = [len(p["prompt"].split()) for p in pairs]
    response_lengths = [len(p["response"].split()) for p in pairs]
    
    return {
        "num_samples": len(pairs),
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "reward_min": np.min(rewards),
        "reward_max": np.max(rewards),
        "prompt_length_mean": np.mean(prompt_lengths),
        "response_length_mean": np.mean(response_lengths),
    }


if __name__ == "__main__":
    # Test the data utilities
    print("Loading OASST1 dataset...")
    dataset = load_oasst1_dataset("train")
    print(f"Loaded {len(dataset)} messages")
    
    print("\nExtracting prompt-response pairs...")
    pairs = extract_prompt_response_pairs(dataset, lang="en", min_quality=0.5, max_samples=1000)
    print(f"Extracted {len(pairs)} pairs")
    
    stats = get_dataset_statistics(pairs)
    print("\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nSample pair:")
    if pairs:
        sample = pairs[0]
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Response: {sample['response'][:100]}...")
        print(f"  Reward: {sample['reward']:.4f}")
