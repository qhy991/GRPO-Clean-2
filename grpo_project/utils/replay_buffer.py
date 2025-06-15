import logging
import random
from collections import deque
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ExperienceBuffer:
    """Experience replay buffer for storing and sampling high-quality training examples."""

    def __init__(self, max_size: int = 1000):
        self.buffer: deque[Dict[str, Any]] = deque(maxlen=max_size) # Type hint for buffer
        self.max_size = max_size
        logger.info(f"ExperienceBuffer initialized with max_size: {self.max_size}")

    def get_buffer_state(self) -> Dict[str, Any]:
        """获取经验回放池的当前状态，用于保存。"""
        return {
            "buffer_content": list(self.buffer),
            "max_size": self.max_size
        }

    def load_buffer_state(self, state_dict: Dict[str, Any]):
        """从字典加载经验回放池的状态。"""
        self.max_size = state_dict.get("max_size", self.max_size)
        # Ensure items are dicts, handle potential errors if state_dict is malformed
        buffer_content_from_state = state_dict.get("buffer_content", [])
        self.buffer = deque(
            (item for item in buffer_content_from_state if isinstance(item, dict)),
            maxlen=self.max_size
        )
        logger.info(f"ExperienceBuffer state loaded. Restored {len(self.buffer)} experiences. Max capacity: {self.max_size}.")

    def add_experience(self, prompt: str, completion: str, reward: float, metadata: Optional[Dict[str, Any]] = None):
        """Add a new experience to the buffer."""
        if metadata is None:
            metadata = {}

        experience: Dict[str, Any] = { # Type hint
            "prompt": prompt,
            "completion": completion,
            "reward": reward,
            "metadata": metadata,
            "timestamp": len(self.buffer) # This timestamp is just an ordinal, consider time.time() for actual time
        }
        self.buffer.append(experience)

    def get_high_reward_examples(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k high reward examples for replay."""
        if not self.buffer: # Check if buffer is empty
            return []

        # Sort by reward, ensuring all rewards are comparable (e.g., floats)
        try:
            sorted_buffer = sorted(self.buffer, key=lambda x: float(x.get("reward", -float('inf'))), reverse=True)
            return sorted_buffer[:min(top_k, len(sorted_buffer))]
        except (TypeError, ValueError) as e:
            logger.error(f"Error sorting buffer by reward: {e}. Returning empty list or unsorted sample.")
            # Fallback or re-raise, depending on desired strictness
            return random.sample(list(self.buffer), min(top_k, len(self.buffer)))


    def sample_experiences(self, num_samples: int) -> List[Dict[str, Any]]:
        """Sample experiences with preference for higher rewards."""
        if not self.buffer: # Check if buffer is empty
            return []

        experiences = list(self.buffer)
        # Ensure rewards are float for weighting and prevent errors with max(0.1, ...)
        try:
            rewards = np.array([float(exp.get("reward", 0.0)) for exp in experiences])
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid reward data in buffer: {e}. Falling back to uniform sampling.")
            return random.sample(experiences, min(num_samples, len(experiences)))

        # Use softmax for more robust weighting, or ensure positive weights for simple normalization
        # weights = np.exp(rewards - np.max(rewards)) # Softmax for stability
        weights = np.maximum(0.1, rewards) # Original approach: ensure positive weights (can skew if rewards are mostly negative)

        if weights.sum() == 0: # If all effective weights are zero
            logger.warning("All experience weights are zero (or near zero). Falling back to uniform sampling.")
            return random.sample(experiences, min(num_samples, len(experiences)))

        weights = weights / weights.sum()

        try:
            # Ensure num_samples does not exceed population size if replace=False
            actual_num_to_sample = min(num_samples, len(experiences))
            indices = np.random.choice(len(experiences), size=actual_num_to_sample, p=weights, replace=False)
            return [experiences[i] for i in indices]
        except ValueError as e: # Handles issues like p not summing to 1, or empty weights
            logger.warning(f"Weighted sampling failed (e.g. probability sum issue): {e}. Falling back to uniform sampling.")
            return random.sample(experiences, min(num_samples, len(experiences)))

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self.buffer: # Check if buffer is empty
            return {"size": 0.0, "mean_reward": 0.0, "max_reward": 0.0, "min_reward": 0.0, "std_reward": 0.0}

        try:
            rewards = [float(exp.get("reward", 0.0)) for exp in self.buffer] # Ensure float
            return {
                "size": float(len(self.buffer)),
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "max_reward": float(np.max(rewards)) if rewards else 0.0,
                "min_reward": float(np.min(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0
            }
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating buffer stats due to invalid reward data: {e}")
            return {"size": float(len(self.buffer)), "mean_reward": 0.0, "max_reward": 0.0, "min_reward": 0.0, "std_reward": 0.0, "error": 1.0}

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return list(self.buffer)[index] # Note: This can be inefficient for large deques if not careful
