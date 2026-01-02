"""
Adapter to wrap B1KPolicyWrapper for direct use with OmniGibson evaluator.
This adapter provides the forward() method expected by the evaluator.
"""
import logging
import torch as th
from typing import Any

logger = logging.getLogger(__name__)


class DirectPolicyAdapter:
    """
    Adapter that wraps a B1KPolicyWrapper to provide the forward() interface
    expected by the OmniGibson evaluator.
    
    The evaluator expects:
    - forward(obs: dict) -> th.Tensor
    - reset() -> None
    """
    
    def __init__(self, wrapped_policy: Any):
        """
        Args:
            wrapped_policy: A B1KPolicyWrapper instance that has act() and reset() methods
        """
        self.wrapped_policy = wrapped_policy
        
    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Forward pass that converts the observation and calls the wrapped policy.
        
        Args:
            obs: Observation dictionary from the evaluator
            
        Returns:
            th.Tensor: Action tensor of shape (action_dim,)
        """
        # The wrapped policy's act() method expects the observation in a specific format
        # and returns a torch.Tensor of shape (1, action_dim) or (action_dim,)
        action = self.wrapped_policy.act(obs)
        
        # Ensure it's a tensor
        if not isinstance(action, th.Tensor):
            action = th.tensor(action)
        
        # Convert to the expected format: squeeze batch dimension if present
        if action.dim() > 1:
            action = action.squeeze(0)
        
        # Ensure it's on CPU and detached (for safety)
        return action.detach().cpu()
    
    def reset(self) -> None:
        """Reset the wrapped policy."""
        self.wrapped_policy.reset()

