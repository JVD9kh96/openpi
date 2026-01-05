"""
Direct evaluation script for B1K tasks without websockets.
This script can be run standalone or imported in a Jupyter notebook.

Usage:
    python scripts/eval_b1k_direct.py \
        --behavior_repo_path /path/to/behavior/repo \
        --config_name pi1_b1k \
        --checkpoint_dir /path/to/checkpoint/38000 \
        --task_name turning_on_radio \
        [other eval args...]

Or in a Jupyter notebook:
    from openpi.scripts.eval_b1k_direct import DirectB1KEvaluator
    
    evaluator = DirectB1KEvaluator(
        behavior_repo_path="/path/to/behavior/repo",
        config_name="pi1_b1k",
        checkpoint_dir="/path/to/checkpoint/38000",
        task_name="turning_on_radio",
    )
    evaluator.run_evaluation(...)
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch as th

# Import openpi modules
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config
from openpi.policies.direct_policy_adapter import DirectPolicyAdapter

logger = logging.getLogger(__name__)


class DirectB1KEvaluator:
    """
    Direct evaluator for B1K tasks that loads the policy directly without websockets.
    Can be used in scripts or Jupyter notebooks.
    """
    
    def __init__(
        self,
        behavior_repo_path: str,
        config_name: str,
        checkpoint_dir: str,
        task_name: str = "turning_on_radio",
        dataset_root: Optional[str] = None,
        default_prompt: Optional[str] = None,
        pytorch_device: Optional[str] = None,
        additional_paths: Optional[list[str]] = None,
    ):
        """
        Initialize the direct evaluator.
        
        Args:
            behavior_repo_path: Path to the behavior repository (for OmniGibson imports)
            config_name: Training config name (e.g., "pi1_b1k")
            checkpoint_dir: Path to checkpoint directory (e.g., "/path/to/checkpoint/38000")
            task_name: Task name for evaluation
            dataset_root: Root directory for dataset (default: "/scr/behavior/2025-challenge-demos")
            default_prompt: Default prompt to use (will be retrieved from dataset if not provided)
            pytorch_device: Device for PyTorch (default: "cuda" if available, else "cpu")
            additional_paths: Additional paths to add to sys.path (e.g., path to gello repository)
        """
        # Add behavior repo to path
        behavior_repo_path = Path(behavior_repo_path).resolve()
        if str(behavior_repo_path) not in sys.path:
            sys.path.insert(0, str(behavior_repo_path))
            logger.info(f"Added {behavior_repo_path} to sys.path")
        
        # Add additional paths (e.g., gello)
        if additional_paths:
            for path in additional_paths:
                path = Path(path).resolve()
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
                    logger.info(f"Added {path} to sys.path")
        
        # Try common locations for gello if not already in path
        gello_in_path = any("gello" in str(p).lower() for p in sys.path)
        if not gello_in_path:
            common_gello_locations = [
                behavior_repo_path.parent / "gello",
                behavior_repo_path / "gello",
                Path("/kaggle/working/gello"),
                Path("/kaggle/input/gello"),
            ]
            for gello_path in common_gello_locations:
                if gello_path.exists() and str(gello_path) not in sys.path:
                    sys.path.insert(0, str(gello_path))
                    logger.info(f"Found and added gello at {gello_path} to sys.path")
                    break
        
        # Import OmniGibson modules after adding to path
        try:
            from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata
            from omnigibson.learning.eval import Evaluator
            from omnigibson.macros import gm
        except ImportError as e:
            error_msg = str(e)
            if "gello" in error_msg.lower():
                raise ImportError(
                    f"Failed to import 'gello' module. {error_msg}\n"
                    f"Please either:\n"
                    f"  1. Install gello: pip install gello\n"
                    f"  2. Clone gello repository and add its path using --additional_paths\n"
                    f"  3. If gello is in a standard location, update the script to find it\n"
                    f"Current sys.path: {sys.path[:5]}..."
                )
            raise ImportError(
                f"Failed to import OmniGibson modules. Make sure behavior_repo_path is correct: {e}"
            )
        
        self.behavior_repo_path = behavior_repo_path
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.task_name = task_name
        self.dataset_root = dataset_root or "/scr/behavior/2025-challenge-demos"
        self.default_prompt = default_prompt
        
        # Determine device
        if pytorch_device is None:
            try:
                pytorch_device = "cuda" if th.cuda.is_available() else "cpu"
            except:
                pytorch_device = "cpu"
        self.pytorch_device = pytorch_device
        
        # Load metadata and get prompt
        metadata = BehaviorLerobotDatasetMetadata(
            repo_id="behavior-1k/2025-challenge-demos",
            root=self.dataset_root,
            tasks=[self.task_name] if self.task_name else ["turning_on_radio"],
            modalities=[],
            cameras=[],
        )
        self.prompt = list(metadata.tasks.values())[0] if not self.default_prompt else self.default_prompt
        logger.info(f"Using prompt: {self.prompt}")
        
        # Load policy
        logger.info(f"Loading policy from config: {config_name}, checkpoint: {checkpoint_dir}")
        train_config = _config.get_config(config_name)
        policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=self.prompt,
            pytorch_device=self.pytorch_device,
        )
        
        # Wrap policy for B1K
        wrapped_policy = B1KPolicyWrapper(policy, text_prompt=self.prompt)
        
        # Create adapter for evaluator interface
        self.policy = DirectPolicyAdapter(wrapped_policy)
        
        # Store Evaluator class for later use
        self.Evaluator = Evaluator
        self.gm = gm
        
    def create_evaluator(self, eval_config: Any) -> Any:
        """
        Create an Evaluator instance with the direct policy.
        
        Args:
            eval_config: Configuration dict/config object for the evaluator
            
        Returns:
            Evaluator instance
        """
        # Modify the config to use our direct policy
        # The evaluator expects cfg.model to be instantiable, but we'll replace it
        original_model_cfg = eval_config.model
        
        # Create a simple config that will return our policy
        class DirectPolicyConfig:
            def __init__(self, policy):
                self.policy = policy
            
            def __call__(self, *args, **kwargs):
                return self.policy
        
        eval_config.model = DirectPolicyConfig(self.policy)
        
        evaluator = self.Evaluator(eval_config)
        
        # Restore original config
        eval_config.model = original_model_cfg
        
        return evaluator
    
    def run_evaluation(
        self,
        eval_config: Any,
        instances_to_run: Optional[list] = None,
        write_video: bool = False,
        video_path: Optional[str] = None,
    ) -> dict:
        """
        Run evaluation with the direct policy.
        
        Args:
            eval_config: Configuration for evaluation (from OmniGibson)
            instances_to_run: List of instance IDs to evaluate (None = use config default)
            write_video: Whether to write videos
            video_path: Path to save videos
            
        Returns:
            Dictionary with evaluation results
        """
        evaluator = self.create_evaluator(eval_config)
        
        # Run evaluation (this follows the same pattern as the original eval.py)
        # The exact implementation depends on how you want to structure it
        # For now, return the evaluator so the user can run it manually
        return evaluator


def main():
    """Main entry point for command-line usage."""
    # Import the evaluation function - we'll use it directly
    from openpi.scripts.eval_b1k_direct_notebook import setup_and_run_evaluation
    
    parser = argparse.ArgumentParser(description="Direct B1K evaluation without websockets")
    
    # Required arguments
    parser.add_argument(
        "--behavior_repo_path",
        type=str,
        required=True,
        help="Path to the behavior repository (for OmniGibson imports)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Training config name (e.g., 'pi1_b1k')",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., '/path/to/checkpoint/38000')",
    )
    
    # Optional arguments
    parser.add_argument(
        "--task_name",
        type=str,
        default="turning_on_radio",
        help="Task name for evaluation",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/scr/behavior/2025-challenge-demos",
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default=None,
        help="Default prompt to use (overrides dataset prompt)",
    )
    parser.add_argument(
        "--pytorch_device",
        type=str,
        default=None,
        help="PyTorch device (default: 'cuda' if available, else 'cpu')",
    )
    parser.add_argument(
        "--additional_paths",
        type=str,
        nargs="+",
        default=None,
        help="Additional paths to add to sys.path (e.g., path to gello repository)",
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--eval_instance_ids",
        type=int,
        nargs="+",
        default=None,
        help="List of instance IDs to evaluate (None = all test instances)",
    )
    parser.add_argument(
        "--eval_on_train_instances",
        action="store_true",
        help="Evaluate on training instances instead of test",
    )
    parser.add_argument(
        "--test_hidden",
        action="store_true",
        help="Evaluate on hidden test instances",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum steps per episode (None = auto from human stats)",
    )
    parser.add_argument(
        "--no-partial_scene_load",
        action="store_true",
        help="Disable partial scene loading",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run with GUI (not headless)",
    )
    parser.add_argument(
        "--write_video",
        action="store_true",
        help="Save evaluation videos",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./eval_logs",
        help="Path to save logs and metrics",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, force=True)
    
    logger.info("Starting direct B1K evaluation...")
    
    # Run evaluation
    results = setup_and_run_evaluation(
        behavior_repo_path=args.behavior_repo_path,
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        task_name=args.task_name,
        dataset_root=args.dataset_root,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
        additional_paths=args.additional_paths,
        eval_instance_ids=args.eval_instance_ids,
        eval_on_train_instances=args.eval_on_train_instances,
        test_hidden=args.test_hidden,
        max_steps=args.max_steps,
        partial_scene_load=not args.no_partial_scene_load,
        headless=not args.no_headless,
        write_video=args.write_video,
        log_path=args.log_path,
    )
    
    # Print results
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"Total trials: {results['total_trials']}")
    logger.info(f"Total success trials: {results['total_success_trials']}")
    logger.info(f"Success rate: {results['success_rate']:.2%}")
    logger.info("=" * 50)
    
    return results


if __name__ == "__main__":
    main()

