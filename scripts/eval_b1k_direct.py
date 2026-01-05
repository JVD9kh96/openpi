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


def setup_and_run_evaluation(
    behavior_repo_path: str,
    config_name: str,
    checkpoint_dir: str,
    task_name: str = "turning_on_radio",
    dataset_root: Optional[str] = None,
    default_prompt: Optional[str] = None,
    pytorch_device: Optional[str] = None,
    eval_instance_ids: Optional[list[int]] = None,
    eval_on_train_instances: bool = False,
    test_hidden: bool = False,
    max_steps: Optional[int] = None,
    partial_scene_load: bool = True,
    headless: bool = True,
    write_video: bool = False,
    log_path: str = "./eval_logs",
    additional_paths: Optional[list[str]] = None,
) -> dict:
    """
    Complete setup and run evaluation without websockets.
    
    This function is embedded directly in this file to make it standalone.
    """
    import json
    import csv
    
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
    
    # Import after adding to path
    try:
        import hydra
        from hydra.utils import instantiate
        from omegaconf import DictConfig, OmegaConf
        from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata
        from omnigibson.learning.eval import Evaluator
        from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
        from omnigibson.macros import gm, create_module_macros
        from pathlib import Path as PathLib
        from inspect import getsourcefile
        import torch as th
    except ImportError as e:
        error_msg = str(e)
        if "gello" in error_msg.lower():
            raise ImportError(
                f"Failed to import 'gello' module. {error_msg}\n"
                f"Please either:\n"
                f"  1. Install gello: pip install gello\n"
                f"  2. Clone gello repository and add its path using additional_paths parameter\n"
                f"  3. If gello is in a standard location, update the script to find it\n"
                f"Current sys.path: {sys.path[:5]}..."
            )
        raise ImportError(
            f"Failed to import required modules. Make sure behavior_repo_path is correct: {e}"
        )
    
    # Load metadata and get prompt
    dataset_root = dataset_root or "/scr/behavior/2025-challenge-demos"
    metadata = BehaviorLerobotDatasetMetadata(
        repo_id="behavior-1k/2025-challenge-demos",
        root=dataset_root,
        tasks=[task_name] if task_name else ["turning_on_radio"],
        modalities=[],
        cameras=[],
    )
    prompt = list(metadata.tasks.values())[0] if not default_prompt else default_prompt
    logger.info(f"Using prompt: {prompt}")
    
    # Load policy
    logger.info(f"Loading policy from config: {config_name}, checkpoint: {checkpoint_dir}")
    train_config = _config.get_config(config_name)
    
    # Determine device
    if pytorch_device is None:
        try:
            pytorch_device = "cuda" if th.cuda.is_available() else "cpu"
        except:
            pytorch_device = "cpu"
    
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=prompt,
        pytorch_device=pytorch_device,
    )
    
    # Wrap policy for B1K
    wrapped_policy = B1KPolicyWrapper(policy, text_prompt=prompt)
    
    # Create adapter for evaluator interface
    direct_policy = DirectPolicyAdapter(wrapped_policy)
    
    # Set up OmniGibson globals
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = True
    gm.HEADLESS = headless
    
    # Register resolvers
    register_omegaconf_resolvers()
    
    # Create evaluator config
    # We need to find the config directory from the behavior repo
    # Try multiple possible locations
    possible_paths = [
        Path(behavior_repo_path) / "OmniGibson" / "omnigibson" / "learning" / "configs",
        Path(behavior_repo_path) / "omnigibson" / "learning" / "configs",
        Path(behavior_repo_path) / "behavior" / "OmniGibson" / "omnigibson" / "learning" / "configs",
    ]
    
    eval_config_dir = None
    for path in possible_paths:
        if path.exists():
            eval_config_dir = path
            logger.info(f"Found config directory at: {eval_config_dir}")
            break
    
    if eval_config_dir is None:
        tried_paths = "\n  - ".join([str(p) for p in possible_paths])
        raise FileNotFoundError(
            f"Could not find evaluator config directory. Tried:\n  - {tried_paths}\n"
            f"Please check that the behavior_repo_path is correct and contains OmniGibson."
        )
    
    # Initialize Hydra
    with hydra.initialize_config_dir(str(eval_config_dir), version_base="1.1"):
        # Compose base config with policy=local override
        # We use "local" policy config and then replace it with our direct policy
        eval_config = hydra.compose("base_config.yaml", overrides=["policy=local"])
        OmegaConf.resolve(eval_config)
    
    # Override config values
    eval_config.task.name = task_name
    eval_config.headless = headless
    eval_config.write_video = write_video
    eval_config.log_path = log_path
    eval_config.max_steps = max_steps
    eval_config.partial_scene_load = partial_scene_load
    eval_config.eval_on_train_instances = eval_on_train_instances
    eval_config.test_hidden = test_hidden
    if eval_instance_ids is not None:
        eval_config.eval_instance_ids = eval_instance_ids
    
    # Create evaluator with the local policy config
    # The config will create a LocalPolicy, which we'll then replace with our direct policy
    logger.info("Creating evaluator...")
    evaluator = Evaluator(eval_config)
    
    # Replace the policy with our direct policy
    # LocalPolicy has a 'policy' attribute that we can set
    # If it's a LocalPolicy, set its policy attribute to our direct policy
    if hasattr(evaluator.policy, 'policy'):
        logger.info("Replacing LocalPolicy.policy with direct policy...")
        evaluator.policy.policy = direct_policy
    else:
        # If it's not a LocalPolicy, try to replace the whole policy
        logger.info("Replacing evaluator policy with direct policy...")
        evaluator.policy = direct_policy
    
    # Determine instances to run
    from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
    
    m = create_module_macros(module_path=__file__)
    m.NUM_EVAL_EPISODES = 1
    m.NUM_TRAIN_INSTANCES = 200
    m.NUM_EVAL_INSTANCES = 10
    
    if eval_on_train_instances:
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        instances_to_run = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(str(int((episode["episode_index"] // 10) % 1e3)))
        if eval_instance_ids is not None:
            instances_to_run = [instances_to_run[i] for i in eval_instance_ids]
    elif test_hidden:
        instances_to_run = (
            eval_instance_ids if eval_instance_ids is not None else list(range(m.NUM_EVAL_INSTANCES))
        )
    else:
        instances_to_run = (
            eval_instance_ids if eval_instance_ids is not None else list(range(m.NUM_EVAL_INSTANCES))
        )
        # Load test instances from CSV
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        test_instances = lines[TASK_NAMES_TO_INDICES[task_name]][2].strip().split(",")
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]
    
    # Set up video path
    if write_video:
        video_path = Path(log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    else:
        video_path = None
    
    # Run evaluation
    logger.info("Starting evaluation...")
    metrics = {}
    metrics_path = Path(log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    from omnigibson.learning.utils.obs_utils import create_video_writer
    
    with evaluator:
        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=test_hidden)
            logger.info(f"Starting task instance {idx} for evaluation...")
            
            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False
                
                if write_video and video_path:
                    video_name = str(video_path) + f"/{task_name}_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(448, 672),
                    )
                
                # Run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)
                
                while not done:
                    terminated, truncated = evaluator.step()
                    if terminated or truncated:
                        done = True
                    
                    if write_video:
                        evaluator._write_video()
                    
                    if evaluator.env._current_step % 1000 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")
                
                # Run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)
                
                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: {terminated}, {truncated}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")
                
                # Gather metric results
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                
                # Save metrics
                with open(metrics_path / f"{task_name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)
                
                # Reset video writer
                if write_video and video_path:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {video_name}")
    
    logger.info("Evaluation complete!")
    return {
        "metrics": metrics,
        "total_trials": evaluator.n_trials,
        "total_success_trials": evaluator.n_success_trials,
        "success_rate": evaluator.n_success_trials / evaluator.n_trials if evaluator.n_trials > 0 else 0.0,
    }


def main():
    """Main entry point for command-line usage."""
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

