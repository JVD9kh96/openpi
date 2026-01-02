"""
Jupyter notebook-friendly direct evaluation script for B1K tasks.

This script provides a complete integration with the OmniGibson evaluator
that can be easily used in Jupyter notebooks.

Example usage in Jupyter:
    from openpi.scripts.eval_b1k_direct_notebook import setup_and_run_evaluation
    
    results = setup_and_run_evaluation(
        behavior_repo_path="/path/to/behavior/repo",
        config_name="pi1_b1k",
        checkpoint_dir="/path/to/checkpoint/38000",
        task_name="turning_on_radio",
        eval_instance_ids=[0, 1, 2],  # or None for all
        write_video=True,
        log_path="./eval_logs",
    )
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, List
import json

logger = logging.getLogger(__name__)


def setup_and_run_evaluation(
    behavior_repo_path: str,
    config_name: str,
    checkpoint_dir: str,
    task_name: str = "turning_on_radio",
    dataset_root: Optional[str] = None,
    default_prompt: Optional[str] = None,
    pytorch_device: Optional[str] = None,
    # Evaluation parameters
    eval_instance_ids: Optional[List[int]] = None,
    eval_on_train_instances: bool = False,
    test_hidden: bool = False,
    max_steps: Optional[int] = None,
    partial_scene_load: bool = True,
    headless: bool = True,
    write_video: bool = False,
    log_path: str = "./eval_logs",
    robot_controllers: Optional[dict] = None,
) -> dict:
    """
    Complete setup and run evaluation without websockets.
    
    Args:
        behavior_repo_path: Path to the behavior repository
        config_name: Training config name (e.g., "pi1_b1k")
        checkpoint_dir: Path to checkpoint directory
        task_name: Task name for evaluation
        dataset_root: Root directory for dataset
        default_prompt: Default prompt (overrides dataset)
        pytorch_device: PyTorch device
        eval_instance_ids: List of instance IDs to evaluate (None = all test instances)
        eval_on_train_instances: Whether to evaluate on training instances
        test_hidden: Whether to evaluate on hidden test instances
        max_steps: Maximum steps per episode (None = auto from human stats)
        partial_scene_load: Whether to use partial scene loading
        headless: Whether to run in headless mode
        write_video: Whether to write videos
        log_path: Path to save logs and metrics
        robot_controllers: Optional robot controller config
        
    Returns:
        Dictionary with evaluation results
    """
    # Add behavior repo to path
    behavior_repo_path = Path(behavior_repo_path).resolve()
    if str(behavior_repo_path) not in sys.path:
        sys.path.insert(0, str(behavior_repo_path))
        logger.info(f"Added {behavior_repo_path} to sys.path")
    
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
        raise ImportError(
            f"Failed to import required modules. Make sure behavior_repo_path is correct: {e}"
        )
    
    # Import openpi modules
    from openpi.policies import policy_config as _policy_config
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.training import config as _config
    from openpi.policies.direct_policy_adapter import DirectPolicyAdapter
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, force=True)
    
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
    eval_config_dir = Path(behavior_repo_path) / "omnigibson" / "learning" / "configs"
    if not eval_config_dir.exists():
        # Try alternative location
        eval_config_dir = Path(behavior_repo_path) / "behavior" / "OmniGibson" / "omnigibson" / "learning" / "configs"
    
    if not eval_config_dir.exists():
        raise FileNotFoundError(
            f"Could not find evaluator config directory. Tried: {eval_config_dir}"
        )
    
    # Initialize Hydra
    with hydra.initialize_config_dir(str(eval_config_dir), version_base="1.1"):
        # Compose base config
        eval_config = hydra.compose("base_config.yaml")
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
    if robot_controllers is not None:
        eval_config.robot.controllers = robot_controllers
    
    # Create a policy config that returns our direct policy
    class DirectPolicyConfig:
        def __init__(self, policy):
            self.policy = policy
        
        def __call__(self, *args, **kwargs):
            return self.policy
    
    # Replace the model config with our direct policy
    eval_config.model = DirectPolicyConfig(direct_policy)
    
    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = Evaluator(eval_config)
    
    # Determine instances to run
    from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
    import csv
    
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

