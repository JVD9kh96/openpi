"""
Example script for running direct B1K evaluation without websockets.

This can be run as a script or imported in a Jupyter notebook.
"""
from openpi.scripts.eval_b1k_direct_notebook import setup_and_run_evaluation

# Configure your paths
# Adjust these paths based on your environment (local, Kaggle, etc.)
BEHAVIOR_REPO_PATH = "/path/to/behavior/repo"  # Update this!
CONFIG_NAME = "pi1_b1k"
CHECKPOINT_DIR = "/path/to/checkpoint/38000"  # Update this!
TASK_NAME = "turning_on_radio"

if __name__ == "__main__":
    # Run evaluation
    results = setup_and_run_evaluation(
        behavior_repo_path=BEHAVIOR_REPO_PATH,
        config_name=CONFIG_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        task_name=TASK_NAME,
        eval_instance_ids=[0, 1, 2],  # Evaluate first 3 instances
        write_video=False,  # Set to True to save videos
        log_path="./eval_logs",
        headless=True,
    )
    
    # Print results
    print(f"Total trials: {results['total_trials']}")
    print(f"Total success trials: {results['total_success_trials']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"\nMetrics: {results['metrics']}")


# Example for Kaggle:
"""
results = setup_and_run_evaluation(
    behavior_repo_path="/kaggle/input/behavior-repo",
    config_name="pi1_b1k",
    checkpoint_dir="/kaggle/working/b1k-baselines/baselines/openpi/outputs/checkpoints/pi1_b1k/pi1_b1k_test/38000",
    task_name="turning_on_radio",
    eval_instance_ids=[0],
    write_video=False,
    log_path="/kaggle/working/eval_logs",
    headless=True,
)
"""

