import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub.constants import HF_HOME
import numpy as np
from datasets import load_dataset

from lerobot.configs import parser
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class PrepareTrainValSplitsConfig:
    repo_id: str
    root: str | None = None
    output_root: str | None = None
    ood_object_task: int = 1
    ood_spatial_task: int = 9
    val_seen_ratio: float = 0.1
    split_seed: int = 42


@parser.wrap()
def prepare_splits(cfg: PrepareTrainValSplitsConfig) -> None:
    default_cache_path = Path(HF_HOME) / "lerobot"
    HF_LEROBOT_HOME = Path(
        os.getenv("HF_LEROBOT_HOME", default_cache_path)
    ).expanduser()
    init_logging()

    if cfg.val_seen_ratio < 0 or cfg.val_seen_ratio > 1:
        raise ValueError(f"val_seen_ratio must be in [0, 1], got {cfg.val_seen_ratio}")
    
    # Determine dataset root path
    root = HF_LEROBOT_HOME / cfg.repo_id if cfg.root is None else Path(cfg.root).expanduser()
    # if cfg.root is not None:
    #     root = Path(cfg.root).expanduser()
    # else:
    #     from huggingface_hub import snapshot_download
    #     root = Path(snapshot_download(repo_id=cfg.repo_id, repo_type="dataset"))
    
    print(f"Loading dataset from {root}...")
    hf_dataset = load_dataset(path=root.as_posix())["train"]
    print(f"Dataset loaded with {len(hf_dataset)} frames.")
    
    if "task_index" not in hf_dataset.column_names:
        raise ValueError("Dataset does not contain 'task_index', cannot build splits.")

    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    task_indices = np.asarray(hf_dataset["task_index"], dtype=np.int64)

    # Map episode to its task index, ensuring no mixed tasks per episode
    episode_to_task: dict[int, int] = {}
    for ep_idx, task_idx in zip(episode_indices.tolist(), task_indices.tolist(), strict=True):
        prev_task = episode_to_task.get(ep_idx)
        if prev_task is None:
            episode_to_task[ep_idx] = task_idx
        elif prev_task != task_idx:
            raise ValueError(
                "Found mixed task_index values within the same episode. "
                "Cannot build splits for this dataset."
            )

    all_episode_indices = sorted(episode_to_task.keys())
    ood_object_set = {ep for ep in all_episode_indices if episode_to_task[ep] == cfg.ood_object_task}
    ood_spatial_set = {ep for ep in all_episode_indices if episode_to_task[ep] == cfg.ood_spatial_task}
    id_episodes = [ep for ep in all_episode_indices if ep not in ood_object_set and ep not in ood_spatial_set]

    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(id_episodes)

    num_val_seen = int(len(id_episodes) * cfg.val_seen_ratio)
    if cfg.val_seen_ratio > 0 and len(id_episodes) > 0:
        num_val_seen = max(1, num_val_seen)

    val_seen_episodes = sorted(id_episodes[:num_val_seen])
    train_episodes = sorted(id_episodes[num_val_seen:])
    ood_object_episodes = sorted(ood_object_set)
    ood_spatial_episodes = sorted(ood_spatial_set)

    splits: dict[str, list[int]] = {}
    if train_episodes:
        splits["train"] = train_episodes
    if val_seen_episodes:
        splits["val_seen"] = val_seen_episodes
    if ood_object_episodes:
        splits["val_unseen_object"] = ood_object_episodes
    if ood_spatial_episodes:
        splits["val_unseen_spatial"] = ood_spatial_episodes

    if not splits:
        raise ValueError("No splits were created from the current dataset/task mapping.")

    # Save episode indices as JSON
    if cfg.output_root is not None:
        output_root = Path(cfg.output_root).expanduser()
    else:


        output_root = HF_LEROBOT_HOME / cfg.repo_id

    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("Saving episode indices to: %s", output_root)
    splits_file = output_root / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)
    logging.info("Saved splits to %s", splits_file)
    for split_name, episode_list in splits.items():
        logging.info("  %s: %d episodes", split_name, len(episode_list))

    logging.info("To train with these splits, set LEROBOT_SPLIT_OUTPUT_ROOT=%s", output_root)



def main():
    register_third_party_plugins()
    prepare_splits()


if __name__ == "__main__":
    main()
