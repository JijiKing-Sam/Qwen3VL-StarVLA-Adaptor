import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader
from deployment.model_server.tools.trace_tools import trace

logger = get_logger(__name__)


def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return None

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        trace(
            "dataloader.build_vla.start",
            dataset_py=dataset_py,
            data_root_dir=vla_dataset_cfg.get("data_root_dir", None),
            data_mix=vla_dataset_cfg.get("data_mix", None),
            action_type=vla_dataset_cfg.get("action_type", None),
            action_mode=vla_dataset_cfg.get("action_mode", None),
            obs=vla_dataset_cfg.get("obs", None),
            per_device_batch_size=vla_dataset_cfg.get("per_device_batch_size", None),
        )

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        trace(
            "dataloader.build_vla.dataset",
            dataset_type=type(vla_dataset).__name__,
            dataset_len=_safe_len(vla_dataset),
            statistics_keys=list(getattr(vla_dataset, "statistics", {}).keys()) if hasattr(vla_dataset, "statistics") else None,
        )
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            # shuffle=True
        )        
        trace(
            "dataloader.build_vla.dataloader",
            dataloader_type=type(vla_train_dataloader).__name__,
            dataset_len=_safe_len(vla_dataset),
            dataloader_len=_safe_len(vla_train_dataloader),
            batch_size=vla_train_dataloader.batch_size,
            num_workers=vla_train_dataloader.num_workers,
        )
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        trace("dataloader.build_vlm.start", dataset_py=dataset_py, dataset_use=cfg.datasets.vlm_data.get("dataset_use", None))
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        trace(
            "dataloader.build_vlm.dataloader",
            dataloader_type=type(vlm_train_dataloader).__name__,
            dataloader_len=_safe_len(vlm_train_dataloader),
            batch_size=getattr(vlm_train_dataloader, "batch_size", None),
            num_workers=getattr(vlm_train_dataloader, "num_workers", None),
        )
        
        return vlm_train_dataloader
