from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict


def clone_task_payload(task_payload):
    return OmegaConf.create(OmegaConf.to_container(task_payload, resolve=False))


def _validate_override_keys(base, override, path=""):
    """Reject unused recipe keys while allowing explicit dataset source mappings."""
    if path in ("dataset.config", "sdf_dataset.config"):
        allowed = {
            "volume_dir",
            "true_dir",
            "ref_masks_dir",
            "few_masks_dir",
            "mean",
            "std",
            "strict_stem_match",
            "train",
            "val",
        }
        unknown = set(override) - allowed
        if unknown:
            raise ValueError(
                f"Unknown dataset source keys in {path}: {sorted(unknown)}"
            )
        for split in ("train", "val"):
            if isinstance(override.get(split), dict):
                _validate_override_keys({}, override[split], path)
        return
    if not isinstance(base, dict) or not isinstance(override, dict):
        return
    for key, value in override.items():
        location = f"{path}.{key}" if path else key
        if key not in base:
            raise ValueError(f"Unknown training configuration key: {location}")
        _validate_override_keys(base[key], value, location)


def wrap_stage_task_config(
    base_cfg,
    task_payload,
    *,
    apply_unet_downsample: bool = False,
):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    with open_dict(cfg):
        if "shared" in base_cfg.task:
            cfg.shared = clone_task_payload(base_cfg.task.shared)
        cfg.task = clone_task_payload(task_payload)
        shared_runtime = base_cfg.task.get("shared", {}).get("runtime", {})
        if shared_runtime:
            for field in ("mean", "std"):
                if field in shared_runtime:
                    cfg.task[field] = shared_runtime[field]
            runtime_resolution = shared_runtime.get("resolution", None)
            unet_downsample = int(shared_runtime.get("unet_downsample", 1) or 1)
            if runtime_resolution is not None:
                cfg.task.resolution = list(runtime_resolution)
                cfg.task.unet_downsample = unet_downsample
                if "scratch" in cfg.task:
                    divisor = unet_downsample if apply_unet_downsample else 1
                    cfg.task.scratch.resolution = [
                        max(1, int(v) // divisor) for v in runtime_resolution
                    ]
    return cfg


def build_bootstrap_task_config(task_cfg, round0_pseudo_root: Path):
    pipeline_cfg = task_cfg.get("pipeline", None)
    bootstrap_cfg = pipeline_cfg.get("bootstrap", None) if pipeline_cfg else None
    shared_cfg = task_cfg.get("shared", None)
    if bootstrap_cfg is None:
        raise ValueError("Missing task.pipeline.bootstrap in unified task config")

    bootstrap_task_cfg = clone_task_payload(bootstrap_cfg)

    with open_dict(bootstrap_task_cfg):
        bootstrap_task_cfg.output_root = str(round0_pseudo_root)
        if shared_cfg is not None:
            bootstrap_task_cfg.volume_dir = shared_cfg.paths.volume_dir
            bootstrap_task_cfg.true_dir = shared_cfg.paths.get("true_dir")
            if shared_cfg.paths.get("annotations"):
                bootstrap_task_cfg.annotations = shared_cfg.paths.annotations
            bootstrap_task_cfg.sam2_config_path = (
                shared_cfg.configs.sam2_bootstrap_config
            )
            bootstrap_task_cfg.sam2_model_path = shared_cfg.paths.sam2_model_path
            if shared_cfg.get("runtime", None) is not None:
                runtime_cfg = shared_cfg.runtime
                if runtime_cfg.get("mean", None) is not None:
                    bootstrap_task_cfg.mean = runtime_cfg.mean
                if runtime_cfg.get("std", None) is not None:
                    bootstrap_task_cfg.std = runtime_cfg.std
                if runtime_cfg.get("resolution", None) is not None:
                    bootstrap_task_cfg.resolution = runtime_cfg.resolution
                if runtime_cfg.get("unet_downsample", None) is not None:
                    bootstrap_task_cfg.unet_downsample = runtime_cfg.unet_downsample

    return bootstrap_task_cfg


def resolve_sam_training_config(cfg):
    args_config = cfg.task.sam2_config_path

    project_root = Path(str(cfg.get("root", Path(__file__).resolve().parent.parent)))
    config_dir = project_root.resolve() / "config"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        sam_cfg = compose(config_name=str(args_config))

    task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    if isinstance(task_cfg, dict):
        task_cfg.pop("shared", None)
        task_cfg.pop("task", None)
        task_cfg.pop("sam2_config_path", None)
        for field in ("resolution", "unet_downsample", "mean", "std"):
            task_cfg.pop(field, None)

        def _copy_common_fields_to_branches(section_name: str) -> None:
            section = task_cfg.get(section_name, {})
            config_root = section.get("config", {})
            if not isinstance(config_root, dict):
                return

            common_fields = {
                field: config_root[field]
                for field in (
                    "volume_dir",
                    "true_dir",
                    "mean",
                    "std",
                    "ref_masks_dir",
                    "few_masks_dir",
                    "strict_stem_match",
                )
                if field in config_root
            }

            for branch_name in ("train", "val"):
                branch = config_root.get(branch_name)
                if not isinstance(branch, dict):
                    branch = {}

                for field, value in common_fields.items():
                    if field not in branch:
                        branch[field] = value

                config_root[branch_name] = branch

                video_dataset = branch.get("video_dataset")
                if isinstance(video_dataset, dict):
                    dataset_config = video_dataset.get("dataset_config")
                    if not isinstance(dataset_config, dict):
                        dataset_config = {}
                    for field, value in branch.items():
                        if field not in dataset_config:
                            dataset_config[field] = value
                    video_dataset["dataset_config"] = dataset_config

        _copy_common_fields_to_branches("dataset")
        _copy_common_fields_to_branches("sdf_dataset")
    _validate_override_keys(OmegaConf.to_container(sam_cfg, resolve=False), task_cfg)
    task_cfg = OmegaConf.create(task_cfg)
    OmegaConf.set_struct(sam_cfg, False)

    launcher_cfg = sam_cfg.get("launcher", None)
    launcher_raw = (
        OmegaConf.to_container(launcher_cfg, resolve=False)
        if launcher_cfg is not None
        else None
    )
    if isinstance(launcher_raw, dict):
        ckpt_path = launcher_raw.get("ckpt_path", None)
        if isinstance(ckpt_path, str) and ckpt_path.startswith("${hydra:runtime.cwd}"):
            launcher_cfg.ckpt_path = str(project_root / ckpt_path.split("/", 1)[1])
        exp_dir = launcher_raw.get("experiment_log_dir", None)
        if isinstance(exp_dir, str) and exp_dir.startswith("${hydra:runtime.cwd}"):
            launcher_cfg.experiment_log_dir = str(
                project_root / exp_dir.split("/", 1)[1]
            )

    sam_cfg = OmegaConf.merge(sam_cfg, task_cfg)

    model_image_size = None
    scratch_cfg = OmegaConf.to_container(sam_cfg.get("scratch", None), resolve=False)
    if isinstance(scratch_cfg, dict):
        raw_resolution = scratch_cfg.get("resolution", None)
        if isinstance(raw_resolution, (list, tuple)) and len(raw_resolution) >= 3:
            model_image_size = int(max(int(raw_resolution[1]), int(raw_resolution[2])))
        elif isinstance(raw_resolution, (int, float)):
            model_image_size = int(raw_resolution)
    if model_image_size is not None and "image_size" in sam_cfg.trainer.model:
        sam_cfg.trainer.model.image_size = model_image_size

    with open_dict(sam_cfg):
        for key in ("root", "datasets_root", "output_root"):
            if key in cfg:
                sam_cfg[key] = cfg[key]

    launcher_cfg = OmegaConf.to_container(sam_cfg.launcher, resolve=False)
    exp_dir = (
        launcher_cfg.get("experiment_log_dir")
        if isinstance(launcher_cfg, dict)
        else None
    )
    if exp_dir is None or (isinstance(exp_dir, str) and exp_dir.startswith("${")):
        sam_cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", str(args_config)
        )

    return sam_cfg
