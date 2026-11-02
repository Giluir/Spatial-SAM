"""Build and execute the training graph with explicit, immutable stage inputs."""

import gc
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from spatialsam.runners.task_config_resolver import (
    build_bootstrap_task_config,
    clone_task_payload,
    resolve_sam_training_config,
    wrap_stage_task_config,
)


STAGE_ORDER = ("sdf_train", "sdf_infer", "sam_train", "sam_infer")


@dataclass(frozen=True)
class Stage:
    """A stage owns one directory and declares every upstream artifact it uses."""

    name: str
    kind: str
    directory: Path
    inputs: tuple[Path, ...]
    outputs: dict[str, Path]
    config: dict


def _check_keys(config, allowed, context):
    unknown = set(config) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {context} keys: {sorted(unknown)}")


def validate_pipeline_config(cfg):
    """Reject ambiguous schedules and unsupported top-level configuration fields."""
    task = cfg.task
    from spatialsam.configuration import validate_output_directory

    validate_output_directory(
        task.shared.paths.output_root,
        task.shared.paths.volume_dir,
        task.shared.paths.get("true_dir"),
        task.shared.paths.get("annotations"),
    )
    _check_keys(
        task,
        (
            "num_rounds",
            "start_round",
            "start_step",
            "cuda_devices",
            "shared",
            "pipeline",
        ),
        "task",
    )
    _check_keys(task.shared, ("runtime", "configs", "launcher", "paths"), "shared")
    _check_keys(
        task.shared.configs,
        ("sam2_bootstrap_config", "sam2_sdf_train_config", "sam2_sam_train_config"),
        "model recipes",
    )
    _check_keys(
        task.shared.paths,
        ("output_root", "volume_dir", "true_dir", "annotations", "sam2_model_path"),
        "paths",
    )
    _check_keys(
        task.shared.runtime,
        ("resolution", "unet_downsample", "mean", "std", "train_mode", "accelerator"),
        "runtime",
    )
    _check_keys(task.pipeline, ("bootstrap", "round_schedule", "stages"), "pipeline")
    if set(task.pipeline.stages) != set(STAGE_ORDER):
        raise ValueError(f"Training stages must be exactly {STAGE_ORDER}")
    _check_keys(
        task.pipeline.bootstrap,
        (
            "enabled",
            "save_masks",
            "save_few_masks",
            "save_xyz_masks",
            "mask_to_binary",
            "slices",
            "to_sdf",
            "select_mask_with_obj",
            "label_start",
            "label_stride",
            "pseudo_blur_sigma",
            "twice",
        ),
        "bootstrap",
    )
    for name, stage in task.pipeline.stages.items():
        allowed = (
            (
                "sam2_config_path",
                "launcher",
                "scratch",
                "sdf_dataset",
                "dataset",
                "trainer",
            )
            if name.endswith("train")
            else (
                "volume_dir",
                "sam2_config_path",
                "save_masks",
                "mask_to_binary",
                "to_sdf",
                "sdf_infer",
            )
        )
        _check_keys(stage, allowed, name)
    if str(task.shared.runtime.train_mode) != "train_only":
        raise ValueError("The training pipeline does not include evaluation stages")
    if (
        int(task.shared.launcher.num_nodes) != 1
        or int(task.shared.launcher.gpus_per_node) != 1
    ):
        raise ValueError("This pipeline requires one process on one GPU")
    if "," in str(task.cuda_devices):
        raise ValueError("cuda_devices must identify one GPU")
    if not 1 <= task.start_round <= task.num_rounds or not 1 <= task.start_step <= 4:
        raise ValueError("Invalid starting round or stage")
    if task.shared.runtime.std <= 0:
        raise ValueError("Normalization standard deviation must be positive")
    if task.shared.runtime.get("accelerator", "cuda") not in ("cpu", "cuda"):
        raise ValueError("accelerator must be cpu or cuda")
    for profile in task.pipeline.round_schedule:
        _check_keys(
            profile,
            ("start_round", "end_round", "sdf_epochs", "sam_epochs"),
            "round schedule",
        )
        if min(profile.sdf_epochs, profile.sam_epochs) <= 0:
            raise ValueError("Epoch counts must be positive")
    for index in range(1, task.num_rounds + 1):
        _round_profile(task, index)


def _round_profile(task, index):
    profiles = [
        profile
        for profile in task.pipeline.round_schedule
        if profile.start_round <= index <= profile.end_round
    ]
    if len(profiles) != 1:
        raise ValueError(f"Round {index} must match exactly one schedule entry")
    return profiles[0]


def build_stages(cfg) -> list[Stage]:
    """Resolve the whole graph without loading volumes, models, or CUDA."""
    from predict.configuration import validate_options

    validate_pipeline_config(cfg)
    task = cfg.task
    root = Path(str(task.shared.paths.output_root))
    images = Path(str(task.shared.paths.volume_dir))
    pretrained = Path(str(task.shared.paths.sam2_model_path))
    bootstrap_dir = root / "bootstrap"
    initial_sdf = bootstrap_dir / "sdf"
    few_masks = bootstrap_dir / "few_masks"
    bootstrap = build_bootstrap_task_config(task, initial_sdf)
    bootstrap.pop("enabled", None)
    bootstrap.strict_stem_match = True
    bootstrap.device = task.shared.runtime.get("accelerator", "cuda")
    validate_options(bootstrap, bootstrap=True)
    bootstrap_inputs = [images]
    bootstrap_outputs = {"sdf": initial_sdf, "few_masks": few_masks}
    if bootstrap.get("annotations"):
        from spatialsam.data.annotations import read_annotations
        from spatialsam.configuration import validate_output_directory
        from utils.misc import VolumeIO

        annotations = Path(bootstrap.annotations)
        mapping = read_annotations(annotations, VolumeIO.list_volume_files(images))
        masks = sorted(
            {Path(entry["mask"]) for entries in mapping.values() for entry in entries}
        )
        validate_output_directory(root, annotations, *masks)
        bootstrap_inputs.extend([annotations, *masks])
        bootstrap_outputs["annotations"] = bootstrap_dir / "annotations_used.json"
    else:
        bootstrap_inputs.append(Path(str(task.shared.paths.true_dir)))
    bootstrap_inputs.append(pretrained)
    stages = [
        Stage(
            "bootstrap",
            "bootstrap",
            bootstrap_dir,
            tuple(bootstrap_inputs),
            bootstrap_outputs,
            OmegaConf.to_container(bootstrap, resolve=True),
        )
    ]
    previous_sdf_checkpoint = None
    previous_sam_checkpoint = pretrained
    previous_sam_sdf = initial_sdf
    for index in range(1, task.num_rounds + 1):
        profile = _round_profile(task, index)
        round_root = root / f"round_{index:03d}"
        sdf_train_root = round_root / "sdf_train"
        sam_train_root = round_root / "sam_train"
        sdf_checkpoint = sdf_train_root / "checkpoints" / "stage.pt"
        sam_checkpoint = sam_train_root / "checkpoints" / "checkpoint.pt"
        sdf_predictions = round_root / "sdf_infer" / "sdf"
        sam_predictions = round_root / "sam_infer" / "sdf"
        for kind in STAGE_ORDER:
            payload = clone_task_payload(task.pipeline.stages[kind])
            directory = round_root / kind
            is_sdf = kind.startswith("sdf")
            if kind.endswith("train"):
                section = "sdf_dataset" if is_sdf else "dataset"
                # SAM supervision remains the bootstrap SDF in every round.
                target = previous_sam_sdf if is_sdf else initial_sdf
                payload[section].config.true_dir = str(target)
                payload[section].config.strict_stem_match = True
                if not is_sdf:
                    payload[section].config.ref_masks_dir = str(sdf_predictions)
                    payload[section].config.few_masks_dir = str(few_masks)
                payload.scratch.num_epochs = (
                    profile.sdf_epochs if is_sdf else profile.sam_epochs
                )
                checkpoint = (
                    previous_sdf_checkpoint if is_sdf else previous_sam_checkpoint
                )
                payload.launcher = {
                    "num_nodes": 1,
                    "gpus_per_node": 1,
                    "ckpt_path": str(checkpoint) if checkpoint else None,
                    "experiment_log_dir": str(directory),
                }
                composed = wrap_stage_task_config(
                    cfg, payload, apply_unet_downsample=is_sdf
                )
                resolved = resolve_sam_training_config(composed)
                resolved.trainer.data.val = None
                resolved.trainer.accelerator = task.shared.runtime.get(
                    "accelerator", "cuda"
                )
                outputs = {"checkpoint": sdf_checkpoint if is_sdf else sam_checkpoint}
                inputs = [images, target]
                if checkpoint:
                    inputs.append(checkpoint)
                if not is_sdf:
                    inputs.extend((sdf_predictions, few_masks))
            else:
                payload.output_root = str(
                    sdf_predictions if is_sdf else sam_predictions
                )
                payload.sdf_model_path = str(sdf_checkpoint)
                payload.restore_output_resolution = False
                payload.strict_stem_match = True
                payload.mode = "sdf_only" if is_sdf else "sam_with_cached_sdf"
                if not is_sdf:
                    payload.sam2_model_path = str(sam_checkpoint)
                    payload.ref_masks_dir = str(sdf_predictions)
                resolved = wrap_stage_task_config(cfg, payload).task
                resolved.device = task.shared.runtime.get("accelerator", "cuda")
                validate_options(resolved)
                inputs = [images, sdf_checkpoint]
                if not is_sdf:
                    inputs.extend((sam_checkpoint, sdf_predictions))
                outputs = {"sdf": sdf_predictions if is_sdf else sam_predictions}
            stages.append(
                Stage(
                    f"round_{index:03d}/{kind}",
                    kind,
                    directory,
                    tuple(inputs),
                    outputs,
                    OmegaConf.to_container(resolved, resolve=True),
                )
            )
        previous_sdf_checkpoint = sdf_checkpoint
        previous_sam_checkpoint = sam_checkpoint
        previous_sam_sdf = sam_predictions
    return stages


def execute_stage(stage: Stage):
    """Execute one stage in the current process, releasing its models afterward."""
    from hydra.utils import instantiate
    import torch

    config = OmegaConf.create(stage.config)
    try:
        if stage.kind.endswith("train"):
            trainer = instantiate(config.trainer, _recursive_=False)
            try:
                trainer.run()
            finally:
                trainer.logger.close()
                del trainer
        elif stage.kind == "bootstrap":
            from predict.pseudo_label_orchestrator import (
                PseudoLabelInitializationOrchestrator,
            )
            from spatialsam.data.io import build_pseudo_label_samples

            PseudoLabelInitializationOrchestrator(config).run_all(
                build_pseudo_label_samples(config)
            )
        else:
            from predict.inference_orchestrator import InferenceOrchestrator
            from spatialsam.data.io import build_inference_samples

            InferenceOrchestrator(config).run_all(build_inference_samples(config))
        if "sdf" in stage.outputs:
            from spatialsam.data.contracts import validate_sdf_outputs

            validate_sdf_outputs(
                stage.outputs["sdf"],
                config.volume_dir,
                config.resolution,
                config.unet_downsample,
            )
    finally:
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def run_pipeline(cfg, *, before_stage=None):
    """Reuse validated stages and resume only artifacts matching their provenance."""
    from spatialsam.stage_cache import StageCache
    from spatialsam.configuration import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    stages = build_stages(cfg)
    # A launcher may already have selected a checked physical GPU by UUID.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.task.cuda_devices))
    cache = StageCache(
        Path(str(cfg.task.shared.paths.output_root)), Path(str(cfg.root))
    )
    start = (
        1
        + (int(cfg.task.start_round) - 1) * len(STAGE_ORDER)
        + int(cfg.task.start_step)
        - 1
    )
    with cache.lock():
        for position, stage in enumerate(stages):
            if before_stage is not None:
                before_stage(stage)
            require_cached = position < start and position != 0
            if position == 0 and not cfg.task.pipeline.bootstrap.enabled:
                require_cached = True
            cache.run(stage, execute_stage, require_cached=require_cached)
