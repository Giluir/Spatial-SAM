from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.misc import VolumeIO


@dataclass
class VolumeDatasetEntry:
    volume_path: str
    masks_path: str
    volume_mean: Any
    volume_std: Any
    ref_masks_path: Optional[str] = None
    few_masks_paths: Optional[List[str]] = None


@dataclass
class InferenceSample:
    volume_path: str
    output_root: str
    dataset_name: Optional[str] = None
    organelle: Optional[str] = None
    split: Optional[str] = None
    volume_mean: Optional[float] = None
    volume_std: Optional[float] = None
    ref_masks_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PseudoLabelInitSample:
    volume_path: str
    masks_path: Optional[str]
    output_root: str
    dataset_name: Optional[str] = None
    organelle: Optional[str] = None
    split: Optional[str] = None
    volume_mean: Optional[float] = None
    volume_std: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def list_volume_files(directory: str, limit_volumes: Optional[int] = None) -> List[str]:
    return VolumeIO.list_volume_files(directory, limit_volumes=limit_volumes)


def _group_few_masks_by_volume(
    volume_path: str, few_masks_files: List[str]
) -> Optional[List[str]]:
    """Group few_masks files by volume.

    Processes files matching pattern: {volume_name}_z_frame_{frame_idx}.{ext}

    Args:
        volume_path: Path to the volume file
        few_masks_files: List of all few_masks file paths

    Returns:
        Sorted list of matching few_mask files, or None if no matches.
        Example: ["volume_001_z_frame_100.tif", "volume_001_z_frame_200.tif"]
    """
    base_name = Path(volume_path).stem
    direction_files = [
        file_path
        for file_path in few_masks_files
        if os.path.basename(file_path).startswith(f"{base_name}_z_frame_")
    ]
    return sorted(direction_files) if direction_files else None


def _build_training_entries_from_simple_config(
    dataset_cfg,
    split: str,
    use_ref_mask: bool = False,
    use_few_masks: bool = False,
) -> List[VolumeDatasetEntry]:
    volume_io = VolumeIO()
    volume_dir = dataset_cfg.get("volume_dir", None)
    masks_dir = dataset_cfg.get("true_dir", None)
    if volume_dir is None or masks_dir is None:
        raise ValueError("dataset_config requires volume_dir and true_dir")

    volume_mean = dataset_cfg.get("mean", 0.0)
    volume_std = dataset_cfg.get("std", 1.0)

    volume_files = volume_io.list_volume_files(volume_dir)
    mask_files = volume_io.list_volume_files(masks_dir)
    mask_files = match_files_by_stem(
        volume_files,
        mask_files,
        "mask",
        strict_stem_match=bool(dataset_cfg.get("strict_stem_match", True)),
    )

    ref_masks_files = None
    if use_ref_mask:
        ref_masks_dir = dataset_cfg.get("ref_masks_dir", None)
        if ref_masks_dir is None:
            raise ValueError(
                "use_ref_mask=True requires ref_masks_dir in dataset_config"
            )
        ref_masks_files = match_files_by_stem(
            volume_files,
            volume_io.list_volume_files(str(ref_masks_dir)),
            "reference mask",
            strict_stem_match=bool(dataset_cfg.get("strict_stem_match", True)),
        )

    few_masks_files = None
    if use_few_masks:
        few_masks_dir = dataset_cfg.get("few_masks_dir", None)
        if few_masks_dir is None:
            raise ValueError(
                "use_few_masks=True requires few_masks_dir in dataset_config"
            )
        few_masks_files = volume_io.list_volume_files(str(few_masks_dir))

    entries: List[VolumeDatasetEntry] = []
    for idx, (volume_path, masks_path) in enumerate(zip(volume_files, mask_files)):
        entries.append(
            VolumeDatasetEntry(
                volume_path=volume_path,
                masks_path=masks_path,
                ref_masks_path=(ref_masks_files[idx] if ref_masks_files else None),
                few_masks_paths=(
                    _group_few_masks_by_volume(volume_path, few_masks_files)
                    if few_masks_files
                    else None
                ),
                volume_mean=volume_mean,
                volume_std=volume_std,
            )
        )

    return entries


def build_training_dataset_entries(
    dataset_cfg,
    split: str,
    use_ref_mask: bool = False,
    use_few_masks: bool = False,
) -> List[VolumeDatasetEntry]:
    if dataset_cfg is None:
        return []

    return _build_training_entries_from_simple_config(
        dataset_cfg,
        split=split,
        use_ref_mask=use_ref_mask,
        use_few_masks=use_few_masks,
    )


def match_files_by_stem(
    primary_paths: List[str],
    other_paths: List[str],
    other_name: str,
    strict_stem_match: bool = True,
) -> List[str]:
    by_stem: Dict[str, str] = {}
    for path in other_paths:
        stem = Path(path).stem
        if stem in by_stem:
            raise ValueError(f"{other_name} contains a duplicate filename stem: {stem}")
        by_stem[stem] = path

    matched: List[Optional[str]] = []
    missing: List[str] = []
    for primary_path in primary_paths:
        stem = Path(primary_path).stem
        path = by_stem.get(stem, None)
        if path is None:
            missing.append(stem)
        matched.append(path)

    if missing:
        if strict_stem_match:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Volume and {other_name} stems do not match; missing examples: {preview}"
            )
        if len(primary_paths) != len(other_paths):
            raise ValueError(
                f"Volume and {other_name} counts differ: {len(primary_paths)} vs {len(other_paths)}"
            )
        return list(other_paths)

    return [str(path) for path in matched if path is not None]


def build_simple_source(task_cfg) -> Dict[str, Any]:
    volume_dir = task_cfg.get("volume_dir", None)
    if volume_dir is None:
        raise ValueError("task.volume_dir is required to build samples")

    true_dir = task_cfg.get("true_dir", None)
    split = str(task_cfg.get("split", "train"))

    return {
        "split": split,
        "volume_dir": str(volume_dir),
        "true_dir": str(true_dir) if true_dir is not None else None,
        "ref_masks_dir": (
            str(task_cfg.get("ref_masks_dir"))
            if task_cfg.get("ref_masks_dir", None)
            else None
        ),
        "few_masks_dir": (
            str(task_cfg.get("few_masks_dir"))
            if task_cfg.get("few_masks_dir", None)
            else None
        ),
        "volume_mean": task_cfg.get("mean", None),
        "volume_std": task_cfg.get("std", None),
    }


def build_inference_samples(
    task_cfg, simple_source: Optional[Dict[str, Any]] = None
) -> List[InferenceSample]:
    limit_volumes = task_cfg.get("limit_volumes", None)
    strict_stem_match = bool(task_cfg.get("strict_stem_match", True))
    output_root = task_cfg.get("output_root")
    if output_root is None:
        raise ValueError("task.output_root is required to build inference samples")

    source = (
        simple_source if simple_source is not None else build_simple_source(task_cfg)
    )
    volume_paths = list_volume_files(source["volume_dir"], limit_volumes=limit_volumes)
    ref_paths = (
        list_volume_files(str(source["ref_masks_dir"]), limit_volumes=limit_volumes)
        if source.get("ref_masks_dir", None)
        else []
    )

    if ref_paths:
        ref_paths = match_files_by_stem(
            volume_paths,
            ref_paths,
            "ref",
            strict_stem_match=strict_stem_match,
        )

    samples: List[InferenceSample] = []
    for idx, volume_path in enumerate(volume_paths):
        samples.append(
            InferenceSample(
                volume_path=volume_path,
                output_root=str(output_root),
                dataset_name=task_cfg.get("dataset_name", None),
                organelle=None,
                split=str(source.get("split", "train")),
                volume_mean=source.get("volume_mean", None),
                volume_std=source.get("volume_std", None),
                ref_masks_path=ref_paths[idx] if ref_paths else None,
                meta={},
            )
        )

    return samples


def build_pseudo_label_samples(
    task_cfg, simple_source: Optional[Dict[str, Any]] = None
) -> List[PseudoLabelInitSample]:
    limit_volumes = task_cfg.get("limit_volumes", None)
    strict_stem_match = bool(task_cfg.get("strict_stem_match", True))
    output_root = task_cfg.get("output_root")
    if output_root is None:
        raise ValueError("task.output_root is required to build bootstrap samples")

    source = (
        simple_source if simple_source is not None else build_simple_source(task_cfg)
    )
    if task_cfg.get("annotations"):
        from spatialsam.data.annotations import read_annotations
        from spatialsam.configuration import validate_output_directory

        mapping = read_annotations(
            task_cfg["annotations"], list_volume_files(source["volume_dir"])
        )
        # Bootstrap writes SDF and few_masks as siblings under its own root.
        validate_output_directory(
            Path(output_root).parent,
            source["volume_dir"],
            task_cfg["annotations"],
            *(entry["mask"] for entries in mapping.values() for entry in entries),
        )
        return [
            PseudoLabelInitSample(
                volume_path=path,
                masks_path=None,
                output_root=str(output_root),
                dataset_name=task_cfg.get("dataset_name"),
                split=str(source.get("split", "train")),
                volume_mean=source.get("volume_mean"),
                volume_std=source.get("volume_std"),
                meta={"annotations": mapping[Path(path).stem]},
            )
            for path in list_volume_files(
                source["volume_dir"], limit_volumes=limit_volumes
            )
        ]
    if not source.get("true_dir", None):
        return []

    all_volume_paths = list_volume_files(source["volume_dir"])
    slice_counts = allocate_prompt_slices(
        len(all_volume_paths), task_cfg.get("slices", 5)
    )
    volume_paths = list_volume_files(source["volume_dir"], limit_volumes=limit_volumes)
    masks_paths = list_volume_files(
        str(source["true_dir"]), limit_volumes=limit_volumes
    )
    masks_paths = match_files_by_stem(
        volume_paths,
        masks_paths,
        "mask",
        strict_stem_match=strict_stem_match,
    )

    samples: List[PseudoLabelInitSample] = []
    for idx, volume_path in enumerate(volume_paths):
        samples.append(
            PseudoLabelInitSample(
                volume_path=volume_path,
                masks_path=masks_paths[idx],
                output_root=str(output_root),
                dataset_name=task_cfg.get("dataset_name", None),
                organelle=None,
                split=str(source.get("split", "train")),
                volume_mean=source.get("volume_mean", None),
                volume_std=source.get("volume_std", None),
                meta={"slices": slice_counts[idx]},
            )
        )

    return samples


def allocate_prompt_slices(volume_count: int, slices_per_volume: float) -> List[int]:
    """Distribute the rounded dataset-wide budget before applying sample limits."""
    if volume_count < 0 or slices_per_volume < 0:
        raise ValueError("Volume count and slice budget must be nonnegative")
    if volume_count == 0:
        return []
    total = round(slices_per_volume * volume_count)
    base, remainder = divmod(total, volume_count)
    return [base + (index < remainder) for index in range(volume_count)]
