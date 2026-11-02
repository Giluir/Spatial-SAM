from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List

import numpy as np

from dataprocess.volume import Volume
from predict.inference_backend import UnifiedInferenceBackend
from predict.configuration import validate_options
from predict.prompt_selector import PromptSelector
from spatialsam.data.io import PseudoLabelInitSample
from spatialsam.data.sdf import blur_logits, mask_to_sdf
from utils.misc import (
    inplane_resolution,
    normalize_resolution,
)


class PseudoLabelInitializationOrchestrator:
    def __init__(self, task_cfg):
        validate_options(task_cfg, bootstrap=True)
        self.task_cfg = task_cfg
        self.backend = UnifiedInferenceBackend(task_cfg)
        self.selector = PromptSelector()
        self.resolution = normalize_resolution(task_cfg.get("resolution", 1024))
        self.annotations_used = {}

    def _save_volume(
        self,
        volume_np: np.ndarray,
        volume_path: str,
        output_root: str,
        suffix: str,
    ) -> str:
        os.makedirs(output_root, exist_ok=True)
        out_volume = Volume(None)
        out_volume.volume = volume_np
        out_volume.rotate_volume("z")

        volume_name = os.path.basename(volume_path)
        output_name = f"{volume_name.rsplit('.', 1)[0]}.tif"
        if suffix != "avg":
            output_root = os.path.join(output_root, suffix)
            os.makedirs(output_root, exist_ok=True)
        output_path = os.path.join(output_root, output_name)
        out_volume.save_volume(output_path)
        return output_path

    def _run_inference(
        self,
        volume_np: np.ndarray,
        mask_np: np.ndarray,
        mean,
        std,
        slices: int | None,
        prompts: dict[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        resolution = self.resolution

        volume_obj = Volume(None)
        volume_obj.volume = volume_np
        volume_obj.scale_volume_to(resolution, order=0)
        volume_obj.volume_to_frames(
            direction="z",
            img_size=inplane_resolution(resolution),
            use_pil=False,
        )

        if prompts is None:
            mask_obj = Volume(None)
            mask_obj.volume = mask_np > 0
            mask_obj.scale_volume_to(resolution, order=0)
            mask_obj.volume_to_frames(
                direction="z",
                img_size=inplane_resolution(resolution),
                use_pil=False,
            )

            prompts = self.selector.select_prompts(
                mask_obj.frames,
                task_cfg=self.task_cfg,
                slices=slices,
            )
        frame_indices = sorted(prompts.keys())
        if not frame_indices:
            return np.zeros_like(volume_obj.volume, dtype=np.float32), {}

        start_frame = frame_indices[len(frame_indices) // 2]
        state = self.backend.create_sam_state(
            frames=volume_obj.frames,
            mean=mean,
            std=std,
        )
        try:
            self.backend.add_mask_prompts(state, prompts, obj_id=0)
            mask = self.backend.propagate_single(
                state,
                start_frame=start_frame,
                twice=bool(self.task_cfg.get("twice", False)),
            )
            if self.task_cfg.get("mask_to_binary", False):
                mask = (mask > 0).astype(np.float32)
            return mask, prompts
        finally:
            self.backend.release_sam_state(state)

    def run_sample(self, sample: PseudoLabelInitSample) -> np.ndarray:
        volume = Volume(sample.volume_path)
        volume.load()

        sparse_prompts = None
        mask_np = None
        annotation_records = None
        if "annotations" in sample.meta:
            from spatialsam.data.annotations import load_annotation_prompts

            sparse_prompts, annotation_records = load_annotation_prompts(
                sample.meta["annotations"], volume.volume.shape, self.resolution
            )
        else:
            masks = Volume(sample.masks_path)
            masks.load()
            masks.volume_to_binary_bool()
            mask_np = masks.volume

        pred, prompts = self._run_inference(
            volume_np=volume.volume,
            mask_np=mask_np,
            mean=sample.volume_mean,
            std=sample.volume_std,
            slices=sample.meta.get("slices", None),
            prompts=sparse_prompts,
        )

        if self.task_cfg.get("save_xyz_masks", False):
            self._save_volume(
                pred,
                sample.volume_path,
                sample.output_root,
                suffix="z",
            )

        # Z-only: use prediction directly without multi-direction merge
        merged = pred

        sigma = float(self.task_cfg.get("pseudo_blur_sigma", 1.0))
        self.backend.offload_sam_model()
        merged = blur_logits(
            merged,
            sigma=sigma,
            device=self.backend.device,
            use_autocast=str(self.backend.device).startswith("cuda"),
        )

        if bool(self.task_cfg.get("to_sdf", False)):
            downsample = int(self.task_cfg.get("unet_downsample", 2)) > 1
            target = (
                self.resolution[0] // int(self.task_cfg.get("unet_downsample", 2)),
                self.resolution[1] // int(self.task_cfg.get("unet_downsample", 2)),
                self.resolution[2] // int(self.task_cfg.get("unet_downsample", 2)),
            )
            merged = mask_to_sdf(merged, target_shape=target if downsample else None)

        if bool(self.task_cfg.get("save_masks", True)):
            self._save_volume(
                merged,
                sample.volume_path,
                sample.output_root,
                suffix="avg",
            )

        if bool(self.task_cfg.get("save_few_masks", False)):
            output_few_dir = str(Path(sample.output_root).parent / "few_masks")
            os.makedirs(output_few_dir, exist_ok=True)

            frame_name = Path(sample.volume_path).stem
            for frame_idx, frame in prompts.items():
                out_path = os.path.join(
                    output_few_dir,
                    f"{frame_name}_z_frame_{frame_idx}.tif",
                )
                out = Volume(None)
                out.volume = frame
                out.save_volume(out_path)

        if annotation_records is not None:
            self.annotations_used[Path(sample.volume_path).stem] = {
                "image": sample.volume_path,
                "original_shape": list(volume.volume.shape),
                "inference_shape": list(self.resolution),
                "slices": annotation_records,
            }
            path = Path(sample.output_root).parent / "annotations_used.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.annotations_used, ensure_ascii=False, indent=2)
            )
            temporary.replace(path)

        return merged

    def run_all(self, samples: List[PseudoLabelInitSample]) -> None:
        # Check every supplied slice before initializing SAM on the first volume.
        from spatialsam.data.annotations import validate_annotation_samples

        validate_annotation_samples(samples, self.resolution)
        for sample in samples:
            self.run_sample(sample)
