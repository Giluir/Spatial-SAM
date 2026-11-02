from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from dataprocess.volume import Volume
from predict.inference_backend import UnifiedInferenceBackend
from predict.configuration import validate_options
from spatialsam.data.io import InferenceSample
from spatialsam.data.sdf import mask_to_sdf
from utils.misc import normalize_resolution


class InferenceOrchestrator:
    def __init__(self, task_cfg):
        validate_options(task_cfg)
        self.task_cfg = task_cfg
        self.backend = UnifiedInferenceBackend(task_cfg)

    def _resolve_mode(self, sample: InferenceSample) -> str:
        explicit_mode = self.task_cfg.get("mode", None)
        if explicit_mode is not None:
            return explicit_mode

        if sample.ref_masks_path:
            return "sam_with_cached_sdf"
        return "end2end"

    def _save_result(
        self,
        result: np.ndarray,
        volume_path: str,
        output_root: str,
        direction: str,
        ori_resolution,
        scale_order: int,
    ) -> str:
        os.makedirs(output_root, exist_ok=True)

        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()

        output_volume = Volume(None)
        output_volume.volume = result

        if direction in ["x", "y", "z"]:
            output_volume.rotate_volume(direction)

        if ori_resolution is not None:
            output_volume.scale_volume_to(ori_resolution, order=scale_order)

        volume_name = os.path.basename(volume_path)
        output_name = f"{volume_name.rsplit('.', 1)[0]}.tif"
        output_path = os.path.join(output_root, output_name)
        output_volume.save_volume(output_path)
        return output_path

    def run_sample(self, sample: InferenceSample) -> np.ndarray:
        source_volume = Volume(sample.volume_path)
        source_volume.load()
        mode = self._resolve_mode(sample)
        convert_to_sdf = mode != "sdf_only" and bool(self.task_cfg.get("to_sdf", False))
        restore_output_resolution = (
            bool(self.task_cfg.get("restore_output_resolution", True))
            and not convert_to_sdf
        )
        ori_resolution = (
            source_volume.get_resolution() if restore_output_resolution else None
        )
        scale_order = 0
        result = self.backend.run(
            mode=mode,
            volume_path=sample.volume_path,
            mean=sample.volume_mean,
            std=sample.volume_std,
            ref_sdf_path=sample.ref_masks_path,
            mask_to_binary=self.task_cfg.get("mask_to_binary", True),
        )
        if convert_to_sdf:
            resolution = normalize_resolution(self.task_cfg.get("resolution", 1024))
            downsample = int(self.task_cfg.get("unet_downsample", 2))
            result.volume = mask_to_sdf(
                result.volume,
                target_shape=tuple(size // downsample for size in resolution),
            )
        elif mode != "sdf_only" and self.task_cfg.get("postprocess", True):
            from predict.postprocess import final_mask

            result.volume = final_mask(
                result.volume,
                result.reference_sdf,
                to_instance=bool(self.task_cfg.get("to_instance", False)),
            )

        if self.task_cfg.get("save_masks", True):
            self._save_result(
                result=result.volume,
                volume_path=sample.volume_path,
                output_root=sample.output_root,
                direction=result.direction,
                ori_resolution=ori_resolution,
                scale_order=scale_order,
            )

        return result.volume

    def run_all(self, samples: List[InferenceSample]) -> None:
        for sample in samples:
            self.run_sample(sample)
