from __future__ import annotations

from typing import Dict

import numpy as np


class PromptSelector:
    """Select few-shot prompt masks from a 3D mask volume."""

    def _select_with_stride(
        self, mask_frames: np.ndarray, label_start: int, label_stride: int
    ) -> Dict[int, np.ndarray]:
        real_start = label_start % label_stride
        selected = mask_frames[real_start::label_stride]

        out: Dict[int, np.ndarray] = {}
        for index, mask in enumerate(selected):
            frame_idx = real_start + index * label_stride
            out[frame_idx] = mask
        return out

    def _select_with_object(
        self, mask_frames: np.ndarray, slices: int
    ) -> Dict[int, np.ndarray]:
        with_object = {
            idx: mask for idx, mask in enumerate(mask_frames) if np.any(mask)
        }
        if len(with_object) <= slices:
            return with_object

        frame_indices = sorted(with_object.keys())
        chosen = np.linspace(0, len(frame_indices) - 1, slices, dtype=int)
        return {frame_indices[i]: with_object[frame_indices[i]] for i in chosen}

    def select_prompts(
        self, mask_frames: np.ndarray, task_cfg, slices: int | None = None
    ) -> Dict[int, np.ndarray]:
        use_object = bool(task_cfg.get("select_mask_with_obj", False))
        label_start = int(task_cfg.get("label_start", 400))
        label_stride = int(task_cfg.get("label_stride", 200))

        if use_object:
            target_slices = int(
                slices if slices is not None else task_cfg.get("slices", 5)
            )
            return self._select_with_object(mask_frames, target_slices)
        return self._select_with_stride(mask_frames, label_start, label_stride)
