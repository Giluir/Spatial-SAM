from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import zoom

from dataprocess.volume import Volume
from sam2.build_sam import build_sam2_video_predictor
from spatialsam.checkpoints import load_model_weights
from spatialsam.models.sdf_unet import SDFUNet3D
from spatialsam.runners.inproc_hydra import local_config_context
from utils.misc import inplane_resolution, normalize_resolution


@dataclass
class BackendResult:
    volume: np.ndarray
    direction: str = "z"
    reference_sdf: Optional[np.ndarray] = None


class UnifiedInferenceBackend:
    """Provide reusable SDF prediction and SAM volume propagation operations."""

    def __init__(self, task_cfg):
        self.task_cfg = task_cfg
        self.device = task_cfg.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.resolution = normalize_resolution(task_cfg.get("resolution", 1024))
        self.unet_downsample = int(task_cfg.get("unet_downsample", 2))

        self._sam_predictor = None
        self._sdf_model = None

    def _sam_infer_context(self):
        device_str = str(self.device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _sdf_infer_context(self):
        cfg = self.task_cfg.get("sdf_infer", {}) or {}
        use_autocast = bool(cfg.get("use_autocast", False))
        if not use_autocast:
            return nullcontext()

        device_str = str(self.device)
        if not device_str.startswith("cuda") or not torch.cuda.is_available():
            return nullcontext()
        if torch.cuda.get_device_properties(self.device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        autocast_dtype = str(cfg.get("autocast_dtype", "bfloat16")).lower()
        dtype = torch.bfloat16 if autocast_dtype == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _resize_volume(
        self, volume: np.ndarray, target_shape: Tuple[int, int, int], order: int
    ) -> np.ndarray:
        scale = np.array(target_shape) / np.array(volume.shape)
        if np.allclose(scale, 1):
            return volume
        original_dtype = volume.dtype
        if original_dtype == np.float16:
            volume = volume.astype(np.float32)
        result = zoom(volume, scale, order=order)
        return result.astype(original_dtype, copy=False)

    def _normalize_volume(
        self, volume: np.ndarray, mean: Optional[float], std: Optional[float]
    ) -> np.ndarray:
        source_dtype = volume.dtype
        volume = volume.astype(np.float32)
        if source_dtype == np.uint8:
            volume = volume / 255.0
        elif source_dtype == np.uint16:
            volume = volume / 65535.0

        if mean is not None and std is not None:
            volume = (volume - float(mean)) / (float(std) + 1e-10)
        return volume

    def _get_sdf_window_cfg(
        self,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        cfg = self.task_cfg.get("sdf_infer", {}) or {}

        patch_shape = tuple(int(v) for v in cfg.get("patch_shape", [128, 128, 128]))
        stride_shape = tuple(int(v) for v in cfg.get("stride_shape", [128, 128, 128]))
        halo_shape = tuple(int(v) for v in cfg.get("halo_shape", [16, 16, 16]))

        if len(patch_shape) != 3 or len(stride_shape) != 3 or len(halo_shape) != 3:
            raise ValueError(
                "task.sdf_infer.{patch_shape,stride_shape,halo_shape} must be lists of length 3"
            )

        return patch_shape, stride_shape, halo_shape

    def _build_starts(self, size: int, patch: int, stride: int) -> List[int]:
        patch = max(1, min(patch, size))
        stride = max(1, stride)

        starts = [0]
        while True:
            current = starts[-1]
            if current + patch >= size:
                break
            nxt = min(current + stride, size - patch)
            if nxt == current:
                break
            starts.append(nxt)
        return starts

    def _predict_sdf_sliding_window(self, volume_np: np.ndarray) -> np.ndarray:
        self._ensure_sdf_model()

        patch_shape, stride_shape, halo_shape = self._get_sdf_window_cfg()
        d, h, w = map(int, volume_np.shape)

        pd, ph, pw = (
            min(patch_shape[0], d),
            min(patch_shape[1], h),
            min(patch_shape[2], w),
        )
        sd, sh, sw = stride_shape
        hd, hh, hw = halo_shape

        z_starts = self._build_starts(d, pd, sd)
        y_starts = self._build_starts(h, ph, sh)
        x_starts = self._build_starts(w, pw, sw)

        # Keep the same halo convention as pytorch3dunet:
        # 1) mirror-pad full volume by halo on each side
        # 2) infer on (patch + 2 * halo)
        # 3) remove halo and write back to [start:start+patch]
        pad_mode = "reflect"
        if any(dim <= 1 for dim in (d, h, w)):
            pad_mode = "edge"
        padded_volume = np.pad(
            volume_np,
            ((hd, hd), (hh, hh), (hw, hw)),
            mode=pad_mode,
        )

        cfg = self.task_cfg.get("sdf_infer", {}) or {}
        output_dtype_cfg = str(cfg.get("output_dtype", "float32")).lower()
        output_dtype = np.float16 if output_dtype_cfg == "float16" else np.float32

        # Each sliding-window accumulation is rounded to the output dtype.
        prediction_map = np.zeros((d, h, w), dtype=output_dtype)
        normalization = np.zeros((d, h, w), dtype=np.uint8)

        with torch.no_grad():
            for z0 in z_starts:
                for y0 in y_starts:
                    for x0 in x_starts:
                        # Read from padded volume so every core patch has halo context.
                        z1, y1, x1 = (
                            z0 + pd + 2 * hd,
                            y0 + ph + 2 * hh,
                            x0 + pw + 2 * hw,
                        )
                        patch = padded_volume[z0:z1, y0:y1, x0:x1]

                        x = (
                            torch.from_numpy(patch)
                            .float()
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(self.device)
                        )

                        with self._sdf_infer_context():
                            pred = (
                                self._sdf_model(x)
                                .squeeze(0)
                                .squeeze(0)
                                .detach()
                                .cpu()
                                .float()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )

                        # Remove halo and write back to the core patch region.
                        core = pred[hd : hd + pd, hh : hh + ph, hw : hw + pw]
                        gz = slice(z0, z0 + pd)
                        gy = slice(y0, y0 + ph)
                        gx = slice(x0, x0 + pw)

                        prediction_map[gz, gy, gx] += core
                        if np.any(normalization[gz, gy, gx] == 255):
                            raise ValueError(
                                "More than 255 patches overlap at one voxel"
                            )
                        normalization[gz, gy, gx] += 1

        if np.any(normalization == 0):
            raise ValueError("The inference patch grid leaves uncovered voxels")
        return (prediction_map / normalization).astype(output_dtype, copy=False)

    def _ensure_sdf_model(self) -> None:
        if self._sdf_model is not None:
            return

        checkpoint_path = self.task_cfg.get("sdf_model_path", None)
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                "SDF checkpoint not found; provide --sdf-checkpoint"
            )

        model_cfg = self.task_cfg.get("sdf_model", {}) or {}
        if not model_cfg:
            model_cfg = {
                "in_channels": 1,
                "out_channels": 1,
                "final_sigmoid": True,
                "f_maps": [32, 64, 128, 256],
                "layer_order": "gcr",
                "num_groups": 8,
                "num_levels": 4,
                "is_segmentation": False,
            }

        self._sdf_model = SDFUNet3D(**model_cfg).to(self.device)
        self._sdf_model.eval()
        load_model_weights(self._sdf_model, checkpoint_path)

    def _ensure_sam_predictor(self) -> None:
        if self._sam_predictor is not None:
            self._sam_predictor.to(self.device)
            return

        config_name = self.task_cfg.get("sam2_config_path")
        checkpoint_path = self.task_cfg.get("sam2_model_path")
        image_size = int(max(self.resolution[1], self.resolution[2]))
        hydra_overrides_extra = [f"++model.image_size={image_size}"]

        # In unified in-process pipeline mode, previous stages may clear Hydra globals.
        # Recreate a local config context so SAM2 compose() is deterministic.
        project_root = Path(
            str(self.task_cfg.get("root", Path(__file__).resolve().parent.parent))
        )
        config_dir = project_root.resolve() / "config"
        with local_config_context(config_dir):
            self._sam_predictor = build_sam2_video_predictor(
                config_name,
                checkpoint_path,
                hydra_overrides_extra=hydra_overrides_extra,
                device=self.device,
            )

    def predict_sdf(
        self, volume_np: np.ndarray, mean: Optional[float], std: Optional[float]
    ) -> np.ndarray:
        target_shape = tuple(
            max(1, dim // self.unet_downsample) for dim in self.resolution
        )
        resized = self._resize_volume(volume_np, target_shape=target_shape, order=0)
        normalized = self._normalize_volume(resized, mean=mean, std=std)
        return self._predict_sdf_sliding_window(normalized)

    def create_sam_state(
        self, frames: np.ndarray, mean: Optional[float], std: Optional[float]
    ):
        self._ensure_sam_predictor()

        normalization_params = None
        if mean is not None and std is not None:
            normalization_params = {"mean": mean, "std": std}

        with self._sam_infer_context():
            return self._sam_predictor.init_state(
                images_np=frames,
                normalization_params=normalization_params,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
            )

    def build_sam_frames(self, volume_np: np.ndarray) -> np.ndarray:
        volume = Volume(None)
        volume.volume = self._resize_volume(
            volume_np,
            target_shape=self.resolution,
            order=0,
        )
        volume.volume_to_frames(
            direction="z", img_size=inplane_resolution(self.resolution), use_pil=False
        )
        return volume.frames

    def add_mask_prompts(
        self, state, prompts: dict[int, np.ndarray], obj_id: int = 0
    ) -> None:
        self._ensure_sam_predictor()
        with self._sam_infer_context():
            for frame_idx, mask in prompts.items():
                self._sam_predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=int(frame_idx),
                    obj_id=obj_id,
                    mask=mask,
                )

    def propagate_single(
        self, state, start_frame: int, twice: bool = False
    ) -> np.ndarray:
        self._ensure_sam_predictor()

        frame_to_mask = {}

        if twice:
            with self._sam_infer_context():
                iterator = self._sam_predictor.propagate_in_volume(state)
                for output in iterator:
                    out_frame_idx, out_obj_ids, out_mask = output[:3]
                    if len(out_obj_ids) == 0:
                        mask = torch.zeros_like(out_mask[0])
                    else:
                        mask = out_mask[0]
                    frame_to_mask[int(out_frame_idx)] = mask.squeeze().to(torch.float16)
        else:
            with self._sam_infer_context():
                forward_iterator = self._sam_predictor.propagate_in_video(
                    state,
                    start_frame_idx=int(start_frame),
                    reverse=False,
                )
                for output in forward_iterator:
                    out_frame_idx, out_obj_ids, out_mask = output[:3]
                    if len(out_obj_ids) == 0:
                        mask = torch.zeros_like(out_mask[0])
                    else:
                        mask = out_mask[0]
                    frame_to_mask[int(out_frame_idx)] = mask.squeeze().to(torch.float16)

                if start_frame > 0:
                    reverse_iterator = self._sam_predictor.propagate_in_video(
                        state,
                        start_frame_idx=int(start_frame),
                        reverse=True,
                    )
                    for output in reverse_iterator:
                        out_frame_idx, out_obj_ids, out_mask = output[:3]
                        if int(out_frame_idx) == start_frame:
                            continue
                        if len(out_obj_ids) == 0:
                            mask = torch.zeros_like(out_mask[0])
                        else:
                            mask = out_mask[0]
                        frame_to_mask[int(out_frame_idx)] = mask.squeeze().to(
                            torch.float16
                        )

        ordered = [frame_to_mask[index] for index in sorted(frame_to_mask.keys())]
        if not ordered:
            return np.zeros((0, 0, 0), dtype=np.float32)

        masks = torch.stack(ordered)
        return masks.detach().cpu().numpy()

    def attach_reference_sdf(self, state, ref_sdf_np: np.ndarray) -> None:
        self._ensure_sam_predictor()

        ref_volume = Volume(None)
        ref_volume.volume = self._resize_volume(
            ref_sdf_np,
            target_shape=self.resolution,
            order=0,
        )
        ref_volume.volume = ref_volume.volume * self.unet_downsample
        ref_volume.volume_to_frames(
            direction="z", img_size=inplane_resolution(self.resolution), use_pil=False
        )

        frame_to_masks = {
            idx: ref_volume.frames[idx] for idx in range(len(ref_volume.frames))
        }
        self._sam_predictor.track_in_both_directions = True
        with self._sam_infer_context():
            self._sam_predictor.add_ref_masks(state, 0, frame_to_masks)

    def propagate_volume_masks(self, state, mask_to_binary: bool) -> np.ndarray:
        self._ensure_sam_predictor()

        masks = []
        with self._sam_infer_context():
            for _, out_obj_ids, out_mask, _ in self._sam_predictor.propagate_in_volume(
                state
            ):
                if len(out_obj_ids) == 0:
                    mask = torch.zeros_like(out_mask[0])
                else:
                    mask = out_mask[0]
                masks.append(mask.squeeze().to(torch.float16))

        output = torch.stack(masks)
        if mask_to_binary:
            output = output > 0
        return output.detach().cpu().numpy()

    def release_sam_state(self, state) -> None:
        self._ensure_sam_predictor()
        self._sam_predictor.reset_state(state)
        self._sam_predictor.clear_state(state)

    def offload_sam_model(self) -> None:
        """Release model memory before the full-volume Gaussian convolution."""
        if self._sam_predictor is not None:
            self._sam_predictor.to("cpu")
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def predict_sam(
        self,
        frames_np: np.ndarray,
        ref_sdf_np: np.ndarray,
        mean: Optional[float],
        std: Optional[float],
        mask_to_binary: bool,
    ) -> np.ndarray:
        state = self.create_sam_state(frames_np, mean=mean, std=std)
        try:
            self.attach_reference_sdf(state, ref_sdf_np)
            return self.propagate_volume_masks(state, mask_to_binary=mask_to_binary)
        finally:
            self.release_sam_state(state)

    def run(
        self,
        mode: str,
        volume_path: str,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        ref_sdf_path: Optional[str] = None,
        mask_to_binary: bool = True,
    ) -> BackendResult:
        volume = Volume(volume_path)
        volume.load()

        if mode == "sdf_only":
            sdf = self.predict_sdf(volume.volume, mean=mean, std=std)
            return BackendResult(volume=sdf, direction="z")

        if mode == "sam_with_cached_sdf":
            if ref_sdf_path is None:
                raise ValueError("mode=sam_with_cached_sdf requires ref_sdf_path")
            ref = Volume(ref_sdf_path)
            ref.load()
            sam_frames = self.build_sam_frames(volume.volume)
            mask = self.predict_sam(
                sam_frames,
                ref.volume,
                mean=mean,
                std=std,
                mask_to_binary=mask_to_binary,
            )
            return BackendResult(volume=mask, direction="z", reference_sdf=ref.volume)

        if mode == "end2end":
            sdf = self.predict_sdf(volume.volume, mean=mean, std=std)
            sam_frames = self.build_sam_frames(volume.volume)
            mask = self.predict_sam(
                sam_frames,
                sdf,
                mean=mean,
                std=std,
                mask_to_binary=mask_to_binary,
            )
            return BackendResult(volume=mask, direction="z", reference_sdf=sdf)

        raise ValueError(f"Unsupported mode: {mode}")
