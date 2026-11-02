"""Paired image and SDF augmentation for 3D U-Net training."""

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from training.batch import VolumePatch


class SDFTransform:
    """Augment paired patches before converting them to native volume tensors.

    Independent random streams retain the original transform correlations.
    Geometric draws are shared by the image and target so both remain aligned.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        seed: int,
        training: bool,
        brightness_probability: float = 0.5,
        contrast_probability: float = 0.5,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        axis_probability: float = 0.5,
        elastic_probability: float = 0.4,
        elastic_alpha: float = 1000,
        elastic_sigma: float = 50,
    ):
        self.mean, self.std = mean, std
        self.training = training
        self.brightness_probability = brightness_probability
        self.contrast_probability = contrast_probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.axis_probability = axis_probability
        self.elastic_probability = elastic_probability
        self.elastic_alpha, self.elastic_sigma = elastic_alpha, elastic_sigma
        self.random_states = {
            name: np.random.RandomState(seed)
            for name in ("brightness", "contrast", "flip", "rotation", "elastic")
        }

    def _intensity(self, image: np.ndarray) -> np.ndarray:
        rng = self.random_states["brightness"]
        if rng.uniform() < self.brightness_probability:
            image = np.clip(image * rng.uniform(*self.brightness_range), 0.0, 1.0)
        rng = self.random_states["contrast"]
        if rng.uniform() < self.contrast_probability:
            factor = rng.uniform(*self.contrast_range)
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0.0, 1.0)
        return image

    def _geometry(self, image, target):
        rng = self.random_states["flip"]
        for axis in range(3):
            if rng.uniform() > self.axis_probability:
                image, target = np.flip(image, axis), np.flip(target, axis)
        rng = self.random_states["rotation"]
        axes = ((1, 0), (2, 1), (2, 0))[rng.randint(3)]
        rotations = rng.randint(0, 4)
        image = np.rot90(image, rotations, axes)
        target = np.rot90(target, rotations, axes)

        rng = self.random_states["elastic"]
        if rng.uniform() < self.elastic_probability:
            height, width = image.shape[-2:]
            dx = (
                gaussian_filter(
                    rng.rand(height, width) * 2 - 1, self.elastic_sigma, mode="reflect"
                )
                * self.elastic_alpha
            )
            dy = (
                gaussian_filter(
                    rng.rand(height, width) * 2 - 1, self.elastic_sigma, mode="reflect"
                )
                * self.elastic_alpha
            )
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            map_x, map_y = np.float32(x + dx), np.float32(y + dy)

            def remap(volume, interpolation):
                return np.stack(
                    [
                        cv2.remap(
                            frame,
                            map_x,
                            map_y,
                            interpolation,
                            borderMode=cv2.BORDER_REFLECT,
                        )
                        for frame in volume
                    ]
                )

            image = remap(image, cv2.INTER_LINEAR)
            target = remap(target, cv2.INTER_NEAREST)
        return image, target

    def __call__(self, image: np.ndarray, target: np.ndarray) -> VolumePatch:
        if self.training:
            image = self._intensity(image)
        image = (image - self.mean) / np.clip(self.std, a_min=1e-10, a_max=None)
        if self.training:
            image, target = self._geometry(image, target)
        return VolumePatch(
            torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(
                0
            ),
            torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)).unsqueeze(
                0
            ),
        )

    def state_dict(self):
        return {name: rng.get_state() for name, rng in self.random_states.items()}

    def load_state_dict(self, state):
        if state.keys() != self.random_states.keys():
            raise ValueError("Augmentation random state keys do not match")
        for name, rng in self.random_states.items():
            rng.set_state(state[name])
