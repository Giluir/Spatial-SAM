from PIL import Image
import numpy as np
from scipy.ndimage import zoom
from utils.misc import VolumeIO


class Volume:
    def __init__(self, volume_path=None, io=None):
        self.volume_path = volume_path
        self.volume = None
        self.frames = None
        self.io = VolumeIO() if io is None else io

    def load(self):
        self.volume = self.io.load(self.volume_path)

    def volume_to_binary_bool(self, threshold=0):
        if self.volume is None:
            self.load()
        if self.volume.dtype != np.bool_:
            self.volume = self.volume > threshold

    def scale_volume(self, scale, order=0, copy=False):
        """
        Scale the volume by a factor

        Args:
            scale (float or tuple): scale factor for each dimension
            order (int): order of interpolation
        """
        is_float16 = self.volume.dtype == np.float16
        if copy:
            volume = self.volume.copy()
            if is_float16:
                volume = volume.astype(np.float32)
            volume = zoom(volume, scale, order=order)
            if is_float16:
                volume = volume.astype(np.float16)
            return volume
        else:
            if is_float16:
                self.volume = self.volume.astype(np.float32)
            self.volume = zoom(self.volume, scale, order=order)
            if is_float16:
                self.volume = self.volume.astype(np.float16)

    def volume_to_frames(
        self,
        img_start=(0, 0),
        img_size=(1024, 1024),
        frame_start=0,
        frame_end=None,
        direction="z",
        use_pil=True,
    ):
        """
        Extracts frames from the volume along the specified
        direction and saves them as either PIL images or numpy arrays

        Args:
            img_start (tuple): starting point of the image (y, x)
            img_size (tuple): size of the image
            frame_start (int): starting frame
            frame_end (int): ending frame
            direction (str): direction of the frames
            use_pil (bool): whether to use PIL images
        """
        if self.volume is None:
            self.load()
        self.frames = []

        if frame_end is None:
            frame_end = self.volume.shape[0]
        # zyx
        if direction == "y":
            self.volume = self.volume.transpose(1, 0, 2)  # yzx

        elif direction == "x":
            self.volume = self.volume.transpose(2, 1, 0)  # xyz

        if use_pil:
            for i in range(frame_start, frame_end):
                img = Image.fromarray(
                    self.volume[
                        i,
                        img_start[0] : img_start[0] + img_size[0],
                        img_start[1] : img_start[1] + img_size[1],
                    ]
                )
                self.frames.append(img)
        else:
            self.frames = self.volume[
                frame_start:frame_end,
                img_start[0] : img_start[0] + img_size[0],
                img_start[1] : img_start[1] + img_size[1],
            ]

        if direction == "y":
            self.volume = self.volume.transpose(1, 0, 2)
        elif direction == "x":
            self.volume = self.volume.transpose(2, 1, 0)

        return {
            "volume_size": self.volume.shape,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "img_start": img_start,
            "img_size": img_size,
            "direction": direction,
        }

    def rotate_volume(self, src_direction, tgt_direction="z"):
        assert src_direction in ["z", "y", "x"]
        assert tgt_direction in ["z", "y", "x"]

        if src_direction == tgt_direction:
            return
        # Rotate the volume to zyx
        if src_direction == "y":
            self.volume = self.volume.transpose(1, 0, 2)
        elif src_direction == "x":
            self.volume = self.volume.transpose(2, 1, 0)

        # Rotate the volume to the target direction
        if tgt_direction == "y":
            self.volume = self.volume.transpose(1, 0, 2)
        elif tgt_direction == "x":
            self.volume = self.volume.transpose(2, 1, 0)

    def scale_volume_to(self, tgt_res, copy=False, order=0):
        res = self.volume.shape
        scale = np.array(tgt_res) / np.array(res)
        if not np.allclose(scale, 1):
            if copy:
                return self.scale_volume(scale, order=order, copy=True)
            else:
                self.scale_volume(scale, order=order)
        else:
            if copy:
                return self.volume

    def save_volume(self, output_path):
        if self.volume is None:
            self.load()

        self.io.save(self.volume, output_path)

    def get_resolution(self):
        return self.volume.shape
