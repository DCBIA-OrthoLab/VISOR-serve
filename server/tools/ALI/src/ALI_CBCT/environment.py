"""The volume an agent navigates: one scan, loaded at every scale.

Ported from ALI_CBCT_utils/environment.py, inference only. Everything the
original carried for training -- loading ground-truth fiducials, sampling
random poses around a known landmark, computing rewards -- is gone; so is the
`sys.version_info >= (3, 10)` branch choosing between monai's
`EnsureChannelFirst` and the `AddChannel` transform removed from monai years
ago. The server pins monai 1.6, where only the former exists.
"""

import logging
import sys

import numpy as np
import SimpleITK as sitk

from base import ToolUnavailableError

from .brain import import_torch

logger = logging.getLogger("ALI.cbct.environment")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)

_MONAI_HINT = (
    "ALI's CBCT engine needs monai. Install it with `pip install -r requirements.txt` "
    "(see server/README.md)."
)


def import_transforms():
    try:
        from monai.transforms import BorderPad, Compose, EnsureChannelFirst, ScaleIntensity, SpatialCrop
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_MONAI_HINT} (missing: monai)") from exc
    return BorderPad, Compose, EnsureChannelFirst, ScaleIntensity, SpatialCrop


class Environment:
    """One patient's scan at every scale, plus the landmarks found in it.

    The agent walks in VOXEL coordinates of the current scale. `padding` (half
    the agent's field of view) is added around the volume so a field of view
    centred near an edge is still a full box, and `physical_position` converts
    a voxel position back to the scanner's own millimetre frame at the end.
    """

    def __init__(self, patient_id: str, padding, device: str):
        BorderPad, Compose, EnsureChannelFirst, ScaleIntensity, SpatialCrop = import_transforms()

        self.patient_id = patient_id
        self.padding = np.asarray(padding).astype(np.int16)
        self.device = device
        self._crop = SpatialCrop
        self._rescale = ScaleIntensity(minv=-1.0, maxv=1.0, factor=None)
        self._transform = Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),
                BorderPad(spatial_border=self.padding.tolist()),
            ]
        )
        self.data = {}
        self.predicted_landmarks = {}

    @property
    def scale_count(self) -> int:
        return len(self.data)

    def load_images(self, images_per_scale: dict) -> None:
        """Read the resampled volume for every scale into memory."""
        torch = import_torch()
        for scale_key, path in images_per_scale.items():
            image = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(image)
            origin = image.GetOrigin()
            self.data[scale_key] = {
                "path": path,
                "image": self._transform(array).to(dtype=torch.int16),
                "spacing": np.array(image.GetSpacing()),
                # SimpleITK reports the origin in (x, y, z) while the array is
                # indexed (z, y, x); reversed once here so every coordinate
                # below is in array order.
                "origin": np.array([origin[2], origin[1], origin[0]]),
                "size": np.array(np.shape(array)),
            }

    def release(self) -> None:
        """Free the volumes. A cohort would not fit in memory otherwise --
        each scan is held at two spacings, the finer one being the larger."""
        self.data = {}

    def spacing(self, scale_key: str):
        return self.data[scale_key]["spacing"]

    def size(self, scale_key: str):
        return self.data[scale_key]["size"]

    def field_of_view(self, scale_key: str, center, crop_size):
        """The rescaled cube of voxels the agent currently sees.

        The centre is shifted by `padding` because `load_images` padded the
        volume: a position in agent coordinates sits `padding` voxels further
        in inside the padded array. (The original expressed this as
        `center.tolist() + self.padding`, which reads like list concatenation
        and is in fact numpy broadcasting the addition.)
        """
        torch = import_torch()
        crop = self._crop((np.asarray(center) + self.padding).tolist(), crop_size)
        return self._rescale(crop(self.data[scale_key]["image"])).type(torch.float32)

    def add_predicted_landmark(self, label: str, position) -> None:
        self.predicted_landmarks[label] = position

    def physical_position(self, scale_key: str, position):
        """A voxel position at `scale_key`, in the scanner's LPS millimetres.

        The inverse of the (x, y, z) -> (z, y, x) flip `load_images` applied,
        so the value returned is in the order a markups file expects.
        """
        reference = self.data[scale_key]
        spacing = reference["spacing"]
        physical_origin = abs(reference["origin"] / spacing)
        coordinates = (np.asarray(position) - physical_origin) * spacing
        return np.array([coordinates[2], coordinates[1], coordinates[0]])
