"""The multi-view network that draws the palatal patch AREG's IOS mode
registers on.

Ported from `AREG_IOS/AREG_IOS_utils/net.py` (MonaiUNetHRes / TimeDistributed).
A mesh is rendered from seven fixed viewpoints above the arch; each rendering
is four channels (the normals as RGB, plus the depth map); a 2D monai UNet
segments all seven; the per-pixel predictions are scattered back onto the faces
they came from through pytorch3d's pixel-to-face map.

Everything the training loop needed is gone -- loss, metrics, optimizer, the
`*_step` methods and the `pytorch_lightning` base class. This module only loads
a checkpoint and runs a forward pass, and the checkpoint stores a plain
`state_dict`, so a `torch.nn.Module` is enough. That keeps `pytorch_lightning`
and `torchmetrics` off the import path of a tool that never trains.

Every heavy import is inside a function: `registry.py` imports every tool at
startup, so a module-level `import pytorch3d` would take AREG out of the
registry on a deployment that lacks it -- CBCT registration included, which
needs none of this.
"""

import logging

from base import ToolUnavailableError

logger = logging.getLogger("AREG")

_INSTALL_HINT = (
    "AREG's IOS mode renders each mesh from several viewpoints, which needs "
    "pytorch3d and monai. pytorch3d has no PyPI distribution and is compiled into "
    "the deployment image; a server without it can still run AREG's CBCT mode."
)

# Where the seven cameras sit, in the unit-sphere space `surfaces.scale_to_unit`
# puts the mesh in: clustered above the arch, looking down at the palate.
_CAMERA_POSITIONS = (
    (0.0, 0.0, 0.9),
    (0.2, 0.0, 0.9),
    (-0.2, 0.0, 0.9),
    (0.0, 0.2, 0.9),
    (0.0, -0.2, 0.9),
    (-0.2, -0.2, 0.9),
    (0.2, -0.2, 0.9),
)

IMAGE_SIZE = 320
OUT_CHANNELS = 2


def check_dependencies() -> None:
    """Import the whole stack once, before any mesh is read.

    A missing dependency belongs to the server, not to one patient: found
    inside the per-mesh loop it makes a 40-patient batch fail 40 times
    identically, each time only after that mesh's preprocessing, and the run
    ends on a summary that buries the one line naming what to install.
    """
    _import_torch()
    _import_monai()
    _import_pytorch3d()


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: torch)") from exc
    return torch


def _import_monai():
    try:
        import monai
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: monai)") from exc
    return monai


def _import_pytorch3d():
    try:
        from pytorch3d.renderer import (
            AmbientLights,
            FoVPerspectiveCameras,
            HardPhongShader,
            MeshRasterizer,
            MeshRenderer,
            RasterizationSettings,
            TexturesVertex,
            look_at_rotation,
        )
        from pytorch3d.structures import Meshes
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: pytorch3d)") from exc
    return {
        "AmbientLights": AmbientLights,
        "FoVPerspectiveCameras": FoVPerspectiveCameras,
        "HardPhongShader": HardPhongShader,
        "MeshRasterizer": MeshRasterizer,
        "MeshRenderer": MeshRenderer,
        "RasterizationSettings": RasterizationSettings,
        "TexturesVertex": TexturesVertex,
        "look_at_rotation": look_at_rotation,
        "Meshes": Meshes,
    }


def build(device: str):
    """The network, on `device`, with no weights loaded yet."""
    torch = _import_torch()
    monai = _import_monai()
    p3d = _import_pytorch3d()

    class TimeDistributed(torch.nn.Module):
        """Apply a 2D module to a (batch, views, ...) tensor."""

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, sequence):
            size = sequence.size()
            flat = sequence.contiguous().view(size[0] * size[1], *size[2:])
            output = self.module(flat)
            return output.contiguous().view(size[0], size[1], *output.size()[1:])

    class MultiViewUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            unet = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=4,
                out_channels=OUT_CHANNELS,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
            self.model = TimeDistributed(unet)
            self.register_buffer(
                "ico_verts", torch.tensor(_CAMERA_POSITIONS, dtype=torch.float32)
            )
            self.renderer = p3d["MeshRenderer"](
                rasterizer=p3d["MeshRasterizer"](
                    cameras=p3d["FoVPerspectiveCameras"](),
                    raster_settings=p3d["RasterizationSettings"](
                        image_size=IMAGE_SIZE,
                        blur_radius=0,
                        faces_per_pixel=1,
                        max_faces_per_bin=200000,
                        perspective_correct=True,
                    ),
                ),
                shader=p3d["HardPhongShader"](
                    cameras=p3d["FoVPerspectiveCameras"](), lights=p3d["AmbientLights"]()
                ),
            )

        def to(self, *args, **kwargs):
            self.renderer = self.renderer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

        def render(self, vertices, faces, colors):
            meshes = p3d["Meshes"](
                verts=vertices, faces=faces, textures=p3d["TexturesVertex"](verts_features=colors)
            )
            images, pixel_to_face = [], []
            for position in self.ico_verts:
                position = position.unsqueeze(0).to(vertices.device)
                rotation = p3d["look_at_rotation"](position, device=vertices.device)
                translation = -torch.bmm(rotation.transpose(1, 2), position[:, :, None])[:, :, 0]

                rendered = self.renderer(meshes_world=meshes.clone(), R=rotation, T=translation)
                fragments = self.renderer.rasterizer(meshes.clone())

                view = torch.cat([rendered[:, :, :, 0:3], fragments.zbuf], dim=-1)
                images.append(view.permute(0, 3, 1, 2).unsqueeze(1))
                pixel_to_face.append(fragments.pix_to_face.permute(0, 3, 1, 2).unsqueeze(1))

            return torch.cat(images, dim=1), torch.cat(pixel_to_face, dim=1)

        def forward(self, batch):
            vertices, faces, colors = batch
            views, pixel_to_face = self.render(vertices, faces, colors)
            return self.model(views), views, pixel_to_face

    return MultiViewUNet().to(device)


def load(checkpoint_path: str, device: str):
    """The network with `checkpoint_path`'s weights, in eval mode.

    `map_location` is the device this server actually runs on. The original
    called a bare `torch.load(path)`, which restores every tensor onto the
    device the checkpoint was SAVED from -- a CUDA checkpoint loaded on a CPU
    deployment raises before the first mesh is read.
    """
    torch = _import_torch()

    model = build(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # The checkpoint was written by a LightningModule that held the same UNet
    # under the same name, plus buffers this inference-only copy does not have.
    missing, unexpected = model.load_state_dict(state, strict=False)
    trained = [name for name in state if name.startswith("model.")]
    if not trained:
        raise ToolUnavailableError(
            f"'{checkpoint_path}' holds no 'model.*' weights: it is not an AREG IOS "
            f"registration checkpoint."
        )
    if missing:
        logger.warning("AREG IOS: %d weight(s) not present in the checkpoint", len(missing))
    logger.debug("AREG IOS: %d unused key(s) in the checkpoint", len(unexpected))
    model.eval()
    return model
