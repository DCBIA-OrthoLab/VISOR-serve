"""Offscreen rendering of a tooth from several viewpoints.

The IOS engine does not look at the mesh directly: it renders the tooth from a
fixed set of camera directions, runs a 2D UNet on each image, and projects the
predicted mask back onto the faces that produced those pixels. This module
builds the renderer and holds the camera directions.

Ported from ALI_IOS_utils/{render,mask_renderer,agent}.py. **The camera
directions are reproduced byte for byte, including their inconsistencies** --
several are divided by the norm of a *different* vector than the one being
normalized, so the resulting directions are not unit length. That is not a
typo to fix: the shipped weights were trained on the views these exact
vectors produce, and "correcting" them silently changes every prediction.

pytorch3d is imported inside the functions that need it. It has no PyPI
distribution at all and must be compiled into the deployment image, so on a
server without it ALI still loads, still publishes its schema, and only an
IOS run fails -- with a message naming what is missing. The CBCT engine is
unaffected.
"""

import numpy as np

from base import ToolUnavailableError

_INSTALL_HINT = (
    "ALI's IOS engine needs pytorch3d, which has no PyPI distribution and must be "
    "compiled into the deployment image against its torch/CUDA build. See "
    "ALI_PORT_CONTEXT.md 4. The CBCT engine works without it."
)


def _norm(vector) -> float:
    return float(np.linalg.norm(vector))


# Camera directions per network and jaw, verbatim from the original
# `dic_cam`. The occlusal ones look straight down (or up) the arch and fan out
# slightly; the cervical ones orbit it from the side, where those points sit.
CAMERA_POSITIONS = {
    "O": {
        "Upper": (
            [0, 0, -1],
            np.array([0.5, 0.0, -1]) / _norm([0.5, 0.5, -1]),
            np.array([-0.5, 0.0, -1]) / _norm([-0.5, -0.5, -1]),
            np.array([0, 0.5, -1]) / _norm([1, 0, -1]),
            np.array([0, -0.5, -1]) / _norm([0, 1, -1]),
        ),
        "Lower": (
            [0, 0, 1],
            np.array([0.5, 0.0, 1.0]) / _norm([0.5, 0.5, 1.0]),
            np.array([-0.5, 0.0, 1.0]) / _norm([-0.5, -0.5, 1.0]),
            np.array([0, 0.5, 1]) / _norm([1, 0, 1]),
            np.array([0, -0.5, 1]) / _norm([0, 1, 1]),
        ),
    },
    "C": {
        "Upper": tuple(
            np.array(vector) / _norm(vector)
            for vector in [
                [1, 0, 0], [-1, 0, 0], [1, -1, 0], [-1, -1, 0], [1, 1, 0], [-1, 1, 0],
                [1, 0, -0.5], [-1, 0, -0.5], [1, -1, -0.5], [-1, -1, -0.5],
                [1, 1, -0.5], [-1, 1, -0.5],
            ]
        ),
        "Lower": tuple(
            np.array(vector) / _norm(vector)
            for vector in [
                [1, 0, 0], [-1, 0, 0], [1, -1, 0], [-1, -1, 0], [1, 1, 0], [-1, 1, 0],
                [1, 0, 0.5], [-1, 0, 0.5], [1, -1, 0.5], [-1, -1, 0.5],
                [1, 1, 0.5], [-1, 1, 0.5],
            ]
        ),
    },
}

# Rasterization settings the models were trained with. Fixed in the original
# CLI's XML and never exposed to the user; they describe the trained networks,
# not a user preference.
IMAGE_SIZE = 224
BLUR_RADIUS = 0
FACES_PER_PIXEL = 1


def import_pytorch3d():
    try:
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            HardPhongShader,
            MeshRasterizer,
            MeshRenderer,
            PointLights,
            RasterizationSettings,
            blending,
            look_at_rotation,
        )
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: {exc.name or 'pytorch3d'})") from exc

    return {
        "FoVPerspectiveCameras": FoVPerspectiveCameras,
        "HardPhongShader": HardPhongShader,
        "MeshRasterizer": MeshRasterizer,
        "MeshRenderer": MeshRenderer,
        "Meshes": Meshes,
        "PointLights": PointLights,
        "RasterizationSettings": RasterizationSettings,
        "TexturesVertex": TexturesVertex,
        "blending": blending,
        "look_at_rotation": look_at_rotation,
    }


def build_renderer(device, image_size: int = IMAGE_SIZE, blur_radius: float = BLUR_RADIUS,
                   faces_per_pixel: int = FACES_PER_PIXEL):
    """A Phong renderer over a 90-degree perspective camera.

    The original built a second, mask-only renderer alongside this one. It is
    gone: nothing on the inference path ever called it (only the training-time
    `GetView(rend=True)` did), and it was the sole reason `mask_renderer.py`
    existed.
    """
    p3d = import_pytorch3d()

    cameras = p3d["FoVPerspectiveCameras"](znear=0.01, zfar=10, fov=90, device=device)
    raster_settings = p3d["RasterizationSettings"](
        image_size=image_size, blur_radius=blur_radius, faces_per_pixel=faces_per_pixel
    )
    rasterizer = p3d["MeshRasterizer"](cameras=cameras, raster_settings=raster_settings)
    shader = p3d["HardPhongShader"](
        device=device,
        cameras=cameras,
        lights=p3d["PointLights"](device=device),
        blend_params=p3d["blending"].BlendParams(background_color=(0, 0, 0)),
    )
    return p3d["MeshRenderer"](rasterizer=rasterizer, shader=shader)


def tooth_center(labels, vertices, tooth_number: int, device):
    """The centroid of a tooth's vertices, which the cameras orbit.

    Returns None when the tooth is not in this mesh -- the original returned
    a zero vector, which is a real position at the centre of the arch, so the
    cameras dutifully rendered the middle of the palate and the network was
    asked to find a cusp in it.
    """
    from ..ALI_CBCT.brain import import_torch

    torch = import_torch()

    label_table = labels[0]
    matches = (label_table == int(tooth_number)).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return None
    return vertices[0][matches].mean(dim=0).unsqueeze(0).to(device)


def render_views(renderer, mesh, center, radius: float, camera_positions, device):
    """Render the tooth from every camera direction.

    Returns (images, pix_to_face): the batch the UNet consumes, and the face
    each rendered pixel came from, which is how a predicted mask gets back
    onto the mesh.
    """
    from ..ALI_CBCT.brain import import_torch

    torch = import_torch()
    p3d = import_pytorch3d()
    look_at_rotation = p3d["look_at_rotation"]

    images = torch.empty(0).to(device)
    pix_to_faces = torch.empty(0).to(device)

    directions = torch.tensor(np.asarray(camera_positions, dtype=np.float32)).to(device)
    for direction in directions:
        camera_position = center + direction * radius
        rotation = look_at_rotation(camera_position, at=center, device=device)
        translation = -torch.bmm(rotation.transpose(1, 2), camera_position[:, :, None])[:, :, 0]

        rendered = renderer(meshes_world=mesh.clone(), R=rotation, T=translation.to(device))
        rendered = rendered.permute(0, 3, 1, 2)[:, :-1, :, :]

        fragments = renderer.rasterizer(mesh.clone())
        depth = fragments.zbuf.permute(0, 3, 1, 2)

        # The network takes 4 channels: the three normal-as-color ones plus
        # depth.
        view = torch.cat([rendered, depth], dim=1)
        images = torch.cat((images, view.unsqueeze(0)), dim=0)
        pix_to_faces = torch.cat((pix_to_faces, fragments.pix_to_face.unsqueeze(0)), dim=0)

    return images.permute(1, 0, 2, 3, 4), pix_to_faces
