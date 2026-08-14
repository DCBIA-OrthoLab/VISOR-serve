"""Predicting the palatal patch ("Butterfly") a pair of intra-oral scans is
registered on, and turning it into the point cloud the ICP consumes.

Ported from `AREG_IOS/AREG_IOS_utils/PredPatch.py`, the preprocessing half of
`dataset.py` and the `vtkMeshTeeth` point-cloud builder in `vtkSegTeeth.py`.

The patch is the rugae area of the palate: the part of an arch orthodontic
treatment does not move, which is why registering two timepoints on it measures
what the teeth did rather than what the whole model did.

Corrections, all of which were silent:

* the device was hardcoded. `PredPatch.__init__` did `torch.device("cuda")` and
  every tensor went through `.cuda()`, so a CPU deployment raised inside the
  first forward pass. The device is `settings.DEVICE` now;
* background pixels voted for the last face. `P_faces[:, pf] += pred` indexes
  with pytorch3d's pixel-to-face map, whose value is **-1** where a pixel hit
  no geometry -- and -1 indexes the last element. Those pixels were zeroed
  before a softmax that turned the zeros back into an even 0.5/0.5 vote. Only
  pixels that hit a face contribute here;
* the face array was assumed to be triangles by a `reshape(-1, 4)` that does
  not fail on anything else, it just reads the wrong indices. The mesh is
  triangulated first (see `postprocess.triangulate`).
"""

import logging
import os
import threading

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from base import ToolArgumentError
from config import settings

from . import net, orientation, postprocess, surfaces

logger = logging.getLogger("AREG")

PATCH_ARRAY_NAME = "Butterfly"

# One request may hold several meshes and the server may be serving several
# requests; the renderer allocates a 320x320x7 rasterization per mesh plus the
# UNet's activations. Sized like CrownSeg's, which does the same kind of work.
_gpu_semaphore = threading.Semaphore(settings.AREG_MAX_GPU_JOBS)


def resolve_device() -> str:
    """"cuda" only when a card is actually there, "cpu" otherwise."""
    torch = net._import_torch()
    if settings.DEVICE.startswith("cuda") and torch.cuda.is_available():
        return settings.DEVICE
    if settings.DEVICE.startswith("cuda"):
        logger.warning("AREG IOS: DEVICE is %s but no GPU is visible, using cpu", settings.DEVICE)
    return "cpu"


CHECKPOINT_EXTENSIONS = (".ckpt", ".pth")


def find_checkpoint(model_path: str) -> str:
    """The checkpoint file inside a hosted model entry.

    A `server_selectable` name resolves to whatever is under
    `DATA/AREG/models/<name>` -- and the published bundle (`AREG_model.zip`) is
    a FOLDER, so what arrives here is a directory, not the `.ckpt` the network
    loads. The Slicer module did the same search
    (`getModel(folder, extension="ckpt")`) and took `[0]` of whatever it found;
    an ambiguity is a 422 naming the candidates instead, because which weights
    registered a patient must never be a surprise.
    """
    if os.path.isfile(model_path):
        return model_path

    found = sorted(
        os.path.join(directory, file_name)
        for directory, _, file_names in os.walk(model_path)
        for file_name in file_names
        if file_name.lower().endswith(CHECKPOINT_EXTENSIONS) and not file_name.startswith(".")
    )
    name = os.path.basename(str(model_path).rstrip(os.sep))
    if not found:
        raise ToolArgumentError(
            f"'{name}' holds no {' or '.join(CHECKPOINT_EXTENSIONS)} checkpoint. Fetch the "
            f"published bundle with `./scripts/setup-models.sh --tool AREG`, or name "
            f"another entry in 'registration_model'."
        )
    if len(found) > 1:
        raise ToolArgumentError(
            f"'{name}' holds several checkpoints "
            f"({', '.join(os.path.basename(path) for path in found)}): "
            f"'registration_model' has to name an entry holding exactly one."
        )
    return found[0]


class PatchPredictor:
    """The network, loaded once and reused for every mesh of a run.

    Loading a checkpoint per mesh is what the original did (a `PredPatch` per
    CLI invocation, and the Slicer module invoked the CLI once per batch);
    holding it here means a forty-patient batch pays for it once.
    """

    def __init__(self, model_path: str, device: str = None):
        self.device = device or resolve_device()
        self.checkpoint = find_checkpoint(str(model_path))
        self.model = net.load(self.checkpoint, self.device)

    def __call__(self, surface: vtk.vtkPolyData) -> tuple:
        """`(mesh carrying the patch array, note or None)`.

        The mesh is returned in its own coordinates: the canonical orientation
        is a rendering convenience and never leaves this function.
        """
        surface = postprocess.triangulate(surface)

        note = None
        try:
            oriented, _matrix = orientation.to_canonical(surface)
        except orientation.OrientationError as exc:
            # Faithful to the original, which carried on with a log line. The
            # difference is that this reaches the run report: a patch predicted
            # from cameras pointed at the wrong part of an unoriented arch is
            # the one failure mode that still produces a plausible-looking
            # result, so it has to be visible next to it.
            oriented = surface
            note = (
                f"patch predicted without the canonical orientation ({exc}); "
                f"check the result if the mesh was not oriented beforehand"
            )
            logger.warning("AREG IOS: %s", note)

        labels = self._predict(oriented)

        adjacency = postprocess.Adjacency(surface)
        labels = postprocess.clean_patch(labels, adjacency)

        array = numpy_to_vtk(np.ascontiguousarray(labels.astype(np.int64)), deep=True)
        array.SetName(PATCH_ARRAY_NAME)
        surface.GetPointData().AddArray(array)
        return surface, note

    def _predict(self, oriented: vtk.vtkPolyData) -> np.ndarray:
        torch = net._import_torch()

        vertices, faces, colors = _to_tensors(oriented, self.device)
        with _gpu_semaphore, torch.no_grad():
            logits, _views, pixel_to_face = self.model((vertices, faces, colors))
            # Zeroed where the pixel hit nothing, so the softmax below is
            # computed on real predictions only.
            probabilities = torch.nn.functional.softmax(logits * (pixel_to_face >= 0), dim=2)

            face_votes = torch.zeros(net.OUT_CHANNELS, faces.shape[1], device=self.device)
            for view_faces, view_probabilities in zip(pixel_to_face.squeeze(0), probabilities.squeeze(0)):
                view_faces = view_faces.squeeze(0)
                hit = view_faces >= 0
                face_votes[:, view_faces[hit]] += view_probabilities[:, hit]

            face_labels = torch.argmax(face_votes, dim=0)

            # A vertex takes the label of a face it is the first corner of;
            # every other vertex stays background. Unchanged from the original,
            # and what the checkpoint was trained to produce.
            vertex_labels = torch.zeros(vertices.shape[1], dtype=torch.int64, device=self.device)
            vertex_labels[faces[0, :, 0]] = face_labels
            return (vertex_labels >= 1).to(torch.int64).cpu().numpy()


def _to_tensors(surface: vtk.vtkPolyData, device: str) -> tuple:
    """The (vertices, faces, colours) batch the network takes.

    The colours are the point normals mapped to RGB, which is the only texture
    the network ever sees -- it segments shape, not appearance.
    """
    torch = net._import_torch()

    scaled = surfaces.compute_normals(surfaces.scale_to_unit(surface))

    vertices = torch.tensor(surfaces.points_of(scaled), dtype=torch.float32)
    faces = torch.tensor(postprocess.faces_of(scaled).astype(np.int64), dtype=torch.int64)
    normals = vtk_to_numpy(scaled.GetPointData().GetArray("Normals"))
    # (n * 0.5 + 0.5) * 255 / 255 -- the original built a uint8 array and
    # divided it back by 255, which is the same mapping through a rounding step.
    colors = torch.tensor(normals * 0.5 + 0.5, dtype=torch.float32)

    return (
        vertices.unsqueeze(0).to(device),
        faces.unsqueeze(0).to(device),
        colors.unsqueeze(0).to(device),
    )


def patch_cloud(surface: vtk.vtkPolyData, array_name: str = PATCH_ARRAY_NAME) -> vtk.vtkPolyData:
    """The patch's points, as a vertex-only mesh for the ICP.

    `array_name` because there are two patches: the palatal one this module
    predicts, and the mucogingival band `mgl.py` builds for the lower arch. They
    carry different names on purpose -- a mandible is never labelled after the
    palate -- and the ICP is pointed at whichever one the mode painted.

    `vtkMeshTeeth` built this by inserting one point and one vertex cell at a
    time in Python, plus a `vtkStringArray` of stringified indices that nothing
    downstream ever read. Same geometry, built in one call.
    """
    labels = vtk_to_numpy(surface.GetPointData().GetArray(array_name))
    points = surfaces.points_of(surface)[labels == 1]
    if points.shape[0] == 0:
        raise surfaces.SurfaceError(
            "the patch prediction selected no point on this mesh, so there is nothing "
            "to register on"
        )

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))

    vertices = vtk.vtkCellArray()
    for index in range(points.shape[0]):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(index)

    cloud = vtk.vtkPolyData()
    cloud.SetPoints(vtk_points)
    cloud.SetVerts(vertices)
    return cloud
