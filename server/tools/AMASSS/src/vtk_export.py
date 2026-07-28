"""Surface (.vtk) generation from a segmentation, for AMASSS's
"generate surface file" option.

This stays SERVER-side on purpose. AMASSS is not only called by the Slicer
module: the architecture is an API, and other modules (AREG today, others
later) call AMASSS programmatically. Generating surfaces in the Slicer client
would make them unavailable to every non-Slicer consumer, so the capability
belongs where the segmentation is produced.

The mesh pipeline itself is kept identical to the original CLI's
(SimpleITK -> temporary .nrrd -> vtkNrrdReader -> vtkDiscreteMarchingCubes ->
vtkSmoothPolyDataFilter -> per-cell colors), because AMASSS surfaces are
already consumed downstream and silently changing the geometry convention
would be a regression, not an improvement. What IS fixed here is the crash:

  the original resolved a structure's label by parsing it back out of the
  output FILE NAME (`base.split('_')[-1]`) and looking it up in
  `LABELS[model_size]` with model_size hardcoded to "LARGE"
  (AMASSS_CLI.py:260) -- a KeyError for any structure absent from that table,
  which aborted the whole scan. The structure code is now passed in
  explicitly by the caller, which knows it for certain.

vtk is imported lazily so the server still boots (and every other tool still
works) if it isn't installed.
"""

import logging
import os

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger("AMASSS.vtk")

_INSTALL_HINT = (
    "Surface generation needs VTK. Install it with "
    "`pip install -r requirements-amasss.txt`, or run AMASSS with "
    "generate_surface=false."
)


def _import_vtk():
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise RuntimeError(_INSTALL_HINT) from exc
    return vtk


def is_available() -> bool:
    try:
        import vtk  # noqa: F401
    except ImportError:
        return False
    return True


def _mesh_from_mask(mask: np.ndarray, reference: sitk.Image, temp_dir: str,
                    smoothing: int, color_rgb):
    """Build a colored surface for one binary mask."""
    vtk = _import_vtk()

    binary = sitk.GetImageFromArray(mask.astype(np.uint8))
    binary.CopyInformation(reference)
    temp_nrrd = os.path.join(temp_dir, "surface_input.nrrd")
    sitk.WriteImage(binary, temp_nrrd)

    reader = vtk.vtkNrrdReader()
    reader.SetFileName(temp_nrrd)
    reader.Update()

    marching_cubes = vtk.vtkDiscreteMarchingCubes()
    marching_cubes.SetInputConnection(reader.GetOutputPort())
    marching_cubes.GenerateValues(1, 1, 1)

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(marching_cubes.GetOutputPort())
    smoother.SetNumberOfIterations(max(0, int(smoothing)))
    smoother.Update()

    polydata = smoother.GetOutput()

    colors = vtk.vtkUnsignedCharArray()
    colors.SetName("Colors")
    colors.SetNumberOfComponents(3)
    colors.SetNumberOfTuples(polydata.GetNumberOfCells())
    for cell_index in range(polydata.GetNumberOfCells()):
        colors.SetTuple(cell_index, color_rgb)
    polydata.GetCellData().SetScalars(colors)
    return polydata


def _write(polydata, output_path: str) -> None:
    vtk = _import_vtk()
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()


def write_separate_surface(mask: np.ndarray, reference: sitk.Image, structure_code: str,
                           label_colors: dict, labels: dict, temp_dir: str,
                           smoothing: int, output_path: str) -> str:
    """One binary structure -> one .vtk.

    `structure_code` is passed in by the caller instead of being parsed back
    out of the file name -- this is the fix for the original KeyError.
    """
    label_index = labels.get(structure_code)
    color = label_colors.get(label_index, (255, 255, 255))
    polydata = _mesh_from_mask(mask, reference, temp_dir, smoothing, color)
    _write(polydata, output_path)
    logger.info("Wrote surface for %s", structure_code)
    return output_path


def write_merged_surface(merged: np.ndarray, reference: sitk.Image, names_from_labels: dict,
                         label_colors: dict, temp_dir: str, smoothing: int,
                         output_path: str) -> str:
    """A multi-label volume -> one .vtk holding every structure's surface."""
    vtk = _import_vtk()

    append = vtk.vtkAppendPolyData()
    surfaces = 0
    for label in sorted(int(value) for value in np.unique(merged)):
        if label == 0:
            continue
        structure_code = names_from_labels.get(label)
        if structure_code is None:
            # Unknown label: skip it rather than raise. The original indexed
            # NAMES_FROM_LABELS directly and died on anything unexpected.
            logger.warning("Skipping unknown label %s while building merged surface", label)
            continue
        color = label_colors.get(label, (255, 255, 255))
        append.AddInputData(
            _mesh_from_mask((merged == label), reference, temp_dir, smoothing, color)
        )
        surfaces += 1

    if surfaces == 0:
        logger.warning("No labels found in merged volume; no surface written")
        return ""

    append.Update()
    _write(append.GetOutput(), output_path)
    logger.info("Wrote merged surface with %d structure(s)", surfaces)
    return output_path
