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
import uuid

import numpy as np
import SimpleITK as sitk

from base import ToolUnavailableError

logger = logging.getLogger("AMASSS.vtk")

_INSTALL_HINT = (
    "Surface generation needs VTK. Install it with "
    "`pip install -r requirements.txt`, or run AMASSS with "
    "generate_surface=false."
)


def _import_vtk():
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(_INSTALL_HINT) from exc
    return vtk


def is_available() -> bool:
    try:
        import vtk  # noqa: F401
    except ImportError:
        return False
    return True


def _mesh_from_mask(mask: np.ndarray, reference: sitk.Image, temp_dir: str,
                    smoothing: int, color_rgb, decimation: int = 0):
    """Build a colored surface for one binary mask.

    `decimation` is the percentage of triangles to drop (0 keeps the raw
    marching-cubes mesh). See the `surface_decimation` argument in AMASSS.py
    for why the default is not 0.
    """
    vtk = _import_vtk()
    from vtk.util.numpy_support import numpy_to_vtk

    binary = sitk.GetImageFromArray(mask.astype(np.uint8))
    binary.CopyInformation(reference)
    # Unique per call. The name used to be fixed, which made every surface in a
    # run write over the same file -- harmless only for as long as surfaces are
    # built one at a time, and a silent corruption the first time they are not.
    temp_nrrd = os.path.join(temp_dir, f"surface_input_{uuid.uuid4().hex}.nrrd")
    sitk.WriteImage(binary, temp_nrrd)

    try:
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

        # Marching cubes works on the ORIGINAL scan grid, so a CBCT at 0.33mm
        # yields a triangle per voxel face: 1.6M for a cranial base, 11.8M for
        # a merged nine-structure volume. That is a level of detail no CBCT
        # segmentation actually carries -- a binary mask is only accurate to
        # about half a voxel to begin with -- and it is what made the results
        # unusable downstream, both to ship and to open.
        #
        # Decimating 90% of a cranial base moves the surface by 0.059mm on
        # average (p95 0.171mm), a fifth of a voxel, well inside the mask's own
        # uncertainty. PreserveTopologyOn keeps thin structures from being
        # punctured; the reduction is a target, not a guarantee.
        reduction = min(max(int(decimation), 0), 99) / 100.0
        if reduction > 0 and polydata.GetNumberOfCells() > 0:
            decimator = vtk.vtkDecimatePro()
            decimator.SetInputData(polydata)
            decimator.SetTargetReduction(reduction)
            decimator.PreserveTopologyOn()
            decimator.SetFeatureAngle(60)
            decimator.Update()
            polydata = decimator.GetOutput()
    finally:
        # The mask can be a few hundred MB; a batch would otherwise keep one
        # copy per surface alive until the whole request is cleaned up.
        try:
            os.remove(temp_nrrd)
        except OSError:
            pass

    # One flat colour for every cell, built in numpy and handed over once
    # rather than a Python-level SetTuple per cell. Same bytes either way; the
    # loop was only ~80ms on a mandible's 590k cells, so this is tidiness and a
    # bounded cost on the bigger structures, not a headline saving.
    cell_colors = np.tile(
        np.asarray(color_rgb, dtype=np.uint8), (polydata.GetNumberOfCells(), 1)
    )
    colors = numpy_to_vtk(cell_colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetName("Colors")
    polydata.GetCellData().SetScalars(colors)
    return polydata


def _write(polydata, output_path: str) -> None:
    vtk = _import_vtk()
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    # `vtkPolyDataWriter` defaults to ASCII, which writes every coordinate as a
    # decimal string. Marching cubes over a CBCT at scan resolution returns
    # millions of triangles, and that default was what made AMASSS responses
    # enormous: the merged surface alone came to 848.5MB, against 6.4MB for
    # every segmentation in the same run. Binary took it to 296.7MB, and a
    # nine-structure run from 1386MB to 629MB.
    #
    # Binary is also the *more* accurate of the two, which is worth stating
    # because the reflex is to assume the opposite. It round-trips the float32
    # vertices exactly; ASCII prints them to about six significant digits, so
    # reading one back moved points by up to 5e-05mm. This writes what marching
    # cubes actually produced.
    writer.SetFileTypeToBinary()
    writer.Write()


def write_separate_surface(mask: np.ndarray, reference: sitk.Image, structure_code: str,
                           label_colors: dict, labels: dict, temp_dir: str,
                           smoothing: int, output_path: str, decimation: int = 0) -> str:
    """One binary structure -> one .vtk.

    `structure_code` is passed in by the caller instead of being parsed back
    out of the file name -- this is the fix for the original KeyError.
    """
    label_index = labels.get(structure_code)
    color = label_colors.get(label_index, (255, 255, 255))
    polydata = _mesh_from_mask(mask, reference, temp_dir, smoothing, color, decimation)
    _write(polydata, output_path)
    logger.info(
        "Wrote surface for %s (%d triangles)", structure_code, polydata.GetNumberOfCells()
    )
    return output_path


def write_merged_surface(merged: np.ndarray, reference: sitk.Image, names_from_labels: dict,
                         label_colors: dict, temp_dir: str, smoothing: int,
                         output_path: str, decimation: int = 0) -> str:
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
            _mesh_from_mask((merged == label), reference, temp_dir, smoothing, color, decimation)
        )
        surfaces += 1

    if surfaces == 0:
        logger.warning("No labels found in merged volume; no surface written")
        return ""

    append.Update()
    _write(append.GetOutput(), output_path)
    logger.info("Wrote merged surface with %d structure(s)", surfaces)
    return output_path
