"""Mesh export for BatchDentalSeg: one segmentation -> .vtk / .stl / .obj.

Ports the local module's export half (`exportSegmentation`, `_exportVTKPerLabel`,
`_exportMergedVTK`), which was the one feature of it the server port left out.
The mesh pipeline is AMASSS's, deliberately: SimpleITK -> temporary .nrrd ->
vtkNrrdReader -> vtkDiscreteMarchingCubes -> vtkSmoothPolyDataFilter ->
vtkDecimatePro, with the decimation defaults it measured (90% costs a fifth of
a voxel on average and buys a factor of ten in size and parse time).

**A second copy of AMASSS's vtk_export.py rather than an import**, for the
reason nnunet_runner.py is: `registry.py` imports every tool at startup, so
importing another tool's module would make one tool's missing dependency take
both out of the registry. What is NOT copied is AMASSS's structure-code
indirection -- here a label is an integer with a name, straight from the
model's own table.

What differs from AMASSS, and why:

* **Three formats, not one.** The local module offered STL/OBJ/VTK and a merged
  VTK, and those are what a downstream tool or a printer actually consumes.
* **Only VTK carries the colours.** STL is geometry with no colour field at
  all, and OBJ needs a companion .mtl that nothing downstream here reads. The
  colour is still recorded in the run report for every format, so a viewer can
  apply it.
* **glTF is not offered.** Upstream produced it through the OpenAnatomy Slicer
  extension; VTK's own exporter needs a live render window, which a headless
  container has no GL context for. Exporting glTF from the segmentation once it
  is loaded in Slicer is the client's job, not a server argument that would
  fail at runtime on most deployments.

vtk is imported lazily so the server still boots without it.
"""

import logging
import os
import uuid

import numpy as np
import SimpleITK as sitk

from base import ToolUnavailableError

logger = logging.getLogger("BatchDentalSeg.surfaces")

_INSTALL_HINT = (
    "Mesh export needs VTK. Install it with `pip install -r requirements.txt`, "
    "or run BatchDentalSeg without any export format selected."
)

# The option names the schema offers, and what each one writes. "VTK (merged)"
# is handled separately: it is one file for the whole scan rather than one per
# structure.
PER_STRUCTURE_FORMATS = {
    "VTK": ".vtk",
    "STL": ".stl",
    "OBJ": ".obj",
}
MERGED_FORMAT = "VTK (merged)"

# Every option, in the order the panel shows them.
ALL_FORMATS = tuple(PER_STRUCTURE_FORMATS) + (MERGED_FORMAT,)


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
                    smoothing: int, color_rgb, decimation: int):
    """Build one coloured surface from one binary mask.

    `decimation` is the percentage of triangles to drop. Marching cubes runs on
    the ORIGINAL scan grid, so a 0.33 mm CBCT gives a triangle per voxel face --
    detail the mask does not carry, it being accurate to about half a voxel.
    """
    vtk = _import_vtk()
    from vtk.util.numpy_support import numpy_to_vtk

    binary = sitk.GetImageFromArray(mask.astype(np.uint8))
    binary.CopyInformation(reference)
    # Unique per call: a fixed name makes every surface of a run write over the
    # same path, which is silent corruption the first time two are built at
    # once.
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

        # PreserveTopologyOn keeps a thin structure -- a mandibular canal, a
        # root apex -- from being punctured; the reduction is a target, not a
        # guarantee.
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

    cell_colors = np.tile(
        np.asarray(color_rgb, dtype=np.uint8), (polydata.GetNumberOfCells(), 1)
    )
    colors = numpy_to_vtk(cell_colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetName("Colors")
    polydata.GetCellData().SetScalars(colors)
    return polydata


def _write(polydata, output_path: str) -> str:
    """Write one polydata and return the path, or "" if nothing landed.

    **The return value is checked against the filesystem, not against the
    writer.** VTK writers report their failures to the console and do not
    raise, and the three formats then disagree on what they leave behind:
    given empty geometry `vtkPolyDataWriter` writes a valid file with a header
    and no cells, while `vtkSTLWriter` and `vtkOBJWriter` write nothing at all.
    Trusting the call would therefore ship an empty .vtk as a success, and list
    .stl paths in the run report that do not exist on disk.
    """
    """Write one polydata, picking the writer from the extension.

    **Every writer here is put in binary mode explicitly.** `vtkPolyDataWriter`
    and `vtkSTLWriter` both default to ASCII, and that default is what made
    AMASSS's responses enormous: a merged surface came to 848.5 MB against 6.4
    MB for every segmentation in the same run. Binary is also the *more*
    accurate of the two -- it round-trips the float32 vertices exactly, while
    ASCII prints about six significant digits and moves points by up to
    5e-05 mm on read-back. OBJ has no binary form; it is offered because
    downstream tools ask for it, not because it is compact.
    """
    vtk = _import_vtk()
    if polydata.GetNumberOfCells() == 0:
        logger.warning("Nothing to write to %s: the mesh has no triangle", output_path)
        return ""
    extension = os.path.splitext(output_path)[1].lower()

    if extension == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileTypeToBinary()
    elif extension == ".obj":
        writer = vtk.vtkOBJWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()

    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        logger.warning("%s writer produced no file", extension or "mesh")
        return ""
    return output_path


def write_surfaces(labels: sitk.Image, model, formats, base: str, output_dir: str,
                   suffix: str, temp_dir: str, smoothing: int, decimation: int) -> list:
    """Every mesh one scan's label volume was asked for. Returns the paths.

    Only the labels PRESENT in this scan are meshed: a UniversalLab run would
    otherwise attempt 55 surfaces per patient, most of them empty, and an empty
    mesh is indistinguishable from a structure the model failed on.

    A structure that fails to mesh is logged and skipped rather than raised: the
    segmentation itself is already written and is the deliverable every
    downstream tool consumes, so one bad surface must not cost the patient.
    """
    wanted = [name for name in ALL_FORMATS if name in set(formats or ())]
    if not wanted:
        return []

    array = sitk.GetArrayViewFromImage(labels)
    present = sorted(int(value) for value in np.unique(array) if value != 0)
    if not present:
        logger.warning("No label found in the segmentation; no surface written")
        return []

    names_from_labels = model.label_names
    rgb_by_label = model.rgb_by_label
    per_structure = [name for name in wanted if name in PER_STRUCTURE_FORMATS]

    written = []
    merged_parts = []
    for value in present:
        name = names_from_labels.get(value)
        if name is None:
            # A label the model's table does not describe. Skipped rather than
            # raised, and named in the log: the alternative is a mesh filed
            # under a structure it is not.
            logger.warning("Skipping label %s: not in the %s table", value, model.name)
            continue
        try:
            polydata = _mesh_from_mask(
                (array == value), labels, temp_dir, smoothing,
                rgb_by_label.get(value, (255, 255, 255)), decimation,
            )
        except Exception as exc:  # noqa: BLE001 - one structure must not end the scan
            logger.exception("Could not mesh '%s': %s", name, exc)
            continue

        # A structure that meshed to nothing. It is not an error worth failing
        # the scan over, but it must not be reported as a file either: a label
        # too thin or too small for marching cubes at this resolution leaves an
        # EMPTY mesh, and an empty mesh delivered as a success is a structure
        # the clinician will believe was found and could not be seen.
        if polydata.GetNumberOfCells() == 0:
            logger.warning("'%s' produced an empty mesh; no file written", name)
            continue

        safe_name = name.replace(" ", "-").replace("/", "-")
        for format_name in per_structure:
            destination = os.path.join(
                output_dir, f"{base}_{suffix}_{safe_name}{PER_STRUCTURE_FORMATS[format_name]}"
            )
            try:
                path = _write(polydata, destination)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Could not write %s: %s", destination, exc)
                continue
            if path:
                written.append(path)

        if MERGED_FORMAT in wanted:
            merged_parts.append(polydata)

    if merged_parts:
        vtk = _import_vtk()
        append = vtk.vtkAppendPolyData()
        for polydata in merged_parts:
            append.AddInputData(polydata)
        append.Update()
        destination = os.path.join(output_dir, f"{base}_{suffix}_merged.vtk")
        try:
            path = _write(append.GetOutput(), destination)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not write the merged surface: %s", exc)
        else:
            if path:
                written.append(path)

    logger.info("Wrote %d surface file(s) for %s", len(written), base)
    return written
