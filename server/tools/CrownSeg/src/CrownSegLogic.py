"""Crown segmentation on intraoral surface scans, via shapeaxi.

Labels every tooth of a mesh with its Universal number, as a point-data array
on a copy of the surface. That array is the precondition for ALI's IOS landmark
identification and for the IOS modes of ASO, AREG and FlexReg, which is why
this is its own tool rather than a helper inside any of them.

Nothing here is ported. The Slicer modules ran the `dentalmodelseg` executable
out of Slicer's own bin directory, and that executable is only the
console-script entry point of the `shapeaxi` PyPI package
(`dentalmodelseg = shapeaxi.dental_model_seg:cml`). Server-side there is no
Slicer binary to shell out to, so `shapeaxi.dental_model_seg.main` is called
directly with the namespace its own `cml()` would have built.

Two entry points, following the AMASSSLogic precedent:

* `segment_crowns(...)` -> `CrownSegRun`, the reusable API: the produced meshes
  and a report, zipping nothing. ALI's IOS engine calls exactly this.
* `main(...)` -> path to the output directory, the schema adapter CrownSeg.py
  uses.
"""

import contextlib
import io
import json
import logging
import os
import sys
import threading
import time
from argparse import Namespace

import file_utils
from base import ToolArgumentError, ToolUnavailableError
from config import settings
from data_store import DataNotFoundError, data_store

logger = logging.getLogger("CrownSeg")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)


# Extensions shapeaxi's reader handles and this tool therefore discovers.
# Advertised in the schema through base.FILE_TYPES["surface_or_zip_file"], so
# the two lists cannot drift into "accepted by the UI, ignored by the engine".
SURFACE_EXTENSIONS = (".vtk", ".stl")

# Point-data array names that already carry per-tooth labels. A mesh holding
# any of them is segmented and is passed through untouched -- re-running the
# network on it would cost minutes and change nothing.
LABEL_ARRAY_NAMES = ("PredictedID", "UniversalID", "Universal_ID")

DEFAULT_ARRAY_NAME = "Universal_ID"
DEFAULT_SUFFIX = "Seg"

# One mesh at a time on the card: the model rasterizes an icosahedron's worth
# of 320x320 views of the whole arch in one batch.
_GPU_SEMAPHORE = threading.BoundedSemaphore(max(1, int(settings.CROWNSEG_MAX_GPU_JOBS)))

_INSTALL_HINT = (
    "CrownSeg needs shapeaxi and pytorch3d. pytorch3d has no PyPI distribution and "
    "must be compiled into the deployment image against its torch/CUDA build; every "
    "shapeaxi release also requires a newer torch than the current image ships. See "
    "ALI_PORT_CONTEXT.md 4."
)


class CrownSegRun:
    """Result of `segment_crowns()`: where the meshes are, and what happened."""

    def __init__(self, output_dir: str, report: dict, meshes: list):
        self.output_dir = output_dir
        self.report = report
        # Absolute paths of every mesh that now carries tooth labels, whether
        # this run produced them or found them already labelled.
        self.meshes = meshes


def _import_vtk():
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise RuntimeError(f"CrownSeg needs VTK: pip install -r requirements.txt") from exc
    return vtk


def _import_dental_model_seg():
    """Import shapeaxi's segmentation module, lazily.

    registry.py imports every tool at startup, so a module-level import here
    would take CrownSeg -- and, through it, nothing else, but still -- out of
    the registry on any server whose image predates the pytorch3d rebuild.
    Deferred, the tool loads, publishes its schema through GET /tools, and only
    an actual run fails, naming what is missing.
    """
    try:
        from shapeaxi import dental_model_seg
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: {exc.name or 'shapeaxi'})") from exc
    return dental_model_seg


def resolve_device(requested: str = None) -> str:
    """The device string shapeaxi accepts: exactly "cpu" or "cuda:0"."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: torch)") from exc

    wanted = (requested or settings.DEVICE or "cpu").strip().lower()
    if wanted.startswith("cuda") and torch.cuda.is_available():
        return "cuda:0"
    if wanted.startswith("cuda"):
        logger.warning("DEVICE=%s requested but CUDA is unavailable; falling back to CPU", wanted)
    return "cpu"


def default_model_path() -> str:
    """The server's configured crown-segmentation checkpoint.

    Exists so a calling tool never has to know where CrownSeg keeps its data:
    ALI asks for crown segmentation, not for a file under DATA/CrownSeg/.

    Handing shapeaxi no model at all is a third option and is deliberately not
    taken -- it downloads the checkpoint from GitHub on the spot, and a server
    holding confidential data does not make outbound calls mid-request.
    """
    try:
        return data_store.resolve_model("CrownSeg", settings.CROWNSEG_MODEL).path
    except DataNotFoundError as exc:
        raise FileNotFoundError(
            f"The crown-segmentation model '{settings.CROWNSEG_MODEL}' is not on this server. "
            f"Fetch it with `./scripts/setup-models.sh --tool CrownSeg`, or set CROWNSEG_MODEL "
            f"to one of the names listed by GET /tools/CrownSeg/data. ({exc})"
        ) from exc


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def is_segmented(mesh_path: str) -> bool:
    """True when the mesh already carries a per-point tooth-label array.

    Reads only the point-data array NAMES, never their values: this decides
    whether to spend minutes of GPU time, and nothing about the patient.
    """
    vtk = _import_vtk()

    extension = os.path.splitext(mesh_path)[1].lower()
    if extension == ".stl":
        # An STL has no point data at all by construction, so it can never be
        # segmented. Reading it anyway keeps one code path.
        reader = vtk.vtkSTLReader()
    elif extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        return False

    reader.SetFileName(mesh_path)
    reader.Update()
    point_data = reader.GetOutput().GetPointData()
    names = {point_data.GetArrayName(index) for index in range(point_data.GetNumberOfArrays())}
    return bool(names & set(LABEL_ARRAY_NAMES))


def discover_meshes(input_path: str, scratch_dir: str) -> list:
    """Resolve the `input` argument into a list of surface files.

    Accepts the three shapes one schema argument can carry: a single mesh, a
    zip archive of a folder of them, or a folder served from the data store.
    Folder scanning is recursive, and the tree is what the output mirrors.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path) and input_path.lower().endswith(".zip"):
        input_path = file_utils.extract_zip(
            input_path,
            os.path.join(scratch_dir, "input_extracted"),
            strip_single_root=True,
            max_total_bytes=settings.MAX_EXTRACTED_MB * 1024 * 1024,
        )

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(SURFACE_EXTENSIONS):
            raise ToolArgumentError(
                f"'{os.path.basename(input_path)}' is not a surface mesh. Expected one of: "
                f"{', '.join(SURFACE_EXTENSIONS)}."
            )
        return [input_path]

    meshes = [
        os.path.join(root, name)
        for root, _dirs, files in os.walk(input_path)
        for name in sorted(files)
        if name.lower().endswith(SURFACE_EXTENSIONS)
    ]
    if not meshes:
        raise ToolArgumentError(
            f"No surface mesh found in the input. Supported extensions: "
            f"{', '.join(SURFACE_EXTENSIONS)}."
        )
    return sorted(meshes)


def _input_root(input_path: str, meshes: list) -> str:
    """The directory output paths are made relative to, so a batch keeps its
    tree instead of being flattened into one folder where two patients named
    scan.vtk overwrite each other."""
    if os.path.isdir(input_path):
        return input_path
    return os.path.dirname(meshes[0])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _run_shapeaxi(csv_path: str, output_dir: str, model_path: str, input_root: str,
                  array_name: str, suffix: str, device: str, fdi: bool) -> None:
    """Call shapeaxi's own `main()` with the namespace its `cml()` would build.

    stdout is swallowed on purpose. shapeaxi prints "Saving results to
    <path>" for every mesh, and that path carries the patient's own file name
    -- which this server never writes to a log (see the note at the top of
    main.py). The exception it raises on a real failure is untouched.
    """
    dental_model_seg = _import_dental_model_seg()

    args = Namespace(
        surf=None,
        csv=csv_path,
        model=model_path,
        suffix=suffix,
        out=output_dir,
        num_workers=max(1, int(settings.CROWNSEG_NUM_WORKERS)),
        crown_segmentation=0,
        array_name=array_name,
        fdi=1 if fdi else 0,
        overwrite=0,
        device=device,
        vtk_folder=input_root,
    )

    with _GPU_SEMAPHORE:
        with contextlib.redirect_stdout(io.StringIO()):
            dental_model_seg.main(args)


def _predicted_path(output_dir: str, csv_stem: str, suffix: str, mesh: str,
                    input_root: str) -> str:
    """Where shapeaxi's csv branch writes the segmented copy of `mesh`.

    Mirrors `save_data_vtk_from_csv`: the input path minus its extension,
    plus `_<suffix>.vtk`, with `vtk_folder` stripped off the front, filed
    under `<out>/<csv stem>_<suffix>/`. Recomputed rather than parsed back out
    of shapeaxi's stdout, which is where the file names this server must not
    log would have had to travel.
    """
    without_extension = os.path.splitext(mesh)[0] + f"_{suffix}.vtk"
    relative = without_extension.replace(input_root, "").lstrip(os.sep)
    return os.path.normpath(os.path.join(output_dir, f"{csv_stem}_{suffix}", relative))


def segment_crowns(
    input_path: str,
    model_path: str = None,
    array_name: str = DEFAULT_ARRAY_NAME,
    suffix: str = DEFAULT_SUFFIX,
    fdi: bool = False,
    skip_segmented: bool = True,
    device: str = None,
    scratch_dir: str = None,
    output_dir: str = None,
) -> CrownSegRun:
    """Label the teeth of one mesh or a whole folder of them.

    `model_path=None` uses the server's configured checkpoint (see
    `default_model_path`), which is how a calling tool asks for crown
    segmentation without knowing where CrownSeg keeps its data.

    `skip_segmented=True` passes an already-labelled mesh through unchanged
    instead of re-running the network on it. That is what makes ALI's IOS mode
    a single button for both raw and pre-segmented input.
    """
    started_at = time.monotonic()

    array_name = (array_name or DEFAULT_ARRAY_NAME).strip() or DEFAULT_ARRAY_NAME
    suffix = (suffix or DEFAULT_SUFFIX).strip() or DEFAULT_SUFFIX

    if scratch_dir is None:
        scratch_dir = file_utils.make_scratch_dir("CrownSeg_")
    os.makedirs(scratch_dir, exist_ok=True)

    if output_dir is None:
        output_dir = os.path.join(scratch_dir, f"CrownSeg_{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    meshes = discover_meshes(input_path, scratch_dir)
    input_root = _input_root(input_path, meshes)

    already_segmented, to_segment = [], []
    for mesh in meshes:
        (already_segmented if skip_segmented and is_segmented(mesh) else to_segment).append(mesh)

    records = {}
    produced = []

    # A mesh that already carries labels is copied, not re-predicted. Copied
    # rather than referenced in place because the caller is handed one output
    # tree, and because the input may live in read-only DATA_DIR.
    for mesh in already_segmented:
        relative = os.path.relpath(mesh, input_root)
        base = os.path.splitext(os.path.basename(mesh))[0]
        destination = os.path.join(
            output_dir, os.path.dirname(relative), f"{base}_{suffix}.vtk"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _write_as_vtk(mesh, destination)
        records[relative] = {"status": "already_segmented", "output": destination}
        produced.append(destination)

    if to_segment:
        model_path = model_path or default_model_path()
        device = resolve_device(device)

        # Written into the scratch dir. The Slicer module wrote this csv into
        # the extension's own source folder, which breaks a read-only install
        # and leaves a file behind pointing at the patient's data.
        csv_path = os.path.join(scratch_dir, "crownseg_input.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("surf\n")
            for mesh in to_segment:
                handle.write(f"{mesh}\n")

        logger.info("CrownSeg: segmenting %d mesh(es) on %s", len(to_segment), device)
        _run_shapeaxi(
            csv_path=csv_path,
            output_dir=output_dir,
            model_path=model_path,
            input_root=input_root,
            array_name=array_name,
            suffix=suffix,
            device=device,
            fdi=fdi,
        )

        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        for mesh in to_segment:
            relative = os.path.relpath(mesh, input_root)
            predicted = _predicted_path(output_dir, csv_stem, suffix, mesh, input_root)
            if os.path.isfile(predicted):
                records[relative] = {"status": "segmented", "output": predicted}
                produced.append(predicted)
            else:
                # One mesh shapeaxi could not write must not cost the batch.
                records[relative] = {
                    "status": "failed",
                    "error": "the segmentation produced no output for this mesh",
                }

    if not produced:
        raise RuntimeError("CrownSeg produced no segmented mesh for any input.")

    report = {
        "tool": "CrownSeg",
        "array_name": array_name,
        "suffix": suffix,
        "numbering": "FDI" if fdi else "Universal",
        "device": device if to_segment else None,
        "meshes": records,
        "summary": {
            "total": len(meshes),
            "segmented": sum(1 for r in records.values() if r["status"] == "segmented"),
            "already_segmented": len(already_segmented),
            "failed": sum(1 for r in records.values() if r["status"] == "failed"),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }

    with open(os.path.join(output_dir, "run_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return CrownSegRun(output_dir=output_dir, report=report, meshes=produced)


def _write_as_vtk(source: str, destination: str) -> None:
    """Copy a mesh to `destination` as .vtk polydata.

    A pre-segmented input can be an .stl, and shapeaxi's outputs are always
    .vtk; normalizing here means the caller gets one format whichever branch a
    mesh took, instead of an .stl that every downstream reader then has to
    special-case. (The Slicer module copied it verbatim, and its landmark CLI
    then discovered only .vtk -- so those meshes were silently never used.)
    """
    vtk = _import_vtk()

    extension = os.path.splitext(source)[1].lower()
    reader = vtk.vtkSTLReader() if extension == ".stl" else vtk.vtkPolyDataReader()
    reader.SetFileName(source)
    reader.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(destination)
    writer.SetInputData(reader.GetOutput())
    writer.Write()


def main(
    input: str,
    model: str = None,
    array_name: str = DEFAULT_ARRAY_NAME,
    suffix: str = DEFAULT_SUFFIX,
    numbering: str = "Universal",
    skip_segmented: bool = True,
) -> str:
    """Schema adapter: run `segment_crowns` and return the folder of results.

    Returns a DIRECTORY: with `output_kind = "files"`, main.py bundles it and
    streams the archive, so no zip code lives in this tool.
    """
    scratch_dir = file_utils.make_scratch_dir("CrownSeg_")

    run = segment_crowns(
        input_path=input,
        model_path=model or None,
        array_name=array_name,
        suffix=suffix,
        fdi=numbering == "FDI",
        skip_segmented=skip_segmented,
        scratch_dir=scratch_dir,
    )
    return run.output_dir
