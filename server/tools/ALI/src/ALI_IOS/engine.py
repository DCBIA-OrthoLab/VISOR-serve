"""The IOS landmark pipeline: render each tooth, predict masks, project back.

Ported from ALI_IOS/ALI_IOS.py. Per tooth, the mesh is rendered from a dozen
viewpoints, a 2D UNet predicts one mask channel per landmark type, and each
mask is projected back onto the mesh faces that produced its pixels; the
landmark is the surface point nearest the centroid of those faces.

Restructured around what is loaded when. The original built a renderer,
instantiated a UNet and called `load_state_dict` **inside the per-tooth
loop** -- 28 model loads per scan per network, plus re-reading and re-scaling
the mesh every time. Here the mesh is read once per scan, the renderer once
per run, and the weights once per (network, jaw). Nothing about the inference
itself changes.

Fixes carried over from the analysis, all of which cost results silently:

* a jaw whose weights are missing from the bundle raised a `KeyError` that was
  caught and discarded, so the jaw simply vanished from the output. It is
  reported now.
* a mesh with no tooth labels fell back to all-zeros, so no tooth was ever
  found and the run ended with no landmarks and no reason. ALILogic segments
  such a mesh through CrownSeg before it gets here.
* `.stl` input was accepted by the UI, copied along, and then never discovered
  by the CLI, which globbed for `.vtk` only.
"""

import logging
import os
import sys
import threading
import time

from base import ToolUnavailableError
from config import settings

from ..markups import MARKUPS_EXTENSION
from ..markups import write as write_markups
from ..ALI_CBCT.brain import import_torch
from . import landmarks as catalog
from . import render, surface

logger = logging.getLogger("ALI.ios")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)

_GPU_SEMAPHORE = threading.BoundedSemaphore(max(1, int(settings.ALI_MAX_GPU_JOBS)))

_MONAI_HINT = (
    "ALI's IOS engine needs monai: pip install -r requirements.txt (see server/README.md)."
)


def _import_unet():
    try:
        from monai.networks.nets import UNet
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_MONAI_HINT} (missing: monai)") from exc
    return UNet


def resolve_device(requested: str = None) -> str:
    torch = import_torch()
    wanted = (requested or settings.DEVICE or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("DEVICE=%s requested but CUDA is unavailable; falling back to CPU", wanted)
        return "cpu"
    return wanted


def check_dependencies() -> None:
    """Import the whole lazy stack once, before any mesh is touched.

    Same reason as the CBCT engine's: a missing dependency belongs to the
    server, not to a patient's mesh, and the per-mesh `except` below would
    otherwise report it once per mesh and then hide it behind "produced no
    landmarks for any mesh". It matters more here -- pytorch3d is absent from
    the current deployment image, so this is the failure an IOS run actually
    hits today, and it has to say so in one clear line.
    """
    import_torch()
    surface.import_vtk()
    render.import_pytorch3d()
    _import_unet()


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def discover_weights(model_path: str):
    """({network: {jaw: path}}, unrecognized file names) for an IOS bundle.

    **One naming rule, stated once.** A checkpoint's base name is split on
    `_`; it must contain a token naming the network (`O` or `C`) and a token
    naming the jaw (`Upper` or `Lower`). The published weights are called
    `Upper_O_model.pth`, `Lower_C_model.pth` and so on, so both rules the
    original used agreed on them -- but they were different rules (the UI
    required the substring `_O_`, the CLI took `basename.split("_")[1]`), and
    a bundle named slightly differently was accepted by one and rejected by
    the other.

    The jaw must be named explicitly. The original treated every file not
    containing "Lower" as upper-jaw weights, so a bundle missing its mandibular
    model quietly predicted the lower arch with the maxillary one.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"IOS model bundle is not a directory: {model_path}")

    weights: dict = {}
    unrecognized = []

    for root, _dirs, files in os.walk(model_path):
        for name in sorted(files):
            if not name.endswith(".pth"):
                continue
            tokens = {token.lower() for token in os.path.splitext(name)[0].split("_")}
            network = next((code for code in catalog.NETWORK_CODES if code.lower() in tokens), None)
            jaw = next((candidate for candidate in catalog.JAWS if candidate.lower() in tokens), None)
            if network is None or jaw is None:
                unrecognized.append(name)
                continue
            weights.setdefault(network, {})[jaw] = os.path.join(root, name)

    return weights, sorted(unrecognized)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_network(checkpoint: str, device: str):
    """The 2D UNet the IOS weights were trained as.

    Every argument here is part of the checkpoint's shape; changing one makes
    `load_state_dict` fail rather than degrade.
    """
    torch = import_torch()
    UNet = _import_unet()

    network = UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=4,
    ).to(device)
    network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    network.eval()
    return network


def _predicted_faces(predictions, pix_to_face, channel: int) -> list:
    """Face indices whose rendered pixels were predicted as `channel`.

    `predictions` is the UNet output, (views, channels, H, W); `pix_to_face`
    says which mesh face each rendered pixel came from, shaped
    (views, 1, H, W, faces_per_pixel=1). A pixel where nothing was rendered
    carries -1, and those are dropped -- the original passed them on, where
    `faces[-1]` silently selected the mesh's last face and dragged the
    landmark's centroid to an arbitrary corner of the arch.

    The class is taken by argmax over the raw logits. The original cast them
    to int16 first, truncating every value toward zero, which turned near ties
    into exact ties that argmax then resolved in favour of the background
    channel -- shrinking every mask for no reason anyone intended.
    """
    torch = import_torch()

    classes = torch.argmax(predictions, dim=1)
    faces = []
    for view_index, row, column in (classes == channel).nonzero(as_tuple=False):
        face = int(pix_to_face[view_index, 0, row, column, 0].item())
        if face >= 0:
            faces.append(face)
    return faces


def _landmark_position(faces, face_table, vertices, locator, scaled_surface):
    """The surface point nearest the centroid of a set of faces.

    Snapped onto an actual mesh point rather than left as the raw centroid,
    which would float inside the tooth: a landmark is a point *on* the crown.
    """
    vertex_ids = [
        int(face_table[0][face][corner].item()) for face in faces for corner in range(3)
    ]
    if not vertex_ids:
        return None

    centroid = sum(vertices[0][vertex_id] for vertex_id in vertex_ids) / len(vertex_ids)
    point_id = locator.FindClosestPoint(centroid.detach().cpu().numpy())
    return scaled_surface.GetPoint(point_id)


def _predict_one_scan(mesh_path, key, record, weights, networks, device, renderer,
                      output_dir, prediction_ID,
                      mesh_index: int = 1, mesh_total: int = 1) -> None:
    """Every requested network, on every tooth this mesh actually carries.

    Logs one line per (network, jaw) pass. Teeth are rendered and predicted one
    at a time, so a full arch is dozens of GPU passes -- without a line in
    between, a run looks like a hang. Counts and anatomical labels only, never
    the mesh's file name.
    """
    torch = import_torch()
    p3d = render.import_pytorch3d()
    vtk, _ = surface.import_vtk()

    raw = surface.read_surface(mesh_path)
    scaled, mesh_center, scale_factor = surface.scale_to_unit(raw)
    vertices, faces, colors, labels = surface.surface_properties(scaled, device)

    mesh = p3d["Meshes"](
        verts=vertices, faces=faces, textures=p3d["TexturesVertex"](verts_features=colors)
    ).to(device)

    locator = vtk.vtkOctreePointLocator()
    locator.SetDataSet(scaled)
    locator.BuildLocator()

    present = {int(value) for value in labels.squeeze(0).unique().tolist()}
    teeth_by_jaw = {jaw: [] for jaw in catalog.JAWS}
    for jaw, numbers in catalog.UNIVERSAL_NUMBERS.items():
        teeth_by_jaw[jaw] = sorted(number for number in numbers.values() if number in present)

    if not any(teeth_by_jaw.values()):
        raise RuntimeError(
            "no known tooth number is present in this mesh's label array "
            "(expected Universal numbering, 2-15 upper and 18-31 lower)"
        )

    positions = {}
    for network in networks:
        for jaw, teeth in teeth_by_jaw.items():
            if not teeth:
                continue
            checkpoint = weights.get(network, {}).get(jaw)
            if checkpoint is None:
                # Reported, not swallowed: the original's KeyError here was
                # caught and discarded, so the jaw silently disappeared.
                record["jaws_without_model"].setdefault(
                    catalog.NETWORK_DISPLAY_NAMES.get(network, network), []
                ).append(jaw)
                continue

            # Loaded once per (network, jaw), not once per tooth.
            pass_started = time.monotonic()
            pass_name = f"{catalog.NETWORK_DISPLAY_NAMES.get(network, network)}/{jaw}"
            before = len(record["landmarks_found"])
            logger.info(
                "mesh %d/%d: %s -- %d tooth/teeth",
                mesh_index, mesh_total, pass_name, len(teeth),
            )

            unet = _build_network(checkpoint, device)
            try:
                for tooth_number in teeth:
                    label_names = catalog.LABELS[network].get(str(tooth_number))
                    if label_names is None:
                        continue
                    try:
                        found = _predict_one_tooth(
                            unet=unet,
                            renderer=renderer,
                            mesh=mesh,
                            network=network,
                            jaw=jaw,
                            tooth_number=tooth_number,
                            label_names=label_names,
                            vertices=vertices,
                            faces=faces,
                            labels=labels,
                            locator=locator,
                            scaled=scaled,
                            mesh_center=mesh_center,
                            scale_factor=scale_factor,
                            device=device,
                        )
                    except Exception as exc:
                        # One tooth failing must not cost the other 27.
                        logger.exception("IOS prediction raised for tooth %d", tooth_number)
                        record["landmarks_failed"][f"{jaw}-{tooth_number}"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue

                    positions.update(found)
                    record["landmarks_found"].extend(sorted(found))
            finally:
                del unet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(
                    "mesh %d/%d: %s -- %d landmark(s) in %.0fs",
                    mesh_index, mesh_total, pass_name,
                    len(record["landmarks_found"]) - before,
                    time.monotonic() - pass_started,
                )

    if not positions:
        raise RuntimeError("no landmark was predicted on this mesh")

    record["landmarks_found"].sort()
    destination = os.path.join(
        output_dir,
        os.path.dirname(key),
        f"{os.path.splitext(os.path.basename(mesh_path))[0]}_lm_{prediction_ID}"
        f"{MARKUPS_EXTENSION}",
    )
    record["files"].append(write_markups(positions, destination))


def _predict_one_tooth(unet, renderer, mesh, network, jaw, tooth_number, label_names,
                       vertices, faces, labels, locator, scaled, mesh_center, scale_factor,
                       device) -> dict:
    """Render one tooth, predict its landmarks, return {label: position}."""
    torch = import_torch()

    center = render.tooth_center(labels, vertices, tooth_number, device)
    if center is None:
        return {}

    with _GPU_SEMAPHORE:
        images, pix_to_face = render.render_views(
            renderer=renderer,
            mesh=mesh,
            center=center,
            radius=catalog.CAMERA_RADIUS[network],
            camera_positions=render.CAMERA_POSITIONS[network][jaw],
            device=device,
        )
        # (1, views, 4, H, W) -> (views, 4, H, W): one image per camera, which
        # is the batch the UNet consumes.
        with torch.no_grad():
            predictions = unet(images[0].float().to(device))

    found = {}
    # Channel 0 of the network output is background; the landmark types start
    # at 1, in the order catalog.NETWORKS declares them.
    for channel, label_name in enumerate(label_names, start=1):
        predicted = _predicted_faces(predictions, pix_to_face, channel)
        kept = surface.faces_on_tooth(faces, predicted, labels, tooth_number)
        position = _landmark_position(kept, faces, vertices, locator, scaled)
        if position is not None:
            found[label_name] = surface.upscale(position, mesh_center, scale_factor)
    return found


def predict_landmarks(
    meshes: list,
    model_path: str,
    networks=None,
    prediction_ID: str = "Pred",
    output_dir: str = None,
    scratch_dir: str = None,
    device: str = None,
) -> dict:
    """Place landmarks on every mesh; return the run report.

    `meshes` is a list of `(absolute path, key)` pairs, the key being the
    path relative to the input root -- so a batch keeps its tree and two
    patients named `scan.vtk` in different folders cannot overwrite each other.
    Every mesh is expected to carry tooth labels already; ALILogic runs
    CrownSeg on the ones that do not.
    """
    started_at = time.monotonic()

    check_dependencies()
    device = resolve_device(device)
    networks = tuple(networks) if networks is not None else catalog.NETWORK_CODES
    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    weights, unrecognized = discover_weights(model_path)
    available = [network for network in networks if network in weights]
    if not available:
        network_names = ", ".join(
            catalog.NETWORK_DISPLAY_NAMES.get(code, code) for code in networks
        )
        raise FileNotFoundError(
            f"'{os.path.basename(model_path)}' has no IOS weights for the selected network(s) "
            f"({network_names}). Checkpoints must be named with an 'O' or 'C' token and an "
            f"'Upper' or 'Lower' token, e.g. Upper_O_model.pth."
            + (f" Unrecognized: {', '.join(unrecognized)}." if unrecognized else "")
        )

    renderer = render.build_renderer(device)
    logger.info(
        "ALI IOS: %d mesh(es), %d network(s), device=%s", len(meshes), len(available), device
    )

    scan_reports = {}
    for mesh_index, (mesh_path, key) in enumerate(meshes, start=1):
        record = {
            "input": os.path.basename(mesh_path),
            "status": "pending",
            "landmarks_found": [],
            "landmarks_failed": {},
            "jaws_without_model": {},
            "files": [],
        }
        scan_reports[key] = record
        scan_started = time.monotonic()
        # Position in the batch, never the mesh's name -- see the CBCT engine.
        logger.info("mesh %d/%d: reading and scaling", mesh_index, len(meshes))

        try:
            _predict_one_scan(
                mesh_path=mesh_path,
                key=key,
                record=record,
                weights=weights,
                networks=available,
                device=device,
                renderer=renderer,
                output_dir=output_dir,
                prediction_ID=prediction_ID,
                mesh_index=mesh_index,
                mesh_total=len(meshes),
            )
            record["status"] = "ok"
        except Exception as exc:
            logger.exception("ALI IOS failed on one mesh")
            record["status"] = "failed"
            record["error"] = str(exc)
        record["duration_seconds"] = round(time.monotonic() - scan_started, 2)
        logger.info(
            "mesh %d/%d: %s -- %d landmark(s), %.0fs",
            mesh_index, len(meshes), record["status"],
            len(record["landmarks_found"]), record["duration_seconds"],
        )

    written = sum(len(record["landmarks_found"]) for record in scan_reports.values())
    logger.info(
        "ALI IOS done: %d/%d mesh(es), %d landmark(s) written, %.0fs",
        sum(1 for r in scan_reports.values() if r["status"] == "ok"),
        len(scan_reports), written, time.monotonic() - started_at,
    )

    processed = [record for record in scan_reports.values() if record["status"] == "ok"]
    if not processed:
        first_error = next(
            (record.get("error") for record in scan_reports.values() if record.get("error")),
            "unknown",
        )
        raise RuntimeError(f"ALI produced no landmarks for any mesh. First error: {first_error}")

    return {
        "mode": "IOS",
        "device": device,
        "prediction_ID": prediction_ID,
        "networks": [catalog.NETWORK_DISPLAY_NAMES.get(code, code) for code in available],
        # Same meaning as the CBCT engine's list, and read by the same client
        # code: a network requested but absent from the bundle.
        "landmarks_without_model": [
            catalog.NETWORK_DISPLAY_NAMES.get(code, code)
            for code in networks
            if code not in weights
        ],
        "models_unrecognized": unrecognized,
        "scans": scan_reports,
        "summary": {
            "total": len(scan_reports),
            "processed": len(processed),
            "failed": len(scan_reports) - len(processed),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }
