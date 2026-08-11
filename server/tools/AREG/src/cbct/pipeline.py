"""The CBCT half of AREG: mask the T1 scan to one anatomical region, register
the T2 onto it, write the registered volume and the transform.

Ported from `AREG_CBCT/AREG_CBCT.py` (the driver) and the registration half of
`AREG_CBCT_utils/utils.py`. The Slicer envelope is gone: no `<filter-progress>`
prints, no `time.sleep(0.2)` progress theatre (0.6 s per patient), no
`sys.exit`, and nothing written into the caller's input tree.

Three behaviours are deliberately different from the original, all of them
visible in what the caller receives:

**The written `.tfm` is usable.** The original registered the T1 against a
*recentred copy* of the T2 and wrote the transform between those two spaces --
while the recentred T2 lived in a `<t2_folder>_Center` directory next to the
user's own data and was never returned. So the one file that says *how* the
scans were aligned referred to a volume the caller did not have. There is no
recentring here (see `elastix._RIGID_PARAMETERS`), so the transform maps the T1
frame to the T2 frame the caller sent, and `AREG_report.json` says so.

**The T2 is interpolated once instead of twice.** Recentring resampled every
moving volume with linear interpolation before the registration resampled it
again. elastix's `AutomaticTransformInitialization` aligns the two volumes'
centres by itself, so the first pass bought nothing and cost a blur.

**A registration that cannot be done is reported, not raised.** The original
caught every per-patient exception into a log line and finished by printing how
many had failed; the archive gave no clue. Each patient gets a report entry.
"""

import logging
import os

import SimpleITK as sitk

from .. import catalogs, pairing
from . import elastix

logger = logging.getLogger("AREG")


def register_patient(
    t1_path: str,
    t2_path: str,
    mask_path: str,
    region: str,
    output_dir: str,
    relative_key: str,
    suffix: str,
    segmentation_label: int = None,
) -> dict:
    """Register one T2 onto one T1 and write the results. Returns a report entry.

    Raises `elastix.RegistrationError` when this patient cannot be registered;
    the caller records that and moves on to the next.
    """
    fixed = sitk.ReadImage(t1_path)
    mask = sitk.ReadImage(mask_path)
    masked, note = elastix.apply_mask(fixed, mask, label=segmentation_label)

    moving = sitk.ReadImage(t2_path)
    transform = elastix.register(masked, moving)

    # Resampled on the MOVING image's own grid, as the original did: the result
    # keeps the T2's resolution and field of view, in the T1's coordinate frame.
    # (Resampling onto the T1 grid instead would crop the T2 to the T1's field
    # of view and re-sample it to the T1's spacing -- a different, and lossier,
    # answer to "where is the T2 now".)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    registered = sitk.Cast(resampler.Execute(moving), sitk.sitkInt16)

    relative_dir, patient = os.path.split(relative_key)
    destination = os.path.join(output_dir, region, relative_dir)
    os.makedirs(destination, exist_ok=True)

    _, extension = pairing.split_scan_extension(os.path.basename(t2_path))
    scan_output = os.path.join(
        destination, f"{patient}_{region}_{suffix}{pairing.compressed_extension(extension)}"
    )
    sitk.WriteImage(registered, scan_output, useCompression=True)

    transform_output = os.path.join(destination, f"{patient}_{region}_{suffix}_transform.tfm")
    sitk.WriteTransform(transform, transform_output)

    entry = {
        "status": "ok",
        "region": catalogs.region_name(region),
        "mask": os.path.basename(mask_path),
        # Stated rather than assumed: getting the direction backwards is silent
        # (the file still loads, and still transforms), and it is the only thing
        # a downstream tool needs to know to reuse it.
        "transform_maps": "T1 space -> T2 space (what sitk.ResampleImageFilter consumes)",
        "outputs": sorted(
            os.path.relpath(path, output_dir) for path in (scan_output, transform_output)
        ),
    }
    if note:
        entry["note"] = note
    return entry


def find_masks(mask_roots: list, region: str, scan_keys=()) -> dict:
    """{patient key: mask path} for one region, across several folders.

    Several roots because a mask can come from three places -- a folder the
    caller sent, the T1 folder itself (which is where the original looked when
    no mask folder was given), or AMASSS's output in the automated modes. The
    first root that has a patient's mask wins, so an explicit mask folder
    always beats one found next to the scans.

    `scan_keys` are the patients the scans were discovered under, and matching
    falls back to the LEAF of a key when the full relative path does not line
    up. It has to: a mask folder a caller sends is rarely laid out like their
    scan folder, and AMASSS's output is never laid out like it -- it writes one
    `<scan>_<id>_SegOut/` directory per scan, so a mask discovered under it
    keys to `P1_seg_SegOut/P1` while its scan keys to `P1`. The fallback is
    only taken when the leaf is unambiguous across the whole tree, so two
    subjects genuinely called `P1` in different folders never borrow each
    other's mask.
    """
    found: dict = {}
    for root in mask_roots:
        if not root or not os.path.isdir(root):
            continue
        for key, path in pairing.discover_masks(root, region).items():
            found.setdefault(key, path)

    missing = [key for key in scan_keys if key not in found]
    if not missing:
        return found

    by_leaf: dict = {}
    for key, path in found.items():
        by_leaf.setdefault(os.path.basename(key), []).append(path)
    for key in missing:
        candidates = by_leaf.get(os.path.basename(key), ())
        if len(candidates) == 1:
            found[key] = candidates[0]
    return found
