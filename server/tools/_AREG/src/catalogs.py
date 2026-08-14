"""What AREG's schema offers: the modes, the anatomical regions it can register
on, and the tokens that name a timepoint or a jaw in a file name.

One table per concept, published to the client through `GET /tools` and read by
the engines. Nothing here is written down twice: the presentation `groups` are
derived from the same dicts the pipelines look codes up in, so a region added
below appears in the panel with no client release.
"""

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

MODALITY_CBCT = "CBCT"
MODALITY_IOS = "IOS"

MODALITY_CHOICES = {MODALITY_CBCT: True, MODALITY_IOS: False}

AUTOMATION_SEMI = "Semi-Automated"
AUTOMATION_FULLY = "Fully-Automated"
AUTOMATION_ORIENTED = "Oriented + Fully-Automated"

# Fully-Automated is the default rather than the Slicer module's
# Or_Auto_CBCT: the oriented mode additionally needs an orientation reference
# bundle, and a default that cannot run without a file the user has not chosen
# yet reads as a broken tool rather than as a default.
AUTOMATION_CHOICES = {
    AUTOMATION_SEMI: False,
    AUTOMATION_FULLY: True,
    AUTOMATION_ORIENTED: False,
}

# Which automation levels each modality actually has. The schema cannot say
# "this option only exists for that modality" -- `visible_when` hides an
# argument, not one option of a choice -- so the pair is checked in
# AREGLogic.main and answered with a 422 naming what is available.
AUTOMATION_BY_MODALITY = {
    MODALITY_CBCT: (AUTOMATION_SEMI, AUTOMATION_FULLY, AUTOMATION_ORIENTED),
    MODALITY_IOS: (AUTOMATION_SEMI, AUTOMATION_FULLY),
}


# ---------------------------------------------------------------------------
# CBCT registration regions
# ---------------------------------------------------------------------------
# The anatomy the voxel-based registration is masked to. One run per selected
# region, each producing its own output tree -- registering on the cranial base
# and on the mandible are two different clinical questions, not two settings of
# one.

REGION_CODES = {
    "Cranial base": "CB",
    "Mandible": "MAND",
    "Maxilla": "MAX",
}

REGION_CHOICES = {name: name == "Cranial base" for name in REGION_CODES}

# The AMASSS structure whose mask each region registers against, used by the
# Fully-Automated modes to ask AMASSS for exactly what will be consumed.
# AMASSS speaks display names on its schema and codes in `segment()`; these are
# the codes (see AMASSSLogic.STRUCTURE_GROUPS["Masks"]).
REGION_MASK_STRUCTURES = {
    "CB": "CBMASK",
    "MAND": "MANDMASK",
    "MAX": "MAXMASK",
}

# Tokens that name a region inside a mask's file name, matched as WHOLE tokens
# of the stem rather than as substrings.
#
# The original matched substrings (`"cb" in basename.lower()`), which makes
# every file whose name contains "CBCT" a cranial-base mask -- and "max" also
# matches a patient named MAX_01, "md" matches almost anything. Token matching
# is what makes `P1_CBCT_seg.nii.gz` not a cranial-base mask.
REGION_TOKENS = {
    "CB": ("cb", "cbmask", "cranialbase", "cranial"),
    "MAND": ("mand", "md", "mandmask", "mandible"),
    "MAX": ("max", "mx", "maxmask", "maxilla"),
}

# A file naming itself a segmentation. Combined with the region tokens above:
# a mask has to say BOTH what it is and which structure it covers.
MASK_TOKENS = ("mask", "seg", "pred", "segmentation")


def region_code(name: str) -> str:
    return REGION_CODES[name]


def region_name(code: str) -> str:
    """The display name of a region code ('CB' -> 'Cranial base')."""
    for name, value in REGION_CODES.items():
        if value == code:
            return name
    return code


# ---------------------------------------------------------------------------
# Timepoints
# ---------------------------------------------------------------------------
# T1 and T2 arrive in two separate folders, so unlike ASO -- where both
# timepoints of a subject sit side by side and stripping "_T1" would merge
# them -- the token here IS the timepoint and stripping it is what pairs the
# two folders.

TIMEPOINT_TOKENS = ("t1", "t2", "t0")

# Suffixes a previous run of AREG, ASO, AMASSS or ALI leaves on a name.
# Longest first, so "_Scanreg" is not cut short by "_Scan".
PATIENT_SUFFIXES = (
    "_lm_Pred", "_Scanreg", "_MERGED", "_OutReg", "_SegOr",
    "_scan", "_Scan", "_Seg", "_seg", "_Or", "_OR", "_lm",
)


# ---------------------------------------------------------------------------
# Jaws (IOS)
# ---------------------------------------------------------------------------
# Same table and the same rule as ASO's: a mesh whose name does not say which
# jaw it is gets refused rather than defaulted. `AREG_IOS_utils.Sort` defaulted
# every non-Upper file to Lower, so a maxillary mesh named `patient1.vtk` was
# registered as a mandible and returned as a success.

JAW_UPPER = "Upper"
JAW_LOWER = "Lower"

JAW_TOKENS = {
    "u": JAW_UPPER, "up": JAW_UPPER, "upper": JAW_UPPER,
    "maxilla": JAW_UPPER, "max": JAW_UPPER, "mx": JAW_UPPER,
    "l": JAW_LOWER, "low": JAW_LOWER, "lower": JAW_LOWER,
    "mandible": JAW_LOWER, "mandibule": JAW_LOWER, "mand": JAW_LOWER, "md": JAW_LOWER,
}


# ---------------------------------------------------------------------------
# IOS registration patch
# ---------------------------------------------------------------------------
# Which stable region the two timepoints are aligned on. The two are different
# arches, not two settings of one thing: the palate exists only on the maxilla
# and the mucogingival line only matters on the mandible, so picking one also
# picks which arch is registered and which is carried along.

PATCH_PALATE = "Palate (upper arch)"
PATCH_MGL = "Mucogingival line (lower arch)"

PATCH_CHOICES = {PATCH_PALATE: True, PATCH_MGL: False}

# The jaw each patch registers.
PATCH_JAW = {PATCH_PALATE: JAW_UPPER, PATCH_MGL: JAW_LOWER}
