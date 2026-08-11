"""The models BatchDentalSeg offers, and what each one labels.

Ported from `BATCHDENTALSEG/BATCHDENTALSEGLib/SegmentationWidget.py`
(MODEL_DESCRIPTIONS, `_modelBasePath` and `_get_active_label_map`), which kept
the same information in three places: a description dict, a chain of `elif
selectedModel == ...` branches building a download URL, and a second chain
returning the label map. One table per model here, so adding one is one entry.

The label VALUES are part of the trained weights, not a presentation choice:
they are the integers the network emits, and the order is the order it was
trained in. Renaming an entry is safe, renumbering one silently mislabels
anatomy.
"""

# Each model is one nnUNet v2 3d_fullres bundle. `folder` is its directory
# under DATA/BatchDentalSeg/models/, matching what scripts/data-manifest.yml
# downloads; `labels` is {segment name: the integer the network emits}.
#
# Upstream downloaded these from GitHub releases on first use. This server
# never makes an outbound call mid-request: the bundles are staged with
# `scripts/setup-models.sh --tool BatchDentalSeg`, and a missing one is an
# error naming that command.

# DentalSegmentator and PediatricDentalsegmentator share this table: the
# maxilla is not a class of its own, it is part of Upper Skull.
_FIVE_LABELS = {
    "Upper Skull": 1,
    "Mandible": 2,
    "Upper Teeth": 3,
    "Lower Teeth": 4,
    "Mandibular canal": 5,
}

# NasoMaxillaDentSeg splits the maxilla out, which shifts every later value.
_NASO_LABELS = {
    "Upper Skull": 1,
    "Mandible": 2,
    "Maxilla": 3,
    "Upper Teeth": 4,
    "Lower Teeth": 5,
    "Mandibular canal": 6,
}

# UniversalLab labels every tooth individually: 1-32 in Universal numbering
# (upper right to lower right), then 33-52 for the deciduous teeth, then the
# three structures.
_UNIVERSAL_LABELS = {
    "Upper-right third molar": 1,
    "Upper-right second molar": 2,
    "Upper-right first molar": 3,
    "Upper-right second premolar": 4,
    "Upper-right first premolar": 5,
    "Upper-right canine": 6,
    "Upper-right lateral incisor": 7,
    "Upper-right central incisor": 8,
    "Upper-left central incisor": 9,
    "Upper-left lateral incisor": 10,
    "Upper-left canine": 11,
    "Upper-left first premolar": 12,
    "Upper-left second premolar": 13,
    "Upper-left first molar": 14,
    "Upper-left second molar": 15,
    "Upper-left third molar": 16,
    "Lower-left third molar": 17,
    "Lower-left second molar": 18,
    "Lower-left first molar": 19,
    "Lower-left second premolar": 20,
    "Lower-left first premolar": 21,
    "Lower-left canine": 22,
    "Lower-left lateral incisor": 23,
    "Lower-left central incisor": 24,
    "Lower-right central incisor": 25,
    "Lower-right lateral incisor": 26,
    "Lower-right canine": 27,
    "Lower-right first premolar": 28,
    "Lower-right second premolar": 29,
    "Lower-right first molar": 30,
    "Lower-right second molar": 31,
    "Lower-right third molar": 32,
    "Upper-right second molar (baby)": 33,
    "Upper-right first molar (baby)": 34,
    "Upper-right canine (baby)": 35,
    "Upper-right lateral incisor (baby)": 36,
    "Upper-right central incisor (baby)": 37,
    "Upper-left central incisor (baby)": 38,
    "Upper-left lateral incisor (baby)": 39,
    "Upper-left canine (baby)": 40,
    "Upper-left first molar (baby)": 41,
    "Upper-left second molar (baby)": 42,
    "Lower-left second molar (baby)": 43,
    "Lower-left first molar (baby)": 44,
    "Lower-left canine (baby)": 45,
    "Lower-left lateral incisor (baby)": 46,
    "Lower-left central incisor (baby)": 47,
    "Lower-right central incisor (baby)": 48,
    "Lower-right lateral incisor (baby)": 49,
    "Lower-right canine (baby)": 50,
    "Lower-right first molar (baby)": 51,
    "Lower-right second molar (baby)": 52,
    "Mandible": 53,
    "Maxilla": 54,
    "Mandibular canal": 55,
}


class Model:
    """One trained bundle: what it is called, and what it labels."""

    def __init__(self, name: str, labels: dict, description: str):
        self.name = name
        self.labels = labels
        self.description = description

    @property
    def label_names(self) -> dict:
        """{integer the network emits: segment name}."""
        return {value: name for name, value in self.labels.items()}


# Keyed by the folder name the manifest downloads each bundle into, which is
# also the name the client picks from GET /tools/BatchDentalSeg/data. That is
# deliberate: the caller chooses ONE thing, the bundle, and its label table
# follows. A separate "which labels" argument would let a caller pair bundle X
# with the labels of Y, and the result would be a plausible volume with every
# structure named wrong.
MODELS = {
    model.name: model
    for model in (
        Model(
            "DentalSegmentator",
            _FIVE_LABELS,
            "Adult dental CT/CBCT. Upper Skull (maxilla included), Mandible, "
            "Mandibular canal, Upper Teeth, Lower Teeth",
        ),
        Model(
            "PediatricDentalSeg",
            _FIVE_LABELS,
            "Paediatric dental CT/CBCT, same five segments as DentalSegmentator",
        ),
        Model(
            "NasoMaxillaDentSeg",
            _NASO_LABELS,
            "Six segments: the maxilla is separated from the Upper Skull",
        ),
        Model(
            "UniversalLab",
            _UNIVERSAL_LABELS,
            "Every tooth labelled individually in Universal numbering, "
            "deciduous teeth included, plus Mandible, Maxilla and Mandibular canal",
        ),
    )
}

MODEL_NAMES = tuple(MODELS)


def get(name: str) -> Model:
    """The model a bundle name identifies. Raises KeyError for an unknown one;
    callers turn that into a 422 listing what is known."""
    return MODELS[name]


def describe_all() -> str:
    """One line per model, for the error that lists what a caller may pick."""
    return "; ".join(f"{model.name} ({model.description})" for model in MODELS.values())
