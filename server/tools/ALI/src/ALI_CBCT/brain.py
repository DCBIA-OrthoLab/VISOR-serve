"""The network one CBCT landmark agent consults, and its weights.

A `Brain` holds one network per scale: the agent asks the coarse one which way
to step at 1 mm, then the fine one at 0.3 mm. The architecture is a monai
DenseNet feature extractor followed by a small fully-connected head that scores
the six possible moves.

Ported from ALI_CBCT_utils/brain.py, inference only -- the optimizers, loss,
tensorboard writers and per-scale training bookkeeping the original carried
are gone, along with the model directories it created as a side effect of
being constructed.
"""

import logging
import sys

from base import ToolUnavailableError
from config import settings

logger = logging.getLogger("ALI.cbct.brain")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)

_INSTALL_HINT = (
    "ALI's CBCT engine needs torch and monai. Install them with "
    "`pip install -r requirements.txt` (see server/README.md)."
)

# Width of the feature vector the DenseNet hands to the decision head. Part of
# the trained weights' shape: changing it makes every shipped checkpoint fail
# to load.
TRANSITION_LAYER_SIZE = 1024


def import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: torch)") from exc
    return torch


def resolve_device(requested: str = None) -> str:
    """The device to actually use, falling back to CPU when CUDA is absent.

    Read from settings rather than decided by `torch.cuda.is_available()` deep
    in the code -- which the original did independently in five modules, so a
    server configured for CPU still used a card that happened to be present.
    Shared by both engines.
    """
    torch = import_torch()
    wanted = (requested or settings.DEVICE or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("DEVICE=%s requested but CUDA is unavailable; falling back to CPU", wanted)
        return "cpu"
    return wanted


def _import_monai_densenet():
    try:
        from monai.networks.nets import DenseNet
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: monai)") from exc
    return DenseNet


def build_network(in_channels: int = TRANSITION_LAYER_SIZE, out_channels: int = 6):
    """The DenseNet + decision-head pair, as one module.

    Defined inside a function rather than as module-level classes because
    `torch.nn.Module` cannot be subclassed without importing torch, and this
    module must import on a server that has none (see the lazy-import rule in
    ADDING_A_TOOL.md 7).
    """
    torch = import_torch()
    nn = torch.nn
    DenseNet = _import_monai_densenet()

    class DecisionHead(nn.Module):
        """Scores each possible move from the extracted features."""

        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.fc0 = nn.Linear(in_features, 512)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, out_features)
            for layer in (self.fc0, self.fc1, self.fc2, self.fc3):
                nn.init.xavier_uniform_(layer.weight)

        def forward(self, x):
            x = nn.functional.relu(self.fc0(x))
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            return nn.functional.relu(self.fc3(x))

    class DNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.featNet = DenseNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=in_features,
                growth_rate=34,
                block_config=(6, 12, 24, 16),
            )
            self.dens = DecisionHead(in_features, out_features)

        def forward(self, x):
            return self.dens(self.featNet(x))

    return DNet(in_channels, out_channels)


class Brain:
    """One network per scale, sharing a device and a set of weights."""

    def __init__(self, scale_keys, device: str, out_channels: int = 6):
        self.scale_keys = tuple(scale_keys)
        self.device = device
        self.networks = []
        for _scale in self.scale_keys:
            network = build_network(TRANSITION_LAYER_SIZE, out_channels)
            network.to(device)
            network.eval()
            self.networks.append(network)

    def load(self, weights_per_scale: dict) -> None:
        """Load one checkpoint per scale from {scale key: path}.

        `weights_only=True` is explicit: these are plain state dicts, they
        arrive from a data store rather than from this codebase, and torch
        changed the default mid-2.x -- so stating it both closes the pickle
        surface and keeps the call meaning the same thing across versions.
        """
        torch = import_torch()
        for index, scale in enumerate(self.scale_keys):
            state_dict = torch.load(
                weights_per_scale[scale], map_location=self.device, weights_only=True
            )
            self.networks[index].load_state_dict(state_dict)

    def predict(self, scale_index: int, state):
        """Index of the move the network favours from this field of view."""
        torch = import_torch()
        network = self.networks[scale_index]
        with torch.no_grad():
            batch = torch.unsqueeze(state, 0).type(torch.float32).to(self.device)
            scores = network(batch)
        return int(torch.argmax(scores))

    def release(self) -> None:
        """Drop the networks and free the card before the next landmark.

        Explicit because the alternative -- waiting for a garbage collection
        that Python makes no promise about -- is what turns a 112-landmark run
        into an out-of-memory failure part of the way through.
        """
        torch = import_torch()
        self.networks = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
