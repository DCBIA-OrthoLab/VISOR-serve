"""Every setting, from the environment or a local .env (see .env.example).

No tool reads os.getenv directly, even for a knob only it uses, so the whole
configuration stays in one file.
"""

import os
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

DISPATCH_INPROCESS = "inprocess"
DISPATCH_SUBPROCESS = "subprocess"
DISPATCH_MODES = (DISPATCH_INPROCESS, DISPATCH_SUBPROCESS)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- core ---------------------------------------------------------
    API_TOKEN: str  # required on /run; no default
    DEVICE: str = "cpu"  # "cuda" on the GPU server
    TEMP_DIR: str = "/tmp/inference_server"  # emptied once a request is served
    DATA_DIR: str = "/data"  # DATA_DIR/<tool>/{models,testfiles}, read-only
    DATA_BACKEND: str = "local"  # see data_store.py to plug in another

    # --- running a tool -----------------------------------------------
    # "subprocess" runs a tool in its own virtualenv through runner.py, so the
    # server never imports torch and releases the tool's VRAM when it exits.
    # "inprocess" is the old path. TEMPORARY, removed once every tool has moved.
    SADT_DISPATCH_MODE: str = DISPATCH_INPROCESS
    TOOLS_DIR: str = os.path.join(_SERVER_DIR, "tools")  # <tool>/{.venv,src}
    # Injected by path, never installed into a tool venv, so runner and server
    # are always the same version.
    RUNNER_PATH: str = os.path.join(_SERVER_DIR, "runner.py")
    # Turns a tool's run() signature into its schema, and must run with THAT
    # tool's interpreter. Absent, only tools shipping a .schema.json are served.
    DESCRIBE_PATH: str = os.path.join(_SERVER_DIR, "tools", "scripts", "describe.py")
    # A .schema.json is a cache, and the tool folders are read-only to the
    # process serving them, so a regenerated one cannot live beside its tool.
    SCHEMA_CACHE_DIR: str = os.path.join(_SERVER_DIR, ".schema-cache")
    DEPLOYMENT_CONFIG: str = os.path.join(_SERVER_DIR, "deployment.toml")
    SADT_API: str = "http://127.0.0.1:8000"  # reaches this server from a tool
    MAX_CONCURRENT_TOOLS: int = 4
    # One counter ACROSS tools: an AMASSS run and a CrownSeg run want the same
    # card. Every run counts as GPU work unless it declares `device` and
    # resolves it to a CPU value.
    MAX_CONCURRENT_GPU_JOBS: int = 1
    TOOL_TIMEOUT_SECONDS: float = 0  # 0 = none; a cohort legitimately takes hours

    # --- uploads and results ------------------------------------------
    MAX_UPLOAD_MB: int = 500  # over this, 413
    MAX_EXTRACTED_MB: int = 2000  # zip-bomb cap on an extracted archive, 400
    UPLOAD_CHUNK_MB: int = 8  # default part size, clamped to [1, 64]
    # Idle timeout, not an age limit: every part written and every range read
    # stamps its directory. Bounds how long an uncollected result stays on disk.
    TRANSFER_TTL_SECONDS: int = 900
    TRANSFER_SWEEP_SECONDS: int = 60
    # Under this, the result is streamed in the /run response, which deletes it
    # when the response ends whatever the client does.
    RESULT_REFERENCE_MIN_MB: int = 16
    # Level 6 compresses at ~30 MB/s against level 1's ~61 MB/s and buys ~3% on
    # the one member kind worth compressing. Already-compressed members are STORED.
    ZIP_COMPRESSLEVEL: int = 1
    # Fallback for a generic "file" argument only; a specific type carries its
    # own extensions. "*" accepts everything.
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".nii", ".nii.gz")

    # --- the in-process tools -----------------------------------------
    # Read by server/tools/ only. A packaged tool takes these as arguments.
    AMASSS_MAX_GPU_JOBS: int = 1  # raising it needs a larger shm_size
    # 195.9s -> 77.0s on five structures. Not numerically free: input resampling
    # drops to spline order 1 (Dice 0.978 CV to 0.998 MAND). false is bit-identical.
    AMASSS_GPU_RESAMPLING: bool = True
    AMASSS_TILE_STEP_SIZE: float = 0.5  # DOES move the segmentation (0.7 -> Dice 0.995)
    ALI_MAX_GPU_JOBS: int = 4  # a CBCT run peaks at 256 MiB
    ALI_SEARCH_MAX_SECONDS: Optional[float] = None  # None: 15s on GPU, 60s on CPU
    BATCHDENTALSEG_MAX_GPU_JOBS: int = 1
    BATCHDENTALSEG_TILE_STEP_SIZE: float = 0.5
    CROWNSEG_MAX_GPU_JOBS: int = 1
    CROWNSEG_NUM_WORKERS: int = 2  # >= 1: shapeaxi sets persistent_workers=True
    CROWNSEG_MODEL: str = "07-21-22_val-loss0.169.pth"  # staged, never downloaded mid-request
    # Every ordered triplet is tried below this (7 landmarks is 210), which
    # beats sampling; above it, a generator seeded with ASO_ICP_SEED. Fixed seed
    # on purpose: an orientation applied to patient data must be reproducible.
    ASO_ICP_MAX_TRIPLETS: int = 2500
    ASO_ICP_SEED: int = 0
    ASO_LANDMARK_TOOL: str = "ALI"

    @field_validator("SADT_DISPATCH_MODE")
    @classmethod
    def _known_dispatch_mode(cls, value: str) -> str:
        if value not in DISPATCH_MODES:
            raise ValueError(f"SADT_DISPATCH_MODE must be one of {DISPATCH_MODES}, got {value!r}")
        return value


settings = Settings()
