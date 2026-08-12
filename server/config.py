"""Server configuration, loaded entirely from environment variables.

No secrets are hardcoded: API_TOKEN and every other setting come from the
environment (or a local .env file, see .env.example).
"""

import os
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

# How a tool run is executed (SADT_DISPATCH_MODE).
DISPATCH_INPROCESS = "inprocess"
DISPATCH_SUBPROCESS = "subprocess"
DISPATCH_MODES = (DISPATCH_INPROCESS, DISPATCH_SUBPROCESS)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Bearer token required on /run/{tool_name}. No default: must be set.
    API_TOKEN: str

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    # "inprocess" imports every tool into the server and calls run() directly,
    # which is what the server has always done. "subprocess" runs it in the
    # tool's own virtualenv through runner.py (see dispatch.py), which is where
    # this is going: the server then never imports torch, is not pinned to the
    # lowest common Python across all tools, and releases a tool's VRAM when
    # its process exits.
    #
    # TEMPORARY. It exists so the subprocess path can be proven tool by tool
    # while the in-process one keeps serving, and is removed once every tool
    # has moved.
    SADT_DISPATCH_MODE: str = DISPATCH_INPROCESS

    # Root of the per-tool folders, each holding .venv/ and src/. The tools
    # currently live inside the server tree; the deployment image puts them at
    # /tools/<name>/ instead.
    TOOLS_DIR: str = os.path.join(_SERVER_DIR, "tools")

    # runner.py, injected into the tool's interpreter BY PATH rather than
    # installed into each venv, so the runner and the server are always the
    # same version.
    RUNNER_PATH: str = os.path.join(_SERVER_DIR, "runner.py")

    # Passed to the tool process as SADT_API. Points at this server from inside
    # the container, which is where a tool process runs; API_TOKEN is
    # deliberately NOT passed along with it.
    SADT_API: str = "http://127.0.0.1:8000"

    # Per-tool server-side configuration: which arguments may be filled from
    # DATA_DIR, and the upload limit for that tool (see deployment.py). A
    # missing file is the normal case, not an error -- it means every path
    # argument is upload-only and MAX_UPLOAD_MB applies everywhere.
    DEPLOYMENT_CONFIG: str = os.path.join(_SERVER_DIR, "deployment.toml")

    # Seconds a tool process may run before it is killed. 0 (the default) means
    # no limit, which is what the in-process path does today -- a full cohort
    # segmentation legitimately takes hours.
    TOOL_TIMEOUT_SECONDS: float = 0

    @field_validator("SADT_DISPATCH_MODE")
    @classmethod
    def _known_dispatch_mode(cls, value: str) -> str:
        # Checked at startup: a typo here would otherwise silently fall back to
        # whichever branch the comparison happened to take.
        if value not in DISPATCH_MODES:
            raise ValueError(f"SADT_DISPATCH_MODE must be one of {DISPATCH_MODES}, got {value!r}")
        return value

    # "cuda" on the GPU server, "cpu" for local development. Tools read this
    # to decide where to run.
    DEVICE: str = "cpu"

    # Upload size limit in MB; a larger request is rejected with 413.
    MAX_UPLOAD_MB: int = 500

    # Cap on the UNCOMPRESSED size of an archive extracted for a "folder"
    # argument. Without it a small upload could expand to hundreds of GB and
    # fill TEMP_DIR (zip bomb). Over this, the archive is rejected with 400.
    MAX_EXTRACTED_MB: int = 2000

    # Temporary input/output files. Everything written here is deleted once
    # the request has been served.
    TEMP_DIR: str = "/tmp/inference_server"

    # Default part size for a chunked upload (see transfer.py), in MB; a
    # client may ask for another value and gets it clamped to [1, 64]. At 8 MB
    # a 100 MB scan is 13 parts: enough to keep four connections busy, small
    # enough that losing one costs little.
    UPLOAD_CHUNK_MB: int = 8

    # How long an upload session or a stored result may sit under TEMP_DIR
    # WITHOUT BEING TOUCHED before the reaper deletes it. An idle timeout, not
    # an age limit: every part written and every range read stamps its
    # directory (transfer.touch), so a transfer still moving is never at risk.
    # This is a confidentiality bound -- it says for how long at most a result
    # the client never collected stays on disk.
    TRANSFER_TTL_SECONDS: int = 900

    # How often the reaper sweeps. It also runs when a session is created, but
    # that alone would mean an idle server never cleans up: the abandoned
    # transfer sits longest exactly when no new request comes in.
    TRANSFER_SWEEP_SECONDS: int = 60

    # Results at least this large are offered as a reference the client fetches
    # over parallel ranges (see transfer.py); smaller ones are streamed in the
    # /run response. The threshold is about cleanup, not speed: the streamed
    # path deletes the file as soon as the response ends, with no dependency on
    # the client coming back, so it is kept for the majority of runs.
    RESULT_REFERENCE_MIN_MB: int = 16

    # Maximum number of tool executions running at the same time. Tool runs
    # happen in worker threads (see main.py); requests beyond the cap wait for
    # a free slot instead of piling unbounded work onto RAM/CPU/GPU.
    MAX_CONCURRENT_TOOLS: int = 4

    # How many AMASSS inferences may touch the GPU at once. One by default: a
    # 3d_fullres model plus its sliding-window buffers already fills a typical
    # card. AMASSS predicts its structures sequentially, so this only arbitrates
    # between concurrent requests. Raising it also needs a larger `shm_size` in
    # docker-compose.yml, since nnUNet's workers pass volumes through shared
    # memory. Read once at import time: a change needs a server restart.
    AMASSS_MAX_GPU_JOBS: int = 1

    # Same for ALI, with its own counter because its GPU footprint is tiny: a
    # CBCT run peaks at 256 MiB (one small DenseNet over a 64^3 crop), so the
    # card is never the constraint. Matches MAX_CONCURRENT_TOOLS, i.e. no extra
    # limit in practice. Read once at import time.
    ALI_MAX_GPU_JOBS: int = 4

    # Seconds a single CBCT landmark agent may walk before it is declared not
    # found. None derives it from DEVICE -- 15s on GPU, 60s on CPU, every step
    # being a forward pass. A per-landmark budget.
    ALI_SEARCH_MAX_SECONDS: Optional[float] = None

    # How many BatchDentalSeg inferences may touch the GPU at once. Same shape
    # of work as AMASSS -- a 3d_fullres nnUNet over a CBCT -- so the same
    # default and its own counter. Read once at import time.
    BATCHDENTALSEG_MAX_GPU_JOBS: int = 1

    # nnUNet's sliding-window overlap for BatchDentalSeg, left at nnUNet's own
    # default. Raising it is faster and DOES move the segmentation.
    BATCHDENTALSEG_TILE_STEP_SIZE: float = 0.5

    # How many CrownSeg segmentations may touch the GPU at once. One mesh at
    # 320x320 over an icosahedron's viewpoints already fills a typical card.
    CROWNSEG_MAX_GPU_JOBS: int = 1

    # DataLoader workers shapeaxi uses to load meshes. Must be >= 1: shapeaxi
    # builds its loader with persistent_workers=True, which PyTorch rejects at 0.
    CROWNSEG_NUM_WORKERS: int = 2

    # Crown-segmentation checkpoint in DATA_DIR/CrownSeg/models/, used when the
    # caller names none -- including ALI's IOS mode. Fetched by
    # scripts/setup-models.sh --tool CrownSeg. Staged server-side rather than
    # downloaded at inference time: a server holding confidential data does not
    # make outbound calls mid-request, so a missing file is an error.
    CROWNSEG_MODEL: str = "07-21-22_val-loss0.169.pth"

    # Where AMASSS resamples volumes. nnUNet's stock resamplers are scipy spline
    # interpolations pinned to one CPU core and cost more than the network they
    # feed (on a 512x512x365 scan at 0.33mm, per structure: 14.6s in, 4.5s of
    # inference, 6.9s out). nnUNet ships torch equivalents; running them on the
    # GPU takes the five-structure run from 195.9s to 77.0s.
    #
    # NOT numerically free: torch has no 3D cubic interpolation, so the input
    # resampling drops from spline order 3 to order 1. Dice against the scipy
    # pipeline: MAND 0.998, UAW 0.997, MAX 0.995, CB 0.991, CV 0.978. Set to
    # false for bit-identical nnUNet output. Ignored on CPU, and skipped for a
    # bundle whose plans ask for a non-default resampler.
    AMASSS_GPU_RESAMPLING: bool = True

    # nnUNet's sliding-window overlap: the window advances by patch_size times
    # this. Only the GPU share of a run scales with it, and unlike
    # AMASSS_GPU_RESAMPLING it DOES move the segmentation (0.7 measures Dice
    # 0.995 against 0.5). Left at nnUNet's own default.
    AMASSS_TILE_STEP_SIZE: float = 0.5

    # How many landmark triplets ASO's coarse alignment evaluates before the ICP
    # that follows. Every ordered triplet is tried when there are at most this
    # many (7 landmarks is 210), which is faster and better than sampling; above
    # it, candidates come from a generator seeded with ASO_ICP_SEED.
    ASO_ICP_MAX_TRIPLETS: int = 2500

    # Seed for that search when it has to sample. Fixed on purpose: an
    # orientation applied to patient data must be reproducible, and the original
    # drew from the process-global numpy generator, so concurrent requests
    # consumed each other's random state.
    ASO_ICP_SEED: int = 0

    # The tool ASO asks for CBCT landmarks in Fully-Automated mode. A server
    # that does not carry it answers 422 (see tools/ASO/src/ali_client.py).
    ASO_LANDMARK_TOOL: str = "ALI"

    # DEFLATE level for the archives file_utils.make_zip builds. 1 by default:
    # level 6 compresses at ~30 MB/s against level 1's ~61 MB/s and buys about
    # 3% of size on the one member kind still worth compressing (binary .vtk).
    # Already-compressed members (.nii.gz, .zip, .xlsx, ...) never see this
    # knob -- they are STORED as-is.
    ZIP_COMPRESSLEVEL: int = 1

    # Fallback whitelist, used only for arguments with a generic "file" type
    # (see base.FILE_TYPES). An argument with a specific type is validated
    # against that type's own extensions instead. "*" accepts everything.
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".nii", ".nii.gz")

    # Root of the read-only server-side data, as DATA_DIR/<tool_name>/
    # {models,testfiles}/. Mounted read-only in docker-compose.yml.
    DATA_DIR: str = "/data"

    # Backend serving the files declared with ArgSpec(server_selectable=...).
    # "local" reads DATA_DIR from disk -- see data_store.py to plug in another
    # backend without changing main.py or any tool.
    DATA_BACKEND: str = "local"


settings = Settings()
