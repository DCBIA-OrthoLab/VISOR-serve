"""Server configuration, loaded entirely from environment variables.

No secrets are hardcoded here: API_TOKEN and all other settings must be
provided via the environment (or a local .env file, see .env.example).
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Bearer token required on /run/{tool_name}. No default: must be set explicitly.
    API_TOKEN: str

    # "cuda" on the GPU server, "cpu" for local development without a GPU.
    # Individual tools read this to decide where to run.
    DEVICE: str = "cpu"

    # Upload size limit, in megabytes. Requests exceeding this are rejected
    # with 413 before the file is fully received.
    MAX_UPLOAD_MB: int = 500

    # Cap on the UNCOMPRESSED size of an archive the server extracts on behalf
    # of a "folder"-typed argument (see base.FILE_TYPES). Without it a small
    # upload passing MAX_UPLOAD_MB could still expand to hundreds of GB and
    # fill TEMP_DIR. Archives over this are rejected with a 400.
    MAX_EXTRACTED_MB: int = 2000

    # Directory used for temporary input/output files. Every file written
    # here is deleted once a request has been served.
    TEMP_DIR: str = "/tmp/inference_server"

    # Maximum number of tool executions allowed to run at the same time.
    # Tool runs happen in worker threads (see main.py) so the server stays
    # responsive while inference is in progress; this caps how many run
    # concurrently -- requests beyond the cap simply wait for a free slot
    # instead of piling unbounded work onto RAM/CPU/GPU.
    MAX_CONCURRENT_TOOLS: int = 4

    # How many AMASSS inferences may touch the GPU at the same time. One by
    # default: a 3d_fullres model plus its sliding-window buffers already fills
    # a typical card. Raise it only on hardware you have actually measured.
    #
    # Two things bound what a higher value buys. AMASSS predicts its structures
    # sequentially, so a single request never exceeds one GPU job -- this only
    # arbitrates between CONCURRENT requests, and there are never more than
    # MAX_CONCURRENT_TOOLS of those. And nnUNet's worker processes pass whole
    # volumes through shared memory, so raising this without raising
    # docker-compose's `shm_size` fails as a SIGBUS in a worker, not as a clean
    # out-of-memory error.
    #
    # Read once, when AMASSS's nnunet_runner module is imported (the semaphore
    # it sizes is a module global), so a change needs a server restart.
    AMASSS_MAX_GPU_JOBS: int = 1

    # Same idea for ALI, with its own counter because the two tools' GPU
    # footprints are unrelated -- and ALI's is tiny. Measured on the real
    # bundle (2026-07-31, RTX 6000 Ada): a CBCT run peaks at **256 MiB**, being
    # one small DenseNet plus a 64^3 crop, so the card is never the constraint;
    # at 1 job, two concurrent requests fully serialize and the second waits
    # ~6.5s per landmark for the first. The default therefore matches
    # MAX_CONCURRENT_TOOLS, which makes this effectively "no extra limit" while
    # still capping the damage if that is ever raised a lot.
    #
    # The figure is a property of the models, not of the card, so it holds on
    # any GPU able to run this at all (4 x 256 MiB = 1 GB). AMASSS keeps its
    # own, lower, limit: nothing here was measured about it.
    #
    # Read once, when the module owning the semaphore is imported, so a change
    # needs a server restart.
    ALI_MAX_GPU_JOBS: int = 4

    # Seconds a single CBCT landmark agent may walk before it is declared not
    # found. None means "derive from DEVICE" -- 15s on a GPU, 60s on CPU, since
    # every step is a forward pass and CPU inference needs several times longer
    # to converge. Set it explicitly only when neither default fits the
    # hardware. A per-landmark budget, so the worst case for a run is roughly
    # this times the number of landmarks times the number of scans.
    ALI_SEARCH_MAX_SECONDS: Optional[float] = None

    # How many CrownSeg segmentations may touch the GPU at the same time. One
    # mesh at 320x320 over an icosahedron's worth of viewpoints already fills a
    # typical card.
    CROWNSEG_MAX_GPU_JOBS: int = 1

    # DataLoader worker processes shapeaxi uses to load meshes. Must be >= 1:
    # shapeaxi builds its loader with persistent_workers=True, which PyTorch
    # rejects at 0.
    CROWNSEG_NUM_WORKERS: int = 2

    # Name of the crown-segmentation checkpoint in DATA_DIR/CrownSeg/models/,
    # used when a caller does not name one -- including ALI's IOS mode, which
    # segments an unlabelled mesh through CrownSeg without ever naming its
    # data. Fetched by scripts/setup-models.sh --tool CrownSeg.
    #
    # The alternative is what shapeaxi does when handed no model: download the
    # checkpoint from GitHub at inference time. A server holding confidential
    # data does not make outbound calls mid-request, so the file is staged
    # server-side instead and a missing one is an error, never a download.
    CROWNSEG_MODEL: str = "07-21-22_val-loss0.169.pth"

    # Where AMASSS resamples volumes. nnUNet's stock resamplers are scipy spline
    # interpolations that run single-threaded on the CPU, and on a CBCT they
    # cost far more than the network they feed. Measured on a 512x512x365 scan
    # at 0.33mm, for ONE structure: 14.6s to resample the input to the model's
    # 0.4mm grid, 4.5s of actual GPU inference, 6.9s to resample the logits back
    # -- the card sits idle for about seven eighths of the run.
    #
    # nnUNet ships torch equivalents of both (resample_torch_fornnunet). Running
    # them on the GPU takes those two steps to 0.14s and 0.40s; the full default
    # five-structure run on that scan goes from 195.9s to 77.0s.
    #
    # This is NOT free numerically. torch has no 3D cubic interpolation, so the
    # input resampling drops from spline order 3 to order 1, and the masks move
    # slightly. Measured against the scipy pipeline on that scan: MAND 0.998,
    # UAW 0.997, MAX 0.995, CB 0.991, CV 0.978 Dice. The cervical vertebra is
    # consistently the outlier -- it is the thinnest structure and the one
    # closest to the edge of the field of view, so a boundary shift costs it
    # proportionally more. Judge that against your own use before trusting it.
    #
    # Set to false to get bit-identical nnUNet behaviour back. Ignored when
    # DEVICE is cpu, and skipped for any model bundle whose plans ask for a
    # non-default resampler (see nnunet_runner._enable_gpu_resampling).
    AMASSS_GPU_RESAMPLING: bool = True

    # nnUNet's sliding-window overlap: the window advances by patch_size times
    # this, so 0.5 covers a voxel roughly eight times over and 0.7 roughly
    # three. Only the GPU share of a run scales with it (4.5s -> 2.0s per
    # structure on the scan above), which is why it buys much less than
    # AMASSS_GPU_RESAMPLING once that is on -- and unlike it, this DOES move the
    # segmentation: 0.7 measures Dice 0.995 against 0.5.
    #
    # Left at nnUNet's own default. Raise it only after looking at the masks.
    AMASSS_TILE_STEP_SIZE: float = 0.5

    # How many landmark triplets ASO's coarse alignment evaluates before the
    # ICP that follows it. Every ordered triplet is tried when there are at most
    # this many (7 landmarks is 210, 14 is 2184), which is both faster and
    # better than sampling; above it, candidates are drawn from a generator
    # seeded with ASO_ICP_SEED. Raising it costs time and buys very little --
    # the search is over a coarse alignment the ICP then refines.
    ASO_ICP_MAX_TRIPLETS: int = 2500

    # Seed for that search when it has to sample. Fixed rather than random on
    # purpose: an orientation applied to patient data must be reproducible, and
    # the original drew its triplets from the process-global numpy generator, so
    # the same request gave a different transform every time and two concurrent
    # requests consumed each other's random state.
    ASO_ICP_SEED: int = 0

    # The tool ASO asks for CBCT landmark predictions in Fully-Automated mode.
    # Deployed here, so that mode is live. A server that does not carry ALI
    # answers 422 with a message saying so (see tools/ASO/src/ali_client.py);
    # Semi-Automated CBCT and both IOS modes never go near it either way.
    ASO_LANDMARK_TOOL: str = "ALI"

    # DEFLATE level for the archives file_utils.make_zip builds (tool outputs,
    # zipped test folders). 1 by default: measured on the real testfiles
    # (2026-08-07, one core), level 6 compresses at ~30 MB/s against level 1's
    # ~61 MB/s, and buys about 3% of size on the one kind of member still worth
    # compressing (binary .vtk, ~2.7:1 at either level). Members that are
    # already compressed (.nii.gz, .zip, .xlsx, ...) never see this knob: they
    # are STORED as-is, because DEFLATE was measured gaining ~0% on them while
    # costing seconds of CPU per request on both ends of the transfer.
    ZIP_COMPRESSLEVEL: int = 1

    # Fallback whitelist, only used for arguments with a generic "file" type
    # (see base.FILE_TYPES). Arguments with a specific type (e.g. "zip_file")
    # are validated against that type's own extensions instead, regardless
    # of this list. "*" accepts everything for the generic fallback.
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".nii", ".nii.gz")

    # Root directory for read-only server-side data (AI models, test files),
    # organized as DATA_DIR/<tool_name>/{models,testfiles}/. Mounted
    # read-only in docker-compose.yml; tools never write here.
    DATA_DIR: str = "/data"

    # Backend serving models/test files declared via
    # ArgSpec(server_selectable=...). "local" reads DATA_DIR from disk --
    # see data_store.py to plug in another backend (e.g. a database or
    # object store) without changing main.py or any tool.
    DATA_BACKEND: str = "local"


settings = Settings()
