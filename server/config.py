"""Every setting, from the environment or a local .env (see .env.example).

No tool reads os.getenv directly, even for a knob only it uses, so the whole
configuration stays in one file.
"""

import os

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

# The catalogue that ships with the server. Always scanned, so the two kinds
# of tool -- one the server imports, one it never does -- can sit side by side.
BUILTIN_TOOLS_DIR = os.path.join(_SERVER_DIR, "tools")

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
    # One catalogue, or several separated by os.pathsep. A catalogue is just a
    # directory of tool folders, so serving a second one is a path, not a code
    # change -- which is the point of a server that holds no tool knowledge.
    # BUILTIN_TOOLS_DIR is always scanned, last, so a packaged tool dropped
    # beside the imported ones is found even when TOOLS_DIR names somewhere
    # else, as the deployment image does.
    TOOLS_DIR: str = BUILTIN_TOOLS_DIR  # <tool>/{.venv,src}

    def tool_roots(self) -> tuple:
        """Every directory scanned for packaged tools, in order, deduplicated.

        Order is precedence: a name found in an earlier root wins, and the
        later duplicate is reported at startup rather than silently shadowing.
        """
        roots = []
        for entry in str(self.TOOLS_DIR).split(os.pathsep):
            entry = entry.strip()
            if entry:
                roots.append(os.path.abspath(entry))
        if os.path.abspath(BUILTIN_TOOLS_DIR) not in roots:
            roots.append(os.path.abspath(BUILTIN_TOOLS_DIR))
        seen, ordered = set(), []
        for root in roots:
            if root not in seen:
                seen.add(root)
                ordered.append(root)
        return tuple(ordered)
    # Injected by path, never installed into a tool venv, so runner and server
    # are always the same version.
    RUNNER_PATH: str = os.path.join(_SERVER_DIR, "execution", "runner.py")
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

    # --- splitting a cohort -------------------------------------------
    #
    # A folder of 20 CBCTs is ~2 GB sent as one archive: the card sits idle
    # until its last byte lands, nothing survives a connection that drops at
    # 95%, and it is over MAX_UPLOAD_MB anyway. A client splits such a folder
    # into batches and sends each as its own run; these two numbers are what
    # this server tells it to split on.
    #
    # **Megabytes, not a file count, and that is the whole design.** What a
    # batch protects is the server's own bandwidth, disk and upload limit --
    # none of which is a property of a tool, so none of which belongs in a
    # per-tool table. Sizing by bytes also gets for free what a fixed count
    # cannot: 20 intraoral surfaces of 5 MB batch together, 8 CBCTs of 50 MB do
    # not, and a cohort of unusually large volumes shrinks its own batches --
    # with nobody having had to write "IOS" or "CBCT" anywhere.
    #
    # Deliberately under MAX_UPLOAD_MB: a batch that cannot be uploaded is a
    # cohort split into pieces that each answer 413.
    BATCH_MAX_MB: int = 400
    # The ceiling bytes cannot supply. 5 000 clinical notes of 20 KB are 100 MB
    # -- one batch, comfortably under the cap, and not one partial result until
    # the last note is done. Whichever of the two binds first wins.
    BATCH_MAX_FILES: int = 25

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

    # --- run progress and cancellation --------------------------------
    # Idle TTL of a run directory, the same shape as TRANSFER_TTL_SECONDS:
    # every event appended and every event read stamps it, so a cohort
    # reporting progress for hours never expires under itself. The normal path
    # removes a run with its request; this bounds the one whose client vanished
    # mid-POST, and a progress message can name a file.
    RUN_TTL_SECONDS: int = 900
    # How often GET /runs/{id}/events tails the file. Small because it is the
    # latency a clinician sees on a progress bar, and cheap because it is one
    # read of the bytes appended since the last poll.
    RUN_EVENT_POLL_SECONDS: float = 0.25
    # How often a run in flight checks whether the client has cancelled it: one
    # stat per interval, per running tool.
    RUN_CANCEL_POLL_SECONDS: float = 1.0
    # Events reported per run. A chatty tool in a 500-patient cohort must not
    # fill TEMP_DIR; past the cap one "not reported" event is delivered and the
    # rest are dropped, terminal events always excepted.
    MAX_RUN_EVENTS: int = 2000


    @field_validator("SADT_DISPATCH_MODE")
    @classmethod
    def _known_dispatch_mode(cls, value: str) -> str:
        if value not in DISPATCH_MODES:
            raise ValueError(f"SADT_DISPATCH_MODE must be one of {DISPATCH_MODES}, got {value!r}")
        return value


settings = Settings()
