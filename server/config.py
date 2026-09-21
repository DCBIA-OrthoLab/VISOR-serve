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
    TOOL_TIMEOUT_SECONDS: float = 0  # 0 = none; a cohort legitimately takes hours
    # How many times a run that died for want of memory is started again, with
    # more room reserved each time. A clinician who asked for a segmentation
    # wants it to wait and then run, not to be told "failed" because the server
    # guessed one job too many -- and a retry here is cheap in a way it is not
    # in an in-process design: the tool is a subprocess, so the failure killed
    # it and nothing else, and its inputs are still staged on disk.
    #
    # 2 rather than a larger number, because the escalation saturates: the
    # third attempt is already asking for the whole budget, which means running
    # alone, and a tool that runs out of memory ALONE does not fit at all.
    # `admission.holds_everything` stops earlier than this count whenever that
    # point is reached. Set to 0 to answer 500 on the first failure.
    MEMORY_RETRIES: int = 2
    # What each retry multiplies the reservation by. The recorded peak is a
    # lower bound after an out-of-memory -- the run died reaching for more --
    # so believing it again would admit the same neighbours and fail the same
    # way.
    MEMORY_RETRY_GROWTH: float = 1.5
    # How many recent runs a tool's measured cost is taken from. An ALL-TIME
    # maximum never comes back down, so one unusual scan raises a tool's
    # reservation for every run that follows it, for ever: measured here, an
    # 8 GiB outlier would pin a tool whose ordinary peak is 0.26 GiB and
    # collapse its parallelism permanently. The worst of the last N runs
    # forgets that outlier once the tool has behaved normally N times.
    #
    # Safe only because MEMORY_RETRIES exists. Forgetting a peak means
    # occasionally admitting one job too many, and that is a retry rather than
    # a failure now.
    COST_WINDOW: int = 20

    # An optional cap on how many channels one run may open. 0, the default,
    # means there is none.
    #
    # It was 8, and 8 was invented: the claim attached to it -- that past
    # roughly eight, contention takes back what the extra channels buy -- was
    # never measured on anything. Two MEASURED things bound a width instead,
    # and between them there is no realistic one left to refuse:
    #
    #   * how many items the request holds (`width_from` in deployment.toml),
    #     so seven landmarks is seven channels and one landmark is one;
    #   * what the width COSTS, from the cost table, which admission refuses
    #     when it does not fit host memory, the card's budget, or what the card
    #     actually has free. ALI_CBCT at 0.78 GiB a channel meets the 33.3 GiB
    #     budget somewhere around thirty.
    #
    # Not the core count either: bounding GPU channels by cores would undo the
    # decoupling that lets a GPU-bound tool spread on a small core share, which
    # is the whole reason ALI_CBCT can open more than two.
    #
    # The residual is a tool with a near-zero measured cost AND a very large
    # cohort. Nothing in the catalogue produces it; a deployment that meets it
    # sets this rather than having a number guessed for it here.
    SADT_MAX_CHANNELS: int = 0

    # The fewest cores a run may be admitted holding, when the machine is busy
    # enough that its full share does not fit.
    #
    # Cores are the one resource here that narrows safely: too few makes a run
    # SLOW, where too little memory makes it die. So a run that would otherwise
    # queue is admitted on fewer cores instead -- it starts, which is what the
    # clinician in front of it is waiting for, and the measured cost of being
    # narrowed is latency rather than failure.
    #
    # 4: below that a tool stops being slow and starts being unusable, and a
    # clinician staring at a bar that has not moved in ten minutes cannot tell
    # it from a server that has died. Admitting a run on one core is degrading
    # all the way rather than refusing -- which sounds right until the run it
    # admits is one nobody would have wanted started on those terms.
    #
    # **Know what 1 costs before leaving it there.** The grant is decided once,
    # at spawn, because a BLAS pool cannot be retuned from outside the process
    # -- so a run admitted on one core keeps one core for its whole life, even
    # if the machine empties a second later. Admitting everything at one core
    # therefore trades a queue for universal slowness, which is the classic way
    # a scheduler thrashes: forty runs each eight times slower finish no sooner
    # than six runs at a time, and every one of them looks broken while they do
    # it. Measured here: six concurrent AMASSS already stretch from 79s to
    # ~145s at seven cores each.
    #
    # Raise it to 2 or 4 on a deployment that would rather queue than crawl.
    # It is a floor on the RESERVATION, not on what a run may spend: a job
    # admitted holding one still gets more threads if the machine was emptier
    # than when it arrived.
    SADT_MIN_CPUS_PER_JOB: int = 4

    # Where a benchmark campaign left its summary, for `GET /benchmarks` and
    # the status page. Normally absent -- a production deployment runs no
    # campaign and the endpoint simply answers "nothing measured here", which
    # is the honest answer and not an error.
    #
    # Read rather than computed: the server does not recalculate a statistic it
    # did not take. `benchmarks/campaign_report.py --json` writes the file, the
    # same call that writes the document, so the page and the report cannot
    # disagree with each other.
    SADT_BENCHMARK_DIR: str = "/benchmarks"

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

    # --- what this machine gives the server ---------------------------
    # Every one of these is optional. Left alone, `resources.resolve` sizes the
    # server to the machine it started on (cgroup limit first, so a container's
    # own `cpus:`/`memory:` are honoured), which is what keeps a deployment one
    # `docker compose up` with nothing to tune.
    #
    # SADT_-prefixed, unlike their neighbours, for a reason specific to this
    # server: `_child_environment` copies the whole environment into every tool
    # process, and a bare `CPU_BUDGET` would be inherited by twenty-two
    # third-party stacks that may read a name that generic.
    #
    # Each accepts three spellings -- "75%" of what was detected, "8" as an
    # absolute, or "16GB" -- so one variable covers every way an operator
    # thinks about it. An unparseable value is ignored with a warning, never
    # fatal: a typo in a knob must not stop a server that can size itself.
    SADT_CPU_BUDGET: str = ""
    SADT_RAM_BUDGET: str = ""
    SADT_VRAM_BUDGET: str = ""
    # How much of the machine the server takes when nothing above says. Under 1
    # on purpose: the deployment shares its host with whatever else runs there,
    # and on a workstation that includes the Slicer the clinician is using.
    SADT_RESOURCE_SHARE: float = 0.75
    # The one number an operator actually knows. It buys the per-job caps
    # below, which are the fairness half: with 10 expected clients a run gets a
    # tenth of the budget and no single cohort can take the machine. With 1 --
    # a single workstation -- there is no cap and every run goes as wide as the
    # hardware allows, which is the right answer for latency.
    #
    # It is a DECLARATION, not a detection: there is one shared API token and
    # no client identity anywhere, so the server cannot tell ten workstations
    # from one clicking ten times.
    #
    # Defaulted to MAX_CONCURRENT_TOOLS rather than to 1 so an untouched
    # deployment keeps the concurrency it already had: at 1 a single job would
    # reserve the whole budget and every run on the machine would serialise
    # behind it, which is stricter than the four tool slots that exist today.
    SADT_EXPECTED_CLIENTS: int = 4
    SADT_CPU_PER_JOB: str = ""
    SADT_RAM_PER_JOB: str = ""
    SADT_VRAM_PER_JOB: str = ""

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
