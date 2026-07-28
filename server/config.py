"""Server configuration, loaded entirely from environment variables.

No secrets are hardcoded here: API_TOKEN and all other settings must be
provided via the environment (or a local .env file, see .env.example).
"""

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
