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

    # Directory used for temporary input/output files. Every file written
    # here is deleted once a request has been served.
    TEMP_DIR: str = "/tmp/inference_server"

    # Fallback whitelist, only used for arguments with a generic "file" type
    # (see base.FILE_TYPES). Arguments with a specific type (e.g. "zip_file")
    # are validated against that type's own extensions instead, regardless
    # of this list. "*" accepts everything for the generic fallback.
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".nii", ".nii.gz")


settings = Settings()
