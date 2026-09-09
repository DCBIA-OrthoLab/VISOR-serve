# Where `describe.py` lands

Empty on purpose. In a real build the `tools` context is the **SADT-VISOR
repository**, whose `scripts/describe.py` is copied to `/tools/scripts/` and
run with each tool's own interpreter to generate its `.schema.json`.

These fixtures ship a pre-generated `.schema.json` instead, so they need no
generator - but the folder has to exist for the image's `COPY` to resolve.
