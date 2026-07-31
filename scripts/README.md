# `scripts/` — populating `DATA/`

The server serves AI models and reference test files from `DATA/`, which is
gitignored (it holds confidential data and hundreds of GB of weights). These
scripts fill it from the public GitHub releases the original Slicer modules
already used, so a fresh machine goes from `git clone` to a working server
without anyone hand-copying model bundles around.

| File | Role |
| --- | --- |
| [`setup-models.sh`](setup-models.sh) | Fetch AI models. Runnable straight from GitHub. |
| [`setup-testfiles.sh`](setup-testfiles.sh) | Fetch reference test files. Runnable straight from GitHub. |
| [`fetch_data.py`](fetch_data.py) | The engine both wrappers call. Standard library only. |
| [`data-manifest.yml`](data-manifest.yml) | What exists, where it comes from, where it goes. |

## Usage

On a machine with nothing checked out — run it from the directory that should
end up holding `DATA/`:

```bash
curl -fsSL https://raw.githubusercontent.com/Jules-GP/slicer-remote-tool-server/main/scripts/setup-models.sh | sh
curl -fsSL https://raw.githubusercontent.com/Jules-GP/slicer-remote-tool-server/main/scripts/setup-testfiles.sh | sh
```

From a clone, the same thing without the network round trip (the wrappers
detect `./scripts/` and use the local manifest, so a local edit takes effect):

```bash
./scripts/setup-models.sh
./scripts/setup-testfiles.sh
```

**Everything is about 29 GB** across 14 tools, 12 GB of which is ALI's CBCT
models. Ask for one tool at a time instead:

```bash
./scripts/setup-models.sh --tool AMASSS --tool SurgMovPred
python3 scripts/fetch_data.py --list          # sizes per tool, downloads nothing
```

Arguments reach the engine through `sh -s --` when piping:

```bash
curl -fsSL .../setup-models.sh | sh -s -- --tool AMASSS
```

Useful flags: `--tool` (repeatable), `--kind models|testfiles`, `--data-dir`,
`--force`, `--list`. `DATA_DIR` works as an environment variable too, and
`REPO`/`REF` point the wrappers at a fork or a branch other than `main`.

## What you get

The layout is exactly the one `server/data_store.py` reads, so a file landing
here is immediately offered by `GET /tools/<tool>/data` — nothing else to
configure:

```
DATA/
├── AMASSS/
│   ├── models/AMASSS_Models/{CB,CBMASK,CV,MAND,MANDMASK,MAX,MAXMASK,SKIN,UAW}/…
│   └── testfiles/MG_test_scan.nii.gz
└── SurgMovPred/
    ├── models/all_models/<target>_Pred/stacking_package.pkl
    └── testfiles/TestFiles/patients_to_predict.xlsx
```

Re-running is cheap and safe: anything already on disk is skipped, so an
interrupted 12 GB download resumes by restarting the command. A download in
flight lives in a temporary folder and is only moved into place once complete,
so a killed run never leaves a truncated model that the next run would mistake
for a finished one.

## Adding an entry

Append to [`data-manifest.yml`](data-manifest.yml) — the header there documents
every field. The short version:

```yaml
  MyTool:
    models:
      - name: weights.zip        # what to download
        url: https://…           # where from
        size: 12345678           # optional, for the size report
        extract: true            # unpack, keep the folder, drop the archive
        dest: MyBundle/weights   # optional, overrides the destination name
```

`dest` exists for two situations worth knowing about before you hit them:

- **Name collisions.** Every nnUNet bundle ships a file literally called
  `checkpoint_final.pth`; without `dest` the second one downloaded would
  overwrite the first.
- **Grouping.** `server_selectable="model"` shows one dropdown entry per
  top-level name and hands the tool exactly one. ALI's 112 landmark archives
  are therefore filed under a single `ALI_CBCT_Models/` bundle rather than
  appearing as 112 separate choices.

## Checksums

`sha256` is optional and mostly absent today. When present it is verified and
a mismatch discards the download rather than installing it. To pin an entry,
run the fetch once, take the hash the script prints, and paste it into the
manifest — the hashes are not published by GitHub, so inventing them would be
worse than leaving the field out.
