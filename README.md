# VISOR-serve

An HTTP server that runs tools it has never imported.

Each tool lives in its own directory with its own Python interpreter and its own
lockfile. The server reads a JSON descriptor generated from the tool's `run()`
signature, validates an incoming request against it, and executes the tool as a
subprocess:

```
<TOOLS_DIR>/<tool>/.venv/bin/python <RUNNER_PATH> --job <job dir>/job.json
```

Nothing about a tool is compiled into the server. Two tools whose dependencies
cannot coexist are therefore not a problem: one can run torch 2.8 while another
runs torch 2.11, in the same deployment, while the server itself runs Python
3.13 and holds no deep-learning stack at all.

This repository is the server. The tools it was built for are in
[`SADT-VISOR`](https://github.com/DCBIA-OrthoLab/SADT-VISOR), and the 3D Slicer
client is [`AutomatedDentalToolsRemote`](https://github.com/DCBIA-OrthoLab/AutomatedDentalToolsRemote).
The server carries no clinical knowledge, so a different catalogue can be served
by pointing `TOOLS_DIR` somewhere else.

## Three rules that explain most of the design

**A tool is a directory with its own virtualenv.** The server never imports one.

**Discovery is declared, not inferred.** A directory becomes a tool when its
`pyproject.toml` carries `[tool.sadt] tool = true`, scanned two levels deep so a
grouping folder can hold several. There is no central registry to keep in sync.

**A tool calls another through the supervisor**, never by importing it. The
supervisor starts the callee in its own interpreter and passes a directory of
files. The declared call graph is verified at every boot: the server refuses to
start when a declared callee is not served.

## Installing a server

One command takes a bare Ubuntu machine to a running server: the server image,
the whole SADT-VISOR tool library as its prebuilt image, every model and test
file, and Tailscale in front of it. Run it from the directory that should hold
the deployment (it needs about 60 GB):

```bash
curl -fsSL https://raw.githubusercontent.com/DCBIA-OrthoLab/VISOR-serve/deploy/scripts/setup-server.sh \
  | sh -s -- --full --tailscale --auto-update apply \
      --tools-image ghcr.io/dcbia-ortholab/sadt-visor-tools:latest
```

What it does on its way, asking before each step that needs sudo:

- **NVIDIA driver.** A card whose driver is missing or too old (below 525, or
  below 570 for an RTX 50xx) gets the driver Ubuntu recommends; reboot, then
  run the same command again. A card older than Turing (RTX 20xx) runs on CPU.
- **Docker** and the **NVIDIA Container Toolkit**, when missing.
- **Tailscale.** Before running the command, open the Tailscale admin console,
  Settings > Keys, and generate an auth key. Paste it when asked: the prompt
  hides it, and it never appears on a command line. The server is then
  reachable at `https://<name>.<tailnet>.ts.net`, from the tailnet only.

Run again, the same command updates the deployment and keeps its tokens. The
two tokens a deployment needs are printed by:

```bash
python3 VISOR-serve/scripts/server_ctl.py token            # for the Slicer client
python3 VISOR-serve/scripts/server_ctl.py token --admin    # for /server-debug
```

## Connecting a workstation

A workstation reaches the server through the same tailnet. On Ubuntu:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

`tailscale up` prints a link; open it and sign in with the lab's Tailscale
account. On Windows or macOS, install the app from
<https://tailscale.com/download> and sign in the same way.

Then, in 3D Slicer, open the Slicer Cloud module and set:

- **URL**: `https://<name>.<tailnet>.ts.net`, as printed at the end of the
  server install (`tailscale serve status` on the server shows it again);
- **token**: the API token above, never the admin one.

## Running it by hand

```bash
cp .env.example .env          # API_TOKEN, TOOLS_DIR, DATA_DIR, DEVICE
docker compose up -d          # the deployment image
```

For a local server against a `SADT-VISOR` checkout, without Docker:

```bash
SADT_TOOLS=/path/to/SADT-VISOR ./run-local.sh    # serves on 127.0.0.1:8001
```

`scripts/README.md` documents the unattended installers for a fresh machine,
the model-weight fetcher and the test-file fetcher.

## Documentation

| File | What it covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The shape of the system and the rules above, in full |
| [`docs/INTERNALS.md`](docs/INTERNALS.md) | Registry, dispatch, transfer and the job lifecycle |
| [`docs/TOOL_GRAPH.md`](docs/TOOL_GRAPH.md) | How one tool declares a call to another |
| [`ADDING_A_TOOL.md`](ADDING_A_TOOL.md) | Adding a tool: one `run()` function, no base class |
| [`MIGRATING_A_TOOL.md`](MIGRATING_A_TOOL.md) | Porting an existing Slicer module into a served tool |
| [`RUN_PROGRESS.md`](RUN_PROGRESS.md) | Reporting a run's progress, and cancelling it |
| [`SECURITY.md`](SECURITY.md) | The threat model this server does and does not address |
| [`server/README.md`](server/README.md) | The API surface, endpoint by endpoint |
| [`benchmarks/`](benchmarks/) | The measurement harness, re-runnable |

## Citation

This server is the artifact described in:

> Grivot Pélisson J, Barret R, Cevidanes L. VISOR: A Descriptor-Driven
> Architecture for Environment-Isolated and Composable Dental Imaging Tools.
> AMIA 2027 Amplify Informatics Summit. Submitted.

The benchmark harness under `benchmarks/` reproduces every runtime number in
that paper.
