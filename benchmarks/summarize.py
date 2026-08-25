"""Summaries, derived from the raw files and nothing else.

Everything under results/summary/ can be deleted and regenerated:

    python -m benchmarks.summarize --campaign b1

Two rules the summaries follow, and both come from the paper's protocol rather
than from taste:

**Median and full range, never mean and standard deviation.** Six repetitions of
a process with a hard floor (the work cannot take less than it takes) and an open
tail (a page fault, a GPU queue) are not normally distributed. A mean is pulled
by the tail and a standard deviation implies a symmetry that is not there. Median
plus min and max says what happened without claiming a distribution.

**Failures are counted in every table.** A row reporting the median of four
successes out of six is a different row from one reporting the median of six, and
a summary that does not distinguish them is misleading by omission.

Warm-up runs are excluded from the statistics and counted separately, so
"repetition 1 was discarded" is visible in the output rather than only in the
protocol.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Optional

from .recording import STATUS_FAILED, STATUS_OK, SUMMARY_SUBDIR, load_records
from .settings import BENCHMARKS_ROOT


def _numbers(values: list) -> dict:
    """median / min / max / n, or an all-null row for an empty group.

    p95 by nearest rank rather than interpolation: with 8 or 16 points an
    interpolated p95 is a number no observation supports, and B4 reports a tail
    that has to be a real job's latency.
    """
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None, "p95": None}
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered) + 0.5)) - 1))
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[rank],
    }


def _measured(records: list) -> list:
    """The runs a statistic may be computed over: succeeded, not warm-up."""
    return [r for r in records if r.get("status") == STATUS_OK and not r.get("warmup")]


def _round(value, digits: int = 4):
    """Four decimals by default: a trivial tool answers in single-digit
    milliseconds on loopback, and rounding those to 3 places collapses the whole
    range into one number."""
    return None if value is None else round(float(value), digits)


# ----------------------------------------------------------------------
# Per-campaign tables
# ----------------------------------------------------------------------

def table_b1(records: list) -> list:
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("tool"), record.get("path"))].append(record)
    rows = []
    for (tool, path), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        measured = _measured(group)
        stats = _numbers([r["total_seconds"] for r in measured])
        rows.append(
            {
                "tool": tool,
                "path": path,
                "runs": len(group),
                "warmup_discarded": sum(1 for r in group if r.get("warmup")),
                "failed": sum(1 for r in group if r.get("status") == STATUS_FAILED),
                "n": stats["n"],
                "median_s": _round(stats["median"]),
                "min_s": _round(stats["min"]),
                "max_s": _round(stats["max"]),
                "first_error": next(
                    ((r.get("error_message") or "")[:160] for r in group
                     if r.get("status") == STATUS_FAILED), ""
                ),
            }
        )
    return rows


def table_b2(records: list) -> list:
    from .recording import PHASES

    groups = defaultdict(list)
    for record in records:
        key = (
            record.get("tool"),
            (record.get("extra") or {}).get("payload_label"),
            (record.get("extra") or {}).get("parallelism"),
        )
        groups[key].append(record)
    rows = []
    for (tool, payload, parallelism), group in sorted(groups.items(), key=lambda i: str(i[0])):
        measured = _measured(group)
        row = {
            "tool": tool,
            "payload": payload,
            "parallelism": parallelism,
            "runs": len(group),
            "failed": sum(1 for r in group if r.get("status") == STATUS_FAILED),
            "n": len(measured),
            "total_s": _round(_numbers([r["total_seconds"] for r in measured])["median"]),
            "uploaded_mb": _round(
                _numbers([
                    (r.get("extra") or {}).get("bytes_uploaded", 0) / 1e6 for r in measured
                ])["median"], 1
            ),
            "downloaded_mb": _round(
                _numbers([
                    (r.get("extra") or {}).get("bytes_downloaded", 0) / 1e6 for r in measured
                ])["median"], 1
            ),
        }
        for phase in PHASES:
            values = [
                (r.get("phases") or {}).get(phase)
                for r in measured
                if (r.get("phases") or {}).get(phase) is not None
            ]
            row[f"{phase}_s"] = _round(_numbers(values)["median"])
        rows.append(row)
    return rows


def table_b3(records: list) -> list:
    rows = []
    chains = defaultdict(list)
    for record in records:
        extra = record.get("extra") or {}
        if extra.get("measurement") == "startup":
            rows.append(
                {
                    "measurement": "startup",
                    "tool": record.get("tool"),
                    "mode": "",
                    # Which local mode the probe ran in. The image's virtualenvs
                    # are hardlink-deduplicated and the host checkout's are not,
                    # so two rows for one tool are two different measurements.
                    "local_mode": extra.get("local_mode", ""),
                    "runs": 1,
                    "failed": 1 if record.get("status") == STATUS_FAILED else 0,
                    "n": 1 if record.get("status") == STATUS_OK else 0,
                    "median_s": None,
                    "interpreter_start_s": _round(extra.get("interpreter_start_seconds")),
                    "import_stack_s": _round(extra.get("import_stack_seconds")),
                    "package": extra.get("package", ""),
                    "error": (record.get("error_message") or "")[:160],
                }
            )
        else:
            chains[(record.get("tool"), extra.get("mode"))].append(record)
    for (tool, mode), group in sorted(chains.items(), key=lambda i: str(i[0])):
        measured = _measured(group)
        stats = _numbers([r["total_seconds"] for r in measured])
        rows.append(
            {
                "measurement": "chain",
                "tool": tool,
                "mode": mode,
                "local_mode": (
                    (measured[0].get("extra") or {}).get("local_mode", "") if measured else ""
                ),
                "runs": len(group),
                "failed": sum(1 for r in group if r.get("status") == STATUS_FAILED),
                "n": stats["n"],
                "median_s": _round(stats["median"]),
                "interpreter_start_s": None,
                "import_stack_s": None,
                "package": "",
                "error": next(
                    ((r.get("error_message") or "")[:160] for r in group
                     if r.get("status") == STATUS_FAILED), ""
                ),
            }
        )
    return rows


def table_b4(records: list) -> list:
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("tool"), (record.get("extra") or {}).get("concurrency"))].append(record)
    rows = []
    for (tool, level), group in sorted(groups.items(), key=lambda i: (str(i[0][0]), i[0][1] or 0)):
        measured = _measured(group)
        stats = _numbers([r["total_seconds"] for r in measured])
        first = (group[0].get("extra") or {}) if group else {}
        vram = first.get("vram") or {}
        peaks = (vram.get("peak_mib") or {}).values()
        baselines = (vram.get("baseline_mib") or {}).values()
        rows.append(
            {
                "tool": tool,
                "concurrency": level,
                "jobs": len(group),
                "failed": sum(1 for r in group if r.get("status") == STATUS_FAILED),
                "n": stats["n"],
                "window_s": _round(first.get("campaign_window_seconds")),
                "throughput_jobs_per_min": _round(first.get("throughput_jobs_per_minute"), 2),
                "p50_s": _round(stats["median"]),
                "p95_s": _round(stats["p95"]),
                "max_s": _round(stats["max"]),
                "peak_vram_mib": max(peaks) if peaks else None,
                "baseline_vram_mib": max(baselines) if baselines else None,
                "vram_samples": vram.get("samples"),
                "vram_unavailable": vram.get("unavailable_reason") or "",
            }
        )
    return rows


def table_b5(records: list) -> list:
    rows = []
    for record in records:
        extra = record.get("extra") or {}
        if extra.get("side") == "local_control":
            # The determinism baseline, kept in its own row shape so it can
            # never be misread as a local-versus-remote result. The second RUN
            # carries the same side marker and no comparison, so it is skipped
            # here rather than printed as an empty row.
            if "parity" not in extra and record.get("status") != STATUS_FAILED:
                continue
            parity = extra.get("parity") or {}
            rows.append(
                {
                    "tool": record.get("tool"),
                    "pair": extra.get("pair"),
                    "status": record.get("status"),
                    "local_mode": extra.get("local_mode", ""),
                    "parity_ok": extra.get("deterministic"),
                    "content_parity_ok": extra.get("deterministic"),
                    "identical": parity.get("identical_count"),
                    "differing": parity.get("differing_count"),
                    "only_local": len(parity.get("only_left") or []),
                    "only_remote": len(parity.get("only_right") or []),
                    "renamed_pairs": 0,
                    "renamed_identical": 0,
                    "differing_files": "LOCAL-VS-LOCAL CONTROL: "
                    + "; ".join(parity.get("differing") or [])[:380],
                    "error": (record.get("error_message") or "")[:200],
                }
            )
            continue
        if extra.get("side") != "comparison":
            continue
        parity = extra.get("parity") or {}
        renamed = parity.get("renamed") or {}
        pairs = renamed.get("pairs") or []
        rows.append(
            {
                "tool": record.get("tool"),
                "pair": extra.get("pair"),
                "status": record.get("status"),
                "local_mode": extra.get("local_mode", ""),
                "parity_ok": extra.get("parity_ok"),
                "content_parity_ok": extra.get("content_parity_ok"),
                "identical": parity.get("identical_count"),
                "differing": parity.get("differing_count"),
                "only_local": len(parity.get("only_left") or []),
                "only_remote": len(parity.get("only_right") or []),
                "renamed_pairs": len(pairs),
                "renamed_identical": sum(1 for p in pairs if p.get("identical")),
                "differing_files": "; ".join(parity.get("differing") or [])[:400],
                "error": (record.get("error_message") or "")[:200],
            }
        )
    return rows


TABLES = {
    "b1": table_b1,
    "b2": table_b2,
    "b3": table_b3,
    "b4": table_b4,
    "b5": table_b5,
}


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def to_markdown(rows: list) -> str:
    if not rows:
        return "_No records._\n"
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _cell(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def parity_details(records: list) -> str:
    """The per-file detail B5 exists to produce, in prose a reviewer can read."""
    blocks = []
    for record in records:
        extra = record.get("extra") or {}
        if extra.get("side") != "comparison":
            continue
        parity = extra.get("parity") or {}
        if parity.get("ok"):
            continue
        blocks.append(
            f"### {record.get('tool')} (pair {extra.get('pair')}, "
            f"local mode {extra.get('local_mode', 'unknown')})\n"
        )
        renamed = parity.get("renamed") or {}
        paired_left = {pair["local"] for pair in (renamed.get("pairs") or [])}
        paired_right = {pair["remote"] for pair in (renamed.get("pairs") or [])}
        for name in parity.get("only_left") or []:
            if name in paired_left:
                continue
            blocks.append(f"- `{name}` -- produced by the local path only.")
        for name in parity.get("only_right") or []:
            if name in paired_right:
                continue
            blocks.append(f"- `{name}` -- produced by the remote path only.")
        for pair in renamed.get("pairs") or []:
            verdict = (
                "byte-identical" if pair.get("identical") else "and the bytes differ too"
            )
            blocks.append(
                f"- `{pair['local']}` vs `{pair['remote']}` -- the same artifact under "
                f"two names (the server stages an uploaded input under its argument "
                f"name); {verdict}."
            )
            detail = pair.get("detail") or {}
            if detail:
                byte_detail = detail.get("bytes") or {}
                blocks.append(
                    f"  - sizes {byte_detail.get('left_size')} vs "
                    f"{byte_detail.get('right_size')} bytes"
                )
        for name in parity.get("differing") or []:
            detail = (parity.get("details") or {}).get(name, {})
            blocks.append(f"- `{name}` -- bytes differ.")
            byte_detail = detail.get("bytes") or {}
            blocks.append(
                f"  - sizes {byte_detail.get('left_size')} vs {byte_detail.get('right_size')}"
                f" bytes; first difference at offset {byte_detail.get('first_differing_offset')}"
                + (
                    f"; {byte_detail.get('differing_bytes')} differing bytes"
                    f" ({100 * (byte_detail.get('differing_fraction') or 0):.4f}% of the file)"
                    if byte_detail.get("differing_bytes") is not None else ""
                )
            )
            if detail.get("json", {}).get("readable"):
                keys = detail["json"].get("differing_keys") or []
                blocks.append(f"  - JSON keys that differ: {', '.join(keys[:40]) or '(none)'}")
            if detail.get("text", {}).get("readable"):
                changed = detail["text"].get("changed_lines") or []
                for change in changed[:10]:
                    blocks.append(
                        f"  - line {change['line']}: `{change['left']}` vs `{change['right']}`"
                    )
            numeric = detail.get("numeric")
            if numeric:
                if not numeric.get("available"):
                    blocks.append(
                        f"  - numeric distance unavailable: {numeric.get('reason')}"
                    )
                else:
                    voxelwise = numeric.get("voxelwise")
                    geometry = numeric.get("geometry") or {}
                    blocks.append(
                        f"  - geometry identical: size={geometry.get('size_equal')}, "
                        f"spacing={geometry.get('spacing_equal')}, "
                        f"origin={geometry.get('origin_equal')}, "
                        f"direction={geometry.get('direction_equal')}"
                    )
                    if voxelwise:
                        blocks.append(
                            f"  - voxelwise: {voxelwise['differing_voxels']} of "
                            f"{voxelwise['voxels']} voxels differ "
                            f"({100 * voxelwise['differing_fraction']:.6f}%), "
                            f"max |diff| {voxelwise['max_abs_difference']:.6g}, "
                            f"mean |diff| {voxelwise['mean_abs_difference']:.6g}, "
                            f"RMS {voxelwise['rms_difference']:.6g}"
                        )
                    else:
                        blocks.append(f"  - no voxelwise comparison: {numeric.get('reason')}")
        blocks.append("")
    return "\n".join(blocks) if blocks else "_Every compared artifact was byte-identical._\n"


def summarize(
    campaign: str,
    root: str = BENCHMARKS_ROOT,
    stamp: Optional[str] = None,
    only: Optional[list] = None,
) -> dict:
    """Regenerate this campaign's CSV and markdown from its raw files.

    `only` restricts the summary to named raw files. Every raw file of the
    campaign is used by default -- which is the right thing when a campaign was
    run in several sittings, and the wrong thing when an exploratory run should
    not be mixed with a published one. The summary always lists the files it read,
    so which of the two happened is never in doubt.
    """
    records = load_records(root, campaign)
    if only:
        wanted = {os.path.basename(name) for name in only}
        records = [record for record in records if record.get("_source_file") in wanted]
    builder = TABLES.get(campaign)
    if builder is None:
        raise SystemExit(f"No summary defined for campaign {campaign!r}. Known: {sorted(TABLES)}")
    rows = builder(records)

    total = len(records)
    failures = sum(1 for record in records if record.get("status") == STATUS_FAILED)
    warmups = sum(1 for record in records if record.get("warmup"))
    sources = sorted({record.get("_source_file", "") for record in records} - {""})

    directory = os.path.join(root, SUMMARY_SUBDIR)
    os.makedirs(directory, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    csv_path = os.path.join(directory, f"{campaign}-{stamp}.csv")
    markdown_path = os.path.join(directory, f"{campaign}-{stamp}.md")

    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    hardware = _hardware_line(records)
    body = [
        f"# {campaign.upper()} summary",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} from "
        f"{len(sources)} raw file(s). This file is DERIVED; delete it and re-run "
        f"`python -m benchmarks.summarize --campaign {campaign}` to rebuild it.",
        "",
        f"- records: **{total}**",
        f"- failed runs: **{failures}**",
        f"- warm-up runs discarded from the statistics: **{warmups}**",
        f"- raw files: {', '.join(f'`{name}`' for name in sources) or '(none)'}",
        "",
        f"Hardware: {hardware}",
        "",
        "Statistics are median with the full range. Mean and standard deviation are "
        "deliberately not reported; see the module docstring.",
        "",
        to_markdown(rows),
    ]
    if campaign == "b5":
        body += ["", "## Where the two paths differ", "", parity_details(records)]
    if failures:
        body += ["", "## Failures", "", to_markdown(_failure_rows(records))]

    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(body))

    return {
        "csv": csv_path if rows else None,
        "markdown": markdown_path,
        "records": total,
        "failures": failures,
        "rows": rows,
    }


def _failure_rows(records: list) -> list:
    return [
        {
            "tool": record.get("tool"),
            "path": record.get("path"),
            "repetition": record.get("repetition"),
            "error_type": record.get("error_type"),
            "error_message": (record.get("error_message") or "")[:300],
        }
        for record in records
        if record.get("status") == STATUS_FAILED
    ]


def _hardware_line(records: list) -> str:
    """The machine every number in this file came from.

    Taken from the records themselves rather than re-probed, because a summary
    regenerated on a different machine must still name the machine that
    MEASURED. Two different fingerprints in one campaign are reported as such
    instead of silently picking one.
    """
    seen = []
    for record in records:
        provenance = record.get("provenance") or {}
        cpu = (provenance.get("cpu") or {}).get("model")
        devices = ((provenance.get("gpu") or {}).get("devices")) or []
        gpu = devices[0]["name"] if devices else "no CUDA device"
        line = f"{provenance.get('hostname')} -- {cpu} -- {gpu}"
        if line not in seen:
            seen.append(line)
    if not seen:
        return "_unknown (no provenance in the records)_"
    if len(seen) == 1:
        return seen[0]
    return "**several machines**: " + "; ".join(seen)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--campaign", required=True, choices=sorted(TABLES))
    parser.add_argument("--root", default=BENCHMARKS_ROOT,
                        help="Directory holding results/ (default: the benchmarks package)")
    parser.add_argument(
        "--raw", action="append", default=None,
        help="summarise only this raw file (repeatable). Default: every raw file "
             "of the campaign.",
    )
    arguments = parser.parse_args(argv)
    outcome = summarize(arguments.campaign, arguments.root, only=arguments.raw)
    print(f"records: {outcome['records']}  failures: {outcome['failures']}")
    if outcome["csv"]:
        print(f"csv     : {outcome['csv']}")
    print(f"markdown: {outcome['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
