"""One module per campaign.

Each module exposes the same two functions, which is what lets run.py stay
free of per-campaign branching:

    build_plan(config, options) -> list[PlanItem]
        Pure. No process is started, nothing is written. This is what
        --dry-run prints, and what the disk guard sizes.

    execute(item, context) -> Iterator[RunRecord]
        Runs one plan item and yields one record per individual run. A record
        is yielded for a FAILED run too, carrying its error.
"""
