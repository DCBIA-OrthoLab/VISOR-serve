"""Public benchmark harness for the SADT remote tool server.

Everything a reviewer needs to re-run the numbers in the paper lives under this
package: the campaign definitions (B1-B5), the two execution paths they compare,
the provenance every record carries, and the summaries derived from the raw
data. See README.md for the reviewer-facing instructions and NOTES-local-path.md
for how the `local` path reproduces what the server's dispatcher does.

Importing this package must not need a GPU, a running server, or a Docker
daemon. Anything that does is reached only when a campaign actually executes,
and the unit tests skip with a stated reason when it is missing.
"""

__all__ = ["__version__"]

# Bumped when a change would make two raw files incomparable: a phase renamed,
# a timing boundary moved, a default changed. It is written into every record,
# so a summary can refuse to mix two versions instead of averaging them.
__version__ = "1.0.0"
