"""Streamed tool runs: progress while it works, results as they are produced.

A client sends `X-Result-Delivery: stream` to `POST /run/{tool}` and, instead
of waiting minutes for one archive, reads a line-delimited JSON stream:

    {"event":"start","total":40}
    {"event":"item","index":1,"name":"p1","status":"running"}
    {"event":"artifact","name":"p1","result_ref":{...}}
    {"event":"item","index":2,"name":"p2","status":"failed","error":"..."}
    ...
    {"event":"done","summary":"38/40 scan(s) segmented"}

Two problems, one mechanism:

* **Nothing to look at during a long run.** The panel could only show elapsed
  time, because a tool run is one request whose response arrives at the end.
* **A batch that fails late lost everything that had succeeded.** Thirty-nine
  patients segmented and the fortieth unreadable used to return one archive --
  or, if the failure was fatal, nothing at all.

**Artifacts travel as references, not inline.** Each finished file is moved out
of the work dir by `transfer.store_result` and announced as a `result_ref`; the
client pulls it from `GET /results/{id}` over parallel range requests *while
the run continues*, and releases it with `DELETE`. That reuses the whole
existing transfer machinery rather than base64-ing megabytes into a JSON line,
and it keeps this stream small enough to never be the bottleneck.

**The status code is committed before the first byte.** Everything that can
answer 4xx -- unknown tool, bad extension, argument validation -- happens in
main.py *before* the response starts. After that a failure can only be an
in-band `error` event, which is why `validate()` is called up front here rather
than through `Tool.invoke`.

**Cancellation is the client hanging up.** The generator stops being consumed,
`_Emitter.cancelled` flips, and the tool stops at its next emit point -- so a
cancelled batch stops burning GPU instead of running to completion with nobody
listening. That is the one thing the blocking contract could never offer.
"""

import json
import logging
import os
import queue
import shutil
import threading
import time
from typing import Optional

import anyio

import file_utils
import transfer
from base import Tool, ToolUnavailableError

logger = logging.getLogger("streaming")

# What a tool passes to `emit`. Anything else is dropped with a warning: an
# event the client cannot parse is worse than no event.
EVENT_ITEM = "item"
EVENT_ARTIFACT = "artifact"
EVENT_MESSAGE = "message"
_TOOL_EVENTS = (EVENT_ITEM, EVENT_ARTIFACT, EVENT_MESSAGE)

# How long the drain loop waits on the queue before looking again at whether
# the tool is still running. Only bounds how quickly the stream notices the END
# of a run -- events themselves are forwarded the moment they are emitted.
_POLL_SECONDS = 0.25

# A run with nothing to say still has to prove the connection is alive: an idle
# TCP connection is indistinguishable from a dead one, and some proxies close
# one that says nothing for a minute. nnUNet loading a checkpoint and preparing
# a large scan is exactly such a silence.
_KEEPALIVE_SECONDS = 15.0


class StreamCancelled(Exception):
    """Raised inside the tool's thread when the client has gone away."""


class _Emitter:
    """What the tool calls. Thread-safe by construction: it only ever puts on a
    queue, and the generator draining it runs on the event loop."""

    def __init__(self):
        self.events: "queue.Queue" = queue.Queue()
        self.cancelled = threading.Event()

    def __call__(self, event: dict) -> None:
        """`emit(...)` inside a tool.

        Raises StreamCancelled when the client has disconnected, so a tool that
        emits between items stops there. A tool that never emits simply runs to
        completion, which is the old behaviour and is not a failure.
        """
        if self.cancelled.is_set():
            raise StreamCancelled()
        if not isinstance(event, dict) or event.get("event") not in _TOOL_EVENTS:
            logger.warning("Dropping a malformed tool event: %r", type(event))
            return
        self.events.put(event)


def _line(payload: dict) -> bytes:
    """One NDJSON record. Compact separators because this can be thousands of
    lines on a big cohort, and `\\n` because that is the frame."""
    return (json.dumps(payload, separators=(",", ":"), default=str) + "\n").encode("utf-8")


def _package(path: str, work_dir: str) -> Optional[str]:
    """A tool's artifact as ONE file ready to be stored.

    A directory is zipped (the archive lands in `work_dir`, never inside the
    directory it is packing); a plain file is passed through. Returns None for
    anything that has gone missing, which is a tool bug rather than a reason to
    kill a run that is otherwise producing results.
    """
    if not path or not os.path.exists(path):
        logger.warning("Tool announced an artifact that is not on disk")
        return None
    if os.path.isdir(path):
        name = os.path.basename(path.rstrip(os.sep)) or "artifact"
        return file_utils.make_zip([path], os.path.join(work_dir, f"{name}.zip"))
    return path


async def stream_run(
    tool: Tool,
    cleaned_args: dict,
    work_dir: str,
    cleanup_paths,
    limiter,
    media_type_of,
):
    """The response body generator.

    `cleaned_args` are ALREADY validated (see the module docstring): by the
    time this runs, the 200 has been sent and there is no status code left to
    change.
    """
    emitter = _Emitter()
    outcome: dict = {}
    cancelled_by_client = False

    def work():
        try:
            outcome["result"] = tool.invoke(cleaned_args, emit=emitter)
        except StreamCancelled:
            outcome["cancelled"] = True
        except ToolUnavailableError as exc:
            # 501's in-band twin. The request was fine and this deployment
            # cannot serve it; the reason names a package, never a path.
            outcome["error"] = str(exc)
            outcome["unavailable"] = True
        except Exception as exc:  # noqa: BLE001 - reported to the client, logged in full here
            logger.exception("streamed run of '%s' failed", tool.name)
            outcome["error"] = type(exc).__name__
        finally:
            outcome["finished"] = True

    started = time.monotonic()
    artifacts = 0
    delivered = 0
    stored_ids = []

    # The tool runs on a plain thread rather than inside an anyio task group:
    # a task group must not span a `yield` in an async generator (its cancel
    # scope would be entered and exited in different tasks), and this generator
    # yields for the whole life of the run. The concurrency cap is honoured
    # explicitly instead, with a borrower token so the acquire and the release
    # need not happen in the same task.
    borrower = object()
    await limiter.acquire_on_behalf_of(borrower)
    worker = threading.Thread(target=work, name=f"stream-{tool.name}", daemon=True)
    worker.start()

    try:
        yield _line({"event": "start", "tool": tool.name})
        last_spoke = time.monotonic()

        while True:
            try:
                event = emitter.events.get_nowait()
            except queue.Empty:
                if outcome.get("finished"):
                    break
                # Nothing to forward: hand the loop back rather than blocking
                # on the queue, and say something occasionally so the
                # connection stays provably alive. An idle TCP connection is
                # indistinguishable from a dead one.
                await anyio.sleep(_POLL_SECONDS)
                if time.monotonic() - last_spoke >= _KEEPALIVE_SECONDS:
                    yield _line(
                        {"event": "heartbeat", "elapsed": round(time.monotonic() - started, 1)}
                    )
                    last_spoke = time.monotonic()
                continue

            if event["event"] == EVENT_ARTIFACT:
                artifacts += 1
                packaged = await anyio.to_thread.run_sync(_package, event.get("path"), work_dir)
                if packaged is None:
                    continue
                try:
                    stored = await anyio.to_thread.run_sync(
                        transfer.store_result, packaged, media_type_of(packaged)
                    )
                except OSError:
                    # The run keeps going: this costs early delivery of one
                    # file, not the file itself.
                    logger.exception("could not store an artifact for streaming")
                    continue
                stored_ids.append(stored.result_id)
                delivered += 1
                # The server-side path never travels: the client gets a
                # reference to fetch, not a location on this machine.
                event = {key: value for key, value in event.items() if key != "path"}
                event["result_ref"] = stored.as_reference()

            yield _line(event)
            last_spoke = time.monotonic()

        if outcome.get("cancelled"):
            # Nothing is yielded here: the client that caused this is gone.
            logger.info("streamed run of '%s' cancelled by the client", tool.name)
        elif outcome.get("error"):
            yield _line(
                {
                    "event": "error",
                    "detail": outcome["error"],
                    "unavailable": bool(outcome.get("unavailable")),
                    "delivered": delivered,
                }
            )
        else:
            yield _line(
                {
                    "event": "done",
                    "artifacts": artifacts,
                    "delivered": delivered,
                    "duration_seconds": round(time.monotonic() - started, 2),
                }
            )
    except anyio.get_cancelled_exc_class():
        # The client hung up mid-stream. Tell the worker to stop at its next
        # emit point, then let the cancellation propagate.
        cancelled_by_client = True
        raise
    except GeneratorExit:
        # Same thing, the way an async generator usually learns about it.
        cancelled_by_client = True
        raise
    finally:
        emitter.cancelled.set()
        # The slot is given back here rather than after joining the worker: the
        # thread cannot be killed and only stops at its next emit, and blocking
        # the event loop waiting for it would be worse than briefly admitting
        # one run over the cap. What actually protects the card is the tool's
        # own GPU semaphore, which the abandoned thread still holds until it
        # notices.
        limiter.release_on_behalf_of(borrower)
        logger.info(
            "endpoint=/run/%s streamed=%d artifact(s) duration=%.2fs%s",
            tool.name,
            delivered,
            time.monotonic() - started,
            " (client gone)" if cancelled_by_client or outcome.get("cancelled") else "",
        )
        # **Only when nobody is coming for them.** A finished run has just
        # announced its last artifact, and the client may still be pulling it:
        # deleting here would take a result away between the announcement and
        # the GET. On a normal run the client DELETEs each reference once it
        # has it, exactly as for `X-Result-Delivery: reference`, and the idle
        # reaper (TRANSFER_TTL_SECONDS) is the backstop.
        if cancelled_by_client or outcome.get("cancelled"):
            for result_id in stored_ids:
                transfer.discard_result(result_id)
        for path in cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)
