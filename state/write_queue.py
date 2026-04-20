"""
Async write queue for Hub job state pushes.
Serializes writes to prevent rapid-succession races within a single job.
The queue is drained by a single background worker coroutine.

Usage in app.py:
    from state.write_queue import enqueue_write, start_queue_worker
    # On startup:
    asyncio.create_task(start_queue_worker())
    # Instead of direct job_store calls in hot path:
    await enqueue_write(job_id, record)
"""
import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

_write_queue: asyncio.Queue | None = None
_flush_fn: Callable[[str, dict], None] | None = None


def get_queue() -> asyncio.Queue:
    """Lazy initialization of the queue to ensure it is bound to the running loop."""
    global _write_queue
    if _write_queue is None:
        _write_queue = asyncio.Queue(maxsize=50)
    return _write_queue


def configure_flush(fn: Callable[[str, dict], None]) -> None:
    """Register the function that performs the actual Hub push."""
    global _flush_fn
    _flush_fn = fn


async def enqueue_write(job_id: str, record: dict) -> None:
    """Non-blocking enqueue. Raises if queue is full."""
    q = get_queue()
    if q.full():
        raise RuntimeError(f"write_queue: queue at capacity ({q.maxsize}).")
    await q.put((job_id, record))


async def start_queue_worker() -> None:
    """Drain the write queue sequentially."""
    if _flush_fn is None:
        raise RuntimeError("write_queue: no flush function configured.")
    q = get_queue()
    logger.info("write_queue: worker started")
    while True:
        job_id, record = await q.get()
        try:
            _flush_fn(job_id, record)
        except RuntimeError as e:
            logger.error(f"write_queue: push failed for {job_id}: {e}")
        except Exception as e:
            logger.error(f"write_queue: unexpected error for {job_id}: {e}")
        finally:
            q.task_done()


def queue_depth() -> int:
    """Returns current number of pending writes."""
    return _write_queue.qsize() if _write_queue else 0
