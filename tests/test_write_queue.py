"""Async write queue behavior characterization tests."""

import asyncio
import pytest
from state.write_queue import configure_flush, enqueue_write, start_queue_worker, queue_depth, get_queue
import state.write_queue

@pytest.fixture(autouse=True)
def reset_queue():
    state.write_queue._write_queue = None

@pytest.mark.asyncio
async def test_queue_import_and_instantiation() -> None:
    # _write_queue is module-level, we just check imports and basic depth
    assert queue_depth() >= 0

@pytest.mark.asyncio
async def test_queue_enqueue_and_drain_behavior() -> None:
    pushes = []
    configure_flush(lambda job_id, record: pushes.append((job_id, record)))

    worker_task = asyncio.create_task(start_queue_worker())

    await enqueue_write("job-1", {"status": "running"})
    await enqueue_write("job-2", {"status": "complete"})

    # Give some time for worker to process
    await asyncio.sleep(0.1)

    assert len(pushes) == 2
    assert pushes[0][0] == "job-1"
    assert pushes[1][0] == "job-2"

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_queue_processing_order_is_sequential_fifo() -> None:
    pushes = []
    configure_flush(lambda job_id, record: pushes.append(job_id))

    worker_task = asyncio.create_task(start_queue_worker())

    for i in range(5):
        await enqueue_write(f"job-{i}", {"val": i})

    await asyncio.sleep(0.1)

    assert pushes == ["job-0", "job-1", "job-2", "job-3", "job-4"]

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
