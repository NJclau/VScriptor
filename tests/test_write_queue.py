"""Write queue behavior characterization tests."""

from __future__ import annotations

from queue import Empty, Queue


def test_queue_import_and_instantiation() -> None:
    write_queue: Queue[tuple[str, dict]] = Queue()
    assert isinstance(write_queue, Queue)
    assert write_queue.qsize() == 0


def test_queue_enqueue_and_drain_behavior() -> None:
    write_queue: Queue[str] = Queue()

    write_queue.put("job-1")
    write_queue.put("job-2")

    drained: list[str] = []
    while True:
        try:
            drained.append(write_queue.get_nowait())
            write_queue.task_done()
        except Empty:
            break

    assert drained == ["job-1", "job-2"]
    assert write_queue.qsize() == 0


def test_queue_processing_order_is_sequential_fifo() -> None:
    write_queue: Queue[int] = Queue()
    for value in range(5):
        write_queue.put(value)

    processed: list[int] = []
    while not write_queue.empty():
        item = write_queue.get()
        processed.append(item)
        write_queue.task_done()

    assert processed == [0, 1, 2, 3, 4]
