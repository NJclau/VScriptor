import asyncio

import pytest

from scheduler.admission import AdmissionController


@pytest.mark.asyncio
async def test_short_job_acquires_short_slot():
    admission = AdmissionController()

    slot = await admission.acquire(duration_secs=5.0)
    admission.release(slot)

    assert slot == "short"


@pytest.mark.asyncio
async def test_long_job_acquires_primary_slot():
    admission = AdmissionController()

    slot = await admission.acquire(duration_secs=120.0)
    admission.release(slot)

    assert slot == "primary"


@pytest.mark.asyncio
async def test_job_longer_than_cap_is_rejected_before_acquire():
    admission = AdmissionController()

    with pytest.raises(ValueError):
        await admission.acquire(duration_secs=100000.0)


@pytest.mark.asyncio
async def test_third_long_job_waits_until_primary_slot_released():
    admission = AdmissionController()
    slot1 = await admission.acquire(duration_secs=90.0)
    slot2 = await admission.acquire(duration_secs=95.0)

    pending = asyncio.create_task(admission.acquire(duration_secs=100.0))
    await asyncio.sleep(0.05)
    assert not pending.done()

    admission.release(slot1)
    slot3 = await pending

    admission.release(slot2)
    admission.release(slot3)
    assert slot3 == "primary"
