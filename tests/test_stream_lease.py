import asyncio

import pytest

from counterdream.stream_lease import AllocationEnded, StreamLease


def test_idle_reconnect_replaces_worker_without_extending_cost_deadline():
    async def scenario():
        now, alive, starts, cancelled = [100.], set(), [], []
        async def start(seconds):
            call = len(starts) + 1
            starts.append(seconds)
            alive.add(call)
            return call, f"destination-{call}"
        async def running(call):
            return call in alive
        async def cancel(call):
            cancelled.append(call)
            alive.discard(call)
        lease = StreamLease(start, running, cancel, clock=lambda: now[0])
        assert lease.deadline is None  # Metadata and opening the page cost no GPU.
        assert await asyncio.gather(lease.get(), lease.get()) == ["destination-1"]*2
        assert starts == [1800]
        alive.clear()  # The remote 90-second idle timeout terminated the worker.
        now[0] += 100
        assert await lease.get() == "destination-2"
        assert starts == [1800, 1700] and cancelled == [1]
        now[0] = 1900
        with pytest.raises(AllocationEnded, match="30-minute"):
            await lease.get()
        assert cancelled == [1, 2] and len(starts) == 2
    asyncio.run(scenario())


def test_repeated_gpu_failures_cannot_create_unbounded_allocations():
    async def scenario():
        starts = []
        async def start(seconds):
            starts.append(seconds)
            raise RuntimeError("startup failed")
        async def unused(call):
            raise AssertionError("No worker was retained")
        lease = StreamLease(start, unused, unused, clock=lambda: 0)
        for _ in range(3):
            with pytest.raises(RuntimeError, match="startup failed"):
                await lease.get()
        with pytest.raises(AllocationEnded, match="three GPU starts"):
            await lease.get()
        assert len(starts) == 3
    asyncio.run(scenario())
