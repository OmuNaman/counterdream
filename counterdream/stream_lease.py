"""Reconnect idle workers without extending the viewer's paid allocation window."""
import asyncio
import time


class AllocationEnded(RuntimeError):
    pass


class StreamLease:
    def __init__(self, start, running, cancel, seconds=1800, max_starts=3, clock=time.monotonic):
        self.start, self.running, self.cancel = start, running, cancel
        self.seconds, self.max_starts, self.clock = seconds, max_starts, clock
        self.deadline = None
        self.starts = 0
        self.call = self.destination = None
        self.lock = asyncio.Lock()

    async def get(self):
        async with self.lock:
            if self.deadline is None:
                self.deadline = self.clock() + self.seconds
            remaining = int(self.deadline - self.clock())
            if remaining <= 0:
                await self.close()
                raise AllocationEnded("This 30-minute viewer allocation has ended.")
            if self.call is not None:
                if await self.running(self.call):
                    return self.destination
                await self.close()
            if self.starts >= self.max_starts:
                raise AllocationEnded("This viewer has used its three GPU starts. Start a new viewer session to continue.")
            self.starts += 1
            self.call, self.destination = await self.start(remaining)
            return self.destination

    async def close(self):
        if self.call is not None:
            await self.cancel(self.call)
            self.call = self.destination = None
