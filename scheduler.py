import asyncio

from inference import batch_decode, metrics, run_prefill
from models import Request

MAX_BATCH_SIZE = 8

active_requests: list[Request] = []


async def scheduler_loop():
    while True:
        if not active_requests:
            await asyncio.sleep(0.01)
            continue

        prefill_requests = [r for r in active_requests if r.kv_cache is None]
        decode_requests = [r for r in active_requests if r.kv_cache is not None]

        for request in prefill_requests:
            await run_prefill(request)

        if decode_requests:
            batch = decode_requests[:MAX_BATCH_SIZE]
            await batch_decode(batch)

        active_requests[:] = [r for r in active_requests if not r.finished]

        metrics.log_batch(
            len(decode_requests[:MAX_BATCH_SIZE]) if decode_requests else 0,
            len(active_requests),
        )

        await asyncio.sleep(0)
