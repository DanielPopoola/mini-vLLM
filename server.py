import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from inference import metrics
from models import GenerationRequest, Request
from scheduler import active_requests, scheduler_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scheduler_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerationRequest):
    request = Request(
        request_id=str(uuid.uuid4()),
        prompt=req.prompt,
        max_tokens=req.max_tokens,
    )

    active_requests.append(request)

    async def token_stream():
        while not request.finished or not request.token_queue.empty():
            token = await request.token_queue.get()
            yield f"data: {token}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.get("/metrics")
async def get_metrics():
    return metrics.to_dict()
