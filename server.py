import asyncio
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

STREAM_DELAY = 0

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int


class Request:
    def __init__(self, request_id, prompt: str, max_tokens: int):
        self.request_id = request_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.generated_tokens = []
        self.kv_cache = None
        self.finished: bool = False

        self.new_token_id = None
        self.token_queue = asyncio.Queue()


active_requests: list[Request] = []


async def scheduler_loop():
    while True:
        if not active_requests:
            await asyncio.sleep(0.01)
            continue

        for request in active_requests:
            if request.kv_cache is None:
                input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids
                with torch.no_grad():
                    outputs = model(input_ids)

                request.kv_cache = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_id = torch.argmax(logits, dim=-1)

            else:
                with torch.no_grad():
                    outputs = model(
                        request.new_token_id.unsqueeze(1),
                        past_key_values=request.kv_cache,
                    )

                request.kv_cache = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_id = torch.argmax(logits, dim=-1)

            request.new_token_id = next_id

            if (
                next_id.item() == tokenizer.eos_token_id
                or len(request.generated_tokens) >= request.max_tokens
            ):
                request.finished = True
            else:
                token_text = tokenizer.decode(next_id, skip_special_tokens=True)
                await request.token_queue.put(token_text)
                request.generated_tokens.append(token_text)

        # Clean up finished requests
        active_requests[:] = [r for r in active_requests if not r.finished]
        await asyncio.sleep(0)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler_loop())


@app.post("/generate")
async def generate(req: GenerationRequest):
    request = Request(
        request_id=str(uuid.uuid4()), prompt=req.prompt, max_tokens=req.max_tokens
    )

    active_requests.append(request)

    async def token_stream():
        # Main loop: get tokens while generating
        while not request.finished:
            token = await request.token_queue.get()
            yield f"data: {token}\n\n"

        # After finished, get any remaining items
        while not request.token_queue.empty():
            token = await request.token_queue.get()
            yield f"data: {token}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
