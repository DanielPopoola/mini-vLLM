import asyncio
import uuid

import torch
import torch.nn.functional as F
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

        prefill_requests = [r for r in active_requests if r.kv_cache is None]
        decode_requests = [r for r in active_requests if r.kv_cache is not None]

        if decode_requests:
            await batch_decode(decode_requests)

        for request in prefill_requests:
            input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                outputs = model(input_ids)

            request.kv_cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1)

            request.next_token_id = next_id
            token_text = tokenizer.decode(next_id, skip_special_tokens=True)
            await request.token_queue.put(token_text)
            request.generated_tokens.append(token_text)

            if (
                next_id.item() == tokenizer.eos_token_id
                or len(request.generated_tokens) >= request.max_tokens
            ):
                request.finished = True
        active_requests[:] = [r for r in active_requests if not r.finished]
        await asyncio.sleep(STREAM_DELAY)


async def batch_decode(requests):
    max_len = max(r.kv_cache[0][0].size(2) for r in requests)

    padded_caches = []
    attention_masks = []

    for request in requests:
        current_len = request.kv_cache[0][0].size(2)
        padding_needed = max_len - current_len

        attention_mask = torch.cat(
            [torch.ones(current_len), torch.zeros(padding_needed)]
        )
        attention_masks.append(attention_mask)

        padded_layers = []
        for layer_idx in range(len(request.kv_cache)):
            key = request.kv_cache[layer_idx][0]
            value = request.kv_cache[layer_idx][1]

            padded_key = F.pad(key, pad=(0, 0, 0, padding_needed), value=0.0)
            padded_value = F.pad(value, pad=(0, 0, 0, padding_needed), value=0.0)

            padded_layers.append((padded_key, padded_value))

        padded_cache = tuple(padded_layers)
        padded_caches.append(padded_cache)

    input_ids = torch.stack([r.next_token_id.unsqueeze(0) for r in requests])

    batched_attention_mask = torch.stack(attention_masks)

    batched_past_kv = []
    num_layers = len(padded_caches[0])

    for layer_idx in range(num_layers):
        keys = [cache[layer_idx][0] for cache in padded_caches]
        values = [cache[layer_idx][1] for cache in padded_caches]

        batched_keys = torch.cat(keys, dim=0)
        batched_values = torch.cat(values, dim=0)

        batched_past_kv.append((batched_keys, batched_values))

    batched_past_kv = tuple(batched_past_kv)

    with torch.no_grad():
        outputs = model(
            input_ids,
            past_key_values=batched_past_kv,
            attention_mask=batched_attention_mask,
        )

    update_caches = []
    for i in range(len(requests)):
        individual_cache = []
        for layer_idx in range(len(outputs.past_key_values)):
            batched_key, batched_value = outputs.past_key_values[layer_idx]

            individual_key = batched_key[i : i + 1]
            individual_value = batched_value[i : i + 1]

            individual_cache.append((individual_key, individual_value))

        update_caches.append(tuple(individual_cache))

    for i, request in enumerate(requests):
        logits = outputs.logits[i, -1, :]
        next_token_id = torch.argmax(logits, dim=-1)

        request.next_token_id = next_token_id
        request.kv_cache = update_caches[i]

        token_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
        await request.token_queue.put(token_text)
        request.generated_tokens.append(token_text)

        if (
            next_token_id.item() == tokenizer.eos_token_id
            or len(request.generated_tokens) >= request.max_tokens
        ):
            request.finished = True


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
