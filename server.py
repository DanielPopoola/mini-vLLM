import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

STREAM_DELAY = 0

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int


async def event_generator(prompt: str, max_tokens: int):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids)

    past_kv_cache = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)

    for _ in range(max_tokens):
       token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
       yield f"data: {token_text}\n\n"

       with torch.no_grad():
           outputs = model(next_token_id, past_key_values=past_kv_cache)

       past_kv_cache = outputs.past_key_values

       logits = outputs.logits[:, -1, :]
       next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
       await asyncio.sleep(STREAM_DELAY)

@app.post("/generate")
async def generate(req: GenerationRequest):
    return StreamingResponse(
        event_generator(req.prompt, req.max_tokens),
        media_type='text/event-stream'
    )
