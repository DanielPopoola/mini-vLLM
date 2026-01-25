import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import Metrics
from models import Request

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
metrics = Metrics()


async def run_prefill(request: Request):
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
    metrics.log_token()

    if (
        next_id.item() == tokenizer.eos_token_id
        or len(request.generated_tokens) >= request.max_tokens
    ):
        request.finished = True


async def batch_decode(requests: list[Request]):
    if not requests:
        return

    # Step 1: Find max sequence length
    max_len = max(r.kv_cache[0][0].size(2) for r in requests)

    # Step 2: Pad caches and create masks
    padded_caches = []
    attention_masks = []

    for request in requests:
        current_len = request.kv_cache[0][0].size(2)
        padding_needed = max_len - current_len

        # Create attention mask
        attention_mask = torch.cat(
            [torch.ones(current_len), torch.zeros(padding_needed)]
        )
        attention_masks.append(attention_mask)

        # Pad KV cache
        padded_layers = []
        for layer_idx in range(len(request.kv_cache)):
            key, value = request.kv_cache[layer_idx]

            padded_key = F.pad(key, pad=(0, 0, 0, padding_needed), value=0.0)
            padded_value = F.pad(value, pad=(0, 0, 0, padding_needed), value=0.0)

            padded_layers.append((padded_key, padded_value))

        padded_caches.append(tuple(padded_layers))

    # Step 3: Batch everything
    input_ids = torch.stack([r.next_token_id.unsqueeze(0) for r in requests], dim=0)
    batched_attention_mask = torch.stack(attention_masks)

    # Batch KV caches
    batched_past_kv = []
    num_layers = len(padded_caches[0])

    for layer_idx in range(num_layers):
        keys = [cache[layer_idx][0] for cache in padded_caches]
        values = [cache[layer_idx][1] for cache in padded_caches]

        batched_key = torch.cat(keys, dim=0)
        batched_value = torch.cat(values, dim=0)

        batched_past_kv.append((batched_key, batched_value))

    batched_past_kv = tuple(batched_past_kv)

    # Step 4: Model call
    with torch.no_grad():
        outputs = model(
            input_ids,
            past_key_values=batched_past_kv,
            attention_mask=batched_attention_mask,
        )

    # Step 5: Extract individual caches
    updated_caches = []
    for i in range(len(requests)):
        individual_cache = []
        for layer_idx in range(len(outputs.past_key_values)):
            batched_key, batched_value = outputs.past_key_values[layer_idx]
            individual_key = batched_key[i : i + 1]
            individual_value = batched_value[i : i + 1]
            individual_cache.append((individual_key, individual_value))

        updated_caches.append(tuple(individual_cache))

    # Step 6: Update each request
    for i, request in enumerate(requests):
        logits = outputs.logits[i, -1, :]
        next_token_id = torch.argmax(logits, dim=-1)

        request.next_token_id = next_token_id
        request.kv_cache = updated_caches[i]

        token_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
        await request.token_queue.put(token_text)
        request.generated_tokens.append(token_text)
        metrics.log_token()

        if (
            next_token_id.item() == tokenizer.eos_token_id
            or len(request.generated_tokens) >= request.max_tokens
        ):
            request.finished = True
