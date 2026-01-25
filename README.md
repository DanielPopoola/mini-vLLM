# Mini-vLLM: Learning LLM Inference Infrastructure

A minimal, educational implementation of a production-style LLM inference engine that demonstrates the core concepts behind systems like vLLM, TGI, and TensorRT-LLM.

## Table of Contents
- [What I Built](#what-i-built)
- [Why This Project Matters](#why-this-project-matters)
- [Key Concepts](#key-concepts)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
- [What I Learned](#what-i-learned)
- [Performance](#performance)
- [How to Run](#how-to-run)
- [Current Limitations & Next Steps](#current-limitations--next-steps)

---

## What I Built
This project is a scoped implementation of vLLM inference engine. The focus is to understand the core of LLM infererence which involves kv cache mgt, prefill/decode separation and continuous batching.

This project implements:
- ✅ **KV Cache Management** - Efficient reuse of computed attention states
- ✅ **Prefill/Decode Separation** - Two-phase generation pipeline
- ✅ **Continuous Batching** - Dynamic batching of concurrent requests
- ✅ **Token Streaming** - Real-time response delivery via Server-Sent Events
- ✅ **Request Scheduling** - Queue management with configurable batch sizes
- ✅ **Basic Metrics** - Observability into system performance

---

## Why This Project Matters

### The Problem with Naive LLM Serving

Most people think LLM inference is simply:
```python
while not done:
    next_token = model(all_previous_tokens)
```

But this approach has critical inefficiencies:

**Example:** Generating 100 tokens from the prompt "Hello"
- Token 1: Process "Hello" (1 token)
- Token 2: Process "Hello world" (2 tokens)
- Token 3: Process "Hello world how" (3 tokens)
- ...
- Token 100: Process "Hello world how..." (100 tokens)

**Total computation:** ~5,000 token processings for 100 tokens of output!

The word "Hello" gets recomputed 100 times, even though its meaning doesn't change.

### The Solution: Production Inference Systems

Modern LLM servers solve this through:
1. **KV Caching** - Store computed states, never recompute
2. **Batching** - Process multiple users on one GPU call
3. **Smart Scheduling** - Maximize throughput while maintaining low latency


This project aligns with my goals in AI infra and I needed to understand LLM inference at scale and how GPUs are able to process thousands of requests concurrently.

---

## Key Concepts

### 1. KV Cache: The Memory Optimization

**What it is:**
KV cache is simply a storage system that saves the key and value vectors for every token already
processed so that they are not re-computed.

**How it works:**
When the transformer processes a token, it computes Key and Value vectors for each attention layer. These vectors represent "how this token should be attended to by future tokens."

**Key insight:** Once computed for a token, these K/V vectors never change (in autoregressive generation).

**Without KV cache:**
```
Generate token 1: Process [prompt]
Generate token 2: Process [prompt, token1]  ← Recomputes prompt!
Generate token 3: Process [prompt, token1, token2]  ← Recomputes everything!
```

**With KV cache:**
```
Prefill: Process [prompt] → Save K/V cache
Generate token 1: Process [token1] + reuse prompt's cache
Generate token 2: Process [token2] + reuse all previous cache
```

**Impact:** Reduces computation from O(n²) to O(n) where n = sequence length.

---

### 2. Prefill vs Decode: The Two-Phase Pipeline

**Prefill (happens once):**
- Input: The full prompt (N tokens)
- Computation: Process all N tokens through the model
- Output: First generated token + KV cache for all prompt tokens
- Characteristic: Computationally expensive, processes many tokens

**Decode (happens repeatedly):**
- Input: One new token + cached K/V states
- Computation: Process 1 token, reusing cached states
- Output: Next token + updated cache
- Characteristic: Computationally cheap, processes one token

This separation is important mainly because of two reasons
1. You can't batch prefill & decode together because their input dimensions are different
2. Prefill is far more computationally expensive than decode, so separating them allows the GPU to be used efficiently.

**Code structure:**
```python
# Prefill
outputs = model(full_prompt)
kv_cache = outputs.past_key_values

# Decode loop
for step in range(max_tokens):
    outputs = model(last_token, past_key_values=kv_cache)
    kv_cache = outputs.past_key_values  # Grow cache
    next_token = sample(outputs.logits)
```

---

### 3. Continuous Batching: Maximizing GPU Utilization

**The GPU Efficiency Problem:**

GPUs are designed for parallel computation. Processing requests sequentially wastes this capability:

```
Sequential (bad):
Request A: ████████ (10ms)
Request B:          ████████ (10ms)
Request C:                   ████████ (10ms)
Total: 30ms
```

```
Batched (good):
Requests A+B+C: ████████████ (15ms)
Total: 15ms
```

**How batching works:**
1. Collect all active requests
2. Pad their KV caches to the same sequence length
3. Create attention masks (to ignore padding)
4. Process all requests in one GPU forward pass
5. Split results back to individual requests

**The padding problem:**
```
Request A: [real, real, real, real, real] (length 5)
Request B: [real, real] (length 2)
Request C: [real, real, real, real, real, real, real] (length 7)

After padding to max_len=7:
Request A: [real, real, real, real, real, PAD, PAD]
Request B: [real, real, PAD, PAD, PAD, PAD, PAD]
Request C: [real, real, real, real, real, real, real]

Attention masks:
Request A: [1, 1, 1, 1, 1, 0, 0]  ← Model ignores padded positions
Request B: [1, 1, 0, 0, 0, 0, 0]
Request C: [1, 1, 1, 1, 1, 1, 1]
```
Batching helps to properly utilize the GPU's capabilities of parallel processing, although it introduces some overhead due to e.g context switching, it allows for very efficient processing.

**Continuous batching** means the batch composition changes dynamically:
- New requests join the batch when they arrive
- Finished requests leave the batch when done
- No need to wait for all requests to complete

---

## Architecture

### High-Level Flow

```
Client Request
     ↓
┌────────────────────────────┐
│   FastAPI Endpoint         │
│   - Create Request object  │
│   - Add to incoming queue  │
│   - Stream tokens back     │
└────────────────────────────┘
     ↓
┌────────────────────────────┐
│   Scheduler Loop           │
│   - Process new requests   │
│   - Separate prefill/decode│
│   - Batch decode requests  │
│   - Cleanup finished       │
└────────────────────────────┘
     ↓
┌────────────────────────────┐
│   Inference Engine         │
│   - Prefill: Full prompt   │
│   - Decode: Batched tokens │
│   - Manage KV caches       │
└────────────────────────────┘
     ↓
┌────────────────────────────┐
│   Token Queue              │
│   - Per-request queues     │
│   - Async communication    │
└────────────────────────────┘
```

### Component Breakdown

**`server.py`** - HTTP interface
- FastAPI endpoints for generation and metrics
- Creates Request objects
- Manages token streaming to clients

**`scheduler.py`** - Request orchestration
- Background loop processing active requests
- Separates prefill vs decode phase requests
- Applies batch size limits
- Cleans up finished requests

**`inference.py`** - Model execution
- `run_prefill()`: Process full prompts
- `batch_decode()`: Batch multiple decode operations
- KV cache padding and attention masking
- Token sampling

**`models.py`** - Data structures
- Request: Holds per-request state (cache, tokens, etc.)
- GenerationRequest: API input schema

**`metrics.py`** - Observability
- Tracks active requests, batch sizes
- Calculates tokens/second throughput

---

## Implementation Details

### Request Lifecycle

1. **Arrival**
   ```python
   request = Request(id, prompt, max_tokens)
   await incoming_requests.put(request)
   ```

2. **Prefill Phase**
   ```python
   if request.kv_cache is None:
       await run_prefill(request)
       # Computes first token + saves KV cache
   ```

3. **Decode Phase**
   ```python
   # Batched with other active requests
   await batch_decode([request_a, request_b, ...])
   # Generates next token, updates cache
   ```

4. **Token Streaming**
   ```python
   # Scheduler puts tokens in queue
   await request.token_queue.put(token)
   
   # Endpoint streams to client
   token = await request.token_queue.get()
   yield f"data: {token}\n\n"
   ```

5. **Completion**
   ```python
   if len(generated_tokens) >= max_tokens:
       request.finished = True
   # Cleaned up by scheduler
   ```

### KV Cache Batching Algorithm

The KV Cache batching algorithm is quite challenging because of the way the model expects its inputs. We need to turn the format from `request` -> #`layer ` -> `(key, value)` to `layer` -> `batch` -> `(key, value)`

```python
def batch_decode(requests):
    # 1. Find longest sequence
    max_len = max(r.kv_cache[0][0].size(2) for r in requests)
    
    # 2. Pad each request's cache to max_len
    for request in requests:
        current_len = request.kv_cache[0][0].size(2)
        padding_needed = max_len - current_len
        
        # Pad with zeros
        padded_cache = F.pad(cache, pad=(0, 0, 0, padding_needed))
        
        # Create mask: 1 for real, 0 for padding
        mask = [1]*current_len + [0]*padding_needed
    
    # 3. Concatenate along batch dimension
    batched_cache = torch.cat([r.cache for r in requests], dim=0)
    
    # 4. Single model call
    outputs = model(batched_input, past_key_values=batched_cache, 
                   attention_mask=batched_masks)
    
    # 5. Split results back
    for i, request in enumerate(requests):
        request.kv_cache = outputs.past_key_values[i:i+1]
```

---

## What I Learned

### Technical Insights

**1. The cost of recomputation**
The cost of recomputation is very expensive because of the full attention architecture, so various optimization methods must be made to improve performance. Without the KV cache, the cost of recomputing would grow exponentially

**2. Why batching matters**
Batching matters because we aim to efficiently process requests. If a request A comes in to the server, and while it's still processing another request B comes in to the server, with sequential
processing we would have to wait till request A finishes before processing request B, this can be extremely inefficient especially at scale.

**3. The prefill/decode dichotomy**
The prefill & decode stage do two different things and need different requirements, so it's optimal to handle both separately in the best optimal fashion.

**4. Async programming for inference**
Because of Python's limitations with threading, I had to use asyncio to setup a scheduler that would run continously polling for requests to process. My current implementation though it works isn't optimal as the model runs in the main event loop.

**5. Tensor manipulation at scale**
This can be very challenging mainly because of the data structures huggingface chose to represent model ouptuts. But it becomes intuitive once you see the pattern.

### Challenges Overcome

[YOUR WORDS: What was hard? How did you solve it?]

**Challenge 1: Understanding KV cache structure**

Understanding the KV cache layout and how it was nested. Solved it by tracing through each layer to figure out how to access & manipulate the cache.

**Challenge 2: Padding variable-length sequences**
This was especially difficult because of the `(last_left, last_right, second_to_last_left, second_to_last_right`, `...`) format that `pad` expected in `torch.nn.functional`. Overcame it by
reading the Pytorch docs

**Challenge 3: Async communication between scheduler and endpoints**
This was an issue of yielding control to the main event loop. Fixed it with `asyncio.sleep(0)` to yield control to the main event loop for `token_stream()` to run. 

---

### Performance Notes

Testing with GPT-2 (124M params) on CPU showed minimal batching speedup.
This is expected because:
- Small model processes quickly even sequentially
- Batching overhead (padding, mask creation) is proportional to benefit
- CPU has limited parallel processing vs GPU

**Where batching matters:** Larger models (7B+) on GPU with many concurrent users.

---

## How to Run

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn torch transformers
```

### Start Server

```bash
uvicorn server:app --reload
```

### Test Generation

```bash
# Single request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}' \
  --no-buffer

# Check metrics
curl http://localhost:8000/metrics
```

### Test Concurrent Batching

Open multiple terminals and run requests simultaneously to see batching in action:

```bash
# Terminal 1
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "Hello", "max_tokens": 30}' --no-buffer

# Terminal 2 (run immediately)
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "Goodbye", "max_tokens": 30}' --no-buffer

# Terminal 3 - Monitor metrics
watch -n 0.5 'curl -s http://localhost:8000/metrics'
```

---

## Current Limitations & Next Steps

### What's Implemented ✅

- KV cache per request
- Prefill/decode separation
- Continuous batching with padding
- Request queue and scheduling
- Token streaming via SSE
- Basic metrics

### What's Missing ⏳

**Not implemented:**
- [ ] **Batch prefill operations** - Currently prefills happen one-at-a-time
- [ ] **Temperature/top-p sampling** - Only greedy decoding (argmax)
- [ ] **Request prioritization** - FIFO only, no priority queues
- [ ] **Memory management** - No max cache size limits
- [ ] **Error handling** - Limited validation and error recovery

**Out of scope:**
- ❌ Paged attention (PagedAttention from vLLM)
- ❌ Multi-GPU distribution
- ❌ Quantization (INT8/INT4)
- ❌ Speculative decoding
- ❌ LoRA adapter support

---

## References & Inspiration

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Paged attention and continuous batching
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - KV cache implementation
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - Production serving patterns

---

## License

MIT License - Educational project for learning LLM inference internals.
