## Product Requirements Document — Mini-vLLM

**Project Name:** Mini-vLLM
**Goal:** Build a minimal, production-style LLM inference engine that demonstrates batching, KV-cache management, prefill/decode separation, and token streaming.

### 1. Problem Statement

Modern LLM serving systems are not “model.forward() in a loop”.
They must:

* Serve multiple users concurrently
* Separate **prefill** (prompt encoding) from **decode** (token generation)
* Reuse **KV cache** efficiently
* Batch requests dynamically
* Stream tokens with low latency

Most ML engineers never build this layer. This project demonstrates mastery of the **core inference infrastructure layer** used in vLLM, TGI, SGLang, and TensorRT-LLM.

---

## 2. Target User

Primary:

* ML Infra / AI Engineer interviewer
* Backend engineer evaluating systems skill

Secondary:

* Yourself — learning GPU scheduling, batching, and memory behavior

This is a **portfolio / learning system**, not a commercial product.

---

## 3. Non-Goals (Important to keep scope sane)

This project will NOT:

* Implement paged attention (too complex)
* Support multi-GPU or distributed inference
* Implement quantization or LoRA
* Achieve production throughput

The goal is **conceptual correctness**, not peak performance.

---

## 4. Core Learning Objectives (the real value)

By completing this project you must understand:

1. Prefill vs Decode separation
2. How KV cache works and why it dominates memory
3. Continuous batching vs static batching
4. Latency vs throughput trade-offs
5. Token streaming mechanics
6. Simple GPU scheduling

If these six are implemented, the project succeeded.

---

## 5. High-Level Architecture

```
Client
  ↓ (HTTP streaming)
Inference Server (FastAPI / Go)
  ↓
Request Queue
  ↓
Batch Scheduler
  ↓
Prefill Engine  ──┐
                  ├─→ Decoder Loop → Token Stream
KV Cache Manager ─┘
  ↓
Transformer Model (HF)
```

---

## 6. Functional Requirements

### 6.1 Model Support

* Use a small open model:

  * `gpt2`, `distilgpt2`, or `TinyLlama-1.1B`

Requirements:

* Load model once at startup
* Run on GPU if available, else CPU fallback

---

### 6.2 API Interface

Implement:

#### Endpoint: `POST /generate`

Request:

```json
{
  "prompt": "Explain transformers simply",
  "max_tokens": 64,
  "temperature": 0.8
}
```

Response:

* **Streaming tokens** via SSE or chunked HTTP

Hard requirement:

* Tokens must stream as they are generated (not buffered)

---

### 6.3 Request Queue

Implement an in-memory queue:

Each request has:

* request_id
* prompt
* current_position
* kv_cache reference
* finished flag

Must support:

* Multiple concurrent users
* Dynamic insertion while decoding is running

---

### 6.4 Prefill Phase (Critical Concept)

Behavior:

* When a new request arrives:

  * Run the full prompt through the model once
  * Store:

    * Hidden states
    * KV cache for each layer

Rules:

* Prefill is done **once per request**
* Prefill requests can be batched together

Deliverable:

* A function: `run_prefill(batch_of_prompts)`

This teaches:

> Prefill is expensive, decode is cheap.

---

### 6.5 KV Cache Manager (Core Infra Skill)

Implement a simple KV cache:

Structure:

* Per request:

  * For each layer:

    * key tensor
    * value tensor

Operations:

* Allocate cache on prefill
* Append new keys/values on each decode step
* Free cache when request finishes

Hard requirement:

* No recomputation of past tokens

Deliverable:

* Class: `KVCacheManager`

This is the **heart of LLM serving**.

---

### 6.6 Decode Loop (Token Generation Engine)

Implement a decode scheduler:

Loop:

1. Select all active requests
2. Build a batch of their **last tokens**
3. Run one forward step
4. For each request:

   * Sample next token
   * Append to output
   * Update KV cache
   * Mark finished if EOS

Properties:

* Continuous batching:
  New requests can join between steps

This teaches:

> How one GPU step serves many users.

---

### 6.7 Streaming Engine

For each request:

* As soon as a token is produced:

  * Send it immediately to the client
* Use:

  * Server-Sent Events (FastAPI)
  * Or chunked responses

Must handle:

* Multiple streams concurrently
* Client disconnect cleanup

---

### 6.8 Simple Scheduler

Implement a naive scheduler:

Policy:

* Max batch size = N (configurable)
* At each decode step:

  * Take up to N active requests
  * Batch them

Optional upgrade:

* Separate prefill queue and decode queue

This teaches:

> Throughput comes from batching, not faster models.

---

## 7. Non-Functional Requirements

### Performance

Target (realistic):

* Latency: first token < 500ms (on small model)
* Throughput: 5–20 tokens/sec total

No strict SLA — correctness over speed.

---

### Reliability

Must handle:

* Client disconnect mid-generation
* Finished request cleanup
* GPU OOM gracefully (catch + reset)

---

### Observability (Very Important for Interviews)

Implement minimal metrics:

Expose `/metrics` or logs:

* Active requests
* Batch size per step
* Tokens/sec
* GPU memory used (if CUDA)

Even simple logging is fine.

---

## 8. Tech Stack Recommendation (for you)

Given your background:

**Language:** Python (first version)
**Framework:** FastAPI
**Model:** HuggingFace Transformers
**Streaming:** SSE via FastAPI
**GPU:** Optional (CPU acceptable initially)

Later optional rewrite:

* Go frontend + Python inference worker via gRPC

---

## 9. Milestones (Critical — achievable path)

### Milestone 1 — Baseline streaming server (1–2 days)

* Load model
* Single request streaming generation
* No batching, no cache reuse

Goal: working end-to-end chat.

---

### Milestone 2 — Prefill + KV cache (2–3 days)

* Implement prefill phase
* Implement KV cache
* Decode using cache

At this point:

> You already understand 70% of vLLM.

---

### Milestone 3 — Continuous batching (3–5 days)

* Request queue
* Batch multiple active decodes
* Handle joining mid-generation

This is the **main infra learning step**.

---

### Milestone 4 — Scheduler + metrics (2 days)

* Batch size control
* Basic throughput metrics
* Cleanup logic

---

## 10. Deliverables (What goes in your portfolio)

Your repo must contain:

### A. README (this matters a lot)

Include:

* Architecture diagram
* Explanation of:

  * Prefill vs decode
  * KV cache
  * Continuous batching
* Performance numbers

This shows systems thinking.

---

### B. Code Structure (suggested)

```
mini_vllm/
  server.py        # FastAPI + streaming
  scheduler.py     # batching + loop
  prefill.py       # prompt encoding
  decode.py        # token loop
  kv_cache.py      # cache manager
  model.py         # HF wrapper
```

---

### C. Demo

Record:

* 2–3 concurrent clients
* Streaming tokens
* Logs showing batching

This is extremely impressive in interviews.
