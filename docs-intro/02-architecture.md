# 2. Architecture in Detail

This document describes vLLM's runtime structure: the layers, the processes, the control
loop, and the data that crosses each boundary.

---

## 2.1 The five layers

```mermaid
flowchart TB
    subgraph L1["① Entrypoint layer — vllm/entrypoints/"]
        E1["LLM (offline batch)<br/>entrypoints/llm.py:62"]
        E2["FastAPI app + routers<br/>entrypoints/launchers/api_server/"]
        E3["Protocol handlers<br/>entrypoints/openai/{chat_completion,completion,responses}/"]
    end

    subgraph L2["② Engine frontend — vllm/v1/engine/"]
        F1["InputProcessor<br/>tokenize, validate, MM hashing"]
        F2["AsyncLLM / LLMEngine<br/>per-request asyncio queues"]
        F3["OutputProcessor + IncrementalDetokenizer<br/>token ids → text, stop strings"]
        F4["EngineCoreClient<br/>ZMQ + msgpack transport"]
    end

    subgraph L3["③ Engine core — vllm/v1/engine/core.py, vllm/v1/core/"]
        C1["EngineCore busy loop"]
        C2["Scheduler"]
        C3["KVCacheManager → KVCacheCoordinator → BlockPool"]
        C4["StructuredOutputManager, EncoderCacheManager"]
    end

    subgraph L4["④ Execution layer — vllm/v1/executor/, vllm/v1/worker/"]
        X1["Executor: Uniproc / Multiproc / Ray"]
        X2["Worker (per rank)"]
        X3["GPUModelRunner: batch prep, forward, sample"]
    end

    subgraph L5["⑤ Model + kernel layer"]
        M1["Model definitions<br/>vllm/model_executor/models/"]
        M2["Layers: Linear, MoE, Attention, RoPE, norms<br/>vllm/model_executor/layers/"]
        M3["Attention backends<br/>vllm/v1/attention/backends/"]
        M4["CUDA/HIP kernels: csrc/ · Triton kernels"]
    end

    L1 --> L2 --> L3 --> L4 --> L5
```

**The boundary that matters most is between ③ and ④.** Everything above it is Python
bookkeeping about *which tokens to compute*. Everything below it is "run this exact batch
on this device". The contract across it is a single object, `SchedulerOutput`
(`vllm/v1/core/sched/output.py:241`), and a single return value, `ModelRunnerOutput`.

---

## 2.2 Process topology

vLLM is multi-process by default. The reason is the GIL: tokenisation, detokenisation, and
HTTP serialisation are CPU-heavy Python, and if they share a thread with the engine's
schedule/execute loop they steal time from the GPU. So the frontend and the engine core run
in **separate OS processes** connected by ZeroMQ.

```mermaid
flowchart TB
    client(["HTTP clients"])

    subgraph FE["Frontend process(es) — API server"]
        direction TB
        H["uvicorn + FastAPI"]
        IP["InputProcessor<br/>tokenizer, MM processor"]
        AL["AsyncLLM"]
        OP["OutputProcessor<br/>detokenizer, stop strings"]
        CC["AsyncMPClient"]
        H --> IP --> AL --> CC
        CC --> OP --> H
    end

    subgraph EC["EngineCore process (one per DP rank)"]
        direction TB
        IQ["input_queue (thread-fed)"]
        BL["run_busy_loop()"]
        SCH["Scheduler"]
        OQ["output_queue (thread-drained)"]
        IQ --> BL --> SCH
        BL --> OQ
    end

    subgraph WK["Worker processes (TP × PP ranks)"]
        direction LR
        W0["rank 0<br/>GPUModelRunner"]
        W1["rank 1"]
        Wn["rank N"]
    end

    DPC["DPCoordinator process<br/>(only when DP > 1)"]

    client <--> H
    CC <-->|"ZMQ: EngineCoreRequest / EngineCoreOutputs<br/>msgpack"| IQ
    OQ -.-> CC
    BL <-->|"MessageQueue: shared memory ring + ZMQ fallback"| W0
    W0 <--> W1
    W1 <--> Wn
    EC <-.-> DPC
```

### Who decides the topology

| Decision | Code | Rule |
| --- | --- | --- |
| In-process vs. separate engine process | `EngineCoreClient.make_client`, `vllm/v1/engine/core_client.py:111` | `InprocClient` for synchronous single-process use; `SyncMPClient`/`AsyncMPClient` otherwise. `asyncio_mode` without `multiprocess_mode` is explicitly unsupported |
| Which executor | `Executor.get_class`, `vllm/v1/executor/abstract.py:66` | `uni` → `UniProcExecutor` (world size 1, worker in-process), `mp` → `MultiprocExecutor`, `ray` → `RayExecutorV2`, `external_launcher` → `ExecutorWithExternalLauncher` |
| Which client under DP | `EngineCoreClient.make_async_mp_client`, `vllm/v1/engine/core_client.py:153` | DP=1 → `AsyncMPClient`; DP>1 internal LB → `DPLBAsyncMPClient`; DP>1 external LB → `DPAsyncMPClient` (one client per rank) |

### Why the engine core is single-threaded

`EngineCore.run_busy_loop` (`vllm/v1/engine/core.py:1794`) is deliberately a plain `while`
loop with no locks:

```python
while self._handle_shutdown():
    # 1) Poll the input queue until there is work to do.
    self._process_input_queue()
    self._maybe_publish_request_counts()
    # 2) Step the engine core and return the outputs.
    self._process_engine_step()
    self._maybe_publish_request_counts()
```

Request arrival and output delivery are pushed onto **background threads** that only touch
`queue.Queue` objects (`process_input_sockets` at `vllm/v1/engine/core.py:2171`). The loop
itself never blocks on IO while a GPU step is outstanding, and all mutation of scheduler
state happens on one thread, which removes a whole class of races.

---

## 2.3 The control loop in detail

```mermaid
sequenceDiagram
    autonumber
    participant FE as Frontend (AsyncLLM)
    participant IQ as input_queue thread
    participant EC as EngineCore loop
    participant SC as Scheduler
    participant KV as KVCacheManager
    participant EX as Executor
    participant MR as GPUModelRunner (rank 0)
    participant OQ as output_queue thread

    FE->>IQ: EngineCoreRequest (ZMQ, msgpack)
    IQ->>EC: input_queue.put(ADD, req)
    EC->>EC: _process_input_queue()
    EC->>SC: add_request(Request)

    loop every engine step
        EC->>SC: schedule(throttle_prefills)
        SC->>KV: get_computed_blocks(req)  %% prefix cache lookup
        KV-->>SC: cached blocks + num hit tokens
        SC->>KV: allocate_slots(req, num_new_tokens, ...)
        KV-->>SC: new blocks  (or None → preempt a victim)
        SC-->>EC: SchedulerOutput
        EC->>EX: execute_model(scheduler_output, non_block=True)
        EC->>SC: get_grammar_bitmask(scheduler_output)
        EX->>MR: SchedulerOutput over shared-memory MessageQueue
        MR->>MR: update req state, prepare_inputs, prepare_attn
        MR->>MR: forward (torch.compile / CUDA graph)
        MR->>MR: sample_tokens(grammar_output)
        MR-->>EX: ModelRunnerOutput
        EX-->>EC: future.result()
        EC->>SC: update_from_output(scheduler_output, model_output)
        SC-->>EC: dict[client_idx, EngineCoreOutputs]
        EC->>OQ: output_queue.put(...)
    end

    OQ-->>FE: EngineCoreOutputs (ZMQ)
    FE->>FE: OutputProcessor: detokenize, stop checks, stream
```

The step function itself is only 30 lines (`EngineCore.step`,
`vllm/v1/engine/core.py:700`). Note the ordering trick on lines 713–718: `execute_model` is
launched **non-blocking**, then the grammar bitmask for structured-output requests is
computed on the CPU *while the GPU runs the forward pass*, and only then is the result
awaited and `sample_tokens` invoked. CPU grammar work is overlapped with GPU compute for
free.

### Two step functions

| Function | When used | Behaviour |
| --- | --- | --- |
| `EngineCore.step` (`core.py:700`) | default | One schedule → one execute → one update, strictly serialised |
| `EngineCore.step_with_batch_queue` (`core.py:745`) | async scheduling / PP | Keeps a `batch_queue` of in-flight steps; schedules step *N+1* before step *N*'s output has been consumed, so the CPU scheduling cost overlaps GPU execution |

`step_with_batch_queue` is where pipeline parallelism gets its bubbles filled: with
`batch_queue_size > 1`, filling the queue takes priority over draining it
(see the docstring at `core.py:748`).

---

## 2.4 The scheduler's contract: `SchedulerOutput`

`vllm/v1/core/sched/output.py:241`. This is the single most important data structure to
understand, because it *is* the plan for one forward pass.

```mermaid
classDiagram
    class SchedulerOutput {
        scheduled_new_reqs: list[NewRequestData]
        scheduled_cached_reqs: CachedRequestData
        num_scheduled_tokens: dict[req_id, int]
        total_num_scheduled_tokens: int
        scheduled_spec_decode_tokens: dict[req_id, list[int]]
        scheduled_encoder_inputs: dict[req_id, list[int]]
        finished_req_ids: set[str]
        free_encoder_mm_hashes: list[str]
        structured_output_request_ids: dict
        kv_connector_metadata
    }
    class NewRequestData {
        req_id
        prompt_token_ids
        mm_features
        sampling_params
        block_ids: tuple[list[int], ...]
        num_computed_tokens
        lora_request
    }
    class CachedRequestData {
        req_ids: list[str]
        resumed_from_preemption: list[bool]
        new_token_ids: list[list[int]]
        new_block_ids: list
        num_computed_tokens: list[int]
    }
    SchedulerOutput --> NewRequestData
    SchedulerOutput --> CachedRequestData
```

The split between `scheduled_new_reqs` and `scheduled_cached_reqs` is the "persistent
batch" optimisation: a **new** request ships its full prompt and parameters once, and from
then on each step ships only the **delta** (new token ids, newly allocated block ids,
updated computed-token count). This keeps per-step serialisation cost proportional to the
number of *new tokens*, not to context length.

---

## 2.5 The worker side

```mermaid
flowchart TB
    SO["SchedulerOutput arrives<br/>over MessageQueue"]

    subgraph state["1 · Persistent state update — model_runner.py:1859-1871"]
        S1["finish_requests()<br/>drop finished rows"]
        S2["add_requests()<br/>allocate batch rows for new reqs"]
        S3["update_requests()<br/>append new token ids, new block ids"]
        S4["block_tables.apply_staged_writes()<br/>one H2D copy for all block-table edits"]
    end

    subgraph dispatch["2 · Shape dispatch — model_runner.py:1884-1932"]
        D1["gather_batch_req_state()"]
        D2["dispatch_cg_and_sync_dp()<br/>pick CUDA graph bucket;<br/>all-reduce agreement across DP ranks"]
    end

    subgraph prep["3 · Input preparation — model_runner.py:1946-1949"]
        P1["prepare_inputs()<br/>input_ids, positions, logits_indices"]
        P2["prepare_attn()<br/>block_tables + slot_mapping per KV group"]
        P3["AttentionMetadataBuilder.build()<br/>per-backend metadata"]
    end

    subgraph fwd["4 · Forward"]
        F1["set_forward_context(attn_metadata, ...)"]
        F2["compiled model / replayed CUDA graph"]
        F3["Attention layers read metadata from the forward context"]
    end

    subgraph samp["5 · Sample — model_runner.py:2262"]
        G1["apply grammar bitmask"]
        G2["logits processors, penalties"]
        G3["Sampler.__call__ → token ids"]
        G4["rejection sampler (if spec decode)"]
    end

    SO --> state --> dispatch --> prep --> fwd --> samp --> OUT["ModelRunnerOutput"]
```

Two things are worth calling out.

**Attention metadata travels through a context variable, not through function arguments.**
`set_forward_context` (`vllm/forward_context.py:274`) installs a `ForwardContext`
(`vllm/forward_context.py:137`) holding the per-layer attention metadata; each `Attention`
module fetches it with `get_forward_context()` inside its `forward`
(`vllm/model_executor/layers/attention/attention.py:528`). This is what lets model
definitions be written as plain `nn.Module`s with a `(hidden_states, positions)` signature
— and it is what makes them `torch.compile`-able without the compiler seeing dynamic
metadata objects.

**Shape dispatch happens before input preparation.** `dispatch_cg_and_sync_dp` decides
which CUDA-graph bucket (padded token count) this step will use and, under data
parallelism, **all-reduces that decision across DP ranks** so every rank runs the same
graph. If one rank diverged, a collective inside the model would deadlock. This is why so
many DP bugs look like hangs rather than wrong answers.

---

## 2.6 Memory layout and startup

Before the first request, the engine has to decide how many KV blocks exist. That number is
a function of free device memory after weights and activations, so it must be *measured*.

```mermaid
flowchart TB
    A["Load model weights on each rank"]
    B["profile_run(): dummy forward at<br/>max_num_batched_tokens with no KV cache<br/>model_runner.py:1091"]
    C["determine_available_memory()<br/>peak-memory snapshot per rank"]
    D["EngineCore._initialize_kv_caches()<br/>core.py:285"]
    E["Build KVCacheConfig:<br/>group layers by KVCacheSpec,<br/>num_blocks = usable_bytes / page_size_bytes"]
    F["initialize_kv_cache(): allocate tensors,<br/>create per-layer views<br/>model_runner.py:682"]
    G["compile_or_warm_up_model():<br/>torch.compile + capture_model() CUDA graphs"]
    H["Engine ready"]

    A --> B --> C --> D --> E --> F --> G --> H
```

Key types:

- `KVCacheSpec` (`vllm/v1/kv_cache_interface.py:189`) — what *one layer* needs per block.
  Subclasses: `AttentionSpec` (:456), `MLAAttentionSpec` (:520), `MambaSpec` (:549),
  `HiSparsePageSpec` (:427).
- `KVCacheGroup` (`vllm/v1/kv_cache_interface.py:688`) — a set of layers that share a
  block table because their specs are compatible. Hybrid models (e.g. Mamba +
  full-attention, or sliding-window + full-attention) produce **multiple groups**, which is
  why nearly every KV API returns *tuples indexed by group*.
- `KVCacheConfig` (`vllm/v1/kv_cache_interface.py:843`) — the final plan: tensors, groups,
  `num_blocks`.

The `determine_available_memory` → `num_blocks` path is also why "it OOMs at step 500, not
at startup" bugs happen: anything that allocates memory *after* profiling (a new CUDA
graph, a fragmentation-sensitive allocator pattern, an NCCL buffer) eats into the margin
the profiler left.

---

## 2.7 Where each concern lives (quick lookup)

| Question | Go here |
| --- | --- |
| "Why was my request not scheduled this step?" | `Scheduler.schedule`, `vllm/v1/core/sched/scheduler.py:694` |
| "Why did I get a cache miss?" | `KVCacheManager.get_computed_blocks:287`, `FullAttentionManager.find_longest_cache_hit:906` |
| "Why did it preempt?" | `allocate_slots` returned `None` → `scheduler.py:901-961` |
| "Why is the batch shape padded to N?" | `dispatch_cg_and_sync_dp`, `vllm/v1/worker/gpu/dp_utils.py:232` |
| "Where are the sampled tokens turned into text?" | `OutputProcessor`, `vllm/v1/engine/output_processor.py:554` |
| "Where does the HTTP response get built?" | `vllm/entrypoints/openai/chat_completion/` |
| "Which kernel actually ran?" | `AttentionBackend` subclass chosen by `vllm/v1/attention/selector.py` |
| "What does rank 1 do differently from rank 0?" | almost nothing — see `_get_output_rank`, `vllm/v1/executor/multiproc_executor.py:684` |

Next: [03-code-map.md](03-code-map.md).
