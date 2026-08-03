# ADR 0001: Hardware Performance Methodology

- Status: Accepted
- Date: 2026-08-04
- Owners: LexiGaze maintainers
- Scope: inference and training performance experiments

## Context

LexiGaze contains several hardware-sensitive paths, including MediaPipe face landmarks, OpenCV image transforms, UniGaze/PyTorch inference, Transformer inference, and ONNX experiments. Existing scripts prove that CUDA kernels can execute, but they do not provide a comparable end-to-end baseline. Some time asynchronous CUDA work with host wall-clock calls and no device synchronization, while others mix model initialization, warm-up, preprocessing, inference, data generation, and correctness checks in one number.

This makes it possible to report an apparent speedup that is actually caused by warm caches, omitted CPU work, background GPU activity, asynchronous timing, a different input, or an unacceptable numerical change.

## Decision

Every hardware optimization must start from a versioned benchmark contract and produce a machine-readable result. The contract requires:

1. A hardware and software fingerprint: CPU, GPU, driver, Python, PyTorch, CUDA, cuDNN, model, input hash, and benchmark revision.
2. Separate cold-start, warm-up, and steady-state measurements.
3. End-to-end latency plus named stages for decode, face preprocessing, tensor transform, host-to-device transfer, model forward, and device-to-host transfer.
4. Correct accelerator timing with CUDA events or explicit synchronization at measurement boundaries.
5. Distribution statistics rather than a single mean: count, minimum, p50, p95, p99, maximum, mean, and standard deviation.
6. Resource observations: peak allocated/reserved VRAM, system GPU memory, utilization, temperature, and power when available.
7. Output parity against eager FP32 on exactly the same input. A faster result that fails its declared tolerance is rejected.
8. A quiet-device preflight. GPU experiments refuse to start when background utilization or memory exceeds the configured budget unless an operator explicitly overrides the guard; overridden results are marked contaminated.
9. Fixed local frames or synthetic tensors for performance work. Question-answer sets, final held-out evaluation sessions, and model-quality benchmarks are not tuning inputs.
10. Raw profiler traces remain local because they can be large or contain paths. Compact JSON summaries and the decision outcome are committed.
11. Every run executes in an isolated worker with a default five-minute wall-clock limit. The supervisor terminates the worker process tree on timeout, and CPU runs hide CUDA from the worker.

The first target is the production gaze inference path. Candidate changes are evaluated in increasing order of risk:

1. `torch.inference_mode()` and removal of redundant image encoding/decoding;
2. transfer behavior and direct-frame execution;
3. automatic mixed precision;
4. `torch.compile` modes;
5. asynchronous MediaPipe/video architecture, micro-batching, ONNX, or TensorRT.

## Acceptance gates

A production change must:

- improve steady-state end-to-end p95 by at least 10% or remove a measured resource bottleneck;
- keep the maximum absolute gaze-output difference within the experiment's declared tolerance;
- avoid increasing peak VRAM by more than 1 GiB unless the gain and deployment budget justify it;
- report cold-start and compilation cost separately;
- preserve API contracts and pass the offline CPU quality gate;
- be rechecked on held-out session data only after the implementation is frozen.

The 10% latency and 1 GiB VRAM thresholds are initial operating rules, not scientific constants. A later ADR may revise them with evidence.

## Alternatives considered

### Optimize the model first

Rejected as the starting point. The current request path contains several CPU and synchronization stages, so model-only acceleration may not improve user-visible latency.

### Reuse existing GPU scripts as the baseline

Rejected. They are useful compatibility checks but do not consistently isolate warm-up, synchronize CUDA timing, fingerprint the run, or emit comparable distributions.

### Measure only end-to-end latency

Rejected. It shows whether a change helped but not why, making regressions and hardware portability difficult to reason about.

### Commit representative user frames

Rejected. Local ignored frames may be used by explicit path and recorded by SHA-256 only. No private image is added to Git.

## Consequences

- Hardware work begins more slowly because measurement infrastructure comes first.
- Results become reproducible, reviewable, and comparable across machines and commits.
- Background workloads cause an intentional pause rather than a misleading benchmark.
- Model-quality evaluation remains isolated from performance tuning, reducing overfitting and leakage risk.
