# Gaze Inference Hardware Baseline

## Status

- Date: 2026-08-04 (Asia/Taipei)
- Branch: `perf/gaze-inference-baseline`
- Stacked base: `codex/test/offline-cpu-quality-gate` at `2ee6fea`
- State: benchmark infrastructure in progress; GPU execution deferred until quiet preflight
- Decision: `docs/adr/0001-hardware-performance-methodology.md`
- Performance contract: `docs/performance/gaze-inference-benchmark.md`

## Goal

Establish a trustworthy stage-level CPU/CUDA baseline for the production gaze inference path, identify the dominant end-to-end bottleneck, and accept only optimizations that preserve eager FP32 output within a declared tolerance.

## Product path inventory

`core/gaze_core/inference.py` currently performs:

1. base64 decode and OpenCV JPEG decode;
2. MediaPipe image-mode face landmark detection;
3. crop, head-pose estimation, perspective normalization, and RGB conversion;
4. NumPy/OpenCV to FP32 PyTorch tensor transform;
5. blocking transfer to the selected device;
6. batch-one UniGaze ViT forward under `torch.no_grad()`;
7. device-to-host copy and Python-list conversion;
8. calibration and viewport mapping.

The offline video route already holds frames as NumPy arrays but encodes each selected frame to JPEG/base64 before calling the same predictor, which then decodes it again. Removing that round trip is the leading low-risk pipeline hypothesis, subject to measurement.

## Baseline risks found

- Existing CUDA compatibility tests execute one forward but do not report latency distributions.
- A legacy model benchmark uses host `time.time()` around asynchronous CUDA work without a synchronization boundary, so its GPU latency/FPS is not an acceptance baseline.
- Background GPU work changes clocks, power, temperature, memory pressure, and scheduling.
- Single-image GPU inference may be dominated by MediaPipe, CPU transforms, transfers, or synchronization rather than model compute.
- Compile and autocast may change cold-start time, VRAM, or numerical outputs even when steady-state model latency improves.

## Initial machine observation

Before any LexiGaze GPU experiment, read-only `nvidia-smi` showed:

| Field | Observation |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| Driver | 581.57 |
| VRAM | 4,059 / 24,463 MiB |
| GPU / memory utilization | 21% / 7% |
| Power | 81 W |
| Temperature | 77 °C |
| Active compute process | external Python 3.11 process plus desktop GPU clients |

This violates the planned 5% utilization and 2,048 MiB existing-memory limits. No performance number will be collected from this state.

## Experiment matrix

| Phase | Workload | Variants | GPU budget | Quality data |
| --- | --- | --- | --- | --- |
| A | model-only synthetic/local normalized frame | eager, inference | CPU only | none |
| B | full fixed local frame pipeline | eager | quiet CUDA, 30 measured iterations | none |
| C | model-only | inference, TF32, AMP FP16/BF16, compile modes | one variant/process, <=5 minutes | output parity only |
| D | full pipeline | winning low-risk variants | quiet CUDA | output parity only |
| E | frozen implementation | held-out sessions | explicit later run | held-out, never tuned |

## Acceptance criteria

- Guard refuses contaminated GPU baselines by default.
- Benchmark JSON is deterministic in shape and records enough environment context to compare runs.
- CUDA stage timing uses events or explicit synchronization.
- Every candidate is compared with eager FP32 on the identical input.
- No repository data/model/report artifact is changed by benchmark tests.
- Offline CPU quality gate remains Torch-free and passes.
- A production optimization needs >=10% end-to-end p95 improvement or a demonstrated resource benefit.

## Results

### Infrastructure validation

- The real GPU preflight observed a 63% utilization peak, 4,059 MiB already allocated, and 79 °C across three samples. It returned `status=refused` before starting a Torch worker, as intended.
- After the background workload ended, a five-sample preflight passed at 0% utilization, 88 MiB, and 52 °C. The first CUDA worker then stopped before model loading because PyTorch 2.13 requires an explicit index in `torch.cuda.set_device`; benchmark device strings are now normalized from `cuda` to `cuda:0` and covered by a regression test.
- The first TF32 run exposed that the eager parity reference inherited the candidate's global matmul precision. Its JSON was discarded; parity now runs the reference under `highest`, restores the candidate precision for the observed output, and restores the prior setting afterward.
- The first isolated CPU smoke run exposed a native crash while collecting cuDNN metadata with CUDA hidden. The model inference had completed; the fault came from calling `torch.backends.cudnn.version()` in a CPU-only worker. Environment collection now queries cuDNN only for CUDA runs.
- The corrected CPU smoke run passed with `CUDA_VISIBLE_DEVICES=-1`, eager FP32 parity at zero maximum absolute error, and no system-GPU telemetry access from the worker.
- A one-second timeout probe returned exit 124 after 1.3 seconds, terminated the worker process tree, and left no LexiGaze benchmark process behind.
- The latest offline CPU quality gate passed 27/27 tests in 0.91 seconds. It imported no Torch, attempted no network or subprocess access in the test worker, and left GPU memory unchanged.

### Phase A smoke measurement

This is a diagnostic measurement, not an acceptance baseline: it used one warm-up and three measured iterations on a dirty worktree. It verifies the measurement path before the benchmark infrastructure is committed.

| Metric | Observation |
| --- | ---: |
| Model load | 2,498.856 ms |
| First iteration end-to-end | 134.037 ms |
| Steady-state end-to-end p50 / p95 | 118.900 / 119.933 ms |
| Model-forward p50 / p95 | 118.281 / 119.251 ms |
| Throughput from p50 / p95 | 8.410 / 8.338 FPS |
| Eager FP32 max absolute error | 0.0 |

### Phase A clean CPU eager baseline

The first acceptance-shaped model workload ran from clean commit `d61a1b0`, with 10 warm-up and 30 measured iterations. The machine-readable result is [`results/2026-08-04-cpu-eager-model.json`](results/2026-08-04-cpu-eager-model.json).

| Metric | Observation |
| --- | ---: |
| Model load | 1,851.796 ms |
| First iteration end-to-end | 104.533 ms |
| Steady-state end-to-end p50 / p95 | 107.372 / 268.984 ms |
| Model-forward p50 / p95 | 106.825 / 265.253 ms |
| Throughput from p50 / p95 | 9.313 / 3.718 FPS |
| Maximum end-to-end latency | 805.016 ms |
| Eager FP32 max absolute error | 0.0 |

The wide CPU tail is a finding, not a speedup: PyTorch used 16 threads on a hybrid-core laptop, and the benchmark currently has no CPU-idleness guard. CPU thread-count/affinity experiments may be worthwhile, but they require repeated runs before any production decision.

### Corrected CUDA smoke measurement

After normalizing the device to `cuda:0`, a dirty-worktree diagnostic run with one warm-up and three measured iterations completed successfully. End-to-end p50/p95 was 7.145/7.297 ms, model-forward p50/p95 was 6.430/6.486 ms, and peak PyTorch allocation was 369.965 MiB. Eager FP32 parity was exact. The 287.815 ms first iteration is kept separate and not counted as steady state.

### Phase B clean CUDA eager baseline

The clean model workload ran from commit `84ff748`, after a five-sample guard passed at 0% utilization, 88 MiB, and 51 °C. It used 10 warm-up and 30 measured iterations. The machine-readable result is [`results/2026-08-04-cuda-eager-model.json`](results/2026-08-04-cuda-eager-model.json).

| Metric | Observation |
| --- | ---: |
| Model load | 1,840.714 ms |
| First iteration end-to-end / forward | 215.723 / 213.876 ms |
| Steady-state end-to-end p50 / p95 | 7.439 / 8.136 ms |
| Model-forward p50 / p95 | 6.578 / 7.283 ms |
| Tensor transform p50 / p95 | 0.467 / 0.540 ms |
| Host-to-device p50 / p95 | 0.132 / 0.178 ms |
| Device-to-host p50 / p95 | 0.146 / 0.199 ms |
| Throughput from p50 / p95 | 134.419 / 122.904 FPS |
| Peak allocated / reserved VRAM | 369.965 / 418.000 MiB |
| Post-run utilization / temperature | 54% / 55 °C |
| Eager FP32 max absolute error | 0.0 |

CUDA reduced model-only p50 by about 14.4× relative to the clean CPU eager run. CPU p95 is too scheduling-sensitive for a credible tail-speedup ratio. The model forward still accounts for roughly 88% of CUDA end-to-end p50, so inference context and numeric-format candidates remain worth measuring.

Next, preserve this baseline in Git and compare `torch.inference_mode()` first. No held-out session or question-answer data will be used while selecting the performance implementation.

### Phase C1: `torch.inference_mode()`

The candidate ran from clean commit `39eeb3b` under a clean guard. The machine-readable result is [`results/2026-08-04-cuda-inference-model.json`](results/2026-08-04-cuda-inference-model.json).

| Metric | Eager | `inference_mode` | Outcome |
| --- | ---: | ---: | --- |
| End-to-end p50 | 7.439 ms | 7.445 ms | 0.07% slower |
| End-to-end p95 | 8.136 ms | 10.174 ms | 25.04% slower |
| Model-forward p50 | 6.578 ms | 6.584 ms | effectively unchanged |
| Peak allocated VRAM | 369.965 MiB | 369.965 MiB | unchanged |
| Max absolute error | 0.0 | 0.0 | exact parity |

Decision: do not promote `torch.inference_mode()` as a performance change. It does not meet the 10% gate, and this run has worse tail latency. A repeated interleaved experiment could distinguish tail noise from a true regression, but the essentially identical p50 leaves no plausible acceptance benefit for this path.

Next, test FP16 and BF16 autocast independently, with the same synthetic input and eager FP32 parity check.

### Phase C2: FP16 autocast

FP16 ran from clean commit `248a9bc` under a clean guard. The machine-readable result is [`results/2026-08-04-cuda-amp-fp16-model.json`](results/2026-08-04-cuda-amp-fp16-model.json).

| Metric | Eager FP32 | AMP FP16 | Outcome |
| --- | ---: | ---: | --- |
| End-to-end p50 | 7.439 ms | 8.074 ms | 8.53% slower |
| End-to-end p95 | 8.136 ms | 8.813 ms | 8.31% slower |
| Model-forward p50 | 6.578 ms | 7.249 ms | 10.19% slower |
| Peak allocated VRAM | 369.965 MiB | 371.332 MiB | +1.367 MiB |
| Max absolute error | 0.0 | 0.000475 | within 0.005 tolerance |

Decision: reject FP16 autocast for batch-one UniGaze inference. Its numerical parity is acceptable, but autocast/cast overhead outweighs any Tensor Core benefit and fails the latency gate.

Next, test BF16 under the same conditions.

### Phase C3: BF16 autocast

BF16 ran from clean commit `6b15957` under a clean guard. The machine-readable result is [`results/2026-08-04-cuda-amp-bf16-model.json`](results/2026-08-04-cuda-amp-bf16-model.json).

| Metric | Eager FP32 | AMP BF16 | Outcome |
| --- | ---: | ---: | --- |
| End-to-end p50 | 7.439 ms | 7.820 ms | 5.12% slower |
| End-to-end p95 | 8.136 ms | 8.350 ms | 2.63% slower |
| Model-forward p50 | 6.578 ms | 7.002 ms | 6.44% slower |
| Peak allocated VRAM | 369.965 MiB | 371.332 MiB | +1.367 MiB |
| Max absolute error | 0.0 | 0.002088 | within 0.005 tolerance |

Decision: reject BF16 autocast. Like FP16, it passes numerical parity but adds latency at batch one. Neither mixed-precision mode should be enabled in production based on these results.

Next, attempt `torch.compile` default and reduce-overhead modes. Compilation remains isolated by process and bounded by the five-minute supervisor timeout.

### Phase C4: `torch.compile` default

The first attempt stopped after 37.5 seconds because the Windows environment lacked Triton. PyTorch documents Triton as the GPU code-generation backend for `torch.compile`, and the current [Triton Windows support matrix](https://github.com/triton-lang/triton-windows) maps PyTorch 2.13 to Triton 3.7 and supports Blackwell. `triton-windows==3.7.1.post27` was therefore installed only in the ignored project virtual environment for this experiment; it has not been promoted to a project dependency.

The successful run used clean commit `d5a31fa`, a brand-new TorchInductor cache, a clean GPU guard, 10 warm-up iterations, and 30 measured iterations. The machine-readable result is [`results/2026-08-04-cuda-compile-default-model.json`](results/2026-08-04-cuda-compile-default-model.json).

| Metric | Eager FP32 | Compile default | Outcome |
| --- | ---: | ---: | --- |
| First iteration | 215.723 ms | 16,067.939 ms | 15.85 s compile penalty |
| End-to-end p50 | 7.439 ms | 6.390 ms | 14.11% faster |
| End-to-end p95 | 8.136 ms | 6.700 ms | 17.66% faster |
| Model-forward p50 | 6.578 ms | 5.547 ms | 15.68% faster |
| Model-forward p95 | 7.283 ms | 5.673 ms | 22.10% faster |
| Peak allocated VRAM | 369.965 MiB | 441.054 MiB | +71.089 MiB |
| Max absolute error | 0.0 | 0.000000238 | within 0.00001 tolerance |

Decision: provisionally accept the steady-state result because it clears latency, parity, and VRAM gates. It is not yet a production decision: a 16-second first inference is unacceptable on the request path unless compilation is completed during controlled startup/pre-warm, and the full MediaPipe pipeline may dilute the model-only gain.

Next, compare reduce-overhead mode and explicit TF32 matmul precision, then re-test the best combination on the full fixed-frame pipeline.

### Phase C5: `torch.compile` reduce-overhead

Reduce-overhead ran from clean commit `4e54b7f` with another brand-new TorchInductor cache. The machine-readable result is [`results/2026-08-04-cuda-compile-reduce-overhead-model.json`](results/2026-08-04-cuda-compile-reduce-overhead-model.json).

| Metric | Eager FP32 | Compile default | Reduce overhead |
| --- | ---: | ---: | ---: |
| First iteration | 215.723 ms | 16,067.939 ms | 14,495.361 ms |
| End-to-end p50 | 7.439 ms | 6.390 ms | 7.587 ms |
| End-to-end p95 | 8.136 ms | 6.700 ms | 10.091 ms |
| Model-forward p50 | 6.578 ms | 5.547 ms | 5.590 ms |
| Tensor-transform p50 | 0.467 ms | 0.473 ms | 1.193 ms |
| Peak allocated / reserved VRAM | 369.965 / 418 MiB | 441.054 / 482 MiB | 441.054 / 550 MiB |
| Max absolute error | 0.0 | 0.000000238 | 0.000000238 |

Decision: reject reduce-overhead. Although its compiled model stage is faster than eager, end-to-end p50 is 1.99% slower than eager and 18.74% slower than compile-default; p95 is also worse, and reserved VRAM is higher. Compile-default remains the leading compiled candidate.

Next, add explicit TF32 matmul precision to the benchmark and test it both without compilation and, only if useful, with the winning compiler mode.

### Fixed-frame video-path diagnostic

The full-pipeline fixture is a local ignored 640×360 JPEG (25,098 bytes, SHA-256 `5b6b97ea27b4bb776524482bdd78105bd2efad2a74f7e7e405a886e0859d66ef`). The image path and participant/session metadata are not recorded. MediaPipe uses the official 3,758,596-byte face-landmarker asset cached under the ignored `web/static/*.task` path (SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`).

A dirty-worktree CPU diagnostic compared the exact offline-video transport at 240-pixel width and JPEG quality 50 with a direct OpenCV-frame path. The legacy steady-state p50 transport components were approximately 0.144 ms resize, 0.103 ms JPEG encode, 0.023 ms base64 encode, 0.010 ms base64 decode, and 0.109 ms JPEG decode. Direct-frame execution retains resize and therefore removes only about 0.244 ms of steady-state transport work.

The fixed-frame gaze output changed from `[-0.155597, -0.255159]` after the legacy lossy round trip to `[-0.161127, -0.302447]` with the direct frame, a maximum absolute difference of 0.047289 radians. That may represent better image fidelity, but it is not output parity and would change calibration semantics.

Decision: do not change the production route for this micro-optimization. The measured transport saving is well below the 10% gate, while the fixed-frame output shift is material. Keep both workloads in the benchmark so a future, separately frozen quality study can revisit the design without tuning on held-out data.

### Phase C6: TF32 `high`

After correcting parity isolation, TF32 `high` ran from clean commit `c283f20` under a clean guard. The machine-readable result is [`results/2026-08-04-cuda-tf32-high-model.json`](results/2026-08-04-cuda-tf32-high-model.json).

| Metric | Eager `highest` | Eager TF32 `high` | Outcome |
| --- | ---: | ---: | --- |
| First iteration | 215.723 ms | 196.871 ms | 8.74% faster |
| End-to-end p50 | 7.439 ms | 5.101 ms | 31.43% faster |
| End-to-end p95 | 8.136 ms | 6.147 ms | 24.45% faster |
| Model-forward p50 | 6.578 ms | 4.303 ms | 34.59% faster |
| Model-forward p95 | 7.283 ms | 5.353 ms | 26.50% faster |
| Peak allocated / reserved VRAM | 369.965 / 418 MiB | 369.965 / 418 MiB | unchanged |
| Max absolute error | 0.0 | 0.000356 | within 0.005 tolerance |

Decision: TF32 provisionally clears latency, parity, and memory gates without a compile-time penalty. However, the discarded pre-fix run had slower performance despite the same precision setting, so clean eager and TF32 replicates are required before promotion. If the gain persists, TF32 is preferable to compilation because it avoids the 16-second first-run cost.

### Model-workload replication

A clean eager `highest` replicate ran from commit `1b732dd`; its result is [`results/2026-08-04-cuda-eager-model-r2.json`](results/2026-08-04-cuda-eager-model-r2.json). End-to-end p50/p95 was 7.525/7.878 ms and model-forward p50/p95 was 6.608/6.828 ms. This is within 1.16% at p50 and 3.18% at p95 of the original eager baseline, supporting a stable FP32 reference.

The matching TF32 replicate ran from commit `e08a90a`; its result is [`results/2026-08-04-cuda-tf32-high-model-r2.json`](results/2026-08-04-cuda-tf32-high-model-r2.json). End-to-end p50/p95 was 5.449/7.245 ms and model-forward p50/p95 was 4.612/6.112 ms, with the same 0.000356 maximum error. Against its paired eager run, TF32 improved p50 by 27.59% and p95 by 8.03%.

Across the first two clean runs, the mean-of-run p50 improves from 7.482 to 5.275 ms (29.49%) and mean-of-run p95 improves from 8.007 to 6.696 ms (16.38%). Because one paired p95 improvement is below the 10% gate, collect a third pair rather than selecting the best run.

The third eager and TF32 runs are [`results/2026-08-04-cuda-eager-model-r3.json`](results/2026-08-04-cuda-eager-model-r3.json) and [`results/2026-08-04-cuda-tf32-high-model-r3.json`](results/2026-08-04-cuda-tf32-high-model-r3.json). The paired end-to-end result was 7.345/7.563 ms versus 5.179/6.519 ms at p50/p95, a 29.50%/13.81% improvement.

| Three-run aggregate | Eager `highest` | TF32 `high` | Improvement |
| --- | ---: | ---: | ---: |
| Mean-of-run end-to-end p50 | 7.437 ms | 5.243 ms | 29.50% |
| Mean-of-run end-to-end p95 | 7.859 ms | 6.637 ms | 15.55% |
| Median-of-run end-to-end p50 | 7.439 ms | 5.179 ms | 30.39% |
| Median-of-run end-to-end p95 | 7.878 ms | 6.519 ms | 17.25% |
| Mean-of-run model-forward p50 | 6.569 ms | 4.423 ms | 32.67% |
| Mean-of-run model-forward p95 | 6.939 ms | 5.707 ms | 17.75% |

Decision: accept TF32 at the model-workload gate. It is reproducibly faster, preserves the declared tolerance, adds no VRAM, and has no compilation penalty. Production promotion still requires fixed-frame full-pipeline validation and regression tests around CUDA-only configuration.

### Phase D1: fixed-frame `video-legacy` eager baseline

The full offline-video path ran from clean commit `c548b33` under a clean GPU guard; its result is [`results/2026-08-04-cuda-video-legacy-eager.json`](results/2026-08-04-cuda-video-legacy-eager.json).

| Metric | Observation |
| --- | ---: |
| End-to-end p50 / p95 | 19.789 / 242.859 ms |
| MediaPipe preprocess p50 | 10.029 ms |
| Model-forward p50 | 7.322 ms |
| Tensor transform p50 | 0.581 ms |
| Resize + JPEG/base64 round trip p50 | 0.449 ms |
| Host/device copies p50 | 0.360 ms |

The median confirms that MediaPipe is now the largest stage and the model is second. The p95 is dominated by simultaneous host scheduling stalls: preprocess reached 121.900 ms, tensor transform 10.003 ms, and model forward 76.841 ms. The GPU itself passed a clean preflight, so GPU-only guarding is insufficient for stable full-pipeline tail comparisons.

Next, run TF32 on the identical fixed frame and conditions. Treat median gains separately from tail behavior, and do not promote until the CPU-contended result is understood.
