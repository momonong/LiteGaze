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
| C | model-only | inference, AMP FP16/BF16, compile modes | one variant/process, <=5 minutes | output parity only |
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
- The first isolated CPU smoke run exposed a native crash while collecting cuDNN metadata with CUDA hidden. The model inference had completed; the fault came from calling `torch.backends.cudnn.version()` in a CPU-only worker. Environment collection now queries cuDNN only for CUDA runs.
- The corrected CPU smoke run passed with `CUDA_VISIBLE_DEVICES=-1`, eager FP32 parity at zero maximum absolute error, and no system-GPU telemetry access from the worker.
- A one-second timeout probe returned exit 124 after 1.3 seconds, terminated the worker process tree, and left no LexiGaze benchmark process behind.
- The latest offline CPU quality gate passed 23/23 tests in 0.76 seconds. It imported no Torch, attempted no network or subprocess access in the test worker, and left GPU memory unchanged.

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

Next, preserve this result in Git and let the quiet-device guard decide whether Phase B/C CUDA baselines may start. No held-out session or question-answer data will be used while selecting the performance implementation.
