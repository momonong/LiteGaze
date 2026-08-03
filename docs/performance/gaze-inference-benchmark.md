# Gaze Inference Benchmark Contract

The opt-in hardware benchmark measures the production UniGaze path without joining the pull-request CPU gate. It never selects or tunes model parameters.

## Workloads

- `model`: transforms one already-normalized RGB image and measures tensor transfer plus UniGaze inference.
- `pipeline`: additionally measures JPEG decode and MediaPipe face normalization from an explicit local image.

The pipeline workload requires `--image`. The image stays outside Git; the summary stores only its SHA-256, dimensions, encoded byte size, and whether a face was detected.

## Variants

| Variant | Device | Purpose |
| --- | --- | --- |
| `eager` | CPU/CUDA | Current FP32 `torch.no_grad()` reference. |
| `inference` | CPU/CUDA | Replace autograd suppression with `torch.inference_mode()`. |
| `amp-fp16` | CUDA | Tensor Core candidate using autocast FP16. |
| `amp-bf16` | CUDA | Tensor Core candidate using autocast BF16. |
| `compile-default` | CUDA | TorchInductor default mode. |
| `compile-reduce-overhead` | CUDA | CUDA-graph-oriented small-batch candidate. |

Each non-reference variant computes one eager FP32 output first and reports maximum/mean absolute difference and `allclose` status.

## Run policy

Default GPU preflight limits:

- maximum background GPU utilization: 5%;
- maximum existing device memory: 2,048 MiB;
- maximum starting temperature: 82 °C;
- five samples at 500 ms intervals;
- 10 warm-up iterations and 30 measured iterations;
- no more than 500 measured or 100 warm-up iterations per invocation.
- a five-minute worker timeout, including compilation and cold start.

Use `--allow-busy-gpu` only when intentionally measuring contention. Such a result is labeled `contaminated=true` and must not be used to accept an optimization.

## JSON result

The result uses schema version 1 and contains:

- `status`, `variant`, `workload`, and `device`;
- `revision` and `input` provenance;
- `environment` hardware/software fingerprint;
- `guard` thresholds, samples, and contamination state;
- `setup_ms`, `first_iteration_ms`, warm-up/measured counts, and per-stage latency distributions;
- `throughput_fps` derived from end-to-end p50 and p95;
- `parity` tolerances and output differences;
- `resources` with PyTorch and system GPU observations;
- `failure` details when the run cannot complete.

The supervisor performs the GPU guard before importing Torch, starts a fresh worker for every variant, and records the worker exit and timeout state. CUDA stage durations use events with one synchronization at the end of each iteration, so end-to-end timing is not inflated by synchronization between stages.

JSON files are written atomically. Profiler traces are optional local artifacts and are not committed.

## Example

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.benchmark_gaze_inference `
  --device cuda `
  --variant eager `
  --workload pipeline `
  --image D:\path\to\local-frame.jpg `
  --json-output D:\temp\lexigaze-gaze-eager.json
```

Run variants one at a time so each process has an independent model/compiler cache. Do not compare a quiet baseline with a contended candidate.
