# Testing LexiGaze Safely

LexiGaze separates fast, deterministic regressions from model and hardware experiments. Pull requests use the fast lane so routine validation cannot consume a developer GPU, API quota, or external benchmark.

## Offline CPU quality gate

From the repository root, run:

```bash
python -X utf8 -m scripts.run_offline_quality_gate
```

On Windows with the project environment:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_offline_quality_gate
```

The command supervises a disposable worker and enforces a 180-second timeout. It:

- exposes only the Flask blueprints needed by the selected tests;
- sets `CUDA_VISIBLE_DEVICES=-1` before the worker starts;
- fails if PyTorch is imported;
- clears common AI-provider credentials and prevents `.env` from restoring them;
- blocks DNS, socket access, and child-process spawning, including attempts caught by application fallback code;
- uses temporary report storage and checks that data, report, model, and output files are unchanged;
- prints a final `QUALITY_GATE_RESULT=<json>` record.

Use `--json-output <path>` to persist the same summary or `--timeout-seconds <n>` to adjust the supervisor limit.

## CI dependency boundary

The workflow installs only `requirements-quality-gate.txt`, then compiles and lints the validation surfaces before running the gate. That dependency file intentionally excludes Torch, CUDA, Transformers, OpenCV, model packages, and the full project dependency graph. Dependency installation can use the package network; the test worker itself is network-denied.

## Heavy and research tests

Hardware/model experiments remain opt-in. Examples include `scripts/test_all_features_gpu.py`, gaze-model training, cognitive model inference, and `TestFusionRoutes.test_cross_attention_method`. Run them only with an explicit experiment plan, GPU budget, held-out evaluation, and a recorded result. Do not add them to the offline gate.

The offline gate validates application contracts and deterministic algorithms. It is not evidence of model accuracy, dataset generalization, or GPU correctness.

## Motion-robustness tests

The offline lane includes three gaze-calibration modules that do not import
PyTorch or open images:

- `scripts.test_gaze_motion_robustness` verifies metadata coverage gates,
  aggregate-only reporting, source hashing, session filtering, and group
  disjointness;
- `scripts.test_gaze_calibration_regression` verifies sample-vs-group leakage,
  standardized ridge fitting, and the frozen motion-conditioned feature schema.
- `scripts.test_gaze_motion_experiment` verifies nested outer-block isolation,
  a positive motion-shift control, a no-shift negative control, and portable GPU
  telemetry parsing.

Audit historical or newly collected manifests with:

```bash
python -X utf8 -m scripts.audit_gaze_motion_coverage
python -X utf8 -m scripts.audit_gaze_motion_coverage --session-id SESSION_ID --fail-on-not-ready
```

The audit is a data-readiness gate, not an accuracy benchmark. A real accuracy
claim still requires a frozen session or participant holdout.

Run the preregistered real-capture comparison only after its audit passes:

```bash
python -X utf8 -m scripts.run_gaze_motion_experiment \
  --session-id SESSION_ID \
  --output-model-name motion_run_001_nested_cpu \
  --json-output docs/experiments/results/motion-run.json
```

This runner forces the training request to CPU, keeps model hubs offline,
monitors `nvidia-smi`, fingerprints every normalized input image, and writes an
aggregate-only result. It refuses to overwrite an existing model artifact.

## Capture-run independence gate

Direct browser samples and frames extracted from the accompanying video are two
artifacts of one physical capture, not two independent sessions. New sessions
therefore persist a `capture_run_id`, `capture_source`, and (for video-derived
sessions) `source_session_id`. Audit the available independent captures with:

```bash
python -X utf8 -m scripts.audit_gaze_session_independence
```

The offline test `scripts.test_gaze_session_independence` verifies that linked
direct/video artifacts can never land on opposite sides of a validation split.
Legacy sessions without provenance are conservatively grouped when the same
participant label appears within the frozen 24-hour window. Participant labels
are operational labels and are not asserted to identify unique people.

## Opt-in hardware benchmark

`scripts/benchmark_gaze_inference.py` is intentionally excluded from routine hardware execution. Its pure statistics, GPU guard, telemetry parser, atomic-output tests, and CUDA runtime-policy tests run inside the offline gate without importing Torch. Actual CPU/CUDA inference is opt-in and follows the methodology in `docs/performance/gaze-inference-benchmark.md`.
