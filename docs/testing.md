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
- runs session persistence and API tests in disposable directories, including invalid input,
  corrupt data, failed atomic publication, and concurrent CRUD coverage;
- prints a final `QUALITY_GATE_RESULT=<json>` record.

Use `--json-output <path>` to persist the same summary or `--timeout-seconds <n>` to adjust the supervisor limit.

## CI dependency boundary

The workflow installs only `requirements-quality-gate.txt`, then compiles and lints the validation surfaces before running the gate. That dependency file intentionally excludes Torch, CUDA, Transformers, OpenCV, model packages, and the full project dependency graph. Dependency installation can use the package network; the test worker itself is network-denied.

## Heavy and research tests

Hardware/model experiments remain opt-in. Examples include `scripts/test_all_features_gpu.py`, gaze-model training, cognitive model inference, and `TestFusionRoutes.test_cross_attention_method`. Run them only with an explicit experiment plan, GPU budget, held-out evaluation, and a recorded result. Do not add them to the offline gate.

The offline gate validates application contracts and deterministic algorithms. It is not evidence of model accuracy, dataset generalization, or GPU correctness.

Session persistence behavior, limits, and recovery guidance are documented in
[`docs/session-storage.md`](session-storage.md).
