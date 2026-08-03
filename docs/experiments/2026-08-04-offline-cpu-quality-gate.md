# Offline CPU Quality Gate — Living Log

- Date: 2026-08-04 (Asia/Taipei)
- Branch: `codex/test/offline-cpu-quality-gate`
- Base: `origin/main` at `b22b637`
- Status: complete

## Objective

Create one deterministic validation entry point that is safe to run locally and in pull requests. It must force CPU-only execution, deny external network access, remove API credentials from the child environment, isolate generated artifacts, and report a machine-readable result.

This work does not train or tune a model and does not read a question-answer benchmark. It therefore cannot optimize against a Q&A test set.

## Completion criteria

1. A single documented command runs the selected offline regression suite.
2. The runner fails closed if a test attempts external network access.
3. CUDA is hidden before application imports and API/provider credentials are removed.
4. Generated reports, sessions, caches, and temporary files do not modify tracked or user data.
5. Pull requests run the same gate in GitHub Actions without installing CUDA packages.
6. Local repeat runs produce the same test count and outcome.
7. The final log records wall time, GPU snapshots, repository cleanliness, discarded approaches, and remaining risks.

## Baseline findings

- `origin/main` has no `.github/workflows` quality gate.
- Regression tests are standalone `unittest` scripts under `scripts/`; there is no canonical safe test command.
- The default dependency graph selects a CUDA PyTorch index on Windows/Linux, even though most web/API regressions do not require a GPU.
- Some scripts automatically select CUDA, and some routes can call external providers when developer credentials are present.
- Existing integration tests can persist reports under the repository before cleaning them, so interruption can leave artifacts behind.
- PyTorch `2.13.0+cu130` on this Windows host exits with access violation `-1073741819` when `torch.cuda.is_available()` runs with `CUDA_VISIBLE_DEVICES=-1`. The fast lane therefore prevents Torch imports entirely; the minimal CI environment does not install it.
- During the initial read-only audit, the shared GPU moved from 21% / 4.95 GiB to 99% / 5.07 GiB. NVIDIA process inspection showed an unrelated Python process plus desktop applications; no LexiGaze model or test had been started. All experiments in this branch will set `CUDA_VISIBLE_DEVICES=-1` before Python starts.

## Candidate directions

| Candidate | Value | Risk / cost | Decision |
| --- | --- | --- | --- |
| More model or dataset experiments | Potential metric gain | Competes for an already busy GPU and risks benchmark overfitting | Deferred |
| Session persistence hardening | High user-data value | Deserves a separate feature branch and focused API contract | Next candidate |
| Offline CPU quality gate | Multiplies reliability of every later optimization | Small CI and test-infrastructure change | Selected |
| Split the large reading template | Improves maintainability | Large review surface without a reliable gate | Deferred until gate exists |

## Experiment ledger

| Time (Asia/Taipei) | Experiment | Result | Next action |
| --- | --- | --- | --- |
| 03:29 | Read-only architecture, test, persistence, and GPU audit | No CI; tests mix safe and unsafe execution modes; shared GPU was already busy | Establish CPU/offline gate before further optimization |
| 03:34 | Created branch from refreshed `origin/main` | Clean independent branch at `b22b637` | Measure the current regression baseline |
| 03:39 | Existing three-module regression command with provider keys cleared and `CUDA_VISIBLE_DEVICES=-1` | No completion after roughly 130 seconds; two verified LexiGaze child processes were stopped. Focused diagnosis tied the path to full-app imports and the Windows CUDA/PyTorch incompatibility | Add focused blueprint loading and supervised worker cleanup |
| 03:47 | Offline gate prototype, run 1 | Failed safely: 4 loader errors, 0 tests executed. Safeguards passed; no artifacts or Torch import. Runtime 68.494 s was dominated by scanning 249,595 unrelated files under `data/`. GPU memory was unchanged at 5069 MiB | Launch worker as a module and monitor only direct session JSON plus small output roots |
| 03:48 | Optimized local gate, first pass | 16/16 passed; 1.444 s total; no network, credential, artifact, Torch, or GPU-memory violation | Repeat and validate a clean minimal environment |
| 03:50 | First isolated-dependency run | 26 CPU packages installed; 16/16 passed; `find_spec('torch')` returned `None`; GPU memory unchanged | Add pinned Ruff and reproduce final dependency set |
| 03:52 | Production app-factory smoke test | Default `create_app()` still registered 7 blueprints and 50 routes; GPU memory remained 5069 MiB | Preserve default behavior while using focused loading only in tests |
| 03:54 | Forced one-second timeout | Expected `status=timeout`, `worker_exit=124`; no worker process or `lexigaze-*` temporary directory remained | Timeout/process-tree cleanup criterion satisfied |
| 03:55 | Subprocess-denial hardening, run 1 | Failed safely in 0.474 s before tests: replacing `subprocess.Popen` with a function broke Windows `asyncio` inheritance | Use a subclassable blocked `Popen` class and repeat the full gate |
| 03:56 | Subprocess-denial hardening, run 2 | 15 tests passed; fatigue fusion import was blocked because Python's Windows `platform.machine()` uses local `cmd /c ver` once while importing pandas | Warm the standard-library OS identity cache before enabling the process guard; keep all test-time spawning denied |
| 03:57 | Final local gate | 16/16 passed in 2.289 s total; socket and process probes blocked; Torch absent from `sys.modules`; no artifact changes; GPU memory 5069 MiB before and after | Validate the same dependency boundary in an isolated environment |
| 03:57 | Final isolated-dependency gate | Installed only 27 pinned packages; 16/16 passed in 4.497 s total; `find_spec('torch')` returned `None`; all safeguards passed | Finalize documentation and delivery checks |
| 03:58 | Final repeat under concurrent CI-command simulation | 16/16 passed again; compile, YAML, and isolated Ruff checks also passed; all safeguards passed and GPU memory stayed at 5069 MiB | Ready for commit and Draft PR |

## Completed experiments

1. Measured the existing candidate command and diagnosed full-application import coupling.
2. Added focused blueprint loading while retaining the complete default application.
3. Split the Torch-backed cross-attention regression from deterministic fatigue fusion.
4. Isolated report writes in temporary storage and removed legacy `sys.path` injection from selected tests.
5. Built a supervised worker with credential clearing, offline environment flags, Python socket/DNS denial, subprocess denial, artifact snapshots, timeout cleanup, GPU snapshots, and JSON output.
6. Created a pinned 27-package validation environment with no Torch/CUDA dependency and reproduced the suite there.
7. Added a least-privilege GitHub Actions workflow with immutable action SHAs, compile checks, Ruff, and the same gate command.
8. Repeated successful runs, a production app-factory smoke test, and a deliberate timeout cleanup test.

## Results

| Metric | Baseline | Final |
| --- | --- | --- |
| Canonical command | None | `python -X utf8 -m scripts.run_offline_quality_gate` |
| Completion | Existing three-module command did not finish within roughly 130 s | 16/16 passed |
| Project-environment wall time | Greater than 130 s | 2.289 s |
| Conservative speedup | N/A | Greater than 56x |
| Fresh isolated environment | Full project graph would select CUDA Torch | 27 pinned packages; 4.497 s after cached install |
| Test-time external access | Provider behavior depended on local environment | Socket, DNS, and process-spawn probes blocked |
| Provider credentials | Local `.env` could influence tests | Seven provider credential variables forced empty before import |
| Generated artifacts | Reports could briefly enter repository directories | No changes across sessions, reports, models, or output scopes |
| Torch / GPU dependency | Full app import pulled Torch into unrelated tests | Torch not installed in isolated env and not imported by worker |
| GPU memory | Shared external workload at 5069 MiB | 5069 MiB before and after every final run |
| CI | No workflow | Python 3.12 offline CPU workflow with `contents: read` only |

The final lane never loaded a model, trained parameters, or read a Q&A benchmark. Its assertions cover application contracts and deterministic cognitive/fusion logic, so the work adds no benchmark-specific optimization pressure.

### Delivered surfaces

- Focused application factory configuration: `LEXIGAZE_BLUEPRINTS`.
- Safe runner and explicit test manifest: `scripts/run_offline_quality_gate.py`.
- Minimal pinned environment: `requirements-quality-gate.txt`.
- GitHub workflow: `.github/workflows/offline-cpu-quality-gate.yml`.
- Isolated inspector report lifecycle tests and separated fusion test tiers.
- Operator documentation: `docs/testing.md` and README entry point.

### Research decisions

- GitHub's official releases showed `actions/checkout` v7.0.1 and `actions/setup-python` v7.0.0; the workflow pins their full immutable commit SHAs rather than floating tags.
- Official NVIDIA and PyTorch documentation define `CUDA_VISIBLE_DEVICES=-1` as hiding all GPUs. The local PyTorch/CUDA build crashes when querying availability under that setting, so the fast lane additionally refuses any Torch import instead of trusting the variable alone.
- Ruff 0.16.1 was selected from its official immutable release and pinned in the validation requirements.

Sources: [actions/checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1), [actions/setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0), [NVIDIA CUDA environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html), [PyTorch CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html), [Ruff 0.16.1](https://github.com/astral-sh/ruff/releases/tag/0.16.1).

## Remaining risks

1. The local host does not have Python 3.12, so the exact Ubuntu/Python 3.12 workflow runtime is validated only after the branch is pushed and GitHub Actions runs. Python 3.11 passed both project and isolated environments.
2. The network guard covers Python DNS/socket APIs and disables worker subprocesses; it is not an operating-system firewall against a native extension that opens its own socket. The minimal lane intentionally excludes such model/runtime packages and receives no secrets in CI.
3. Heavy gaze, cognitive-model, and cross-attention tests remain opt-in. This gate does not claim model accuracy, GPU correctness, or dataset generalization.
4. Shared GPU utilization is noisy because an unrelated training process remained active. Stable VRAM plus `torch_imported=false`/`find_spec('torch') is None` is the attribution evidence for this lane.
5. Session persistence still uses direct JSON writes, broad exception swallowing, and naive timestamps. Atomic persistence, schema/path validation, and corruption reporting are the recommended next feature branch.
6. Immutable GitHub Action and Python-package pins need an explicit dependency-update cadence.
