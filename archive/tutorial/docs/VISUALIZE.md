# 👁️ Gaze Visualization Guide

Use `visualize.py` to generate animated GIFs of gaze trajectories. This helps you see how the Neuro-Symbolic engine corrects noisy webcam data.

## Quick Start
Generate animations for all available data:
```bash
python visualize.py
```

## Usage Options
Use the `--target` flag to select a specific data stage:

| Target | Data Stage | Description |
| :--- | :--- | :--- |
| `raw` | Perception | Original noisy webcam coordinates. |
| `baseline` | Baseline | Simple Moving Average filtering. |
| `calibrated` | Fusion | Final result after Neuro-Symbolic calibration. |
| `raw-backup` | Sample | Pre-recorded noisy data (for testing). |
| `all` | Default | Processes all existing CSV files in `data/`. |

### Example Commands
- **Visualize your own calibrated results:**
  ```bash
  python visualize.py --target calibrated
  ```

- **See why raw data is difficult to use:**
  ```bash
  python visualize.py --target raw
  ```

## Output
All GIF animations are saved in the `figures/` folder.
