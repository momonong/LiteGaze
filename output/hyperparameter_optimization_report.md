# 🔬 LexiGaze Decoder Hyperparameter Optimization Report

This report presents grid search evaluation results over Viterbi snapping parameters using empirical subject webcam trials.

## 1. Grid Search Performance Table

| Sigma Gaze (px) | Alpha CM (Prior Weight) | Mean Strict Acc (%) | Mean Group Acc (%) |
|---|---|---|---|
| 25.0 | 0.8 | 1.77% | 14.85% |
| 55.0 | 0.6 | 4.14% | 14.74% |
| 55.0 | 0.8 | 4.14% | 14.74% |
| 55.0 | 1.0 | 4.14% | 14.74% |
| 45.0 | 1.0 | 3.52% | 14.42% |
| 25.0 | 0.6 | 2.41% | 14.15% |
| 15.0 | 0.8 | 1.07% | 14.15% |
| 25.0 | 1.0 | 1.77% | 14.11% |
| 55.0 | 0.0 | 3.06% | 14.00% |
| 55.0 | 0.2 | 3.06% | 14.00% |
| 35.0 | 0.8 | 2.41% | 13.82% |
| 15.0 | 0.0 | 2.09% | 13.81% |
| 25.0 | 0.0 | 2.44% | 13.78% |
| 35.0 | 0.6 | 3.52% | 13.78% |
| 45.0 | 0.6 | 3.17% | 13.78% |
| 45.0 | 0.8 | 3.52% | 13.78% |
| 55.0 | 0.4 | 3.06% | 13.65% |
| 15.0 | 0.4 | 2.44% | 13.43% |
| 25.0 | 0.4 | 2.44% | 13.43% |
| 15.0 | 1.0 | 1.07% | 13.41% |
| 35.0 | 1.0 | 1.77% | 13.41% |
| 15.0 | 0.2 | 2.44% | 13.10% |
| 25.0 | 0.2 | 2.44% | 13.10% |
| 15.0 | 0.6 | 2.11% | 13.10% |
| 35.0 | 0.4 | 2.44% | 12.73% |
| 45.0 | 0.4 | 2.09% | 12.73% |
| 35.0 | 0.0 | 2.44% | 12.41% |
| 45.0 | 0.0 | 2.09% | 12.41% |
| 35.0 | 0.2 | 2.44% | 12.08% |
| 45.0 | 0.2 | 2.09% | 11.71% |

## 2. Conclusion & Integration
- The mathematically optimal configuration is **Sigma Gaze = 25.0 px** and **Alpha CM = 0.8**.
- Setting a moderate foveal search radius filters out minor gaze deviations while preserving layout sequence alignment.
