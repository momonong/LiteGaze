# 🔬 Real Subject Snapping Accuracy Experiment

This report evaluates the accuracy of gaze-to-word snapped mapping comparing baseline Euclidean snapping against the newly integrated **Adaptive POM Viterbi Auto-Calibration Decoder**.

| Subject ID | WPM | Proficiency | Baseline Group Acc | Static Viterbi Group Acc | Dynamic Sliding Group Acc | Systematic Drift (Y) | Dynamic vs Baseline Improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subject001 | 73.7 | 0.0 | 15.79% | 8.77% | 24.56% | (-62.5px, -62.0px) | +8.77% |
| subject002 | 43.5 | 0.0 | 3.23% | 6.45% | 4.84% | (-56.0px, -83.5px) | +1.61% |
| subject003 | 33.3 | 0.3 | 7.69% | 10.26% | 7.69% | (-38.0px, 102.0px) | +0.00% |
| subject004 | 29.0 | 0.225 | 18.52% | 14.81% | 5.56% | (-69.5px, 61.0px) | -12.96% |
| subject005 | 31.6 | 0.15 | 10.71% | 19.64% | 14.29% | (-48.5px, 71.0px) | +3.57% |


*Evaluation conducted on real participant ground-truth coordinates (`subject001`-`subject005`).*