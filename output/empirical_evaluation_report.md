# 🔬 Real Subject Snapping Accuracy Experiment

This report evaluates the accuracy of gaze-to-word snapped mapping comparing baseline Euclidean snapping against the newly integrated **Adaptive POM Viterbi Auto-Calibration Decoder**.

| Subject ID | WPM | Proficiency | Baseline Group Acc | Static Viterbi Group Acc | Dynamic Sliding Group Acc | Systematic Drift (Y) | Dynamic vs Baseline Improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subject001 | 73.7 | 0.0 | 15.79% | 17.54% | 8.77% | (-1.5px, -30.0px) | -7.02% |
| subject002 | 43.5 | 0.0 | 3.23% | 1.61% | 3.23% | (5.0px, -101.5px) | +0.00% |
| subject003 | 33.3 | 0.3 | 7.69% | 7.69% | 7.69% | (-23.5px, 102.0px) | +0.00% |
| subject004 | 29.0 | 0.225 | 18.52% | 14.81% | 5.56% | (-60.5px, 61.5px) | -12.96% |
| subject005 | 31.6 | 0.15 | 10.71% | 7.14% | 12.50% | (26.5px, 60.5px) | +1.79% |


*Evaluation conducted on real participant ground-truth coordinates (`subject001`-`subject005`).*