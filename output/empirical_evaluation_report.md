# 🔬 Real Subject Snapping Accuracy Experiment

This report evaluates the accuracy of gaze-to-word snapped mapping comparing baseline Euclidean snapping against the newly integrated **Adaptive POM Viterbi Auto-Calibration Decoder**.

| Subject ID | WPM | Proficiency | Baseline Snap Acc | Adaptive Viterbi Acc | Systematic Drift (Y) | Improvement |
| --- | --- | --- | --- | --- | --- | --- |
| subject001 | 73.7 | 0.0 | 3.51% | 3.51% | (-62.5px, -62.0px) | +0.00% |
| subject002 | 43.5 | 0.0 | 0.00% | 0.00% | (-56.0px, -83.5px) | +0.00% |
| subject003 | 33.3 | 0.3 | 2.56% | 5.13% | (-38.0px, 102.0px) | +2.56% |
| subject004 | 29.0 | 0.225 | 1.85% | 3.70% | (-69.5px, 61.0px) | +1.85% |
| subject005 | 31.6 | 0.15 | 1.79% | 5.36% | (-48.5px, 71.0px) | +3.57% |


*Evaluation conducted on real participant ground-truth coordinates (`subject001`-`subject005`).*