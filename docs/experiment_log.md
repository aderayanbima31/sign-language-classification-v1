# Experiment Log

## Phase 1 configuration

- Sign system: BISINDO
- Classes: A, B, C, D, E
- Image preprocessing: optional YOLO crop, grayscale, 128×128
- Random seed: 42
- Final test: stratified image-level test split
- Primary tuning metric: macro F1

## Results

| ID | Features | Model | Validation accuracy | Validation macro F1 |
|---|---|---|---:|---:|
| E0 | resized grayscale pixels | KNN | 0.9085 | 0.9082 |
| E1 | HOG | RBF SVM | 0.9817 | 0.9817 |
| E2 | HOG | Random Forest | 0.9756 | 0.9756 |
| E3 | HOG + Hu/contour shape | RBF SVM | 0.9817 | 0.9817 |

## Final test

E1 HOG + SVM was selected. On 328 final test images it achieved accuracy **0.9909** and macro F1 **0.9909**. The confusion matrix is saved at `outputs/phase1/figures/final_confusion_matrix.png`; the complete report is in `outputs/phase1/metrics/final_test.json`.

The observed errors were one A predicted as B, one B predicted as E, and one D predicted as B. These should be investigated further with participant-held-out data before drawing broader conclusions.

## Full-image retraining

The initial model trained on YOLO-cropped images but terminal prediction used full images. A separate model was retrained with `--no-crop` so both stages use the same preprocessing. It correctly predicts the previously reported `flip140.jpg` sample as `A`.

The full-image model achieved validation accuracy 0.9512 and final-test accuracy 0.9573. This is lower than the cropped model, but it is the consistent model for images without annotation files and for the future camera workflow.
