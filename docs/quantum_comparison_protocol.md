# Q0: BISINDO Classical-vs-Quantum Experiment Protocol

This document locks the experiment contract for comparing the completed Phase 1
classical BISINDO classifier with quantum-computing models. Later tickets may
implement the contract, but they must not silently change it. The canonical
machine-readable version is
[`configs/quantum_comparison.json`](../configs/quantum_comparison.json).

## Scope and research question

The benchmark asks whether a quantum-kernel classifier can match or improve the
Phase 1 classical baseline on the same five static BISINDO letters. It is a
comparative experiment, not a claim of quantum advantage. VQC and real hardware
are separate follow-up tracks and are not required for Q0.

## Dataset and primary benchmark

- Sign system: BISINDO.
- Classes: `A`, `B`, `C`, `D`, and `E`.
- Primary benchmark: the YOLO-cropped image pipeline used by Phase 1.
- Source samples are discovered from `images/train/<class>` and
  `images/val/<class>`, combined, and ordered by dataset-relative path.
- Labels come from the class directory. The benchmark preserves the Phase 1
  preprocessing policy: a missing or invalid YOLO label falls back to the full
  image, and the sample remains eligible for the shared split. This means the
  reproduced classical baseline and every quantum run use the same effective
  input policy; the comparison is not restricted to a newly selected
  strict-label subset.
- A label is valid when its first non-blank row has at least five whitespace-
  separated fields, fields 2--5 are finite numeric values, and its width and
  height are positive after parsing and clipping. The first field (YOLO class
  ID) is ignored for class assignment because the parent directory is the
  authoritative class label, matching Phase 1. Multiple rows are allowed but
  ignored after the first non-blank row (first-box authority). A missing label,
  blank file, unreadable label, malformed first row, or empty clipped box is an
  invalid label and triggers the full-image fallback. An unreadable image is
  not a label fallback case and must fail the run clearly.
- There is no augmentation. The split unit is an image, and this protocol does
  not provide participant-held-out evaluation because participant metadata is not
  currently available.

## Shared preprocessing

Both model families start from exactly the same input:

1. Read the image and its first YOLO box.
2. Expand the box by a 10% width and height margin, clipped to image bounds.
   Margin pixels use integer truncation exactly as Phase 1:
   `int(box_width * 0.10)` and `int(box_height * 0.10)`.
3. Convert to grayscale.
4. Resize to `128 x 128` with the repository's existing area interpolation.

The classical baseline extracts HOG from a `64 x 64` working resize using 9
orientations, `8 x 8` pixels per cell, `2 x 2` cells per block, and `L2-Hys`
normalization. The baseline model is the Phase 1 E1 RBF SVM with training-only
macro-F1 grid search over the existing C and gamma values.

Quantum inputs use the same HOG vector, then apply a training-only PCA reducer
and a training-fitted min-max scaler to `[-pi, pi]`, with transformed
validation/test values clipped to that range. The reduced dimension is
equal to the qubit count, so the planned configurations are 4, 6, and 8 qubits
with one reduced feature encoded per qubit. Reducers and scalers must never be
fit on validation or test samples.

## Split and fitting policy

- Random seed: `42`.
- Fractions: 70% train, 15% validation, 15% test.
- Combine the repository's source `train` and `val` partitions before splitting.
- First, stratify 85% train+validation versus 15% test.
- Then, stratify train+validation into 70% train and 15% validation of the full
  dataset (validation is `0.15 / 0.85` of the first-stage remainder).
- Persist the exact sample identifiers, labels, class list, seed, counts, and
  manifest hash in `outputs/quantum_comparison/split_manifest.json`.
- Use train only for preprocessing fitting, hyperparameter selection, and model
  fitting during validation. After selection, refit on train+validation and
  evaluate once on the locked test set.
- Classical and quantum runs must consume the same manifest. A run with a
  different class list or manifest hash is incompatible and must not be merged
  into the comparison report.

## Quantum execution contract

The first quantum model is a multiclass quantum-kernel classifier using a shared
parameterized angle feature map. Ideal local simulation is the default, with
seed `42`, transpilation optimization level `1`, and analytic/statevector
execution (`shots: null`). Noisy simulation is opt-in, uses a shot-based local
simulator with `1024` shots, and must record its explicit noise model. Hardware
execution is outside Q0 and must never be the default test path.

## Evaluation and required reporting

Macro F1 is the primary metric because all five class results should contribute
equally. Report accuracy, per-class precision/recall/F1/support, and a confusion
matrix for both validation and final test results. Every run must also report:

- run/model identity, dataset identity, split manifest path and SHA-256;
- preprocessing and feature set/dimension;
- qubit count, backend, shots, noise model, circuit depth, and gate count;
- random seed, train/validation/test counts;
- fit and prediction runtime; and
- the metric values and confusion matrix.

Results must say whether the quantum model is better, equal, or worse on the
primary metric. They must not claim quantum advantage from a small benchmark,
an ideal simulator, or accuracy alone.

## Output and Phase 1 protection

New artifacts belong under `outputs/quantum_comparison/`:

- `split_manifest.json` for the shared split;
- `classical/` for the reproduced baseline;
- `quantum/ideal/` and `quantum/noisy/` for simulator runs;
- `preprocessors/` for fitted reducers/scalers; and
- `reports/` for comparison tables and summaries.

Every run should be self-contained with configuration, metrics, predictions,
and confusion-matrix artifacts. The existing Phase 1 outputs are protected and
must not be overwritten, deleted, or used as the destination for new results:

- `outputs/phase1/`
- `outputs/phase1_fullimage/`

The Phase 1 artifacts remain the historical baseline; Q0 introduces only the
contract and does not execute or modify experiments.
