# Q4 quantum feature reduction

`src.quantum_preprocessing.QuantumFeaturePreprocessor` converts the shared
classical feature vectors into the fixed-width angle inputs required by the
quantum tickets.

The caller supplies the Q2 manifest and feature records keyed by the manifest
sample IDs. The processor fits a full, non-whitened PCA with `n_components`
equal to the selected qubit count (4, 6, or 8) using only
`manifest.splits.train`. Validation and test vectors are transformed with that
already-fitted PCA.

The training split must contain at least as many samples as the selected qubit
count because PCA requires `n_samples >= n_components`. Therefore the 4-, 6-,
and 8-qubit configurations require at least 4, 6, and 8 training samples,
respectively; smaller inputs fail with a clear contract error.

Each reduced component is independently min-max scaled using its training
minimum and maximum to `[-pi, pi]`. Values outside the training range are
clipped before scaling. A constant training component maps to angle `0`.
These rules avoid split leakage and make the encoded angle contract explicit.

`metadata()` / `save_metadata()` persist the PCA components, mean, explained
variance, training extrema, selected qubit count, input dimension, fit scope,
and training IDs. `load_metadata()` validates the schema and all array shapes
before restoring a processor for inference.
