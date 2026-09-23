# SDD ledger — plan: docs/superpowers/plans/2026-09-23-bisindo-classical-ml-implementation.md

Workspace note: this project is not a Git repository, so checkpoints are recorded here instead of commits.

Pre-flight: no shared Git interfaces; task dependencies are ordered through Python imports and tested incrementally.

Task 1: complete (tests: `./.venv/bin/python -m pytest tests/test_config.py -q` → 2 passed)
Task 2: complete (tests: `./.venv/bin/python -m pytest tests/test_dataset.py -q` → 5 passed; real scan found 2,182 A–E samples)
Task 3: complete (tests: `./.venv/bin/python -m pytest tests/test_features.py -q` → 4 passed)
Task 4: complete (unit tests: 3 passed; real Phase 1 run produced E0–E3 metrics, final test report, confusion matrix, and best model)
Task 4: Ruling: reduced HOG working image to 64×64, pixel baseline to 32×32, and GridSearchCV to one process after the original run exceeded several minutes without artifacts; this preserves the experiment matrix and leakage-safe 5-fold macro-F1 tuning while keeping PyCharm execution practical.
Task 5: complete (tests: `./.venv/bin/python -m pytest tests/test_predict.py -q` → 1 passed)
Task 6: complete (tests: `./.venv/bin/python -m pytest tests/test_phase2_config.py -q` → 1 passed)
Task 6: Ruling: the explicit A–Z class override was already implemented in Task 1, so the Phase 2 test passed on its first run rather than failing first; the test still verifies non-destructive Phase 2 configuration.
Final verification: `MPLCONFIGDIR=... ./.venv/bin/python -m pytest -q` → 16 passed, 14 dependency deprecation warnings.
Final verification: `python -m src.train --help` and `python -m src.predict --help` → both usage screens printed.
Final verification: prediction CLI loaded `outputs/phase1/models/best_model.joblib` and returned a class for a real validation JPG.
Bug investigation: `flip140.jpg` is labeled A; the original cropped model predicted E on full-image input but predicted A when given its YOLO crop, confirming a train/predict preprocessing mismatch.
Fix: added `ProjectConfig.use_crop`, `--no-crop`, and artifact-aware prediction preprocessing; trained `outputs/phase1_fullimage/models/best_model.joblib`.
Fix verification: full-image model predicts `flip140.jpg` as A; complete suite → 17 passed, 14 dependency deprecation warnings.
