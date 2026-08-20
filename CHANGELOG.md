# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-08-20

### Changed

- Temperature calibration now uses scikit-learn's native
  `CalibratedClassifierCV(method="temperature")` when available, while keeping
  the package backport for older scikit-learn releases.
- Python 3.10 support has been restored.
- Runtime dependencies now allow older scikit-learn and aeon releases again,
  and add `packaging` for scikit-learn version detection.

### Deprecated

- Support for scikit-learn `<1.8.0` in `TemperatureCalibratedClassifierCV` is
  deprecated and will be removed in v2.0.0.

## [1.4.0] - 2026-05-22

### Changed

- Runtime API can now take either integer, iterable or cross-validation generator by `cv` argument.

## Deprecated

- `n_splits` argument for runtime API is deprecated. Use `cv` instead.

  This argument is still valid for command line API.

## [1.3.0] - 2026-05-21

### Added

- `n-jobs` can now be specified for training.
- Add `temperature` to calibration option.
- Add `sigmoid_ovo` to calibration option.
- Add `isotonic_ovo` to calibration option.

## [1.2.0] - 2026-05-04

### Added

- Calibration options can now be specified.

## [1.1.1] - 2026-03-23

### Fixed

- `get_sample_path()` now builds non-subprocess recipes without error.

## [1.1.0] - 2026-03-23

### Added

- `classify-train` command now takes `--label-format` argument.
- `classify-predict` command now takes `--output-format` argument.
- `classify-predict` command now takes `--label-type` argument.

## [1.0.1] - 2026-03-19

### Added

- Normalize profiles by default.

## [1.0.0] - 2026-03-18

### Added

- MiniRocket-based classification model.
