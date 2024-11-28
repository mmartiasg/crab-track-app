# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- Option to use interpolation and de-normalization from the config file
- Option to run only render video for a given stat or interpolation or de-normalization instead the whole track process
- Option to run just the track process following the settings in the config file about de-normalization and interpolation.
- Bugfix confidence threshold and nms threshold was not being used.
- Optimized version of the video render

### Changed
- Add support for multiple object in the render video step.

### Fixed
- video render had a bug from open cv where the number of frames is estimated round back to 1 less frame than the actual video had.

### Removed
- Features or functionality that have been removed.

## [0.3.1] - 2024-11-27


