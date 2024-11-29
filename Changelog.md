# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2024-11-29
### Added
- Option to use interpolation and de-normalization from the config file.
- Option to run only render video for a given stat or interpolation or de-normalization instead the whole track process.
- Option to run just the track process. You can enable interpolation or de-normalization post action when running the tracking procedure from the config file
- Coordinate columns can be set from the config file.
- Select which videos to render from the config file in the option **render_videos**.
- Callback logs before and after to keep track of the execution of each one.
- Added class token and confidence for each frame detection in the output stats csv.
- Added command section explanation in README.

### Changed
- Add support for multiple object in the render video step.
- Optimized version of the video render now is up to 2 to 3 times faster.
- Moved Class tracker to tracking **__init__.py** from **main.py** this seems more natural since this could be used from another file other than main.py.

### Fixed
- Open cv where the number of frames is estimated round back to 1 less frame than the actual video had.
- Confidence threshold and nms threshold was not being used.
- De-interpolation was not using the limit from the config file.

### Removed
- Hard-coded code from main.
