# Remove OpenCV from `neuriplo-tasks` — Migration Plan

**Status:** Approved
**Strategy:** Option A (full break) + public stb-backed load/save helpers
**Target version:** `v0.6.0`
**Branch:** `feat/remove-opencv-stb` (shared across all 4 repos)

## Locked design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Branch name | `feat/remove-opencv-stb` | Shared across `neuriplo-tasks`, `neuriplo-infer`, `neuriplo-track`, `tritonic` |
| Image ownership | `Image` (owning) + `ImageView` (non-owning zero-copy view) | Consumers wrap `cv::Mat::data` zero-copy; preprocessors own output |
| `INTER_AREA` / `INTER_CUBIC` | Hand-roll both, numerically faithful | Avoids silent numeric drift in depth/flow outputs |
| Release version | `v0.6.0` | Pre-1.0 minor bump per `docs/Versioning.md` convention |
| Public load/save | Yes — `loadImage` / `saveImage` backed by stb | Lets consumers be OpenCV-free end-to-end |
| `ModelInfo::input_types` | `std::vector<PixelType>` enum (was `std::vector<int>` of CV_* codes) | Removes OpenCV macro leak through public struct |

## Background

`neuriplo-tasks` currently has OpenCV as its only runtime dependency (`find_package(OpenCV REQUIRED)` at `CMakeLists.txt:59`). OpenCV leaks through the public API in three load-bearing seams:

1. `TaskInterface::preprocess(std::vector<cv::Mat>&)` (`task_interface.hpp:74`)
2. `TaskInterface::postprocess(const cv::Size&, ...)` (`task_interface.hpp:82`)
3. `ModelInfo::input_types` carrying `CV_*` integer codes (`model_info.hpp:23,54`)

The existing facade (`image_matrix.hpp`, `bounding_box.hpp`, `opencv_interop.hpp`) only insulates the `Result` variant types — the input path, postprocess control path, and `ModelInfo` dtype field all require consumers to link OpenCV.

Three downstream consumers pin `neuriplo-tasks` at `v0.5.0` via `FetchContent` and hit the same API seams.

## OpenCV usage surface in `neuriplo-tasks` (from inventory)

**Distinct symbols used (~50):** `Mat`, `Size`, `Rect`, `Point`, `Point2f`, `Scalar`, `Vec3b`; constants `CV_8UC1/CV_8UC3/CV_32F/CV_32S/CV_32FC1/CV_32FC2/CV_32FC3/CV_32FC(n)/CV_PI`, `INTER_LINEAR/INTER_AREA/INTER_CUBIC`, `COLOR_BGR2RGB`, `THRESH_BINARY`; free functions `cvtColor`, `resize`, `split`, `merge`, `threshold`, `minMaxLoc`, `cartToPolar`, `rectangle` (tests only), `dnn::NMSBoxes`; Mat members `clone`, `copyTo`, `convertTo`, `isContinuous`, `total`, `channels`, `elemSize`, `empty`, `rows`, `cols`, `size`, `type`, `ptr<T>`, `at<T>`, `data`, `operator()(Rect)`; Mat expressions `*=`/`/=`/`- scalar`/`/ scalar`; facade helpers `toCvMat`/`mutableCvMat`/`fromCvMat`/`toCvRect`/`fromCvRect`.

**Image I/O (`imread`/`imwrite`/`imdecode`/`imencode`):** NOT USED. Consumers hand in already-decoded `cv::Mat`s. stb is needed only for the new public `loadImage`/`saveImage` helpers and internally for nothing (the library never loads files today).

**Already hand-rolled:** A complete greedy IoU NMS exists at `src/object_detection/yolo_postprocessor.cpp:413-454` — promote this to replace the 2 `cv::dnn::NMSBoxes` sites.

## Consumer coupling (per repo)

### `neuriplo-track` (lightest)
- Pin: `versions.env:6` = `v0.5.0`, sibling-override active.
- Links `neuriplo-tasks::neuriplo-tasks` (`app/CMakeLists.txt:37`, `trackers/CMakeLists.txt:58` PUBLIC).
- Single-frame object-detection only:
  - 1 `preprocess({frame})` site (`MultiObjectTrackingApp.cpp:122`).
  - 1 `postprocess(frame.size(), ...)` site (`:133`).
  - `toCvRect(detection.bbox)` at 3 sites (`MultiObjectTrackingApp.cpp:182`, `SortWrapper.cpp:22`, `BoTSORTWrapper.cpp:22`); ByteTrack bypasses and reads `bbox.x/.y/.width/.height` directly.
- `ModelInfo` construction already OpenCV-free (copies `name`/`shape`/`batch_size` from `neuriplo::InferenceMetadata`).
- No `ImageMatrix`/`toCvMat`/batch API usage.
- OpenCV remains load-bearing for SORT (`cv::KalmanFilter`), BoTSORT (`cv::dnn` ReID, `cv::videostab` GMC, `cv::features2d`), app-level `VideoCapture`/drawing.

### `neuriplo-infer` (heaviest single-frame)
- Pin: `versions.env:22` = `v0.5.0`.
- Links **un-namespaced** target `neuriplo-tasks` (`app/CMakeLists.txt:47-53`).
- 7 `preprocess({cv::Mat})` + 7 `postprocess(cv::Size,...)` call sites in `app/src/CLICommands.cpp`.
- Uses all 10 `Result` alternatives in `ResultRenderer.cpp`; `toCvMat` for `mask`/`flow`/`depth` at `:113,:140,:196,:198` + `CLICommands.cpp:322`; `toCvRect` at `:49,61,106`.
- `ModelInfo::input_types[0] = CV_32F` at `InferencePipeline.cpp:90`.
- Two dead files (`NeuriploInferProcessing.cpp`, `NeuriploInferRendering.cpp`) not in `APP_LIB_SOURCES` — delete or migrate.
- No batch API usage.

### `tritonic` (heaviest, batch path + dtype plumbing)
- Pin: `CMakeLists.txt:54` = `v0.5.0`.
- Coupling concentrated in `include/App.hpp` + `src/main/App.cpp` (~50 sites).
- Only consumer of batch API: `BatchRequest{vector<cv::Mat>}` (`App.cpp:420`), `batchPreprocess` (`:421`), `batchPostprocess(*task_, cv::Size, ...)` (`:370`).
- 3 `postprocess(cv::Size,...)` sites (`:258, :370, :387`).
- `toCvMat`/`toCvRect` at 7 sites in `renderPrediction` (`:547, 691, 803, 812, 820, 841, 847`).
- `tritonic::triton::ModelInfo` defaults `type1_{CV_32FC1}`/`type3_{CV_32FC3}` (`include/tritonic/triton/model_info.hpp:21-22`); emits `CV_32F/CV_32S/CV_8U` in `Triton.cpp:200-212, 318-332, 420-457`. Forwarded verbatim at `App.cpp:248`.
- Legacy `include/TaskInterface.hpp` (tritonic's own, OpenCV-typed virtuals) — verify dead, delete if so.
- No tests touch neuriplo-tasks; API breakage won't be caught by unit tests.

## Execution phases

### Phase 0 — Setup
- [ ] Create `feat/remove-opencv-stb` branch in all 4 repos from `develop`.
- [ ] Sibling-override paths in each consumer's CMake already point at `/home/oli/repos/neuriplo-tasks` — all four build against the in-flight branch with zero fetch.

### Phase 1 — Vendor stb + new core types (additive, non-breaking)
- [ ] Vendor `3rdparty/stb/{stb_image.h, stb_image_resize2.h, stb_image_write.h}` (single-header, MIT).
- [ ] CMake: add INTERFACE include-only target `neuriplo::stb`.
- [ ] Add `include/neuriplo/tasks/core/image.hpp`:
  - `enum class PixelType : uint8_t { UInt8, Float32, Int32 }` — replaces `CV_8U/CV_32F/CV_32S`.
  - `struct Size { int width; int height; }` — replaces `cv::Size` (>=100 sites).
  - `class Image` — owning contiguous buffer (rows x cols x channels x PixelType); methods `rows()/cols()/channels()/pixelType()/empty()/clone()/subregion(BoundingBox)/convertTo(PixelType, alpha, beta)/data<T>()/size_bytes()`.
  - `class ImageView` — non-owning view over external memory; same accessor surface.
- [ ] Add `include/neuriplo/tasks/core/image_io.hpp`:
  - `Image loadImage(const std::string& path, int desired_channels = 3)` — wraps `stbi_load`.
  - `bool saveImage(const std::string& path, const Image&)` — wraps `stbi_write_*` by extension.
- [ ] Add private `src/core/image_ops.cpp`:
  - `resizeLinear`, `resizeArea` (hand-rolled area-pixel downsample), `resizeCubic` (hand-rolled bicubic).
  - `cvtColorBgrRgb` (channel swap).
  - `thresholdBinary`.
  - `nms` — promote existing implementation from `yolo_postprocessor.cpp:413-454`.
- [ ] Add new headers to `CORE_HEADERS` in `CMakeLists.txt`.

### Phase 2 — Internal `src/` rewrite (behind still-OpenCV public API)
Domain-by-domain, swap internal `cv::Mat` engine for `Image`. Order = lightest surface first.
- [ ] `src/core/preprocessor.cpp` (canonical reference, 8 cv calls) — template for preprocessors.
- [ ] `src/{object_detection,classification,video_classification,optical_flow,depth_estimation}/...preprocessor.cpp`.
- [ ] `src/{object_detection,instance_segmentation,pose_estimation,open_vocab_detection,optical_flow,depth_estimation,gaussian_splatting,image_understanding}/...postprocessor.cpp`.
- [ ] Replace `cv::dnn::NMSBoxes` (`yolo_postprocessor.cpp:321`, `yolo_pose_postprocessor.cpp:67`) with `image_ops::nms`.
- [ ] `src/core/image_matrix.cpp` — `Impl` swaps `cv::Mat` -> `Image`; `type()` returns `PixelType`.
- [ ] `src/core/bbox_processor.cpp` — pure coordinate math; drop OpenCV include.

### Phase 3 — Public API flip (breaking — single coordinated commit)
- [ ] `TaskInterface::preprocess(vector<cv::Mat>)` -> `preprocess(vector<Image>)` (`task_interface.hpp:74`).
- [ ] `TaskInterface::postprocess(cv::Size, ...)` -> `postprocess(Size, ...)` (`task_interface.hpp:82`).
- [ ] `BaseTask`, 5 postprocessor base classes, all concrete task headers, `PreprocessConfig::input_size`, `Preprocessor::preprocess`, `batchPostprocess(...)`.
- [ ] `BatchRequest::images` -> `vector<Image>` (`batch_types.hpp:21`).
- [ ] `ModelInfo::input_types` type changes from `vector<int>` to `vector<PixelType>` (`model_info.hpp:23,54`); default `CV_32F` -> `PixelType::Float32`.
- [ ] Delete `include/neuriplo/tasks/core/opencv_interop.hpp`. Remove `cv::Mat` friend declarations in `image_matrix.hpp:33-34`.
- [ ] CMake: drop `find_package(OpenCV REQUIRED)` (`CMakeLists.txt:59`) and `${OpenCV_LIBS}` (`:201-204`). Library now has zero runtime deps.

### Phase 4 — Tests
- [ ] Drop `${OpenCV_LIBS}` from `tests/CMakeLists.txt:46` and all per-target lists.
- [ ] Rewrite ~80 `cv::Mat::zeros`/`cv::Mat(...)` constructor sites to `Image::zeros`/`Image(...)`.
- [ ] Remove `cv::rectangle` at `test_optical_flow.cpp:32,35` and `test_result_types.cpp:159`.
- [ ] Fix `.type() == CV_32FC2` assertions in `test_result_types.cpp:165`, `test_depth_estimation.cpp:64-65` -> `PixelType::Float32` (+ channel count).

### Phase 5 — Docs
- [ ] `AGENTS.md` — remove OpenCV references (dev setup, code quality table, conventions, the one-local-gate command).
- [ ] `README.md` — Features list, dependency statement.
- [ ] `REPO_META.yaml` — allowed change classes, build/test commands.
- [ ] `docs/ROADMAP.md`, `docs/batch_support_matrix.md` — references.
- [ ] This document is the design note referenced in Phase 0.

### Phase 6 — Consumer migrations (parallelizable, one PR per repo)

#### `neuriplo-track`
- [ ] `MultiObjectTrackingApp.cpp:122` `preprocess({frame})` -> `preprocess({wrap(frame)})`.
- [ ] `MultiObjectTrackingApp.cpp:133` `frame.size()` -> `Size{frame.cols, frame.rows}`.
- [ ] 3 sites — `toCvRect(bbox)` -> `cv::Rect(bbox.x, bbox.y, bbox.width, bbox.height)`.
- [ ] Add consumer-side adapter `Image wrap(const cv::Mat&)` (non-owning view over `frame.data`).
- [ ] Bump `versions.env:6` -> `v0.6.0`; `CHANGELOG.md`; tag.

#### `neuriplo-infer`
- [ ] `InferencePipeline.cpp:90` `model_info.input_types[0] = CV_32F` -> `PixelType::Float32`.
- [ ] 7 `preprocess({cv::Mat})` sites in `CLICommands.cpp` -> wrap.
- [ ] 7 `postprocess(cv::Size,...)` sites -> `Size{...}`.
- [ ] `ResultRenderer.cpp:113,140,196,198` + `CLICommands.cpp:322` — `toCvMat(...)` -> consumer-side `unwrap()` helper.
- [ ] `ResultRenderer.cpp:49,61,106` — `toCvRect(bbox)` -> direct ctor.
- [ ] Delete or migrate dead files `NeuriploInferProcessing.cpp`/`NeuriploInferRendering.cpp`.
- [ ] Bump `versions.env:22` -> `v0.6.0`; `CHANGELOG.md`; tag.

#### `tritonic`
- [ ] `App.cpp:255,258,370,387,420-421` — wrap/Size/batch.
- [ ] `App.cpp:547,691,820,841` `toCvMat`; `:803,812,847` `toCvRect` -> adapter/direct ctor.
- [ ] `tritonic::triton::ModelInfo` (`include/tritonic/triton/model_info.hpp:21-22`): replace `type1_{CV_32FC1}`/`type3_{CV_32FC3}` and `Triton.cpp:200-212,318-332,420-457` int mappings with `PixelType`.
- [ ] Verify legacy `include/TaskInterface.hpp` is dead; delete if so.
- [ ] `App.cpp:241-252 convertToNeuriploTasksModelInfo()` — simplify once both sides use `PixelType`.
- [ ] Bump `CMakeLists.txt:54` pin -> `v0.6.0`; `CHANGELOG.md`; tag.

### Phase 7 — Release
- [ ] `neuriplo-tasks`: `CHANGELOG.md` entries under `### Changed` / `### Removed` / `### Added`; bump `VERSION` to `0.6.0`; tag `v0.6.0`; `gh release create` with changelog notes (no `--generate-notes`).
- [ ] Consumers in order: neuriplo-track -> neuriplo-infer -> tritonic (lightest to heaviest).
- [ ] `neuriplo-tasks` tag must exist first (FetchContent resolves the tag).

## Consumer adapter pattern

Since `neuriplo-tasks` will not ship an OpenCV interop header under Option A, each consumer adds a ~10-line adapter:

```cpp
// Consumer-side: cv::Mat -> neuriplo_tasks::ImageView (zero-copy)
neuriplo_tasks::ImageView wrap(const cv::Mat& m) {
    return neuriplo_tasks::ImageView(m.data, m.cols, m.rows, m.channels(),
        m.depth() == CV_32F ? neuriplo_tasks::PixelType::Float32
                            : neuriplo_tasks::PixelType::UInt8);
}

// Consumer-side: neuriplo_tasks::ImageMatrix -> cv::Mat (copy)
cv::Mat unwrap(const neuriplo_tasks::ImageMatrix& matrix, int cv_type) {
    cv::Mat m(matrix.rows(), matrix.cols(), cv_type);
    std::memcpy(m.data, matrix.data(), static_cast<size_t>(m.total()) * m.elemSize());
    return m;
}
```

## Validation gates

### Per-commit (neuriplo-tasks)
- clang-format-18, clang-tidy-18, cppcheck, `-DWERROR=ON`, ctest, valgrind (the full local gate from `AGENTS.md`).

### Per-tag (each repo)
- The local gate above.
- Consumer builds against the new `neuriplo-tasks` tag via sibling-override.
- `CHANGELOG.md` entry and GitHub Release created with `--notes` (never `--generate-notes`).

## Risk register

| Risk | Mitigation |
|---|---|
| `INTER_CUBIC` depth results drift after hand-roll | Keep OpenCV build available in a side branch for numeric diff testing during Phase 2 |
| `PixelType` enum ABI change breaks tritonic's `vector<int>` field | Tritonic migration in same release window; coordinated tag bump |
| Dead file `NeuriploInferRendering.cpp` references undeclared `getTaskType` (`:237`) | Already non-building; delete in Phase 6 |
| Legacy tritonic `TaskInterface.hpp` blocks include cleanup | Verify no `.cpp` includes it; delete if dead |
| stb's `stbi_load` defaulting to RGB vs OpenCV's BGR | `Image` stores channel order implicitly as RGB (stb convention); consumers wrapping BGR `cv::Mat` must swap or document |
