cmake_minimum_required(VERSION 3.15)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_RAW)
string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION_RAW)
# Extract semver (strip -dev or other suffixes)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PROJECT_SEMVER "${PROJECT_VERSION_RAW}")

project(neuriplo-tasks VERSION ${PROJECT_SEMVER} LANGUAGES CXX)

include(cmake/CompileSpeed.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTS "Build tests" OFF)

# --- Compiler Warnings (applied to project targets only, not third-party) ---
set(PROJECT_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
)

# Treat warnings as errors in CI (opt-in via -DWERROR=ON)
option(WERROR "Treat compiler warnings as errors" OFF)
if(WERROR)
    list(APPEND PROJECT_WARNING_FLAGS -Werror)
endif()

# Enable sanitizers in debug builds (opt-in via -DSANITIZERS=ON)
option(SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(SANITIZERS)
    if(MSVC)
        message(WARNING "SANITIZERS is not configured for MSVC in this project; skipping sanitizer flags")
    else()
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
        message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
    endif()
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# stb single-header libraries (vendored, header-only, public domain).
# Added as a PRIVATE include path to avoid export-set complications.
set(NEURIPLO_STB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb)

# All library sources - UNIFIED ARCHITECTURE
set(CORE_SOURCES
    # Core
    src/core/batch_preprocess.cpp
    src/core/batch_postprocess.cpp
    src/core/base_task.cpp
    src/core/bbox_processor.cpp
    src/core/image_matrix.cpp
    src/core/image_io.cpp
    src/core/image_ops.cpp
    src/core/preprocessor.cpp
    src/core/task_interface.cpp
    src/core/task_factory.cpp
    src/core/task_pipeline.cpp

    # Unified Tasks (no separate postprocessors - all internal)
    src/object_detection/object_detection_task.cpp
    src/object_detection/yolo_postprocessor.cpp
    src/object_detection/rtdetr_postprocessor.cpp
    src/object_detection/rfdetr_postprocessor.cpp
    src/object_detection/edgecrafter_postprocessor.cpp
    src/classification/classification_task.cpp
    src/classification/classification_postprocessor.cpp
    src/classification/torchvision_postprocessor.cpp
    src/classification/vit_postprocessor.cpp
    src/classification/tensorflow_postprocessor.cpp
    src/instance_segmentation/instance_segmentation_task.cpp
    src/instance_segmentation/yolo_segmentation_postprocessor.cpp
    src/instance_segmentation/rfdetr_segmentation_postprocessor.cpp
    src/instance_segmentation/edgecrafter_segmentation_postprocessor.cpp
    src/optical_flow/optical_flow_task.cpp
    src/optical_flow/raft_postprocessor.cpp
    src/open_vocab_detection/clip_tokenizer.cpp
    src/open_vocab_detection/open_vocab_detection_task.cpp
    src/open_vocab_detection/owlv2_postprocessor.cpp
    src/open_vocab_detection/bert_tokenizer.cpp
    src/open_vocab_detection/grounding_dino_postprocessor.cpp
    src/pose_estimation/pose_estimation_task.cpp
    src/pose_estimation/vit_pose_postprocessor.cpp
    src/pose_estimation/yolo_pose_postprocessor.cpp
    src/pose_estimation/edgecrafter_pose_postprocessor.cpp
    src/pose_estimation/rfdetr_pose_postprocessor.cpp
    src/video_classification/video_classification_task.cpp
    src/video_classification/video_classification_preprocessor.cpp
    src/video_classification/video_classification_postprocessor.cpp

    # Preprocessors (still separate as they're reusable)
    src/object_detection/detection_preprocessor.cpp
    src/classification/classification_preprocessor.cpp
    src/optical_flow/optical_flow_preprocessor.cpp
    src/depth_estimation/depth_estimation_task.cpp
    src/depth_estimation/depth_estimation_preprocessor.cpp
    src/depth_estimation/depth_anything_v2_postprocessor.cpp
    src/gaussian_splatting/gaussian_splatting_task.cpp
    src/gaussian_splatting/gaussian_splatting_preprocessor.cpp
    src/gaussian_splatting/lgm_postprocessor.cpp
    src/image_understanding/image_understanding_task.cpp
)

set(CORE_HEADERS
    # Core
    include/neuriplo/tasks/core/bounding_box.hpp
    include/neuriplo/tasks/core/image.hpp
    include/neuriplo/tasks/core/image_io.hpp
    include/neuriplo/tasks/core/image_matrix.hpp
    include/neuriplo/tasks/core/opencv_interop.hpp
    include/neuriplo/tasks/core/result_types.hpp
    include/neuriplo/tasks/core/batch_types.hpp
    include/neuriplo/tasks/core/batch_preprocess.hpp
    include/neuriplo/tasks/core/batch_postprocess.hpp
    include/neuriplo/tasks/core/base_task.hpp
    include/neuriplo/tasks/core/bbox_processor.hpp
    include/neuriplo/tasks/core/preprocessor.hpp
    include/neuriplo/tasks/core/model_info.hpp
    include/neuriplo/tasks/core/task_interface.hpp
    include/neuriplo/tasks/core/task_factory.hpp
    include/neuriplo/tasks/core/task_pipeline.hpp

    # Unified Task Headers (postprocessing is internal)
    include/neuriplo/tasks/object_detection/object_detection_task.hpp
    include/neuriplo/tasks/object_detection/postprocessor.hpp
    include/neuriplo/tasks/object_detection/yolo_postprocessor.hpp
    include/neuriplo/tasks/object_detection/rtdetr_postprocessor.hpp
    include/neuriplo/tasks/object_detection/rfdetr_postprocessor.hpp
    include/neuriplo/tasks/object_detection/edgecrafter_postprocessor.hpp
    include/neuriplo/tasks/classification/classification_task.hpp
    include/neuriplo/tasks/classification/classification_logits_layout.hpp
    include/neuriplo/tasks/classification/classification_postprocessor.hpp
    include/neuriplo/tasks/classification/torchvision_postprocessor.hpp
    include/neuriplo/tasks/classification/vit_postprocessor.hpp
    include/neuriplo/tasks/classification/tensorflow_postprocessor.hpp
    include/neuriplo/tasks/instance_segmentation/instance_segmentation_task.hpp
    include/neuriplo/tasks/instance_segmentation/segmentation_postprocessor.hpp
    include/neuriplo/tasks/instance_segmentation/yolo_segmentation_postprocessor.hpp
    include/neuriplo/tasks/instance_segmentation/rfdetr_segmentation_postprocessor.hpp
    include/neuriplo/tasks/instance_segmentation/edgecrafter_segmentation_postprocessor.hpp
    include/neuriplo/tasks/optical_flow/optical_flow_task.hpp
    include/neuriplo/tasks/optical_flow/optical_flow_postprocessor.hpp
    include/neuriplo/tasks/optical_flow/raft_postprocessor.hpp
    include/neuriplo/tasks/open_vocab_detection/clip_tokenizer.hpp
    include/neuriplo/tasks/open_vocab_detection/open_vocab_detection_task.hpp
    include/neuriplo/tasks/open_vocab_detection/owlv2_postprocessor.hpp
    include/neuriplo/tasks/open_vocab_detection/postprocessor.hpp
    include/neuriplo/tasks/open_vocab_detection/bert_tokenizer.hpp
    include/neuriplo/tasks/open_vocab_detection/grounding_dino_postprocessor.hpp
    include/neuriplo/tasks/pose_estimation/pose_estimation_task.hpp
    include/neuriplo/tasks/pose_estimation/pose_postprocessor.hpp
    include/neuriplo/tasks/pose_estimation/vit_pose_postprocessor.hpp
    include/neuriplo/tasks/pose_estimation/yolo_pose_postprocessor.hpp
    include/neuriplo/tasks/pose_estimation/edgecrafter_pose_postprocessor.hpp
    include/neuriplo/tasks/pose_estimation/rfdetr_pose_postprocessor.hpp
    include/neuriplo/tasks/video_classification/video_classification_task.hpp
    include/neuriplo/tasks/video_classification/video_classification_preprocessor.hpp
    include/neuriplo/tasks/video_classification/video_classification_postprocessor.hpp

    # Preprocessor Headers (still separate as they're reusable)
    include/neuriplo/tasks/object_detection/detection_preprocessor.hpp
    include/neuriplo/tasks/classification/classification_preprocessor.hpp
    include/neuriplo/tasks/optical_flow/optical_flow_preprocessor.hpp
    include/neuriplo/tasks/depth_estimation/depth_estimation_task.hpp
    include/neuriplo/tasks/depth_estimation/depth_estimation_preprocessor.hpp
    include/neuriplo/tasks/depth_estimation/depth_estimation_postprocessor.hpp
    include/neuriplo/tasks/depth_estimation/depth_anything_v2_postprocessor.hpp
    include/neuriplo/tasks/gaussian_splatting/gaussian_splatting_task.hpp
    include/neuriplo/tasks/gaussian_splatting/gaussian_splatting_preprocessor.hpp
    include/neuriplo/tasks/gaussian_splatting/gaussian_splatting_postprocessor.hpp
    include/neuriplo/tasks/gaussian_splatting/lgm_postprocessor.hpp
    include/neuriplo/tasks/image_understanding/image_understanding_task.hpp
)

# Create library
add_library(neuriplo-tasks ${CORE_SOURCES} ${CORE_HEADERS})
add_library(neuriplo-tasks::neuriplo-tasks ALIAS neuriplo-tasks)

if(WIN32 AND BUILD_SHARED_LIBS)
    set_target_properties(neuriplo-tasks PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(neuriplo-tasks
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${NEURIPLO_STB_INCLUDE_DIR}
)

target_link_libraries(neuriplo-tasks
    PUBLIC
        ${OpenCV_LIBS}
)

# Set compile features and warnings
target_compile_features(neuriplo-tasks PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(neuriplo-tasks PRIVATE /W4)
    if(WERROR)
        target_compile_options(neuriplo-tasks PRIVATE /WX)
    endif()
else()
    target_compile_options(neuriplo-tasks PRIVATE ${PROJECT_WARNING_FLAGS})
endif()

# Install rules
include(GNUInstallDirs)

install(TARGETS neuriplo-tasks
    EXPORT neuriplo-tasks-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT neuriplo-tasks-targets
    FILE neuriplo-tasks-targets.cmake
    NAMESPACE neuriplo-tasks::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuriplo-tasks
)

# Create config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/neuriplo-tasks-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/neuriplo-tasks-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuriplo-tasks
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/neuriplo-tasks-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/neuriplo-tasks-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/neuriplo-tasks-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuriplo-tasks
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "neuriplo-tasks configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
