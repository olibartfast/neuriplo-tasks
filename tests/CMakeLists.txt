# Ensure minimum policies
cmake_minimum_required(VERSION 3.15)

option(NEURIPLO_TASKS_BUNDLED_GTEST "Download and build googletest" ON)

if(NEURIPLO_TASKS_BUNDLED_GTEST)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Provide consistent target names for the rest of the build
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main ALIAS gtest_main)
else()
    find_package(GTest REQUIRED)
endif()

include(GoogleTest)

function(target_link_whole_archive target library_target)
    if(MSVC)
        target_link_libraries(${target} ${library_target})
        target_link_options(${target} PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:${library_target}>")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_libraries(${target}
            "-Wl,--whole-archive"
            ${library_target}
            "-Wl,--no-whole-archive"
        )
    else()
        target_link_libraries(${target} ${library_target})
    endif()
endfunction()

# Common test libraries
set(TEST_LIBS
    neuriplo-tasks::neuriplo-tasks
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Core tests
add_executable(test_result_types test_result_types.cpp)
target_link_libraries(test_result_types ${TEST_LIBS})

add_executable(test_batch_types test_batch_types.cpp)
target_link_libraries(test_batch_types ${TEST_LIBS})

add_executable(test_batch_preprocess test_batch_preprocess.cpp)
target_link_libraries(test_batch_preprocess
    -Wl,--whole-archive neuriplo-tasks -Wl,--no-whole-archive
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

add_executable(test_batch_postprocess test_batch_postprocess.cpp)
target_link_libraries(test_batch_postprocess
    -Wl,--whole-archive neuriplo-tasks -Wl,--no-whole-archive
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

add_executable(test_classification_batch test_classification_batch.cpp)
target_link_libraries(test_classification_batch
    -Wl,--whole-archive neuriplo-tasks -Wl,--no-whole-archive
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

add_executable(test_object_detection_batch test_object_detection_batch.cpp)
target_link_libraries(test_object_detection_batch
    -Wl,--whole-archive neuriplo-tasks -Wl,--no-whole-archive
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

add_executable(test_depth_estimation_batch test_depth_estimation_batch.cpp)
target_link_libraries(test_depth_estimation_batch
    -Wl,--whole-archive neuriplo-tasks -Wl,--no-whole-archive
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

add_executable(test_pose_estimation_batch test_pose_estimation_batch.cpp)
target_link_libraries(test_pose_estimation_batch
    -Wl,--whole-archive neuriplo-tasks -Wl,--no-whole-archive
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

add_executable(test_batch_integration test_batch_integration.cpp)
target_link_libraries(test_batch_integration
    -Wl,--whole-archive neuriplo-tasks -Wl,--no-whole-archive
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

add_executable(test_task_interface test_task_interface.cpp)
target_link_libraries(test_task_interface ${TEST_LIBS})

add_executable(test_task_factory test_task_factory.cpp)
target_link_libraries(test_task_factory ${TEST_LIBS})

add_executable(test_task_pipeline test_task_pipeline.cpp)
target_link_libraries(test_task_pipeline ${TEST_LIBS})

# Postprocessor tests
add_executable(test_yolo_postprocessor test_yolo_postprocessor.cpp)
target_link_libraries(test_yolo_postprocessor ${TEST_LIBS})

add_executable(test_rtdetr_postprocessor test_rtdetr_postprocessor.cpp)
target_link_libraries(test_rtdetr_postprocessor ${TEST_LIBS})

add_executable(test_yolo_segmentation_postprocessor test_yolo_segmentation_postprocessor.cpp)
target_link_libraries(test_yolo_segmentation_postprocessor ${TEST_LIBS})

add_executable(test_bbox_processor test_bbox_processor.cpp)
target_link_libraries(test_bbox_processor ${TEST_LIBS})

# TaskConfig tests (struct defaults + factory integration + behavioral)
add_executable(test_task_config test_task_config.cpp)
target_link_whole_archive(test_task_config neuriplo-tasks)
target_link_libraries(test_task_config
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Concrete task implementation tests
add_executable(test_concrete_tasks test_concrete_tasks.cpp)
target_link_whole_archive(test_concrete_tasks neuriplo-tasks)
target_link_libraries(test_concrete_tasks
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Optical flow tests
add_executable(test_optical_flow test_optical_flow.cpp)
target_link_whole_archive(test_optical_flow neuriplo-tasks)
target_link_libraries(test_optical_flow
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Pose estimation tests
add_executable(test_pose_estimation test_pose_estimation.cpp)
target_link_whole_archive(test_pose_estimation neuriplo-tasks)
target_link_libraries(test_pose_estimation
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Depth estimation tests
add_executable(test_depth_estimation test_depth_estimation.cpp)
target_link_whole_archive(test_depth_estimation neuriplo-tasks)
target_link_libraries(test_depth_estimation
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Open-vocabulary detection tests
add_executable(test_open_vocab_detection test_open_vocab_detection.cpp)
target_link_whole_archive(test_open_vocab_detection neuriplo-tasks)
target_link_libraries(test_open_vocab_detection
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Gaussian splatting tests
add_executable(test_gaussian_splatting test_gaussian_splatting.cpp)
target_link_whole_archive(test_gaussian_splatting neuriplo-tasks)
target_link_libraries(test_gaussian_splatting
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# EdgeCrafter postprocessor and task tests
add_executable(test_edgecrafter test_edgecrafter.cpp)
target_link_whole_archive(test_edgecrafter neuriplo-tasks)
target_link_libraries(test_edgecrafter
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Docs/code contract: every quoted alias in README Supported Model Types block
# must route via TaskFactory. Locks neuriplo-tasks/README.md to task_factory.cpp.
add_executable(test_readme_model_types test_readme_model_types.cpp)
target_link_whole_archive(test_readme_model_types neuriplo-tasks)
target_link_libraries(test_readme_model_types
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)
# Paths are relative to this tests/ directory so they resolve correctly whether
# neuriplo-tasks is the top-level project or a FetchContent sub-project of another
# repo (CMAKE_SOURCE_DIR would point at the consumer's root in the latter case).
target_compile_definitions(test_readme_model_types
    PRIVATE
        NEURIPLO_TASKS_README_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../README.md"
        NEURIPLO_TASKS_TASK_FACTORY_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../src/core/task_factory.cpp"
)

# Add tests manually for better compatibility
add_test(NAME test_result_types COMMAND test_result_types)
add_test(NAME test_batch_types COMMAND test_batch_types)
add_test(NAME test_batch_preprocess COMMAND test_batch_preprocess)
add_test(NAME test_batch_postprocess COMMAND test_batch_postprocess)
add_test(NAME test_classification_batch COMMAND test_classification_batch)
add_test(NAME test_object_detection_batch COMMAND test_object_detection_batch)
add_test(NAME test_depth_estimation_batch COMMAND test_depth_estimation_batch)
add_test(NAME test_pose_estimation_batch COMMAND test_pose_estimation_batch)
add_test(NAME test_batch_integration COMMAND test_batch_integration)
add_test(NAME test_task_interface COMMAND test_task_interface)
add_test(NAME test_task_factory COMMAND test_task_factory)
add_test(NAME test_task_pipeline COMMAND test_task_pipeline)
add_test(NAME test_yolo_postprocessor COMMAND test_yolo_postprocessor)
add_test(NAME test_rtdetr_postprocessor COMMAND test_rtdetr_postprocessor)
add_test(NAME test_yolo_segmentation_postprocessor COMMAND test_yolo_segmentation_postprocessor)
add_test(NAME test_bbox_processor COMMAND test_bbox_processor)
add_test(NAME test_task_config COMMAND test_task_config)
add_test(NAME test_concrete_tasks COMMAND test_concrete_tasks)
add_test(NAME test_optical_flow COMMAND test_optical_flow)
add_test(NAME test_pose_estimation COMMAND test_pose_estimation)
add_test(NAME test_depth_estimation COMMAND test_depth_estimation)
add_test(NAME test_open_vocab_detection COMMAND test_open_vocab_detection)
add_test(NAME test_gaussian_splatting COMMAND test_gaussian_splatting)
add_test(NAME test_edgecrafter COMMAND test_edgecrafter)
add_test(NAME test_readme_model_types COMMAND test_readme_model_types)
