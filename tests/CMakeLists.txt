# Ensure minimum policies
cmake_minimum_required(VERSION 3.15)

option(VISION_CORE_BUNDLED_GTEST "Download and build googletest" ON)

if(VISION_CORE_BUNDLED_GTEST)
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

# Common test libraries
set(TEST_LIBS
    vision-core::vision-core
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Core tests
add_executable(test_result_types test_result_types.cpp)
target_link_libraries(test_result_types ${TEST_LIBS})

add_executable(test_task_interface test_task_interface.cpp)
target_link_libraries(test_task_interface ${TEST_LIBS})

add_executable(test_task_factory test_task_factory.cpp)
target_link_libraries(test_task_factory ${TEST_LIBS})

# Postprocessor tests
add_executable(test_yolo_postprocessor test_yolo_postprocessor.cpp)
target_link_libraries(test_yolo_postprocessor ${TEST_LIBS})

add_executable(test_rtdetr_postprocessor test_rtdetr_postprocessor.cpp)
target_link_libraries(test_rtdetr_postprocessor ${TEST_LIBS})

add_executable(test_bbox_processor test_bbox_processor.cpp)
target_link_libraries(test_bbox_processor ${TEST_LIBS})

# Concrete task implementation tests
add_executable(test_concrete_tasks test_concrete_tasks.cpp)
target_link_libraries(test_concrete_tasks
    -Wl,--whole-archive vision-core -Wl,--no-whole-archive
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Add tests manually for better compatibility
add_test(NAME test_result_types COMMAND test_result_types)
add_test(NAME test_task_interface COMMAND test_task_interface)
add_test(NAME test_task_factory COMMAND test_task_factory)
add_test(NAME test_yolo_postprocessor COMMAND test_yolo_postprocessor)
add_test(NAME test_rtdetr_postprocessor COMMAND test_rtdetr_postprocessor)
add_test(NAME test_bbox_processor COMMAND test_bbox_processor)
add_test(NAME test_concrete_tasks COMMAND test_concrete_tasks)
