cmake_minimum_required(VERSION 3.15)
project(vision-core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options for testing
option(BUILD_TESTS "Build tests" OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# All library sources - UNIFIED ARCHITECTURE
set(CORE_SOURCES
    # Core
    src/core/bbox_processor.cpp
    src/core/preprocessor.cpp
    src/core/task_interface.cpp
    src/core/task_factory.cpp

    # Unified Tasks (no separate postprocessors - all internal)
    src/object_detection/object_detection_task.cpp
    src/object_detection/yolo_postprocessor.cpp
    src/object_detection/rtdetr_postprocessor.cpp
    src/classification/classification_task.cpp
    src/classification/classification_postprocessor.cpp
    src/instance_segmentation/instance_segmentation_task.cpp
    src/instance_segmentation/yolo_segmentation_postprocessor.cpp
    src/instance_segmentation/rfdetr_segmentation_postprocessor.cpp
    src/optical_flow/optical_flow_task.cpp
    src/optical_flow/raft_postprocessor.cpp

    # Preprocessors (still separate as they're reusable)
    src/object_detection/detection_preprocessor.cpp
    src/classification/classification_preprocessor.cpp
    src/optical_flow/optical_flow_preprocessor.cpp
)

set(CORE_HEADERS
    # Core
    include/vision-core/core/result_types.hpp
    include/vision-core/core/bbox_processor.hpp
    include/vision-core/core/preprocessor.hpp
    include/vision-core/core/model_info.hpp
    include/vision-core/core/task_interface.hpp
    include/vision-core/core/task_factory.hpp

    # Unified Task Headers (postprocessing is internal)
    include/vision-core/object_detection/object_detection_task.hpp
    include/vision-core/object_detection/postprocessor.hpp
    include/vision-core/object_detection/yolo_postprocessor.hpp
    include/vision-core/object_detection/rtdetr_postprocessor.hpp
    include/vision-core/classification/classification_task.hpp
    include/vision-core/classification/classification_postprocessor.hpp
    include/vision-core/instance_segmentation/instance_segmentation_task.hpp
    include/vision-core/instance_segmentation/segmentation_postprocessor.hpp
    include/vision-core/instance_segmentation/yolo_segmentation_postprocessor.hpp
    include/vision-core/instance_segmentation/rfdetr_segmentation_postprocessor.hpp
    include/vision-core/optical_flow/optical_flow_task.hpp
    include/vision-core/optical_flow/optical_flow_postprocessor.hpp
    include/vision-core/optical_flow/raft_postprocessor.hpp

    # Preprocessor Headers (still separate as they're reusable)
    include/vision-core/object_detection/detection_preprocessor.hpp
    include/vision-core/classification/classification_preprocessor.hpp
    include/vision-core/optical_flow/optical_flow_preprocessor.hpp
)

# Create library
add_library(vision-core ${CORE_SOURCES} ${CORE_HEADERS})
add_library(vision-core::vision-core ALIAS vision-core)

target_include_directories(vision-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(vision-core
    PUBLIC
        ${OpenCV_LIBS}
)

# Set compile features and warnings
target_compile_features(vision-core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(vision-core PRIVATE /W4 /WX)
else()
    target_compile_options(vision-core PRIVATE 
        -Wall -Wextra -Wpedantic -Werror
        -Wno-unused-parameter
    )
endif()

# Install rules
include(GNUInstallDirs)

install(TARGETS vision-core
    EXPORT vision-core-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT vision-core-targets
    FILE vision-core-targets.cmake
    NAMESPACE vision-core::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vision-core
)

# Create config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vision-core-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vision-core-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vision-core
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vision-core-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vision-core-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vision-core-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vision-core
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "vision-core configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
