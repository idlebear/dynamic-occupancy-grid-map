cmake_minimum_required(VERSION 3.9)
project(dogm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(OpenGL_GL_PREFERENCE "GLVND")
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--shared --compiler-options '-fPIC'" )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
find_package(OpenGL REQUIRED)

find_package(GLM)
if (NOT GLM_FOUND)
	set(GLM_INCLUDE_DIR "" CACHE PATH "")
endif()

find_package(GLFW3)
if (NOT GLFW3_FOUND)
	set(GLFW3_INCLUDE_DIR "" CACHE PATH "")
	set(GLFW3_LIBRARY "" CACHE FILEPATH "" )
endif()

if (NOT WIN32)
	find_package(GLEW)
endif()
if (NOT GLEW_FOUND)
	set(GLEW_INCLUDE_DIRS "" CACHE PATH "")
	set(GLEW_LIBRARIES "" CACHE FILEPATH "" )
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

LIST (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
INCLUDE(GoogleTestDownloadAndBuild)

include_directories(
	SYSTEM
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${GLFW3_INCLUDE_DIR}
	${GLEW_INCLUDE_DIRS}
	${GLM_INCLUDE_DIR}
	${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)

set(HEADER_FILES
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/init.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/init_new_particles.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/mass_update.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/particle_to_grid.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/predict.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/resampling.h
        ${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/resampling_parallel.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/statistical_moments.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/update_persistent_particles.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/ego_motion_compensation.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/common.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/cuda_utils.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/dogm.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/dogm_types.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/mapping/laser_to_meas_grid.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/mapping/kernel/measurement_grid.h
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADER_FILES})

set(SRC_FILES
	${CMAKE_HOME_DIRECTORY}/src/kernel/init.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/init_new_particles.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/mass_update.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/particle_to_grid.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/predict.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/resampling.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/resampling_parallel.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/statistical_moments.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/update_persistent_particles.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/ego_motion_compensation.cu
		${CMAKE_HOME_DIRECTORY}/src/dogm.cu
		${CMAKE_HOME_DIRECTORY}/src/mapping/laser_to_meas_grid.cu
		${CMAKE_HOME_DIRECTORY}/src/mapping/kernel/measurement_grid.cu
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

add_library(dogm SHARED
    ${HEADER_FILES}
	${SRC_FILES}
)

target_include_directories(dogm PUBLIC include)

# support for gpu lambdas
string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")
# supress glm warnings
#string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set_target_properties(dogm PROPERTIES PUBLIC_HEADER "include/dogm/dogm.h;include/dogm/dogm_types.h")

target_link_libraries(dogm
	${OPENGL_LIBRARY}
	${GLFW3_LIBRARY}
	${GLEW_LIBRARIES}
	${OpenCV_LIBS}
)

#INSTALL(TARGETS dogm
#		ARCHIVE DESTINATION lib/dogm
#        LIBRARY DESTINATION lib/dogm
#        PUBLIC_HEADER DESTINATION include/dogm
#)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

add_subdirectory(demo)
add_subdirectory(python)

# Install package.xml for catkin
install(FILES package.xml DESTINATION share/${PROJECT_NAME}/)

set(CONF_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")
set(DOGM_CMAKE_DIR share/${PROJECT_NAME}/cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
	dogm-config.cmake.in
	${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}/dogm-config.cmake
	PATH_VARS DOGM_CMAKE_DIR
	INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}
)

install(
	FILES ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}/dogm-config.cmake
	DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS dogm
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
)

install(
	DIRECTORY include/
	DESTINATION include/
)
