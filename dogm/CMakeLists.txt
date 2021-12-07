cmake_minimum_required(VERSION 3.9)
project(DOGM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(OpenGL_GL_PREFERENCE "GLVND")
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--shared --compiler-options '-fPIC'" )

option(BUILD_TESTS "Build unit tests" ON)

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

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

LIST (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
INCLUDE(GoogleTestDownloadAndBuild)

#if(WIN32)
#    install(FILES "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}d.dll" DESTINATION ${CMAKE_INSTALL_BINARY}/debug CONFIGURATIONS Debug)
#    install(FILES "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll" DESTINATION ${CMAKE_INSTALL_BINARY} CONFIGURATIONS RelWithDebInfo Release)
#endif(WIN32)

include_directories(
	SYSTEM
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${GLFW3_INCLUDE_DIR}
	${GLEW_INCLUDE_DIRS}
	${GLM_INCLUDE_DIR}
	${EIGEN3_INCLUDE_DIR}
	${CMAKE_HOME_DIRECTORY}/include
)

set(HEADER_FILES
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/init.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/init_new_particles.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/mass_update.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/particle_to_grid.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/predict.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/resampling.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/statistical_moments.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/update_persistent_particles.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/kernel/ego_motion_compensation.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/common.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/cuda_utils.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/dogm.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/dogm_types.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/mapping/laser_to_meas_grid.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/mapping/kernel/measurement_grid.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/mapping/opengl/framebuffer.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/mapping/opengl/polygon.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/mapping/opengl/shader.h
		${CMAKE_HOME_DIRECTORY}/include/dogm/mapping/opengl/texture.h
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADER_FILES})

set(SRC_FILES
	${CMAKE_HOME_DIRECTORY}/src/kernel/init.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/init_new_particles.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/mass_update.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/particle_to_grid.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/predict.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/resampling.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/statistical_moments.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/update_persistent_particles.cu
		${CMAKE_HOME_DIRECTORY}/src/kernel/ego_motion_compensation.cu
		${CMAKE_HOME_DIRECTORY}/src/dogm.cu
		${CMAKE_HOME_DIRECTORY}/src/mapping/laser_to_meas_grid.cu
		${CMAKE_HOME_DIRECTORY}/src/mapping/kernel/measurement_grid.cu
		${CMAKE_HOME_DIRECTORY}/src/mapping/opengl/framebuffer.cpp
		${CMAKE_HOME_DIRECTORY}/src/mapping/opengl/polygon.cpp
		${CMAKE_HOME_DIRECTORY}/src/mapping/opengl/renderer.cpp
		${CMAKE_HOME_DIRECTORY}/src/mapping/opengl/shader.cpp
		${CMAKE_HOME_DIRECTORY}/src/mapping/opengl/texture.cpp
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

set_target_properties(dogm PROPERTIES PUBLIC_HEADER "include/dogm.h;include/dogm_types.h")

target_link_libraries(dogm
	PRIVATE ${OPENGL_LIBRARY}
	PRIVATE ${GLFW3_LIBRARY}
	PRIVATE ${GLEW_LIBRARIES}
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
