cmake_minimum_required(VERSION 2.8.3)
project(openface_ros)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  image_transport
  cv_bridge
  std_msgs
  geometry_msgs
  sensor_msgs
  roscpp
  tf2
  tf2_ros
)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_library(OPENFACE_FA_LIB FaceAnalyser PATH_SUFFIXES openface)
find_library(OPENFACE_LD_LIB LandmarkDetector PATH_SUFFIXES openface)
set(OPENFACE_LIBS ${OPENFACE_FA_LIB} ${OPENFACE_LD_LIB})

find_file(OPENLIB_FA_INCLUDE FaceAnalyser.h PATH_SUFFIXES openface)
get_filename_component(OPENLIB_PATH ${OPENLIB_FA_INCLUDE} DIRECTORY)
message(${OPENLIB_PATH})
get_filename_component(OPENLIB_INCLUDE_DIR ${OPENLIB_PATH} DIRECTORY)
message(${OPENLIB_INCLUDE_DIR})
message(${CMAKE_C_COMPILER})

find_package(dlib REQUIRED)

add_message_files(FILES ActionUnit.msg Face.msg Faces.msg)
generate_messages(DEPENDENCIES std_msgs geometry_msgs)

add_definitions(-std=c++11)
add_definitions(-g -pg)
link_directories(${Boost_LIBRARY_DIRS})
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OPENLIB_INCLUDE_DIR}
  ${DLIB_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
add_executable(openface_ros src/openface_ros.cpp)
add_dependencies(openface_ros ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(openface_ros
  ${catkin_LIBRARIES}

  ${OPENFACE_LIBS}
  lapack
  openblas
  dlib
  ${DLIB_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  tbb
  tbbmalloc
  tbbmalloc_proxy
)

install(TARGETS openface_ros
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
