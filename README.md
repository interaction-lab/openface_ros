# OpenFace for ROS

This is a small ROS node that exposes OpenFace over ROS. In particular, given a image of a face, it will output:
  * Eye Gaze Vectors
  * Head Pose
  * 2D Landmarks
  * 3D Landmarks
  * Action Units
  * Debug Visualization (optional)

This repository expects [Interaction Lab's fork of OpenFace](https://github.com/interaction-lab/OpenFace) and [OpenCV 3](https://github.com/opencv/opencv) to be installed.

Dependencies of OpenFace include:
  * `dlib`
  * Boost Filesystem
  * `tbb`
  * OpenCV 3

## ROS Parameters

### Required
  * `~image_topic` - The topic the image is provided on (e.g. `/usb_cam/image_raw`).

### Optional
  * `~clnf_model_path` - Provide an alternate CLNF model to OpenFace.
  * `~tri_model_path` - Provide an alternate tri model to OpenFace.
  * `~au_model_path` - Provide an alternate AU model to OpenFace.
  * `~publish_viz` - Set to `true` to publish a debug visualization (default: `false`).

## Installation
  * Install all dependencies (see below)
  * Clone OpenFace, follow installation instructions, but run `sudo make install` at the end
  * Clone & re-compile `cv_bridge` (in `vision_opencv` stack, `http://wiki.ros.org/vision_opencv`)
  * Clone `usb_cam` ros node (`http://wiki.ros.org/usb_cam`) or other ros node of your choice for interfacing with USB camera

### Dependencies
  * OpenCV 3 (If installing from source, make sure to run cmake as follows: `cmake -DBUILD_SHARED_LIBS=ON ..﻿⁠⁠⁠⁠`
  * dlib (You should be able to install from apt-get, if not, clone and install from source)

### Running
  * `roscore`
  * `rosrun usb_cam usb_cam_node`
  * `rosrun openface_ros openface_ros _image_topic:="/usb_cam/image_raw"`

### Notes

This node requires `cv_bridge` *and* OpenCV 3. You must ensure that `cv_bridge` is also linked against OpenCV 3. If you get a warning during compilation, you may have to manually clone the `vision_opencv` repository and re-build `cv_bridge`.

## Messages

### FaceFeatures
```
std_msgs/Header header

geometry_msgs/Vector3 left_gaze
geometry_msgs/Vector3 right_gaze

geometry_msgs/Pose head_pose

geometry_msgs/Point[] landmarks_3d
geometry_msgs/Point[] landmarks_2d

openface_ros/ActionUnit[] action_units
```

### ActionUnit
```
string name
float64 presence
float64 intensity
```
