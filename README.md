# OpenFace for ROS

This is a small ROS node that exposes OpenFace over ROS. In particular, given a image of a face, it will output:
  * Eye Gaze Vectors
  * Head Pose
  * 2D Landmarks
  * 3D Landmarks
  * Action Units
  * Debug Visualization (optional)

This repository expects interaction lab's fork of OpenFace to be installed to /usr/local and OpenCV 3.
