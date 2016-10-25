#include <ros/ros.h>
#include <ros/package.h>
#include <stdio.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <tuple>
#include <set>

#include <exception>

#include <opencv2/videoio/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "openface_ros/ActionUnit.h"
#include "openface_ros/FaceFeatures.h"

#include <sensor_msgs/Image.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <openface/LandmarkCoreIncludes.h>
#include <openface/Face_utils.h>
#include <openface/FaceAnalyser.h>
#include <openface/GazeEstimation.h>

using namespace std;
using namespace ros;
using namespace cv;

namespace
{
  static geometry_msgs::Quaternion toQuaternion(double pitch, double roll, double yaw)
  {
    geometry_msgs::Quaternion q;
    double t0 = std::cos(yaw * 0.5f);
    double t1 = std::sin(yaw * 0.5f);
    double t2 = std::cos(roll * 0.5f);
    double t3 = std::sin(roll * 0.5f);
    double t4 = std::cos(pitch * 0.5f);
    double t5 = std::sin(pitch * 0.5f);

    q.w = t0 * t2 * t4 + t1 * t3 * t5;
    q.x = t0 * t3 * t4 - t1 * t2 * t5;
    q.y = t0 * t2 * t5 + t1 * t3 * t4;
    q.z = t1 * t2 * t4 - t0 * t3 * t5;
    return q;
  }
}

namespace openface_ros
{

  
  class OpenFaceRos
  {
  public:
    OpenFaceRos(NodeHandle &nh)
      : nh_(nh)
      , it_(nh_)
    {
      NodeHandle pnh("~");
      if(!pnh.getParam("image_topic", image_topic_)) throw invalid_argument("Expected ~image_topic parameter");
      
      const auto base_path = package::getPath("openface_ros");

      
      pnh.param<string>("clnf_model_path", clnf_model_path_, base_path + "/model/" + model_params_.model_location);
      pnh.param<string>("tri_model_path", tri_model_path_, base_path + "/model/model/tris_68_full.txt");
      pnh.param<string>("au_model_path", au_model_path_, base_path + "/model/AU_predictors/AU_all_best.txt");
      pnh.param<bool>("publish_viz", publish_viz_, false);

      camera_sub_ = it_.subscribeCamera(image_topic_, 1, &OpenFaceRos::process_incoming_, this);
      face_features_pub_ = nh_.advertise<FaceFeatures>("face_features", 1000);
      if(publish_viz_) viz_pub_ = it_.advertise("openface/viz", 1);
      init_openface_();
    }
    
    ~OpenFaceRos()
    {
      
    }
    
  private:
    void init_openface_()
    {
      model_params_.track_gaze = true;

      clnf_ = unique_ptr<LandmarkDetector::CLNF>(new LandmarkDetector::CLNF(clnf_model_path_));
      face_analyser_ = unique_ptr<FaceAnalysis::FaceAnalyser>(new FaceAnalysis::FaceAnalyser(vector<Vec3d>(), 0.7, 112, 112, au_model_path_, tri_model_path_));

      ROS_INFO("OpenFace initialized!");
    }

    void process_incoming_(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::CameraInfoConstPtr &cam)
    {
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      }
      catch(const cv_bridge::Exception &e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
			
      if(!LandmarkDetector::DetectLandmarksInVideo(cv_ptr->image, *clnf_, model_params_))
      {
        ROS_INFO("No face detected");
        clnf_->Reset();
        face_analyser_->Reset();
        return;
      }

      double fx = cam->K[0];
      double fy = cam->K[4];
      double cx = cam->K[2];
      double cy = cam->K[5];


      if(fx == 0 || fy == 0)
      {
        fx = 500.0 * cv_ptr->image.cols / 640.0;
        fy = 500.0 * cv_ptr->image.rows / 480.0;
        fx = (fx + fy) / 2.0;
        fy = fx;
      }

      if(cx == 0) cx = cv_ptr->image.cols / 2.0;
      if(cy == 0) cy = cv_ptr->image.rows / 2.0;
      

      FaceFeatures features;
      features.header.frame_id = img->header.frame_id;
      if(model_params_.track_gaze && clnf_->eye_model)
      {
        Point3f left(0, 0, -1);
			  Point3f right(0, 0, -1);
				FaceAnalysis::EstimateGaze(*clnf_, left, fx, fy, cx, cy, true);
				FaceAnalysis::EstimateGaze(*clnf_, right, fx, fy, cx, cy, false);

        features.left_gaze.x = left.x;
        features.left_gaze.y = left.y;
        features.left_gaze.z = left.z;

        features.right_gaze.x = right.x;
        features.right_gaze.y = right.y;
        features.right_gaze.z = right.z;
      }

      const auto head_pose = LandmarkDetector::GetCorrectedPoseWorld(*clnf_, fx, fy, cx, cy);
      features.head_pose.position.x = head_pose[0];
      features.head_pose.position.y = head_pose[1];
      features.head_pose.position.z = head_pose[2];
      features.head_pose.orientation = toQuaternion(head_pose[4], head_pose[3], head_pose[5]);

      if(clnf_->tracking_initialised)
      {
        const auto &landmarks = clnf_->detected_landmarks;
        for(unsigned i = 0; i < clnf_->pdm.NumberOfPoints(); ++i)
        {
          
          geometry_msgs::Point p;
          p.x = landmarks.at<double>(i);
          p.y = landmarks.at<double>(clnf_->pdm.NumberOfPoints() + i);
          features.landmarks_2d.push_back(p);
        }

        cv::Mat_<double> shape_3d = clnf_->GetShape(fx, fy, cx, cy);
        for(unsigned i = 0; i < clnf_->pdm.NumberOfPoints(); ++i)
        {
          geometry_msgs::Point p;
          p.x = shape_3d.at<double>(i);
          p.y = shape_3d.at<double>(clnf_->pdm.NumberOfPoints() + i);
          p.z = shape_3d.at<double>(clnf_->pdm.NumberOfPoints() * 2 + i);
          features.landmarks_3d.push_back(p);
        }
      }

      vector<std::pair<std::string, double>> aus_reg;
      vector<std::pair<std::string, double>> aus_class;
      tie(aus_reg, aus_class) = face_analyser_->PredictStaticAUs(cv_ptr->image, *clnf_);

      unordered_map<string, ActionUnit> aus;
      for(const auto &au_reg : aus_reg)
      {
        auto it = aus.find(get<0>(au_reg));
        if(it == aus.end())
        {
          ActionUnit u;
          u.name = get<0>(au_reg);
          u.intensity = get<1>(au_reg);
          aus.insert({ get<0>(au_reg), u});
          continue;
        }

        it->second.intensity = get<1>(au_reg);
      }

      for(const auto &au_class : aus_class)
      {
        auto it = aus.find(get<0>(au_class));
        if(it == aus.end())
        {
          ActionUnit u;
          u.name = get<0>(au_class);
          u.presence = get<1>(au_class);
          aus.insert({ get<0>(au_class), u});
          continue;
        }

        it->second.presence = get<1>(au_class);
      }

      for(const auto &au : aus) features.action_units.push_back(get<1>(au));
      
		  if(publish_viz_)
      {
        cv_bridge::CvImagePtr viz_ptr;
        try
        {
          viz_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        }
        catch(const cv_bridge::Exception &e)
        {
          ROS_ERROR("cv_bridge viz exception: %s", e.what());
          return;
        }
        decltype(viz_ptr->image) viz_img = viz_ptr->image.clone();

        for(const auto &p : features.landmarks_2d)
        {
          circle(viz_img, Point(p.x, p.y), 3, Scalar(255, 0, 0), -1);
        }

        if(model_params_.track_gaze && clnf_->eye_model)
        {
          const Point3f left(features.left_gaze.x, features.left_gaze.y, features.left_gaze.z);
          const Point3f right(features.right_gaze.x, features.right_gaze.y, features.right_gaze.z);
          FaceAnalysis::DrawGaze(viz_img, *clnf_, left, right, fx, fy, cx, cy);
        }
        auto viz_msg = cv_bridge::CvImage(img->header, "bgr8", viz_img).toImageMsg();
        viz_pub_.publish(viz_msg);
      }

      face_features_pub_.publish(features);
    }

    

    LandmarkDetector::FaceModelParameters model_params_;
    unique_ptr<LandmarkDetector::CLNF> clnf_;
    unique_ptr<FaceAnalysis::FaceAnalyser> face_analyser_;

    string image_topic_;
    string clnf_model_path_;
    string tri_model_path_;
    string au_model_path_;

    NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::CameraSubscriber camera_sub_;
    Publisher face_features_pub_;

    bool publish_viz_;
    image_transport::Publisher viz_pub_;
  };
}

int main(int argc, char *argv[])
{
  init(argc, argv, "openface_ros");
  
  using namespace openface_ros;

  NodeHandle nh;

  try
  {
    OpenFaceRos openface_(nh);
    spin();
  }
  catch(const exception &e)
  {
    ROS_FATAL("%s", e.what());
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
