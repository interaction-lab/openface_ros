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
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <chrono>
#include <exception>

#include <tbb/tbb.h>

#include <opencv2/videoio/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "openface_ros/ActionUnit.h"
#include "openface_ros/Face.h"
#include "openface_ros/Faces.h"

#include <sensor_msgs/Image.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <openface/LandmarkCoreIncludes.h>
#include <openface/Face_utils.h>
#include <openface/FaceAnalyser.h>
#include <openface/GazeEstimation.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <Eigen/Dense>

using namespace std;
using namespace ros;
using namespace cv;

namespace
{
  static geometry_msgs::Quaternion toQuaternion(double pitch, double roll, double yaw)
  {
    double t0 = std::cos(yaw * 0.5f);
    double t1 = std::sin(yaw * 0.5f);
    double t2 = std::cos(roll * 0.5f);
    double t3 = std::sin(roll * 0.5f);
    double t4 = std::cos(pitch * 0.5f);
    double t5 = std::sin(pitch * 0.5f);

    geometry_msgs::Quaternion q;
    q.w = t0 * t2 * t4 + t1 * t3 * t5;
    q.x = t0 * t3 * t4 - t1 * t2 * t5;
    q.y = t0 * t2 * t5 + t1 * t3 * t4;
    q.z = t1 * t2 * t4 - t0 * t3 * t5;
    return q;
  }

  static geometry_msgs::Quaternion operator *(const geometry_msgs::Quaternion &a, const geometry_msgs::Quaternion &b)
  {
    geometry_msgs::Quaternion q;
    
    q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;  // 1
    q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;  // i
    q.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;  // j
    q.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;  // k
    return q;
  }

  void non_overlaping_detections(const vector<LandmarkDetector::CLNF *> &clnf_models, vector<Rect_<double>> &face_detections)
  {
    if(face_detections.empty()) return;
    
    for(size_t model = 0; model < clnf_models.size(); ++model)
    {
      // See if the detections intersect
      cv::Rect_<double> model_rect = clnf_models[model]->GetBoundingBox();

      for(int detection = face_detections.size() - 1; detection >= 0; --detection)
      {
        double intersection_area = (model_rect & face_detections[detection]).area();
        double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

        // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
        if(intersection_area / union_area <= 0.5) continue;
        face_detections.erase(face_detections.begin() + detection);
      }
    }
  }

  vector<ssize_t> redundant_detections(const vector<LandmarkDetector::CLNF *> &clnf_models)
  {
    vector<ssize_t> ret(clnf_models.size(), -1);
    for(size_t modeli = 0; modeli < clnf_models.size(); ++modeli)
    {
      if(ret[modeli] >= 0) continue;
      // See if the detections intersect
      const Rect_<double> modeli_rect = clnf_models[modeli]->GetBoundingBox();

      for(size_t modelj = 0; modelj < clnf_models.size(); ++modelj)
      {
        if(modeli == modelj || ret[modelj] >= 0) continue;
        
        const Rect_<double> modelj_rect = clnf_models[modelj]->GetBoundingBox();

        double intersection_area = (modeli_rect & modelj_rect).area();
        double union_area = modeli_rect.area() + modelj_rect.area() - 2 * intersection_area;

        // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
        if(intersection_area / union_area <= 0.5) continue;

        ret[modelj] = modeli;
      }
    }

    return ret;
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

      
      pnh.param<string>("clnf_model_path", clnf_model_path_, base_path + "/" + model_params_.model_location);
      pnh.param<string>("tri_model_path", tri_model_path_, base_path + "/model/tris_68_full.txt");
      pnh.param<string>("au_model_path", au_model_path_, base_path + "/model/AU_predictors/AU_all_best.txt");
      pnh.param<string>("haar_model_path", haar_model_path_, base_path + "/model/classifiers/haarcascade_frontalface_alt.xml");
      pnh.param<bool>("publish_viz", publish_viz_, false);

      int max_faces = 0;
      pnh.param<int>("max_faces", max_faces, 3);
      if(max_faces <= 0) throw invalid_argument("~max_faces must be > 0");
      
      max_faces_ = max_faces;

      camera_sub_ = it_.subscribeCamera(image_topic_, 1, &OpenFaceRos::process_incoming_, this);
      faces_pub_ = nh_.advertise<Faces>("faces", 1000);
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

      LandmarkDetector::CLNF clnf(clnf_model_path_);
      clnf.face_detector_HAAR.load(haar_model_path_);
	    clnf.face_detector_location = haar_model_path_;

      FaceAnalysis::FaceAnalyser face_analyser(vector<Vec3d>(), 0.7, 112, 112, au_model_path_, tri_model_path_);

      actives_ = vector<bool>(max_faces_, false);
      clnfs_ = vector<LandmarkDetector::CLNF>(max_faces_, clnf);
      face_analysers_ = vector<FaceAnalysis::FaceAnalyser>(max_faces_, face_analyser);

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

      vector<Rect_<double>> face_detections;

      if(cam->header.seq % 30 == 0)
      {
        LandmarkDetector::DetectFaces(face_detections, cv_ptr->image, clnfs_[0].face_detector_HAAR);
        vector<LandmarkDetector::CLNF *> active_clnfs;
        for(unsigned i = 0; i < max_faces_; ++i)
        {
          if(!actives_[i]) continue;
          active_clnfs.push_back(&clnfs_[i]); 
        }
        
        non_overlaping_detections(active_clnfs, face_detections);
        ROS_INFO("new face detections %lu", face_detections.size());
        
      }

      vector<tbb::atomic<bool>> face_detections_used(face_detections.size(), false);
      tbb::parallel_for(0, (int)clnfs_.size(), [&](int i){
				if(clnfs_[i].failures_in_a_row > 4)
				{
					actives_[i] = false;
					clnfs_[i].Reset();
        }
				if(!actives_[i])
				{
					for(size_t j = 0; j < face_detections.size(); ++j)
					{
						// if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
						if(face_detections_used[j].compare_and_swap(true, false)) continue;

            ROS_INFO("Allocating CLNF on face_detection %ld", j);
						
            clnfs_[i].Reset();

            // This ensures that a wider window is used for the initial landmark localisation
            clnfs_[i].detection_success = false;
            LandmarkDetector::DetectLandmarksInVideo(cv_ptr->image, face_detections[j], clnfs_[i], model_params_);
            actives_[i] = true;
            break;
					}
				}
				else
				{
					LandmarkDetector::DetectLandmarksInVideo(cv_ptr->image, clnfs_[i], model_params_);
				}
			});

      vector<LandmarkDetector::CLNF *> active_clnfs;
      for(unsigned i = 0; i < max_faces_; ++i)
      {
        if(!actives_[i]) continue;
        active_clnfs.push_back(&clnfs_[i]); 
      }
      
      const auto redundancies = redundant_detections(active_clnfs);
			
      for(size_t i = 0; i < redundancies.size(); ++i)
      {
        if(redundancies[i] < 0) continue;

        cout << "Detected redundant CLNF at " << i << endl;
        actives_[i] = false;
        clnfs_[i].Reset();
      }

      Faces faces;
      for(unsigned i = 0; i < max_faces_; ++i)
      {
        if(!actives_[i]) continue;

        auto &clnf = clnfs_[i];
        auto &face_analyser = face_analysers_[i];

        Face face;
        face.header.frame_id = img->header.frame_id;
        face.header.stamp = Time::now();
        if(model_params_.track_gaze && clnf.eye_model)
        {
          Point3f left(0, 0, -1);
          Point3f right(0, 0, -1);
          FaceAnalysis::EstimateGaze(clnf, left, fx, fy, cx, cy, true);
          FaceAnalysis::EstimateGaze(clnf, right, fx, fy, cx, cy, false);

          face.left_gaze.x = left.x;
          face.left_gaze.y = left.y;
          face.left_gaze.z = left.z;

          face.right_gaze.x = right.x;
          face.right_gaze.y = right.y;
          face.right_gaze.z = right.z;
        }

        const auto head_pose = LandmarkDetector::GetCorrectedPoseWorld(clnf, fx, fy, cx, cy);
        face.head_pose.position.x = head_pose[0];
        face.head_pose.position.y = head_pose[1];
        face.head_pose.position.z = head_pose[2];
        const auto head_orientation = toQuaternion(head_pose[3], head_pose[5], head_pose[4]);

        face.head_pose.orientation = toQuaternion(M_PI / 2, 0, M_PI / 2);
        face.head_pose.orientation = face.head_pose.orientation * head_orientation;

        // tf
        {
          geometry_msgs::TransformStamped transform;
          transform.header = face.header;
          stringstream out;
          out << "head" << i;
          transform.child_frame_id = out.str();
          transform.transform.translation.x = face.head_pose.position.x / 1000.0;
          transform.transform.translation.y = face.head_pose.position.y / 1000.0;
          transform.transform.translation.z = face.head_pose.position.z / 1000.0;
          transform.transform.rotation = face.head_pose.orientation;
          tf_br_.sendTransform(transform);
        }

        if(clnf.tracking_initialised)
        {
          const auto &landmarks = clnf.detected_landmarks;
          for(unsigned i = 0; i < clnf.pdm.NumberOfPoints(); ++i)
          {
            geometry_msgs::Point p;
            p.x = landmarks.at<double>(i);
            p.y = landmarks.at<double>(clnf.pdm.NumberOfPoints() + i);
            face.landmarks_2d.push_back(p);
          }

          cv::Mat_<double> shape_3d = clnf.GetShape(fx, fy, cx, cy);
          for(unsigned i = 0; i < clnf.pdm.NumberOfPoints(); ++i)
          {
            geometry_msgs::Point p;
            p.x = shape_3d.at<double>(i);
            p.y = shape_3d.at<double>(clnf.pdm.NumberOfPoints() + i);
            p.z = shape_3d.at<double>(clnf.pdm.NumberOfPoints() * 2 + i);
            face.landmarks_3d.push_back(p);
          }
        }

        vector<std::pair<std::string, double>> aus_reg;
        vector<std::pair<std::string, double>> aus_class;
        tie(aus_reg, aus_class) = face_analyser.PredictStaticAUs(cv_ptr->image, clnf);

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

        for(const auto &au : aus) face.action_units.push_back(get<1>(au));

        Point min(100000, 100000);
        Point max(0, 0);
        for(const auto &p : face.landmarks_2d)
        {
          if(p.x < min.x) min.x = p.x;
          if(p.y < min.y) min.y = p.y;
          if(p.x > max.x) max.x = p.x;
          if(p.y > max.y) max.y = p.y;
        }

        faces.faces.push_back(face);
      }

      faces_pub_.publish(faces);

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

        for(unsigned i = 0; i < faces.faces.size(); ++i)
        {
          const auto &face = faces.faces[i];
          auto &clnf = clnfs_[i];
          for(const auto &p : face.landmarks_2d)
          {
            circle(viz_img, Point(p.x, p.y), 3, Scalar(255, 0, 0), -1);
          }

          if(model_params_.track_gaze && clnf.eye_model)
          {
            const Point3f left(face.left_gaze.x, face.left_gaze.y, face.left_gaze.z);
            const Point3f right(face.right_gaze.x, face.right_gaze.y, face.right_gaze.z);
            FaceAnalysis::DrawGaze(viz_img, clnf, left, right, fx, fy, cx, cy);
          }
        }
        
        auto viz_msg = cv_bridge::CvImage(img->header, "bgr8", viz_img).toImageMsg();
        viz_pub_.publish(viz_msg);
      }
    }

    tf2_ros::TransformBroadcaster tf_br_;

    LandmarkDetector::FaceModelParameters model_params_;
    vector<bool> actives_;
    vector<LandmarkDetector::CLNF> clnfs_;
    vector<FaceAnalysis::FaceAnalyser> face_analysers_;

    string image_topic_;
    string clnf_model_path_;
    string tri_model_path_;
    string au_model_path_;
    string haar_model_path_;
    unsigned max_faces_;

    NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::CameraSubscriber camera_sub_;
    Publisher faces_pub_;

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
