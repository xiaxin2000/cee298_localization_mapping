/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 Localization program using Normal Distributions Transform

 Yuki KITSUKAWA
 */

#include <pthread.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pcl/registration/ndt.h>

#include <autoware_config_msgs/ConfigNDT.h>

#include <autoware_msgs/NDTStat.h>

struct pose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

enum class MethodType
{
  PCL_GENERIC = 0,
};

static MethodType _method_type = MethodType::PCL_GENERIC;

static pose initial_pose, predict_pose, previous_pose,
    ndt_pose, current_pose, localizer_pose;

static double offset_x, offset_y, offset_z, offset_yaw;  // current_pos - previous_pose

static pcl::PointCloud<pcl::PointXYZ> map;

// If the map is loaded, map_loaded will be 1.
static int map_loaded = 0;

static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

// Default values
static int max_iter = 30;        // Maximum iterations
static float ndt_res = 1.0;      // Resolution
static double step_size = 0.1;   // Step size
static double trans_eps = 0.01;  // Transformation epsilon
static double filter_res = 1;  // Downsample parameter for Voxel Filter
static std::string PathName_of_map = "/home/carma/CEE_298_Mapping/map.pcd";  // Path of the generated map

static std::string _output_tf_frame_id = "rslidar";

static ros::Publisher ndt_pose_pub;
static geometry_msgs::PoseStamped ndt_pose_msg;

static ros::Duration scan_duration;

static double exe_time = 0.0;
static bool has_converged;
static int iteration = 0;
static double fitness_score = 0.0;
static double trans_probability = 0.0;

static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw = 0.0;

static double current_velocity = 0.0, previous_velocity = 0.0, previous_previous_velocity = 0.0;  // [m/s]
static double current_velocity_x = 0.0, previous_velocity_x = 0.0;
static double current_velocity_y = 0.0, previous_velocity_y = 0.0;
static double current_velocity_z = 0.0, previous_velocity_z = 0.0;
static double current_velocity_smooth = 0.0;

static double angular_velocity = 0.0;

static std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;

static ros::Publisher time_ndt_matching_pub;
static std_msgs::Float32 time_ndt_matching;

static ros::Publisher ndt_stat_pub;
static autoware_msgs::NDTStat ndt_stat_msg;

// hold transfrom from baselink to primary lidar
static Eigen::Matrix4f tf_btol;

static tf2::Stamped<tf2::Transform> local_transform;

static ros::Publisher ndt_map_pub;

pthread_mutex_t mutex;

static pose convertPoseIntoRelativeCoordinate(const pose &target_pose, const pose &reference_pose)
{
    tf2::Quaternion target_q;
    target_q.setRPY(target_pose.roll, target_pose.pitch, target_pose.yaw);
    tf2::Vector3 target_v(target_pose.x, target_pose.y, target_pose.z);
    tf2::Transform target_tf(target_q, target_v);

    tf2::Quaternion reference_q;
    reference_q.setRPY(reference_pose.roll, reference_pose.pitch, reference_pose.yaw);
    tf2::Vector3 reference_v(reference_pose.x, reference_pose.y, reference_pose.z);
    tf2::Transform reference_tf(reference_q, reference_v);

    tf2::Transform trans_target_tf = reference_tf.inverse() * target_tf;

    pose trans_target_pose;
    trans_target_pose.x = trans_target_tf.getOrigin().getX();
    trans_target_pose.y = trans_target_tf.getOrigin().getY();
    trans_target_pose.z = trans_target_tf.getOrigin().getZ();
    tf2::Matrix3x3 tmp_m(trans_target_tf.getRotation());
    tmp_m.getRPY(trans_target_pose.roll, trans_target_pose.pitch, trans_target_pose.yaw);

    return trans_target_pose;
}


static double calcDiffForRadian(const double lhs_rad, const double rhs_rad)
{
  double diff_rad = lhs_rad - rhs_rad;
  if (diff_rad >= M_PI)
    diff_rad = diff_rad - 2 * M_PI;
  else if (diff_rad < -M_PI)
    diff_rad = diff_rad + 2 * M_PI;
  return diff_rad;
}


static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZ>(map));

  static bool is_first_map = true;
  if (is_first_map == true)
  {
    int ret = pcl::io::loadPCDFile<pcl::PointXYZ> (PathName_of_map, *map_ptr);
    ndt.setTransformationEpsilon(trans_eps);
    ndt.setStepSize(step_size);
    ndt.setResolution(ndt_res);
    ndt.setMaximumIterations(max_iter);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*map_ptr, *map_ptr, indices);

    ndt.setInputTarget(map_ptr);
    is_first_map = false;
    map_loaded=1;

    sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
    map_ptr->header.frame_id = "map";
    pcl::toROSMsg(*map_ptr, *map_msg_ptr);
    ndt_map_pub.publish(*map_msg_ptr);

    previous_pose.x = 0;
    previous_pose.y = 0;
    previous_pose.z = 0;
    previous_pose.roll = 0;
    previous_pose.pitch = 0;
    previous_pose.yaw = 0;
  }

  if (map_loaded == 1)
  {
    matching_start = std::chrono::system_clock::now();

    static tf2_ros::TransformBroadcaster br;
    tf2::Transform transform;
    tf2::Quaternion ndt_q, current_q, localizer_q;

    pcl::PointXYZ p;
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_scan(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan(new pcl::PointCloud<pcl::PointXYZ>());

    ros::Time current_scan_time = input->header.stamp;
    static ros::Time previous_scan_time = current_scan_time;

    pcl::fromROSMsg(*input, *raw_scan);

    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(filter_res, filter_res, filter_res);
    voxel_grid_filter.setInputCloud(raw_scan);
    voxel_grid_filter.filter(*filtered_scan);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>(*filtered_scan));
    int scan_points_num = filtered_scan_ptr->size();

    Eigen::Matrix4f t(Eigen::Matrix4f::Identity());   // base_link
    Eigen::Matrix4f t2(Eigen::Matrix4f::Identity());  // localizer

    std::chrono::time_point<std::chrono::system_clock> align_start, align_end, getFitnessScore_start,
        getFitnessScore_end;
    static double align_time, getFitnessScore_time = 0.0;

    pthread_mutex_lock(&mutex);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*filtered_scan_ptr, *filtered_scan_ptr, indices);

    ndt.setInputSource(filtered_scan_ptr);

    // Guess the initial gross estimation of the transformation
    double diff_time = (current_scan_time - previous_scan_time).toSec();

    offset_x = 0.0;
    offset_y = 0.0;
    offset_z = 0.0;
    offset_yaw = 0.0;

    predict_pose.x = previous_pose.x + offset_x;
    predict_pose.y = previous_pose.y + offset_y;
    predict_pose.z = previous_pose.z + offset_z;
    predict_pose.roll = previous_pose.roll;
    predict_pose.pitch = previous_pose.pitch;
    predict_pose.yaw = previous_pose.yaw + offset_yaw;

    pose predict_pose_for_ndt;
    predict_pose_for_ndt = predict_pose;

    Eigen::Translation3f init_translation(predict_pose_for_ndt.x, predict_pose_for_ndt.y, predict_pose_for_ndt.z);
    Eigen::AngleAxisf init_rotation_x(predict_pose_for_ndt.roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf init_rotation_y(predict_pose_for_ndt.pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf init_rotation_z(predict_pose_for_ndt.yaw, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f init_guess = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x) * tf_btol;

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    align_start = std::chrono::system_clock::now();
    ndt.align(*output_cloud, init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = ndt.hasConverged();

    t = ndt.getFinalTransformation();
    iteration = ndt.getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    fitness_score = ndt.getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = ndt.getTransformationProbability();

    align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end - align_start).count() / 1000.0;

    t2 = t * tf_btol.inverse();

    getFitnessScore_time =
        std::chrono::duration_cast<std::chrono::microseconds>(getFitnessScore_end - getFitnessScore_start).count() /
        1000.0;

    pthread_mutex_unlock(&mutex);

    tf2::Matrix3x3 mat_l;  // localizer
    mat_l.setValue(static_cast<double>(t(0, 0)), static_cast<double>(t(0, 1)), static_cast<double>(t(0, 2)),
                   static_cast<double>(t(1, 0)), static_cast<double>(t(1, 1)), static_cast<double>(t(1, 2)),
                   static_cast<double>(t(2, 0)), static_cast<double>(t(2, 1)), static_cast<double>(t(2, 2)));

    // Update localizer_pose
    localizer_pose.x = t(0, 3);
    localizer_pose.y = t(1, 3);
    localizer_pose.z = t(2, 3);
    mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw, 1);

    tf2::Matrix3x3 mat_b;  // base_link
    mat_b.setValue(static_cast<double>(t2(0, 0)), static_cast<double>(t2(0, 1)), static_cast<double>(t2(0, 2)),
                   static_cast<double>(t2(1, 0)), static_cast<double>(t2(1, 1)), static_cast<double>(t2(1, 2)),
                   static_cast<double>(t2(2, 0)), static_cast<double>(t2(2, 1)), static_cast<double>(t2(2, 2)));

    // Update ndt_pose
    ndt_pose.x = t2(0, 3);
    ndt_pose.y = t2(1, 3);
    ndt_pose.z = t2(2, 3);
    mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);

    current_pose.x = ndt_pose.x;
    current_pose.y = ndt_pose.y;
    current_pose.z = ndt_pose.z;
    current_pose.roll = ndt_pose.roll;
    current_pose.pitch = ndt_pose.pitch;
    current_pose.yaw = ndt_pose.yaw;

    // Compute the velocity and acceleration
    diff_x = current_pose.x - previous_pose.x;
    diff_y = current_pose.y - previous_pose.y;
    diff_z = current_pose.z - previous_pose.z;
    diff_yaw = calcDiffForRadian(current_pose.yaw, previous_pose.yaw);
    diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

    const pose trans_current_pose = convertPoseIntoRelativeCoordinate(current_pose, previous_pose);

    current_velocity = (diff_time > 0) ? (diff / diff_time) : 0;
    current_velocity =  (trans_current_pose.x >= 0) ? current_velocity : -current_velocity;
    current_velocity_x = (diff_time > 0) ? (diff_x / diff_time) : 0;
    current_velocity_y = (diff_time > 0) ? (diff_y / diff_time) : 0;
    current_velocity_z = (diff_time > 0) ? (diff_z / diff_time) : 0;
    angular_velocity = (diff_time > 0) ? (diff_yaw / diff_time) : 0;

    current_velocity_smooth = (current_velocity + previous_velocity + previous_previous_velocity) / 3.0;
    if (std::fabs(current_velocity_smooth) < 0.2)
    {
      current_velocity_smooth = 0.0;
    }

    // Set values for publishing pose
    ndt_q.setRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw);

    ndt_pose_msg.header.frame_id = "map";
    ndt_pose_msg.header.stamp = current_scan_time;
    ndt_pose_msg.pose.position.x = ndt_pose.x;
    ndt_pose_msg.pose.position.y = ndt_pose.y;
    ndt_pose_msg.pose.position.z = ndt_pose.z;
    ndt_pose_msg.pose.orientation.x = ndt_q.x();
    ndt_pose_msg.pose.orientation.y = ndt_q.y();
    ndt_pose_msg.pose.orientation.z = ndt_q.z();
    ndt_pose_msg.pose.orientation.w = ndt_q.w();


    current_q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

    ndt_pose_pub.publish(ndt_pose_msg);

    // Send TF "base_link" to "map"
    transform.setOrigin(tf2::Vector3(current_pose.x, current_pose.y, current_pose.z));
    transform.setRotation(current_q);

    tf2::Stamped<tf2::Transform> tf(transform, current_scan_time, "map");
    geometry_msgs::TransformStamped tf_msg = tf2::toMsg(tf);
    tf_msg.child_frame_id = _output_tf_frame_id;

    br.sendTransform(tf_msg);

    matching_end = std::chrono::system_clock::now();
    exe_time = std::chrono::duration_cast<std::chrono::microseconds>(matching_end - matching_start).count() / 1000.0;
    time_ndt_matching.data = exe_time;
    time_ndt_matching_pub.publish(time_ndt_matching);


    // Set values for /ndt_stat
    ndt_stat_msg.header.stamp = current_scan_time;
    ndt_stat_msg.exe_time = time_ndt_matching.data;
    ndt_stat_msg.iteration = iteration;
    ndt_stat_msg.score = fitness_score;
    ndt_stat_msg.velocity = current_velocity;

    ndt_stat_pub.publish(ndt_stat_msg);


    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Sequence: " << input->header.seq << std::endl;
    std::cout << "Timestamp: " << input->header.stamp << std::endl;
    std::cout << "Frame ID: " << input->header.frame_id << std::endl;
    //    std::cout << "Number of Scan Points: " << scan_ptr->size() << " points." << std::endl;
    std::cout << "Number of Filtered Scan Points: " << scan_points_num << " points." << std::endl;
    std::cout << "NDT has converged: " << has_converged << std::endl;
    std::cout << "Fitness Score: " << fitness_score << std::endl;
    std::cout << "Transformation Probability: " << trans_probability << std::endl;
    std::cout << "Execution Time: " << exe_time << " ms." << std::endl;
    std::cout << "Number of Iterations: " << iteration << std::endl;
    std::cout << "(x,y,z,roll,pitch,yaw): " << std::endl;
    std::cout << "(" << current_pose.x << ", " << current_pose.y << ", " << current_pose.z << ", " << current_pose.roll
              << ", " << current_pose.pitch << ", " << current_pose.yaw << ")" << std::endl;
    std::cout << "Transformation Matrix: " << std::endl;
    std::cout << t << std::endl;
    std::cout << "Align time: " << align_time << std::endl;
    std::cout << "Get fitness score time: " << getFitnessScore_time << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;


    // Update previous_***
    previous_pose.x = current_pose.x;
    previous_pose.y = current_pose.y;
    previous_pose.z = current_pose.z;
    previous_pose.roll = current_pose.roll;
    previous_pose.pitch = current_pose.pitch;
    previous_pose.yaw = current_pose.yaw;

    previous_scan_time = current_scan_time;

    previous_previous_velocity = previous_velocity;
    previous_velocity = current_velocity;
    previous_velocity_x = current_velocity_x;
    previous_velocity_y = current_velocity_y;
    previous_velocity_z = current_velocity_z;

  }
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "ndt_matching");
  pthread_mutex_init(&mutex, NULL);

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // Geting parameters

  private_nh.getParam("output_tf_frame_id", _output_tf_frame_id);
  private_nh.getParam("max_iter", max_iter);
  private_nh.getParam("ndt_res", ndt_res);
  private_nh.getParam("step_size", step_size);
  private_nh.getParam("trans_eps", trans_eps);
  private_nh.getParam("filter_res", filter_res);
  private_nh.getParam("PathName_of_map", PathName_of_map);

  std::string lidar_frame;
  nh.param("localizer", lidar_frame, std::string("rslidar"));
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);
  geometry_msgs::TransformStamped tf_baselink2primarylidar;

  tf2::Vector3 tf_trans(0, 0, 0);
  tf2::Quaternion tf_quat;
  tf_quat.setRPY(0, 0, 0);
  tf_baselink2primarylidar.transform.translation = tf2::toMsg(tf_trans);
  tf_baselink2primarylidar.transform.rotation = tf2::toMsg(tf_quat);
  tf_btol = tf2::transformToEigen(tf_baselink2primarylidar).matrix().cast<float>();

  initial_pose.x = 0.0;
  initial_pose.y = 0.0;
  initial_pose.z = 0.0;
  initial_pose.roll = 0.0;
  initial_pose.pitch = 0.0;
  initial_pose.yaw = 0.0;

  // Publishers
  ndt_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/ndt_pose", 10);
  time_ndt_matching_pub = nh.advertise<std_msgs::Float32>("/time_ndt_matching", 10);
  ndt_stat_pub = nh.advertise<autoware_msgs::NDTStat>("/ndt_stat", 10);
  ndt_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/ndt_map", 10);

  // Subscribers
  ros::Subscriber points_sub = nh.subscribe("/rslidar_points", 100, points_callback);
  ros::spin();

  return 0;
}
