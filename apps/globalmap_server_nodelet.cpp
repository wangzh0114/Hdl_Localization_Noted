#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>

namespace hdl_localization {

// 继承nodelet::Nodelet
class GlobalmapServerNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  GlobalmapServerNodelet() {
  }
  virtual ~GlobalmapServerNodelet() {
  }
  // 重写onInit，利用计时器触发回调函数
  void onInit() override {
    // 初始化三个句柄
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    // publish globalmap with "latched（锁存的）" publisher
    globalmap_pub = nh.advertise<sensor_msgs::PointCloud2>("/globalmap", 5, true);
    map_update_sub = nh.subscribe("/map_request/pcd", 10,              &GlobalmapServerNodelet::map_update_callback, this);

    globalmap_pub_timer = nh.createWallTimer(ros::WallDuration(1.0), &GlobalmapServerNodelet::pub_once_cb, this, true, true);
  }

private:
  // globalmap_server_nodelet::initialize_params()，读取pcd文件，对地图降采样
  void initialize_params() {
    // read globalmap from a pcd file
    std::string globalmap_pcd = private_nh.param<std::string>("globalmap_pcd", "");
    globalmap.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
    globalmap->header.frame_id = "map";

    // utm文件 类似与经纬度的坐标文件
    // https://zhuanlan.zhihu.com/p/255036625
    std::ifstream utm_file(globalmap_pcd + ".utm");
    if (utm_file.is_open() && private_nh.param<bool>("convert_utm_to_local", true)) {
      double utm_easting;
      double utm_northing;
      double altitude;
      // 右移可以完成换位？
      utm_file >> utm_easting >> utm_northing >> altitude;
      for(auto& pt : globalmap->points) {
        pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utm_northing, altitude);
      }
      ROS_INFO_STREAM("Global map offset by UTM reference coordinates (x = "
                      << utm_easting << ", y = " << utm_northing << ") and altitude (z = " << altitude << ")");
    }

    // downsample globalmap
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    // VoxelGrid体素采样
    // https://blog.csdn.net/Small_Munich/article/details/108348164
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    // 设置每个体素的大小，可用 Eigen::Vector3f& 或 float x, y, z
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap);
    // 定义pcl::PointCloud<PointT>::Ptr 接收体素采样结果
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap = filtered;
  }
  // 每隔0.05s发布一次（onInit）
  void pub_once_cb(const ros::WallTimerEvent& event) {
    globalmap_pub.publish(globalmap);
  }
  // 更新globalmap TODO:why?
  void map_update_callback(const std_msgs::String &msg){
    ROS_INFO_STREAM("Received map request, map path : "<< msg.data);
    std::string globalmap_pcd = msg.data;
    globalmap.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
    globalmap->header.frame_id = "map";

    // downsample globalmap
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap = filtered;
    globalmap_pub.publish(globalmap);
  }

private:
  // ROS 3个句柄，1个发布，1个订阅，1个计时器，1个globalmap变量
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  ros::Publisher globalmap_pub;
  ros::Subscriber map_update_sub;

  ros::WallTimer globalmap_pub_timer;
  pcl::PointCloud<PointT>::Ptr globalmap;
};

}


PLUGINLIB_EXPORT_CLASS(hdl_localization::GlobalmapServerNodelet, nodelet::Nodelet)
