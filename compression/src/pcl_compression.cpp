#include "compression.h"
#include "utils.h"
#include "timer.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/compression/octree_pointcloud_compression.h>
#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>

CompressionResult compress_pcl(const std::vector<Point3D>& points) {
    CompressionResult result;
    result.compression_time_ms = 0.0;
    
    // Convert points to PCL PointCloud format
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    
    for (size_t i = 0; i < points.size(); ++i) {
        // Convert uint32_t coordinates to float for PCL
        cloud->points[i].x = static_cast<float>(points[i].x);
        cloud->points[i].y = static_cast<float>(points[i].y);
        cloud->points[i].z = static_cast<float>(points[i].z);
    }
    
    // Configure octree compression for fastest compression
    // LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR uses high resolution (faster compression)
    pcl::io::compression_Profiles_e compression_profile = pcl::io::LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR;
    
    // Create compression encoder
    pcl::io::OctreePointCloudCompression<pcl::PointXYZ>* pointCloudEncoder = 
        new pcl::io::OctreePointCloudCompression<pcl::PointXYZ>(compression_profile, false);
    
    // Compress point cloud
    std::stringstream compressed_data;
    StopWatch sw;
    
    pointCloudEncoder->encodePointCloud(cloud, compressed_data);
    
    result.compression_time_ms = sw.ElapsedMs();
    
    // Copy compressed data to result
    std::string compressed_str = compressed_data.str();
    result.compressed_data.resize(compressed_str.size());
    std::memcpy(result.compressed_data.data(), compressed_str.data(), compressed_str.size());
    
    delete pointCloudEncoder;
    
    return result;
}

DecompressionResult decompress_pcl(const std::vector<uint8_t>& compressed_data, const std::string& output_path) {
    DecompressionResult result;
    result.output_path = output_path;
    result.success = false;
    result.decompression_time_ms = 0.0;
    
    // Create compression decoder (same profile as encoder)
    pcl::io::compression_Profiles_e compression_profile = pcl::io::LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR;
    pcl::io::OctreePointCloudCompression<pcl::PointXYZ>* pointCloudDecoder = 
        new pcl::io::OctreePointCloudCompression<pcl::PointXYZ>(compression_profile, false);
    
    // Create point cloud for decompression
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Convert compressed_data to stringstream
    std::stringstream compressed_stream;
    compressed_stream.write(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
    
    // Measure decompression time (excluding file I/O)
    StopWatch sw;
    
    // Decompress
    pointCloudDecoder->decodePointCloud(compressed_stream, cloud);
    
    result.decompression_time_ms = sw.ElapsedMs();
    
    // Convert PCL point cloud to Point3D format
    std::vector<Point3D> points;
    points.reserve(cloud->points.size());
    for (const auto& p : cloud->points) {
        points.emplace_back(static_cast<uint32_t>(p.x), 
                           static_cast<uint32_t>(p.y), 
                           static_cast<uint32_t>(p.z));
    }
    
    delete pointCloudDecoder;
    
    // Save decompressed point cloud using unified PLY writer (file I/O excluded from timing)
    if (!save_ply_geometry(points, output_path)) {
        std::cerr << "Error: Could not save file " << output_path << std::endl;
        return result;
    }
    
    result.success = true;
    return result;
}

