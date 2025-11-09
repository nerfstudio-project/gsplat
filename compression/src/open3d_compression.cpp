#include "compression.h"
#include "utils.h"
#include "timer.h"
#include <open3d/Open3D.h>
#include <open3d/utility/IJsonConvertible.h>
#include <open3d/geometry/Octree.h>
#include <iostream>
#include <exception>
#include <cstring>
#include <vector>

// Include jsoncpp header - use system version to match linked library
#include <json/json.h>

using namespace open3d;
using namespace open3d::geometry;
using namespace open3d::io;

CompressionResult compress_open3d(const std::vector<Point3D>& points) {
    CompressionResult result;
    result.compression_time_ms = 0.0;
    
    // Convert points to Open3D PointCloud format
    PointCloud cloud;
    cloud.points_.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        // Convert uint32_t coordinates to double for Open3D
        cloud.points_[i] = Eigen::Vector3d(static_cast<double>(points[i].x), 
                                           static_cast<double>(points[i].y), 
                                           static_cast<double>(points[i].z));
    }
    
    // For fastest compression, use octree with shallow depth (low max_depth = faster)
    // Lower max_depth means less compression but faster processing
    const int max_depth = 8; // Shallow octree for fastest compression
    
    // Create octree from point cloud
    Octree octree(max_depth);

    StopWatch sw;
    // size_expand parameter: smaller value = higher resolution but faster for shallow trees
    octree.ConvertFromPointCloud(cloud, 0.01);
    
    result.compression_time_ms = sw.ElapsedMs();
    
    // Serialize octree using JSON (Open3D doesn't have direct binary serialization)
    Json::Value json_value;
    if (!octree.ConvertToJsonValue(json_value)) {
        std::cerr << "Error: Could not convert octree to JSON" << std::endl;
        return result;
    }
    
    // Convert JSON to string using Open3D's utility function
    std::string json_str = utility::JsonToString(json_value);
    
    // Copy compressed data
    result.compressed_data.resize(json_str.size());
    std::memcpy(result.compressed_data.data(), json_str.data(), json_str.size());
    
    // Calculate sizes (geometry only: 3 uint32_t = 24 bytes per point in Open3D format)
    result.original_size_bytes = points.size() * 3 * sizeof(uint32_t);
    result.compressed_size_bytes = result.compressed_data.size();
    
    return result;
}

DecompressionResult decompress_open3d(const std::vector<uint8_t>& compressed_data, const std::string& output_path) {
    DecompressionResult result;
    result.output_path = output_path;
    result.success = false;
    result.decompression_time_ms = 0.0;
    
    // Deserialize octree from JSON data
    std::string json_str(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
    
    // Parse JSON string using Open3D's utility function
    Json::Value json_value;
    try {
        json_value = utility::StringToJson(json_str);
    } catch (const std::exception& e) {
        std::cerr << "Error: Could not parse JSON: " << e.what() << std::endl;
        return result;
    }
    
    // Reconstruct octree from JSON
    StopWatch sw;
    Octree octree;
    if (!octree.ConvertFromJsonValue(json_value)) {
        std::cerr << "Error: Could not convert JSON to octree" << std::endl;
        return result;
    }
    
    // Extract points from octree by traversing leaf nodes
    PointCloud cloud;
    std::vector<Eigen::Vector3d> extracted_points;
    
    octree.Traverse([&extracted_points](const std::shared_ptr<OctreeNode>& node,
                                        const std::shared_ptr<OctreeNodeInfo>& node_info) {
        // Check if node is a leaf node (not an internal node)
        // Leaf nodes don't have children, so we check by dynamic_cast
        if (node && std::dynamic_pointer_cast<OctreeInternalNode>(node) == nullptr) {
            // Extract the origin (center) of leaf nodes as compressed point locations
            extracted_points.push_back(node_info->origin_);
        }
        return false; // Continue traversal
    });
    
    // Create point cloud from extracted points
    cloud.points_ = extracted_points;
    
    result.decompression_time_ms = sw.ElapsedMs();
    
    // Convert Open3D point cloud to Point3D format
    std::vector<Point3D> points;
    points.reserve(extracted_points.size());
    for (const auto& p : extracted_points) {
        points.emplace_back(static_cast<uint32_t>(p.x()), 
                           static_cast<uint32_t>(p.y()), 
                           static_cast<uint32_t>(p.z()));
    }
    
    // Save decompressed point cloud using unified PLY writer (file I/O excluded from timing)
    if (!save_ply_geometry(points, output_path)) {
        std::cerr << "Error: Could not save file " << output_path << std::endl;
        return result;
    }
    
    result.success = true;
    return result;
}

