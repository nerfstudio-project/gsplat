#include "compression.h"
#include "utils.h"
#include "timer.h"
#include <draco/compression/encode.h>
#include <draco/compression/decode.h>
#include <draco/point_cloud/point_cloud_builder.h>
#include <draco/io/ply_reader.h>
#include <draco/io/ply_encoder.h>
#include <draco/attributes/attribute_octahedron_transform.h>
#include <iostream>
#include <memory>
#include <cstring>
#include <vector>

CompressionResult compress_draco(const std::vector<Point3D>& points) {
    CompressionResult result;
    result.compression_time_ms = 0.0;
    
    // Convert points to Draco PointCloud format
    draco::PointCloudBuilder builder;
    builder.Start(points.size());
    
    // Add position attribute
    int pos_attr_id = builder.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32);
    if (pos_attr_id < 0) {
        std::cerr << "Error: Failed to add position attribute" << std::endl;
        return result;
    }
    
    // Copy position data
    for (size_t i = 0; i < points.size(); ++i) {
        // Convert uint32_t coordinates to float for Draco
        float pos[3] = {static_cast<float>(points[i].x), 
                       static_cast<float>(points[i].y), 
                       static_cast<float>(points[i].z)};
        builder.SetAttributeValueForPoint(pos_attr_id, draco::PointIndex(i), pos);
    }
    
    std::unique_ptr<draco::PointCloud> geometry_only_pc = builder.Finalize(false);
    if (!geometry_only_pc) {
        std::cerr << "Error: Failed to create geometry-only point cloud" << std::endl;
        return result;
    }
    
    // Create encoder
    draco::Encoder encoder;
    
    // Set compression level to SPEED for fastest compression
    encoder.SetSpeedOptions(10, 10); // Maximum speed
    encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 11); // Lower quantization = faster
    
    // Compress
    StopWatch sw;
    
    draco::EncoderBuffer buffer;
    draco::Status status = encoder.EncodePointCloudToBuffer(*geometry_only_pc, &buffer);
    if (!status.ok()) {
        std::cerr << "Error: Failed to encode point cloud: " << status.error_msg() << std::endl;
        return result;
    }
    
    result.compression_time_ms = sw.ElapsedMs();
    
    // Copy compressed data
    result.compressed_data.resize(buffer.size());
    std::memcpy(result.compressed_data.data(), buffer.data(), buffer.size());
    
    return result;
}

DecompressionResult decompress_draco(const std::vector<uint8_t>& compressed_data, const std::string& output_path) {
    DecompressionResult result;
    result.output_path = output_path;
    result.success = false;
    result.decompression_time_ms = 0.0;
    
    // Create decoder
    draco::Decoder decoder;
    
    // Measure decompression time (excluding file I/O)
    StopWatch sw;
    
    // Decode
    draco::DecoderBuffer decoder_buffer;
    decoder_buffer.Init(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
    
    draco::StatusOr<std::unique_ptr<draco::PointCloud>> status_or_pc = decoder.DecodePointCloudFromBuffer(&decoder_buffer);
    if (!status_or_pc.ok()) {
        std::cerr << "Error: Failed to decode point cloud" << std::endl;
        return result;
    }
    
    std::unique_ptr<draco::PointCloud> pc = std::move(status_or_pc).value();
    
    // Extract points from Draco point cloud
    const draco::PointAttribute* pos_attr = pc->GetNamedAttribute(draco::GeometryAttribute::POSITION);
    if (!pos_attr) {
        std::cerr << "Error: Point cloud has no position attribute" << std::endl;
        return result;
    }
    
    std::vector<Point3D> points;
    points.reserve(pc->num_points());
    
    // Extract position data - Draco stores positions as float32
    for (draco::PointIndex i(0); i < pc->num_points(); ++i) {
        draco::AttributeValueIndex val_index = pos_attr->mapped_index(i);
        // Get the raw data pointer and read 3 floats
        const uint8_t* data_ptr = pos_attr->GetAddress(draco::AttributeValueIndex(val_index));
        if (!data_ptr) {
            std::cerr << "Error: Failed to get position data pointer" << std::endl;
            return result;
        }
        const float* pos = reinterpret_cast<const float*>(data_ptr);
        points.emplace_back(static_cast<uint32_t>(pos[0]), 
                           static_cast<uint32_t>(pos[1]), 
                           static_cast<uint32_t>(pos[2]));
    }
    
    result.decompression_time_ms = sw.ElapsedMs();
    
    // Save decompressed point cloud using unified PLY writer (file I/O excluded from timing)
    if (!save_ply_geometry(points, output_path)) {
        std::cerr << "Error: Could not save file " << output_path << std::endl;
        return result;
    }
    
    result.success = true;
    return result;
}

