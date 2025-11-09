#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include "utils.h"
#include "compression.h"

namespace fs = std::filesystem;

int main() {
    // Paths
    std::string input_folder = "/ssd1/rajrup/Project/gsplat/data/redandblack/Ply";
    std::string output_base = "/ssd1/rajrup/Project/gsplat/results/redandblack";
    
    // Create output directories
    fs::create_directories(output_base + "/pcl");
    fs::create_directories(output_base + "/open3d");
    fs::create_directories(output_base + "/draco");
    fs::create_directories(output_base + "/gpu_octree");
    
    // Get list of PLY files and sort them
    std::vector<std::string> ply_files;
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ply") {
            ply_files.push_back(entry.path().string());
        }
    }
    
    // Sort files to get first 10
    std::sort(ply_files.begin(), ply_files.end());
    
    // Process first 10 files
    int num_files = std::min(10, static_cast<int>(ply_files.size()));
    
    std::cout << "Processing " << num_files << " PLY files..." << std::endl;
    std::cout << "=========================================" << std::endl;
    
    for (int i = 0; i < num_files; ++i) {
        std::string input_file = ply_files[i];
        fs::path input_path(input_file);
        std::string filename = input_path.stem().string(); // e.g., "redandblack_vox10_1450"
        
        // Extract point cloud number (last part after underscore)
        size_t last_underscore = filename.find_last_of('_');
        std::string ptcl_number = (last_underscore != std::string::npos) ? 
                                  filename.substr(last_underscore + 1) : filename;
        
        std::cout << "\nProcessing file " << (i + 1) << "/" << num_files << ": " << input_path.filename().string() << std::endl;
        std::cout << "Point cloud number: " << ptcl_number << std::endl;
        
        // Read point cloud geometry once
        std::cout << "Reading point cloud geometry..." << std::endl;
        std::vector<Point3D> points = read_ply_geometry(input_file);
        if (points.empty()) {
            std::cerr << "Error: Failed to read point cloud from " << input_file << std::endl;
            continue;
        }
        std::cout << "Loaded " << points.size() << " points" << std::endl;
        
        // Validate that all coordinates are within [0, 1023] range for voxelized coordinates
        for (const auto& pt : points) {
            assert(pt.x <= 1023u && pt.y <= 1023u && pt.z <= 1023u &&
                   "Point coordinates must be in [0, 1023] range for voxelized coordinates");
        }
        
        // PCL Compression
        std::cout << "\n--- PCL Compression ---" << std::endl;
        CompressionResult pcl_result = compress_pcl(points);
        if (pcl_result.compression_time_ms > 0) {
            std::cout << "Original size: " << pcl_result.original_size_bytes << " bytes (" 
                      << (pcl_result.original_size_bytes / 1024.0) << " KB)" << std::endl;
            std::cout << "Compressed size: " << pcl_result.compressed_size_bytes << " bytes (" 
                      << (pcl_result.compressed_size_bytes / 1024.0) << " KB)" << std::endl;
            std::cout << "Compression ratio: " << (double)pcl_result.original_size_bytes / pcl_result.compressed_size_bytes 
                      << ":1" << std::endl;
            std::cout << "Compression time: " << pcl_result.compression_time_ms << " ms" << std::endl;
            std::string pcl_output = output_base + "/pcl/" + ptcl_number + ".ply";
            DecompressionResult pcl_decomp = decompress_pcl(pcl_result.compressed_data, pcl_output);
            if (pcl_decomp.success) {
                std::cout << "Decompression time: " << pcl_decomp.decompression_time_ms << " ms" << std::endl;
                std::cout << "Decompressed and saved to: " << pcl_output << std::endl;
            } else {
                std::cerr << "Failed to decompress PCL" << std::endl;
            }
        } else {
            std::cerr << "Failed to compress with PCL" << std::endl;
        }
        
        // Open3D Compression
        std::cout << "\n--- Open3D Compression ---" << std::endl;
        CompressionResult open3d_result = compress_open3d(points);
        if (open3d_result.compression_time_ms > 0) {
            std::cout << "Original size: " << open3d_result.original_size_bytes << " bytes (" 
                      << (open3d_result.original_size_bytes / 1024.0) << " KB)" << std::endl;
            std::cout << "Compressed size: " << open3d_result.compressed_size_bytes << " bytes (" 
                      << (open3d_result.compressed_size_bytes / 1024.0) << " KB)" << std::endl;
            std::cout << "Compression ratio: " << (double)open3d_result.original_size_bytes / open3d_result.compressed_size_bytes 
                      << ":1" << std::endl;
            std::cout << "Compression time: " << open3d_result.compression_time_ms << " ms" << std::endl;
            std::string open3d_output = output_base + "/open3d/" + ptcl_number + ".ply";
            DecompressionResult open3d_decomp = decompress_open3d(open3d_result.compressed_data, open3d_output);
            if (open3d_decomp.success) {
                std::cout << "Decompression time: " << open3d_decomp.decompression_time_ms << " ms" << std::endl;
                std::cout << "Decompressed and saved to: " << open3d_output << std::endl;
            } else {
                std::cerr << "Failed to decompress Open3D" << std::endl;
            }
        } else {
            std::cerr << "Failed to compress with Open3D" << std::endl;
        }
        
        // Draco Compression
        std::cout << "\n--- Draco Compression ---" << std::endl;
        CompressionResult draco_result = compress_draco(points);
        if (draco_result.compression_time_ms > 0) {
            std::cout << "Original size: " << draco_result.original_size_bytes << " bytes (" 
                      << (draco_result.original_size_bytes / 1024.0) << " KB)" << std::endl;
            std::cout << "Compressed size: " << draco_result.compressed_size_bytes << " bytes (" 
                      << (draco_result.compressed_size_bytes / 1024.0) << " KB)" << std::endl;
            std::cout << "Compression ratio: " << (double)draco_result.original_size_bytes / draco_result.compressed_size_bytes 
                      << ":1" << std::endl;
            std::cout << "Compression time: " << draco_result.compression_time_ms << " ms" << std::endl;
            std::string draco_output = output_base + "/draco/" + ptcl_number + ".ply";
            DecompressionResult draco_decomp = decompress_draco(draco_result.compressed_data, draco_output);
            if (draco_decomp.success) {
                std::cout << "Decompression time: " << draco_decomp.decompression_time_ms << " ms" << std::endl;
                std::cout << "Decompressed and saved to: " << draco_output << std::endl;
            } else {
                std::cerr << "Failed to decompress Draco" << std::endl;
            }
        } else {
            std::cerr << "Failed to compress with Draco" << std::endl;
        }
        
        // GPU Octree Compression
        std::cout << "\n--- GPU Octree Compression ---" << std::endl;
        const uint32_t octree_depth = 10;  // 2^octree_depth
        CompressionResult gpu_octree_result = compress_gpu_octree(points, octree_depth);
        if (gpu_octree_result.compression_time_ms > 0) {
            std::cout << "Original size: " << gpu_octree_result.original_size_bytes << " bytes (" 
                      << (gpu_octree_result.original_size_bytes / 1024.0) << " KB)" << std::endl;
            std::cout << "Compressed size: " << gpu_octree_result.compressed_size_bytes << " bytes (" 
                      << (gpu_octree_result.compressed_size_bytes / 1024.0) << " KB)" << std::endl;
            std::cout << "Compression ratio: " << (double)gpu_octree_result.original_size_bytes / gpu_octree_result.compressed_size_bytes 
                      << ":1" << std::endl;
            std::cout << "Compression time: " << gpu_octree_result.compression_time_ms << " ms" << std::endl;
            std::string gpu_octree_output = output_base + "/gpu_octree/" + ptcl_number + ".ply";
            DecompressionResult gpu_octree_decomp = decompress_gpu_octree(gpu_octree_result.compressed_data, gpu_octree_output, octree_depth);
            if (gpu_octree_decomp.success) {
                std::cout << "Decompression time: " << gpu_octree_decomp.decompression_time_ms << " ms" << std::endl;
                std::cout << "Decompressed and saved to: " << gpu_octree_output << std::endl;
            } else {
                std::cerr << "Failed to decompress GPU Octree" << std::endl;
            }
        } else {
            std::cerr << "Failed to compress with GPU Octree" << std::endl;
        }
        
        std::cout << "\n=========================================" << std::endl;
    }
    
    std::cout << "\nAll files processed!" << std::endl;
    return 0;
}

