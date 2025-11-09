#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <cstdint>

/**
 * Simple 3D point structure (geometry only)
 * Uses uint32_t for voxelized integer coordinates
 */
struct Point3D {
    uint32_t x, y, z;
    
    Point3D() : x(0), y(0), z(0) {}
    Point3D(uint32_t x, uint32_t y, uint32_t z) : x(x), y(y), z(z) {}
};

/**
 * Result structure for compression operations
 */
struct CompressionResult {
    std::vector<uint8_t> compressed_data;
    double compression_time_ms;
};

/**
 * Result structure for decompression operations
 */
struct DecompressionResult {
    std::string output_path;
    bool success;
    double decompression_time_ms;
};

/**
 * Read PLY file and extract geometry (points only, no color)
 * @param file_path Path to PLY file
 * @return Vector of Point3D containing geometry data
 */
std::vector<Point3D> read_ply_geometry(const std::string& file_path);

/**
 * Save point cloud to PLY file in the same format as the original redandblack PLY files
 * @param points Vector of 3D points (geometry only)
 * @param output_path Path to save the PLY file
 * @return true if successful, false otherwise
 */
bool save_ply_geometry(const std::vector<Point3D>& points, const std::string& output_path);

#endif // UTILS_H

