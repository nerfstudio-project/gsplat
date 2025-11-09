#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>

std::vector<Point3D> read_ply_geometry(const std::string& file_path) {
    std::vector<Point3D> points;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return points;
    }
    
    std::string line;
    bool header_end = false;
    int vertex_count = 0;
    int vertex_property_count = 0;
    bool has_x = false, has_y = false, has_z = false;
    int x_idx = -1, y_idx = -1, z_idx = -1;
    
    // Read header
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        
        if (token == "element" && line.find("vertex") != std::string::npos) {
            // Extract vertex count
            std::string dummy;
            iss >> dummy; // "vertex"
            iss >> vertex_count;
        } else if (token == "property") {
            std::string type, name;
            iss >> type;
            iss >> name;
            
            if (name == "x") {
                has_x = true;
                x_idx = vertex_property_count;
            } else if (name == "y") {
                has_y = true;
                y_idx = vertex_property_count;
            } else if (name == "z") {
                has_z = true;
                z_idx = vertex_property_count;
            }
            vertex_property_count++;
        } else if (token == "end_header") {
            header_end = true;
            break;
        }
    }
    
    if (!header_end || !has_x || !has_y || !has_z) {
        std::cerr << "Error: Invalid PLY header or missing x/y/z properties" << std::endl;
        file.close();
        return points;
    }
    
    // Reserve space for points
    points.reserve(vertex_count);
    
    // Read vertex data (ASCII format)
    for (int i = 0; i < vertex_count; ++i) {
        if (!std::getline(file, line)) {
            break;
        }
        
        std::istringstream iss(line);
        std::vector<float> values(vertex_property_count);
        
        for (int j = 0; j < vertex_property_count; ++j) {
            iss >> values[j];
        }
        
        // Extract x, y, z and convert to uint32_t (voxelized integer coordinates)
        Point3D pt(static_cast<uint32_t>(values[x_idx]), 
                   static_cast<uint32_t>(values[y_idx]), 
                   static_cast<uint32_t>(values[z_idx]));
        points.push_back(pt);
    }
    
    file.close();
    return points;
}

bool save_ply_geometry(const std::vector<Point3D>& points, const std::string& output_path) {
    std::ofstream ply_file(output_path);
    if (!ply_file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output_path << std::endl;
        return false;
    }

    // Sort vertices by x, then y, then z
    std::vector<Point3D> sorted_points = points;
    std::sort(sorted_points.begin(), sorted_points.end(), 
        [](const Point3D& a, const Point3D& b) {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.z < b.z;
        });

    // Write PLY header matching the original format
    ply_file << "ply\n";
    ply_file << "format ascii 1.0\n";
    ply_file << "element vertex " << sorted_points.size() << "\n";
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "end_header\n";

    // Write vertices (convert uint32_t to float for PLY format)
    for (const auto& pt : sorted_points) {
        ply_file << static_cast<float>(pt.x) << " "
                 << static_cast<float>(pt.y) << " "
                 << static_cast<float>(pt.z) << "\n";
    }

    ply_file.close();
    
    if (!ply_file.good()) {
        std::cerr << "Error: Failed to write PLY data to file " << output_path << std::endl;
        return false;
    }

    return true;
}

