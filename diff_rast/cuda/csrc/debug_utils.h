#pragma once

#include <string>
#include <utility>
#include <vector>

float percent_error(float exp_val, float true_val);

// takes a vector of float pairs, where first value is experimental, 
// and second is true value
std::vector<float> percent_errors(
    const std::vector<std::pair<float,float>>& values
);

// takes a vector of float pairs, where pair's first value is 
// "experimental value", and second is "true value". If any pair's percent exceeds 
// error threshold, the entire vector will be displayed in a nice lil format
void print_errors(
    const std::vector<std::pair<float, float>> &values,
    const std::string &name,
    int check_index,
    float error_threshold = 0.01f // percent error (0.01%)
);