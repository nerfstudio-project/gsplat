#include "debug_utils.h"

#include <iostream>
#include <iomanip>

float percent_error(float exp_val, float true_val) {
    return (abs(true_val - exp_val) / true_val ) * 100.f;
}

std::vector<float> percent_errors(
    const std::vector<std::pair<float,float>>& values
) {
    std::vector<float> p_errors;

    for (int i = 0; i < values.size(); ++i) {
        const float exp_val  = values[i].first;
        const float true_val = values[i].second;
        const float p_error{ percent_error(exp_val, true_val) };
        p_errors.push_back(p_error);
    }

    return p_errors;
}

void print_errors(
    const std::vector<std::pair<float, float>> &values,
    const std::string &name,
    int check_index,
    float error_threshold // by default percent error (0.01%)
) {
    const std::vector<float> p_errors{percent_errors(values)};

    // only print if input is unexpected
    bool data_within_error{true};
    for (float p : p_errors)
        if (p > error_threshold)
            data_within_error = false;

    if (data_within_error)
        return;

    // format cout for how we want to display data
    std::cout << std::setprecision(2);

    std::cout << "Error on check: " << check_index << '\n'
              << name << ":\n"
              << "     ours:     refs:\n";

    for (int i = 0; i < values.size(); ++i) {
        const float d1 = values[i].first;
        const float d2 = values[i].second;
        std::cout << '[' << i << "]: " << std::scientific 
                  << std::setw(10) << d1 << ", " 
                  << std::setw(10) << d2
                  << "\t(percent error=" << std::fixed << p_errors[i] << ")\n";
    }
    std::cout << '\n';
}