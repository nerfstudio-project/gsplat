#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

namespace stream {

class CudaTimer {
private:
    std::vector<cudaEvent_t> start_events;
    std::vector<cudaEvent_t> end_events;
    std::vector<std::string> stage_names;
    std::vector<float> stage_times;
    bool timing_enabled;
    int current_stage;

public:
    CudaTimer(bool enabled = true) : timing_enabled(enabled), current_stage(0) {
        if (timing_enabled) {
            // Pre-allocate events for common number of stages
            reserve_events(20);
        }
    }
    
    ~CudaTimer() {
        cleanup();
    }
    
    void reserve_events(int num_stages) {
        if (!timing_enabled) return;
        
        start_events.resize(num_stages);
        end_events.resize(num_stages);
        stage_names.resize(num_stages);
        stage_times.resize(num_stages, 0.0f);
        
        for (int i = 0; i < num_stages; i++) {
            cudaEventCreate(&start_events[i]);
            cudaEventCreate(&end_events[i]);
        }
    }
    
    void start_stage(const std::string& name) {
        if (!timing_enabled) return;
        
        if (current_stage >= start_events.size()) {
            std::cerr << "Warning: Too many timing stages, skipping: " << name << std::endl;
            return;
        }
        
        stage_names[current_stage] = name;
        cudaEventRecord(start_events[current_stage]);
    }
    
    void end_stage() {
        if (!timing_enabled) return;
        
        if (current_stage >= end_events.size()) {
            return;
        }
        
        cudaEventRecord(end_events[current_stage]);
        current_stage++;
    }
    
    void synchronize_and_compute() {
        if (!timing_enabled) return;
        
        // Synchronize all events
        for (int i = 0; i < current_stage; i++) {
            cudaEventSynchronize(end_events[i]);
        }
        
        // Compute elapsed times
        for (int i = 0; i < current_stage; i++) {
            cudaEventElapsedTime(&stage_times[i], start_events[i], end_events[i]);
        }
    }
    
    void print_results(const std::string& title = "CUDA Timing Results") {
        if (!timing_enabled) return;
        
        synchronize_and_compute();
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << title << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        float total_time = 0.0f;
        for (int i = 0; i < current_stage; i++) {
            total_time += stage_times[i];
        }
        
        for (int i = 0; i < current_stage; i++) {
            float percentage = (total_time > 0) ? (stage_times[i] / total_time * 100.0f) : 0.0f;
            std::cout << std::left << std::setw(35) << stage_names[i] 
                      << std::right << std::setw(8) << std::fixed << std::setprecision(2) << stage_times[i] << " ms"
                      << std::setw(8) << std::fixed << std::setprecision(1) << percentage << "%" << std::endl;
        }
        
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::left << std::setw(35) << "TOTAL TIME"
                  << std::right << std::setw(8) << std::fixed << std::setprecision(2) << total_time << " ms"
                  << std::setw(8) << "100.0%" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    
    float get_total_time() {
        if (!timing_enabled) return 0.0f;
        
        synchronize_and_compute();
        float total = 0.0f;
        for (int i = 0; i < current_stage; i++) {
            total += stage_times[i];
        }
        return total;
    }
    
    float get_stage_time(int stage_idx) {
        if (!timing_enabled || stage_idx >= current_stage) return 0.0f;
        synchronize_and_compute();
        return stage_times[stage_idx];
    }
    
    void reset() {
        current_stage = 0;
        std::fill(stage_times.begin(), stage_times.end(), 0.0f);
    }
    
    void cleanup() {
        if (!timing_enabled) return;
        
        for (auto& event : start_events) {
            cudaEventDestroy(event);
        }
        for (auto& event : end_events) {
            cudaEventDestroy(event);
        }
        start_events.clear();
        end_events.clear();
    }
};

// Convenience class for automatic stage timing
class ScopedCudaTimer {
private:
    CudaTimer* timer;
    
public:
    ScopedCudaTimer(CudaTimer* t, const std::string& stage_name) : timer(t) {
        if (timer) timer->start_stage(stage_name);
    }
    
    ~ScopedCudaTimer() {
        if (timer) timer->end_stage();
    }
};

#define CUDA_TIME_STAGE(timer, name) ScopedCudaTimer _scoped_timer(timer, name)

} // namespace stream

#endif // CUDA_TIMER_H
