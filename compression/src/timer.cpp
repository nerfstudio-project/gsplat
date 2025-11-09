/* 
 * File:   StopWatch.cpp
 * Author: KjellKod
 * From: https://github.com/KjellKod/StopWatch
 * 
 * Created on 2014-02-07 
 */

 #include "timer.h"
 #include <unistd.h>
 
 StopWatch::StopWatch() : mStart(clock::now()) {
     static_assert(std::chrono::steady_clock::is_steady, "Serious OS/C++ library issues. Steady clock is not steady");
     // FYI:  This would fail  static_assert(std::chrono::high_resolution_clock::is_steady(), "High Resolution Clock is NOT steady on CentOS?!");
 }
 
 StopWatch::StopWatch(const StopWatch& other): mStart(other.mStart) { 
 }
 
 /// @return StopWatch::StopWatch&  - assignment operator.
 StopWatch& StopWatch::operator=(const StopWatch& rhs) {
         mStart = rhs.mStart;
         return *this;
 }
 
 /// @return the elapsed microseconds since start
 uint64_t StopWatch::ElapsedUs() const {
     return std::chrono::duration_cast<microseconds>(clock::now() - mStart).count();
 }
 
 /// @return the elapsed milliseconds since start
 uint64_t StopWatch::ElapsedMs() const {
     return std::chrono::duration_cast<milliseconds>(clock::now() - mStart).count();
 }
 
 /// @return the elapsed seconds since start
 uint64_t StopWatch::ElapsedSec() const {
     return std::chrono::duration_cast<seconds>(clock::now() - mStart).count();
 }
 /**
  * Resets the start point
  * @return the updated start point
  */
 std::chrono::steady_clock::time_point StopWatch::Restart() {
     mStart = clock::now();
     return mStart;
 }
 
 /// Print current timestamp
 uint64_t StopWatch::Curr() {
     return std::chrono::system_clock::now().time_since_epoch() / 
      std::chrono::milliseconds(1);
 }
 
 FPSCounter::FPSCounter() {
     last_counter_s = 0;
     curr_counter_s = 0;
     last_s = curr();
     curr_s = last_s;
 }
 
 void FPSCounter::counter(int count, int fps) {
     curr_counter_s = count;
     curr_s = curr();
     // rate limiting
     if(fps) {
         if (curr_counter_s - last_counter_s >= fps) {
             while(curr_s == last_s) {
                 sleep(0.01);
                 curr_s = curr();
             }
         }
     }
     
 
     if (curr_s > last_s) {
         std::cout << "FPS: " << float(curr_counter_s - last_counter_s) * 1.0f / (curr_s - last_s) << " " << curr() << std::endl;
         last_counter_s = curr_counter_s;
         last_s = curr_s;
     }
     
     return;
 }
 
 void FPSCounter::rate_limit(int count, int fps) {
     curr_counter_s = count;
     curr_s = curr();
     // rate limiting
     if (curr_counter_s - last_counter_s >= fps) {
         while(curr_s == last_s) {
             sleep(0.01);
             curr_s = curr();
         }
     }
 
     if (curr_s > last_s) {
         std::cout << "Rate limited FPS: " << float(curr_counter_s - last_counter_s) * 1.0f / (curr_s - last_s) << " " << curr() << std::endl;
         last_counter_s = curr_counter_s;
         last_s = curr_s;
     }
     
     return;
 }
 
 uint64_t FPSCounter::curr() {
     return std::chrono::system_clock::now().time_since_epoch() / 
      std::chrono::milliseconds(1000);
 }
 
 FPSCounter2::FPSCounter2(float fps) 
 {
     this->fps = fps;
     frame_interval = 1000.0 / fps;
     last_frame_time = sw.ElapsedMs();
     last_delay = 0;
     start = false;
 }
 
 float FPSCounter2::fps_counter(uint64_t curr_time)
 {
     float fps_now = 0.0;
     if(curr_time == 0)
         curr_time = sw.ElapsedMs();
     
     if(frame_times.size() < FPS_COUNTER_CAPACITY)
     {
         frame_times.push(curr_time);
         fps_now = 1000.0 / (frame_times.back() - frame_times.front()) * (frame_times.size() - 1);
     }
     else
     {
         frame_times.pop();
         frame_times.push(curr_time);
         fps_now = 1000.0 / (frame_times.back() - frame_times.front()) * (frame_times.size() - 1);
     }
     return fps_now;
 }
 
 /**
  * @brief Limits the FPS to the specified value
  * Adjusts the delay to achieve the specified FPS
  * Note: Don't call fps_counter() and rate_limit() on the same FPSCounter2 object, since they share the same frame_times queue
  * Use different objects for fps_counter() and rate_limit()
  * @return float fps achieved
  */
 float FPSCounter2::rate_limit()
 {
     // This to avoid the first frame delay. Since object is created before the first frame is processed, so last_delay will be quite large
     if(!start)
     {
         last_frame_time = sw.ElapsedMs();
         start = true;
         return fps_counter(last_frame_time);
     }
 
     int64_t elapsed_time = (int64_t)sw.ElapsedMs() - last_frame_time;
     int64_t cur_delay = (frame_interval - elapsed_time) + last_delay;
     if(cur_delay > 0)
     {
         sleep_ms(cur_delay);
         last_delay = 0;
     }
     else
         last_delay = cur_delay;
     last_frame_time = sw.ElapsedMs();
     return fps_counter(last_frame_time);
 }