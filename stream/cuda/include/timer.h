/* 
 * File:   StopWatch.h
 * Author: KjellKod
 * From: https://github.com/KjellKod/StopWatch
 * 
 * Created on 2014-02-07 
 */
 #pragma once
 #include <chrono>
 #include <boost/asio.hpp>
 #include <boost/date_time.hpp>
 #include <boost/chrono.hpp>
 #include <boost/chrono/chrono_io.hpp>
 #include <thread>
 #include <queue>
 
 using namespace boost::posix_time;
 using namespace boost::gregorian;
 
 typedef boost::chrono::system_clock::time_point time_point;
 typedef boost::chrono::nanoseconds time_ns;
 typedef boost::chrono::microseconds time_us;
 typedef boost::chrono::milliseconds time_ms;
 typedef boost::chrono::seconds time_s;
 
 #define time_now() boost::chrono::system_clock::now()
 #define time_now_ns() boost::chrono::duration_cast<time_ns>(time_now().time_since_epoch()).count()
 #define time_now_us() boost::chrono::duration_cast<time_us>(time_now().time_since_epoch()).count()
 #define time_now_ms() boost::chrono::duration_cast<time_ms>(time_now().time_since_epoch()).count()
 #define time_now_s() boost::chrono::duration_cast<time_s>(time_now().time_since_epoch()).count()
 
 #define duration_ns(start, end) boost::chrono::duration_cast<time_ns>(end - start).count()
 #define duration_us(start, end) boost::chrono::duration_cast<time_us>(end - start).count()
 #define duration_ms(start, end) boost::chrono::duration_cast<time_ms>(end - start).count()
 #define duration_s(start, end) boost::chrono::duration_cast<time_s>(end - start).count()
 
 #define sleep_ms(x) std::this_thread::sleep_for(std::chrono::milliseconds(x))
 
 class StopWatch {
 public:
    typedef std::chrono::steady_clock clock;
    typedef std::chrono::microseconds microseconds;
    typedef std::chrono::milliseconds milliseconds;
    typedef std::chrono::seconds seconds;
 
    StopWatch();
    StopWatch(const StopWatch&);
    StopWatch& operator=(const StopWatch& rhs);
 
    uint64_t ElapsedUs() const;
    uint64_t ElapsedMs() const;
    uint64_t ElapsedSec() const;
 
    std::chrono::steady_clock::time_point Restart();
    uint64_t Curr();
 
 private:
    clock::time_point mStart;
 };
 
 
 // fps counter
 class FPSCounter 
 {
 public:
    FPSCounter();
    void counter(int count, int fps = 0);
    void rate_limit(int count, int fps);
 
 private:
    uint64_t curr();
    uint64_t last_s;
    uint64_t curr_s;
    int last_counter_s;
    int curr_counter_s;
 };
 
 #define FPS_COUNTER_CAPACITY 30
 class FPSCounter2
 {
 public:
    FPSCounter2(float fps = 30.0);
    float fps_counter(uint64_t curr_time = 0);
    float rate_limit();
 private:
    float fps;
    int64_t frame_interval;
    int64_t last_frame_time;
    int64_t last_delay;
    StopWatch sw;
    bool start;
 
    std::queue<uint64_t> frame_times;
 };