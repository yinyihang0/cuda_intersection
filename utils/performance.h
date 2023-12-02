#include <iostream>
#include <chrono>

// 开始计时的宏
#define START_TIMER auto start = std::chrono::high_resolution_clock::now();

// 结束计时并报告结果的宏
#define STOP_TIMER(msg) \
    auto stop = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double, std::milli> duration = stop - start; \
    std::cout << (msg) << " took " << duration.count() << " milliseconds.\n";



// function name getter
#define GET_FUNC_NAME(x) #x


#define TIMER_FUNC(func) do { \
    START_TIMER \
    func; \
    STOP_TIMER(GET_FUNC_NAME(func)) \
} while(0)