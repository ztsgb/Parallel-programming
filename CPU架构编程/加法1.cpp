#include <iostream>
#include <chrono>

using namespace std;

constexpr int N = 1 << 30; // 2^19 = 524,288

// 平凡链式加法（顺序执行）
uint64_t sequential_sum(const uint64_t* data) {
    uint64_t sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += data[i]; // 串行依赖链
    }
    return sum;
}

// 三链路超标量优化加法（指令级并行）
uint64_t superscalar_sum(const uint64_t* data) {
    uint64_t sum1 = 0, sum2 = 0, sum3 = 0;
    int i;
    // 主循环处理三个独立链路
    for (i = 0; i < N - 2; i += 3) {
        sum1 += data[i];   // 独立链路1
        sum2 += data[i + 1]; // 独立链路2
        sum3 += data[i + 2]; // 独立链路3
    }
    // 处理剩余元素（最多2个）
    for (; i < N; ++i) {
        sum1 += data[i];
    }
    return sum1 + sum2 + sum3;
}

int main() {
    // 动态分配内存避免栈溢出
    uint64_t* data = new uint64_t[N];
    for (int i = 0; i < N; ++i) {
        data[i] = 1; // 初始化全1数组
    }
    for(int p=0;p<300;p++)
     sequential_sum(data);

    delete[] data;

    return 0;
}