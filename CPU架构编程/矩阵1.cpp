#include <iostream>
using namespace std;

const int n = 1024; // 矩阵维度



void init_matrix(int** a) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            a[j][i] = j + i; // 整数赋值
        }
    }
}

void init_vector(int* v) {
    for (int j = 0; j < n; ++j) {
        v[j] = j + 1;
    }
}
// 方法1：逐列遍历（缓存不友好）
void column_major(int** a, int* v, int* col_sums) {
    for (int i = 0; i < n; ++i) {
        int sum = 0;
        for (int j = 0; j < n; ++j) {
            sum += a[j][i] * v[j]; // 整数运算
        }
        col_sums[i] = sum;
    }
}

// 方法2：缓存优化（按行遍历）
void cache_optimized(int** a, int* v, int* sums) {
    for (int i = 0; i < n; i++)
        sums[i] = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sums[i] += a[j][i] * v[j];
}

int main() {
    // 动态内存分配
    int** a = new int* [n];
    for (int i = 0; i < n; ++i) a[i] = new int[n];

    int* v = new int[n];
    int* col_sums1 = new int[n](); // 初始化为0
    //int* col_sums2 = new int[n]();

    // 初始化数据
    init_matrix(a);
    init_vector(v);
    for(int i=0;i<10000;i++)
    column_major(a, v, col_sums1);

    // 释放内存
    for (int i = 0; i < n; ++i) delete[] a[i];
    delete[] a;
    delete[] v;
    delete[] col_sums1;

    return 0;
}
