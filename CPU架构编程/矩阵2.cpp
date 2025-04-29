#include <iostream>
using namespace std;

const int n = 1024; //     ά  



void init_matrix(int** a) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            a[j][i] = j + i; //       ֵ
        }
    }
}

void init_vector(int* v) {
    for (int j = 0; j < n; ++j) {
        v[j] = j + 1;
    }
}
//     1     б        治 Ѻã 
void column_major(int** a, int* v, int* col_sums) {
    for (int i = 0; i < n; ++i) {
        int sum = 0;
        for (int j = 0; j < n; ++j) {
            sum += a[j][i] * v[j]; //         
        }
        col_sums[i] = sum;
    }
}

//     2       Ż      б     
void cache_optimized(int** a, int* v, int* sums) {
    for (int i = 0; i < n; i++)
        sums[i] = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sums[i] += a[j][i] * v[j];
}

int main() {
    //   ̬ ڴ    
    int** a = new int* [n];
    for (int i = 0; i < n; ++i) a[i] = new int[n];

    int* v = new int[n];
/*    int* col_sums1 = new int[n]();*/ //   ʼ  Ϊ0
    int* col_sums2 = new int[n]();

    //   ʼ      
    init_matrix(a);
    init_vector(v);
    for(int i =0;i<10000;i++)
    cache_optimized(a, v, col_sums2);

    //  ͷ  ڴ 
    for (int i = 0; i < n; ++i) delete[] a[i];
    delete[] a;
    delete[] v;
    delete[] col_sums2;

    return 0;
}
