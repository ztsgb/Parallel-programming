#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O2

int main()
{
    MPI_Init(NULL, NULL);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    if (rank != 0) cout.setstate(ios_base::failbit); // 只禁用非主进程的cout输出
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    if (rank != 0) cout.clear(); // 恢复非主进程的cout输出
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    q.init();
    if (rank == 0) cout << "here" << endl;
    // 广播模型和优先队列初始化信息（如有需要，可补充序列化/反序列化）
    // 这里只做简单同步，实际复杂模型需自定义广播
    MPI_Barrier(MPI_COMM_WORLD);
    int curr_num = 0;
    // 只让每个进程处理属于自己的PT
    int local_history = 0;
    double local_time_hash = 0;
    int local_total_guesses = 0;
    bit32 state[4];
    auto start = system_clock::now();
// 设定全局猜测上限
int generate_n = 10000000;
int local_limit = generate_n / size;
if (rank == size - 1) local_limit += generate_n % size; // 最后一个进程多分一点
int local_generated = 0;

for (size_t i = 0; i < q.priority.size(); ++i) {
    if (i % size != rank) continue; // 只处理分配给本进程的PT
    if (local_generated >= local_limit) break; // 达到本地上限就退出
    // 处理本PT
    q.PopNext();
    q.total_guesses = q.guesses.size();
    // 处理哈希
    if (q.total_guesses > 0) {
        auto start_hash = system_clock::now();
        for (string pw : q.guesses) {
            MD5Hash(pw, state);
        }
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        local_time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        local_history += q.total_guesses;
        local_total_guesses += q.total_guesses;
        local_generated += q.total_guesses;
        q.guesses.clear();
    }
}
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    double local_time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    // 汇总全局统计
    int global_history = 0;
    double global_time_guess = 0;
    MPI_Reduce(&local_history, &global_history, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time_guess, &global_time_guess, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Total wall time: " << global_time_guess << " seconds" << endl;
        cout << "Train time: " << time_train << " seconds" << endl;
        cout << "Total guesses: " << global_history << endl;
    }
    MPI_Finalize();
    return 0;
}