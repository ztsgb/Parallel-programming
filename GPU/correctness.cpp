#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;


    
    // 加载一些测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("./input/Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    int cracked=0;

    q.init();
    cout << "here" << endl;
    int history = 0;
    auto start = system_clock::now();
    while (!q.priority.empty())
    {
        q.PopNext();
        // 立即处理本批guesses
        auto start_hash = system_clock::now();
        bit32 state[4];
        for (const string& pw : q.guesses) {
            if (test_set.find(pw) != test_set.end()) {
                cracked += 1;
            }
            MD5Hash(pw, state);
        }
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        history += q.guesses.size();
        q.guesses.clear();
        // 输出进度
        if (history % 100000 < q.guesses.size()) {
            cout << "Guesses generated: " << history << endl;
        }
        // 终止条件
        if (history > 10000000) {
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            double time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
            cout << "Hash time:" << time_hash << "seconds" << endl;
            cout << "Train time:" << time_train << "seconds" << endl;
            cout << "Cracked:" << cracked << endl;
            break;
        }
    }
    // 循环外补一次，防止遗漏
    if (!q.guesses.empty()) {
        auto start_hash = system_clock::now();
        bit32 state[4];
        for (const string& pw : q.guesses) {
            if (test_set.find(pw) != test_set.end()) {
                cracked += 1;
            }
            MD5Hash(pw, state);
        }
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
    }
    cout << "Final Cracked: " << cracked << endl;
}
