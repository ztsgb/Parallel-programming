#include <windows.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    // 输出CPU核心数
    SYSTEM_INFO sysinfo;
    ZeroMemory(&sysinfo, sizeof(SYSTEM_INFO));
    GetSystemInfo(&sysinfo);
    int cpu_cores = sysinfo.dwNumberOfProcessors;
    std::cout << "CPU Cores: " << cpu_cores << std::endl;

    // 输出CUDA设备信息
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, i);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
            continue;
        }
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        int coresPerSM = 0;
        switch (deviceProp.major) {
            case 2: coresPerSM = (deviceProp.minor == 1) ? 48 : 32; break; // Fermi
            case 3: coresPerSM = 192; break; // Kepler
            case 5: coresPerSM = 128; break; // Maxwell
            case 6: coresPerSM = (deviceProp.minor == 1 || deviceProp.minor == 2) ? 128 : 64; break; // Pascal
            case 7: coresPerSM = 64; break; // Volta/Turing
            case 8: coresPerSM = 64; break; // Ampere
            default: coresPerSM = 64; // 估算
        }
        int totalCores = deviceProp.multiProcessorCount * coresPerSM;
        std::cout << "CUDA Cores: " << totalCores << std::endl;
        std::cout << "Global Memory: " << (deviceProp.totalGlobalMem >> 20) << " MB" << std::endl;
    }
    return 0;
}
