#include <nvml.h>

// #include <chrono>
// using namespace std;
// using namespace std::chrono;
#include <sys/types.h>
#include <unistd.h>
// #include <cstdlib>
#include <string>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <array>

#undef min
#undef max

int start_measure_cpu();
void stop_measure_cpu(int pid);
unsigned long long start_measure_gpu(nvmlDevice_t device);
unsigned long long stop_measure_gpu(nvmlDevice_t device, unsigned long long start_energy);
void pm_cpu();
void adj_cpu(int clock);
void pm_gpu(nvmlDevice_t device);
void reset_gpu(nvmlDevice_t device);
void offset_gpu(int offset);
int adj_gpu(nvmlDevice_t device, int clock, int power);