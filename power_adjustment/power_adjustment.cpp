#include "power_adjustment.h"

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int start_measure_cpu() {
    pid_t  pid;
    int    status;
    char * cmd[2];
    cmd[0]="cpu-energy-meter";
    cmd[1]=NULL;

    if ((pid = fork()) < 0) {     
        printf("*** ERROR: forking child process failed\n");
        exit(1);
    }
    else if (pid == 0) {         
        if (execvp("/home/jieyang/software/cpu-energy-meter/cpu-energy-meter", cmd) < 0) {     
            printf("*** ERROR: exec failed\n");
            exit(1);
        }
    }
    return pid;
}

void stop_measure_cpu(int pid) {
    std::string pid_str = std::to_string(pid);
    std::string kill_cmd = "sudo kill -2 "+ pid_str;
    printf("executing: %s\n", kill_cmd.c_str());
    //system(kill_cmd.c_str());
    std:: string result = exec(kill_cmd.c_str());
    //printf("result");
}

unsigned long long start_measure_gpu(nvmlDevice_t device) {
    unsigned long long energy;
    nvmlDeviceGetTotalEnergyConsumption(device, &energy);
    return energy;
}

unsigned long long stop_measure_gpu(nvmlDevice_t device, unsigned long long start_energy) {
    unsigned long long stop_energy;
    nvmlDeviceGetTotalEnergyConsumption(device, &stop_energy);
    return stop_energy - start_energy;
    //printf("GPU energy: %llu\n", stop_energy - start_energy);
}

void pm_cpu() {
    system("sudo cpupower frequency-set --governor performance");
}

void adj_cpu(int clock) {
    std::string clock_str = std::to_string(clock*1000);
    std::string clock_cmd = "sudo cpupower frequency-set -u "+ clock_str + " > /dev/null";
    system(clock_cmd.c_str());
}


void pm_gpu(nvmlDevice_t device) {
    nvmlReturn_t result;
    result = nvmlDeviceSetPersistenceMode(device, NVML_FEATURE_ENABLED);
    if (NVML_SUCCESS != result)
    {
        printf("Set GPU PM error\n");
    }
}

void reset_gpu(nvmlDevice_t device) {
    nvmlDeviceResetGpuLockedClocks(device);
}

void offset_gpu(int offset) {
    std::string offset_str = std::to_string(offset);
    std::string offset_cmd = "sudo nvidia-settings -a [gpu:0]/GPUGraphicsClockOffset[4]="+ offset_str + " > /dev/null";
    system(offset_cmd.c_str());
}

int adj_gpu(nvmlDevice_t device, int clock, int power)
{
    
    nvmlReturn_t result;
    result = nvmlDeviceSetPowerManagementLimit (device, power);
    if (NVML_SUCCESS != result)
    {
      // printf("Failed to set power limit of device %i: %s\n", i, nvmlErrorString(result));
      return -1;
    }
    // unsigned int get_power_limit;
    // result = nvmlDeviceGetEnforcedPowerLimit(device, &get_power_limit);
    // if (NVML_SUCCESS != result) {
    //     printf("Failed to get power limit of device %i: %s\n", i, nvmlErrorString(result));
    //     return;
    // }
    // printf("Power limit set to: %d\n", get_power_limit);


    result = nvmlDeviceSetGpuLockedClocks ( device, clock, clock );
    if (NVML_SUCCESS != result)
    {
      // printf("Failed to set clock of device %i: %s\n", i, nvmlErrorString(result));
      return -1;
    }
    unsigned int get_gpu_clock;
    result = nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &get_gpu_clock);
    if (NVML_SUCCESS != result)
    {
      // printf("Failed to get GPU clock of device %i: %s\n", i, nvmlErrorString(result));
      return -1;
    }
    //printf("Clock is set to: %u, %u\n", get_gpu_clock, get_mem_clock);
    return get_gpu_clock;

    // nvmlEnableState_t set = NVML_FEATURE_DISABLED;
    // result = nvmlDeviceSetAutoBoostedClocksEnabled(device, set);
    // if (NVML_SUCCESS != result)
    // {
    //   printf("Failed to disable autoboost of device %i: %s\n", i, nvmlErrorString(result));
    //   return;
    // }

}
