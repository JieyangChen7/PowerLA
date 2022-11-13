## Energy-saving matrix decomposition framework (PowerLA)
### Hardware requirements
* x86 CPU and NVIDIA GPUs (tested on a server with Intel Core i7-9700K with NVIDIA RTX 2080 Ti)

### OS requirements
* Linux operating system (tested on Ubuntu 18.04)

### Software dependencies/configurations
* For measuring CPU power: `cpu-energy-meter`(https://github.com/sosy-lab/cpu-energy-meter).
* For adjusting CPU clock frequency: `cpupower`(one of Linux tools).
* For adjusting CPU core voltage: `intel-undervolt`(https://wiki.archlinux.org/title/Undervolting_CPU).
* For running GPU code: `CUDA 11.4+`.
* measuring GPU power, control GPU clock offset: `NVIDIA GPU driver 450.80.02+`.
* For enabling GPU overclocking, set `Coolbits` to the maximum allowed (https://wiki.archlinux.org/title/NVIDIA/Tips\_and\_tricks). The `Coolbits` on the tested system was set to `28`.
* For compliation: `GCC 7.5.0+` and `NVCC 11.4+`.
* For building the project: `CMake 2.8+`.

### Building our PowerLA framework
* The PowerLA framework was built based on the MAGMA library v 2.5.4, so it uses the same build system as the MAGMA library. Please follow the `READMD.md` in the current directory to build PowerLA.

### Running optimized matrix decompositions

##### 1. The major three one-sided matrix decomposition algorithms (Cholesky, LU, and QR) were optimized. They are implemented in:

* Cholesky: ./src/dportf\_gpu.cpp; ./src/sportf\_gpu.cpp
* LU: ./src/dgetrf\_gpu.cpp; ./src/sgetrf\_gpu.cpp
* QR: ./src/dgeqrf\_gpu.cpp; ./src/sgeqrf\_gpu.cpp

In each source code file, we added the following variables to control the energy-saving and fault-tolerance behavior of each matrix decomposition.

* `int tmu_curr_freq` and `int tmu_base_freq`: set the current and based clock frequency of GPU. They should be the same.
* `int tmu_base_offset`: set the base clock offset of GPU.
* `int tmu_opt_offset`: set the optimized clock offset of GPU.
* `adj_gpu(device, tmu_base_freq, 338000);`: set the power limit of GPU.
* `int pd_curr_freq` and `int pd_base_freq`: set the current and based clock frequency of CPU. They should be the same.
* `bool reclaim_slack`: control if we want to enable Slack Reclamation (BSR or SR).
* `double reclaimnation_ratio`: control how much of the slack is reclaimed by the task on the critical path.
* `bool overclock`: control if we want to overclock with undervolting.
* `bool autoboost`: control if we want to enable hardware R2H.
* `bool COL_FT` and `bool ROW_FT` control if we want to enable ABFT (single-side or full checksum)

##### 2. Once the PowerLA framework is built, the MAGMA testing binary executables can be used to run each matrix decomposition with a specificed input matrix size.

The executables can be run with `<build dir>/testing/testing_*_gpu -N <matrix size>`

##### 3. Configuring the variables for different modes

||Original|R2H|SR|BSR|
|---|---|---|---|---|
| `reclaim_slack` |false|false|true|true|
|`reclaimnation_ratio `|N/A|N/A|0|0-1|
| `overclock` |false|false|false|true|
| `autoboost` |false|true|false|false|
| `COL_FT/ROW_FT` |false|false|false|true|

##### 4. When each test finishes execution it will output:

* Energy consumption of CPU and GPU (total)
* Time cost (per operation & total)
* Predicted time cost (per iteration)
* The slack prediction error (total average)
* Clock frequency of CPU and GPU (per iteration)
* Decisions on slack reclamation (per iteration)
