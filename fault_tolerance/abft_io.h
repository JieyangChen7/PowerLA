#include <string>
#include <nvml.h>
//#define GENERATE_GROUNDTRUTH
#define FAULT_ANALYSIS
#define VOID_PROPAGATION
#define RECORD_TIME_AND_ENERGY
#define POWER_LIMIT 38000
#define GRAPHICS_CLOCK 1500
#define MEMORY_CLOCK 5400
void store_matrix(int m, int n, double * dA, int ldda, magma_queue_t stream, std::string file_prefix);
void load_matrix(int m, int n, double * A, int lda, std::string file_prefix);
void load_matrix_to_dev(int m, int n, double * dA, int ldda, magma_queue_t stream, std::string file_prefix);
void compare_matrices(int m, int n, std::string file_prefix1, std::string file_prefix2, std::string output_prefix);

void store_matrix(int m, int n, float * dA, int ldda, magma_queue_t stream, std::string file_prefix);
void load_matrix(int m, int n, float * A, int lda, std::string file_prefix);
void load_matrix_to_dev(int m, int n, float * dA, int ldda, magma_queue_t stream, std::string file_prefix);
void compare_matrices_float(int m, int n, std::string file_prefix1, std::string file_prefix2, std::string output_prefix);

void record_time_and_energy_start(double * time, magma_queue_t stream, nvmlDevice_t device);
void record_time_and_energy_end(double * time, magma_queue_t stream, std::string output_prefix, nvmlDevice_t device);