#include "magma_internal.h"
#undef max
#undef min
#include "abft_checker.h"
#include "abft_io.h"
#include <string>

void abft_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    int m, int n,
    float alpha,
    float * dA, int ldda,
    float * dB, int lddb,
    int nb,
    float * dA_colchk,    int ldda_colchk,
    float * dA_rowchk,    int ldda_rowchk,
    float * dA_colchk_r,  int ldda_colchk_r,
    float * dA_rowchk_r,  int ldda_rowchk_r,
    float * dB_colchk,    int lddb_colchk,
    float * dB_rowchk,    int lddb_rowchk,
    float * dB_colchk_r,  int lddb_colchk_r,
    float * dB_rowchk_r,  int lddb_rowchk_r,
    float * chk_v,        int ld_chk_v, 
    bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    magma_queue_t stream1, magma_queue_t stream2) {


	if (FT & CHECK_BEFORE) {
		// abft_checker_colchk(dA, ldda, n, n, nb,
		// 				    dA_colchk,   ldda_colchk,
  //   					    dA_colchk_r, ldda_colchk_r,
  //   					    chk_v,       ld_chk_v,
  //   					    DEBUG,
  //   					    stream1);
        if (DEBUG) printf("strsm-before-check-B-col\n");
		abft_checker_colchk(dB, lddb, m, n, nb,
						    dB_colchk,   lddb_colchk,
    					    dB_colchk_r, lddb_colchk_r,
    					    chk_v,       ld_chk_v,
    					    DEBUG,
    					    stream1);
	}

#ifdef RECORD_TIME_AND_ENERGY
    if (nvmlInit () != NVML_SUCCESS){
        printf("init error");
        return;
    }
    int i = 0;
    nvmlReturn_t result;
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (NVML_SUCCESS != result){
      printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
      return;
    }
    double time = 0.0;
    record_time_and_energy_start(&time, stream1, device);
#endif

	magma_strsm(side, uplo, transA, diag,
					m, n,
					alpha,
					dA, ldda,
					dB, lddb,
					stream1);

#ifdef RECORD_TIME_AND_ENERGY
    std::string time_energy_name = "strsm-" +
                                    std::to_string(m) + "-" +
                                    std::to_string(n) +
                                    "-time-and-energy";
    record_time_and_energy_end(&time, stream1, time_energy_name, device);
#endif
    
	if (FT) {
		magma_strsm( side, uplo, transA, diag,
					 (m / nb) * 2, n,
					 alpha,
                     dA, ldda,
				     dB_colchk, lddb_colchk,
                     stream2);
	}




#ifdef GENERATE_GROUNDTRUTH
    magma_queue_sync( stream1 );
    std::string name = "strsm-" +
                        std::to_string(m) + "-" +
                        std::to_string(n) +
                        "-groundtruth";
    store_matrix(m, n, dB, lddb, stream1, name);
#endif

#ifdef FAULT_ANALYSIS
    magma_queue_sync( stream1 );
    std::string name = "strsm-" +
                        std::to_string(m) + "-" +
                        std::to_string(n) +
                        "-groundtruth";
    std::string name2 = "strsm-" +
                        std::to_string(m) + "-" +
                        std::to_string(n) +
                        "-current";
    std::string name3 = "strsm-" +
                        std::to_string(m) + "-" +
                        std::to_string(n) +
                        "-diff";
    store_matrix(m, n, dB, lddb, stream1, name2);
    compare_matrices_float(m, n, name, name2, name3);
#endif


	if (FT & CHECK_BEFORE) {
        if (DEBUG) printf("strsm-after-check-B-col\n");
		abft_checker_colchk(dB, lddb, m, n, nb,
						    dB_colchk,   lddb_colchk,
    					    dB_colchk_r, lddb_colchk_r,
    					    chk_v,       ld_chk_v,
    					    DEBUG,
    					    stream1);
#ifdef FAULT_ANALYSIS
    magma_queue_sync( stream1 );
    std::string name4 = "strsm-" +
                        std::to_string(m) + "-" +
                        std::to_string(n) +
                        "-current-after-abft";
    std::string name5 = "strsm-" +
                        std::to_string(m) + "-" +
                        std::to_string(n) +
                        "-diff-after-abft";
    store_matrix(m, n, dB, lddb, stream1, name4);
    compare_matrices_float(m, n, name, name4, name5);
#endif


	}

#ifdef VOID_PROPAGATION
    magma_queue_sync( stream1 );
    std::string groundtruth_name = "strsm-" +
                                   std::to_string(m) + "-" +
                                   std::to_string(n) +
                                   "-groundtruth";
                                    
    load_matrix_to_dev(m, n, dB, lddb, stream1, groundtruth_name);

#endif

}








