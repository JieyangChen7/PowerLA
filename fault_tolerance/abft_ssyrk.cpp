#include "magma_internal.h"
#undef max
#undef min
#include"abft_checker.h"
#include "abft_io.h"
#include <string>

//ssyrk with FT

/**
 * n: number of row of A
 * m: number of col of A
 */
void abft_ssyrk(magma_uplo_t uplo, magma_trans_t trans,
                 int n, int k, 
                 float alpha,
                 float * dA, int ldda,
                 float beta,
                 float * dC, int lddc,
                 int nb,
                 float * dA_colchk,    int ldda_colchk,
                 float * dA_rowchk,    int ldda_rowchk,
                 float * dA_colchk_r,  int ldda_colchk_r,
                 float * dA_rowchk_r,  int ldda_rowchk_r,
                 float * dC_colchk,    int lddc_colchk,
                 float * dC_rowchk,    int lddc_rowchk,
                 float * dC_colchk_r,  int lddc_colchk_r,
                 float * dC_rowchk_r,  int lddc_rowchk_r,
                 float * chk_v,        int ld_chk_v, 
                 bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
                 magma_queue_t stream1, magma_queue_t stream2){
    
    /*         k                n
     * ******************   *********
     * *        A       * =>*   C   * n
     * *                *   *       *
     * ******************   *********
     */
    
    if (FT && CHECK_BEFORE) {
        if (DEBUG) printf("ssyrk-before-check-A-col\n");
        abft_checker_colchk(dA, lddc, n, k, nb,
                            dA_colchk,   ldda_colchk,
                            dA_colchk_r, ldda_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);

        if (DEBUG) printf("ssyrk-before-check-C-col\n");
        abft_checker_colchk(dC, lddc, n, n, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
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

    if (FT) {
        magma_sgemm(
                MagmaNoTrans, MagmaTrans,
                n, n, k,
                MAGMA_D_ONE * (-1),
                dA, ldda, dA, ldda,
                MAGMA_D_ONE,
                dC, lddc,
                stream1);
    } else {
        magma_ssyrk(uplo, trans, n, k,
                    alpha, dA, ldda,
                    beta,     dC, lddc,
                    stream1);
    }

#ifdef RECORD_TIME_AND_ENERGY
    std::string time_energy_name = "ssyrk-" +
                                    std::to_string(n) + "-" +
                                    std::to_string(k) +
                                    "-time-and-energy";
    record_time_and_energy_end(&time, stream1, time_energy_name, device);
#endif

#ifdef GENERATE_GROUNDTRUTH
    magma_queue_sync( stream1 );
    std::string name = "ssyrk-" +
                        std::to_string(n) + "-" +
                        std::to_string(k) +
                        "-groundtruth";
    store_matrix(n, n, dC, lddc, stream1, name);
#endif

#ifdef FAULT_ANALYSIS
    magma_queue_sync( stream1 );
    std::string name = "ssyrk-" +
                        std::to_string(n) + "-" +
                        std::to_string(k) +
                        "-groundtruth";
    std::string name2 = "ssyrk-" +
                        std::to_string(n) + "-" +
                        std::to_string(k) +
                        "-current";
    std::string name3 = "ssyrk-" +
                        std::to_string(n) + "-" +
                        std::to_string(k) +
                        "-diff";
    store_matrix(n, n, dC, lddc, stream1, name2);
    compare_matrices_float(n, n, name, name2, name3);
#endif
    
    if(FT){
        //update checksums on GPU
        magma_sgemm(
                    MagmaNoTrans, MagmaTrans,
                    2, n, k,
                    MAGMA_D_ONE * (-1),
                    dA_colchk,  ldda_colchk,
                    dA,         ldda,
                    MAGMA_D_ONE,
                    dC_colchk,   lddc_colchk,
                    stream1);
    }


    if (FT && CHECK_AFTER) {
        //verify C after update
        if (DEBUG) printf("ssyrk-after-check-C-col\n");
        abft_checker_colchk(dC, lddc, n, n, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1); 
#ifdef FAULT_ANALYSIS
    magma_queue_sync( stream1 );
    std::string name4 = "ssyrk-" +
                        std::to_string(n) + "-" +
                        std::to_string(k) +
                        "-current-after-abft";
    std::string name5 = "ssyrk-" +
                        std::to_string(n) + "-" +
                        std::to_string(k) +
                        "-diff-after-abft";
    store_matrix(n, n, dC, lddc, stream1, name4);
    compare_matrices_float(n, n, name, name4, name5);
#endif
    }

#ifdef VOID_PROPAGATION
    magma_queue_sync( stream1 );
    std::string groundtruth_name = "ssyrk-" +
                                    std::to_string(n) + "-" +
                                    std::to_string(k) +
                                    "-groundtruth";
                                    
    load_matrix_to_dev(n, n, dC, lddc, stream1, groundtruth_name);

#endif


}








