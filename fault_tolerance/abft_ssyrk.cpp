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
                 magma_queue_t stream1, magma_queue_t stream2
                 ){
    
    /*         k                n
     * ******************   *********
     * *        A       * =>*   C   * n
     * *                *   *       *
     * ******************   *********
     */

    if (FT && CHECK_BEFORE) {
        if (DEBUG) printf("ssyrk-before-check-A-col\n");
        abft_checker_colchk(dA, ldda, n, k, nb,
                            dA_colchk,   ldda_colchk,
                            dA_colchk_r, ldda_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);

        if (DEBUG) printf("ssyrk-before-check-C-col\n");
        abft_checker_colchk(dC, ldda, n, n, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);   
    }

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
        abft_checker_colchk(dC, ldda, n, n, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1); 
    }
    
}








