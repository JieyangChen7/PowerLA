#include "magma_internal.h"
#undef max
#undef min
#include "abft_checker.h"

void abft_dtrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    int m, int n,
    double alpha,
    double * dA, int ldda,
    double *       dB, int lddb,
    double *       dX, int lddx,
    int flag,
    double * d_dinvA, int dinvA_length,
    int nb,
    double * dA_colchk,    int ldda_colchk,
    double * dA_rowchk,    int ldda_rowchk,
    double * dA_colchk_r,  int ldda_colchk_r,
    double * dA_rowchk_r,  int ldda_rowchk_r,
    double * dB_colchk,    int lddb_colchk,
    double * dB_rowchk,    int lddb_rowchk,
    double * dB_colchk_r,  int lddb_colchk_r,
    double * dB_rowchk_r,  int lddb_rowchk_r,
    double * chk_v,        int ld_chk_v, 
    bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    magma_queue_t stream1, magma_queue_t stream2) {


	if (FT & CHECK_BEFORE) {
		// abft_checker_colchk(dA, ldda, n, n, nb,
		// 				    dA_colchk,   ldda_colchk,
  //   					    dA_colchk_r, ldda_colchk_r,
  //   					    chk_v,       ld_chk_v,
  //   					    DEBUG,
  //   					    stream1);
		abft_checker_colchk(dB, lddb, m, n, nb,
						    dB_colchk,   lddb_colchk,
    					    dB_colchk_r, lddb_colchk_r,
    					    chk_v,       ld_chk_v,
    					    DEBUG,
    					    stream1);
	}

	magmablas_dtrsm_work(side, uplo, transA, diag,
						 m, n,
						 alpha,
						 dA, ldda,
						 dB, lddb,
						 dX, lddx,
						 flag,
						 d_dinvA, dinvA_length,
						 stream1);
	if (FT) {
		magma_dtrsm( side, uplo, transA, diag,
					 (m / nb) * 2, n,
					 alpha,
                     dA, ldda,
				     dB_colchk, lddb_colchk,
                     stream2);
	}


	if (FT & CHECK_BEFORE) {
		abft_checker_colchk(dB, lddb, m, n, nb,
						    dB_colchk,   lddb_colchk,
    					    dB_colchk_r, lddb_colchk_r,
    					    chk_v,       ld_chk_v,
    					    DEBUG,
    					    stream1);
	}

}