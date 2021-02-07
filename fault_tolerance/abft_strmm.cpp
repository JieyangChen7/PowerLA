#include "magma_internal.h"
#undef max
#undef min
#include "abft_checker.h"
#include "abft_io.h"
#include <string>
#include "cuda_runtime.h"
#include "../fault_tolerance/abft_printer.h"

extern "C" void
abft_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n, 
    float alpha,
    float * dA, int ldda,
    float * dB, int lddb,
    magma_int_t nb,
    float * dA_colchk,   int ldda_colchk,
    float * dA_rowchk,   int ldda_rowchk,
    float * dA_colchk_r, int ldda_colchk_r,
    float * dA_rowchk_r, int ldda_rowchk_r,
    float * dB_colchk,   int lddb_colchk,
    float * dB_rowchk,   int lddb_rowchk,
    float * dB_colchk_r, int lddb_colchk_r,
    float * dB_rowchk_r, int lddb_rowchk_r,
    float * chk_v, int ld_chk_v, 
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    magma_queue_t stream1, magma_queue_t stream2) {

	    /* Constants */
    const float c_zero    = MAGMA_D_ZERO;
    const float c_one     = MAGMA_D_ONE;
    const float c_neg_one = MAGMA_D_NEG_ONE;

	int mem_row;
	int mem_col;
	if (side == MagmaRight && trans == MagmaNoTrans && diag == MagmaNonUnit) {
		if (CHECK_BEFORE) {
			if (COL_FT) {
				mem_row = m;
				mem_col = n;
				if (DEBUG) printf("strmm-before-check-B-col\n");
				abft_checker_colchk(dB, lddb, mem_row, mem_col, nb,
		                            dB_colchk,   lddb_colchk,
		                            dB_colchk_r, lddb_colchk_r,
		                            chk_v,       ld_chk_v,
		                            DEBUG,
		                            stream1);
			}
			if (ROW_FT) {
				mem_row = m;
				mem_col = n;
				if (DEBUG) printf("strmm-before-check-B-row\n");
				abft_checker_rowchk(dB, lddb, mem_row, mem_col, nb,
		                            dB_rowchk,   lddb_rowchk,
		                            dB_rowchk_r, lddb_rowchk_r,
		                            chk_v,       ld_chk_v,
		                            DEBUG,
		                            stream1);
			}
		}

		if (ROW_FT) { // since B is update we need to update checksum before main calculation

			// printf( "input matrix Tx:\n" );
   //      	printMatrix_gpu(dA, ldda, n, n, nb, nb, stream2);
   //      	printf( "column chk:\n" );
	  //       printMatrix_gpu(dA_colchk, ldda_colchk, 
	  //                       (n / nb) * 2, n, 2, nb, stream2);
	  //       printf( "row chk:\n" );
	  //       printMatrix_gpu(dA_rowchk, ldda_rowchk,  
	  //                       n, (n / nb) * 2, nb, 2, stream2);

	  //       printf( "input matrix Bxx:\n" );
   //      	printMatrix_gpu(dB, lddb, m, n, nb, nb, stream2);
   //      	printf( "column chk:\n" );
	  //       printMatrix_gpu(dB_colchk, lddb_colchk, 
	  //                       (m / nb) * 2, n, 2, nb, stream2);
	  //       printf( "row chk:\n" );
	  //       printMatrix_gpu(dB_rowchk, lddb_rowchk,  
	  //                       m, (n / nb) * 2, nb, 2, stream2);

			magma_sgemm(MagmaNoTrans, MagmaNoTrans,
						m, (n/nb)*2, n,
						c_one,
						dB, lddb, dA_rowchk, ldda_rowchk,
						c_zero,
						dB_rowchk, lddb_rowchk,
						stream2);
		}

		magma_strmm( MagmaRight, uplo, MagmaNoTrans, MagmaNonUnit,
	                 m, n,
	                 c_one, dA,  ldda,
	                 dB, lddb, stream1 );

		if (COL_FT) {
			magma_strmm( MagmaRight, uplo, MagmaNoTrans, MagmaNonUnit,
	                 (m/nb)*2, n,
	                 c_one, dA,  ldda,
	                 dB_colchk, lddb_colchk, stream2 );
		}

		

		if (CHECK_AFTER) {
			if (COL_FT) {
				mem_row = m;
				mem_col = n;
				if (DEBUG) printf("strmm-after-check-B-col\n");
				abft_checker_colchk(dB, lddb, mem_row, mem_col, nb,
		                            dB_colchk,   lddb_colchk,
		                            dB_colchk_r, lddb_colchk_r,
		                            chk_v,       ld_chk_v,
		                            DEBUG,
		                            stream1);
			}
			if (ROW_FT) {
				mem_row = m;
				mem_col = n;
				if (DEBUG) printf("strmm-after-check-B-row\n");
				abft_checker_rowchk(dB, lddb, mem_row, mem_col, nb,
		                            dB_rowchk,   lddb_rowchk,
		                            dB_rowchk_r, lddb_rowchk_r,
		                            chk_v,       ld_chk_v,
		                            DEBUG,
		                            stream1);
			}
		}
	} else {
		printf("ABFT-strmm in this case is not implemented yet.\n");
	}


}