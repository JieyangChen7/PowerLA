#include "magma_internal.h"
#undef max
#undef min
#include "abft_checker.h"
#include "abft_io.h"
#include <string>
#include "cuda_runtime.h"
#include "../fault_tolerance/abft_printer.h"
//dgemm with FT

/**
 * m: number of row of A (N-i-B)
 * n: number of row of B (B)
 * k: number of col of A / col of B (i)
 */
void abft_dgemm( magma_trans_t transA, magma_trans_t transB,
			  int m, int n, int k, 
			  double alpha, 
			  double * dA, int ldda,
		      double * dB, int lddb, 
			  double beta, 
			  double * dC, int lddc,
			  int nb,
			  double * dA_colchk,	int ldda_colchk,
			  double * dA_rowchk,   int ldda_rowchk,
			  double * dA_colchk_r,	int ldda_colchk_r,
			  double * dA_rowchk_r, int ldda_rowchk_r,
			  double * dB_colchk,   int lddb_colchk,
			  double * dB_rowchk,   int lddb_rowchk,
			  double * dB_colchk_r, int lddb_colchk_r,
			  double * dB_rowchk_r, int lddb_rowchk_r,
			  double * dC_colchk,   int lddc_colchk,
			  double * dC_rowchk,   int lddc_rowchk,
			  double * dC_colchk_r, int lddc_colchk_r,
			  double * dC_rowchk_r, int lddc_rowchk_r,
			  double * chk_v, int ld_chk_v, 
			  bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, 
			  magma_queue_t stream1, magma_queue_t stream2) {

	

	int mem_row = 0; // number of row and col of B stored in memory(no trans operation)
	int mem_col = 0;

	if (COL_FT && CHECK_BEFORE) {

		// number of row and col of A stored in memory(no trans operation)
		if (transA == MagmaNoTrans) {
			mem_row = m;
			mem_col = k;
			if (DEBUG) printf("dgemm-before-check-A-col\n");
			abft_checker_colchk(dA, ldda, mem_row, mem_col, nb,
	                            dA_colchk,   ldda_colchk,
	                            dA_colchk_r, ldda_colchk_r,
	                            chk_v,       ld_chk_v,
	                            DEBUG,
	                            stream1);


		} else if (transA == MagmaTrans || transA == MagmaConjTrans) {
			mem_row = k;
			mem_col = m;
			if (DEBUG) printf("dgemm-before-check-A-row\n");
			abft_checker_rowchk(dA, ldda, mem_row, mem_col, nb,
	                            dA_rowchk,   ldda_rowchk,
	                            dA_rowchk_r, ldda_rowchk_r,
	                            chk_v,       ld_chk_v,
	                            DEBUG,
	                            stream1);
		}

		mem_row = m;
		mem_col = n;
		if (DEBUG) printf("dgemm-before-check-C-col\n");
		abft_checker_colchk(dC, lddc, mem_row, mem_col, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);

	}
	
	if (ROW_FT && CHECK_BEFORE)	{
		//verify B before use
		if (transB == MagmaNoTrans) {
			mem_row = k;
			mem_col = n;
			if (DEBUG) printf("dgemm-before-check-B-row\n");
			abft_checker_rowchk(dB, lddb, mem_row, mem_col, nb,
	                            dB_rowchk,   lddb_rowchk,
	                            dB_rowchk_r, lddb_rowchk_r,
	                            chk_v,       ld_chk_v,
	                            DEBUG,
	                            stream1);

		} else if (transB == MagmaTrans || transB == MagmaConjTrans) {
			mem_row = n;
			mem_col = k;
			if (DEBUG) printf("dgemm-before-check-B-col\n");
			abft_checker_colchk(dB, lddb, mem_row, mem_col, nb,
	                            dB_colchk,   lddb_colchk,
	                            dB_colchk_r, lddb_colchk_r,
	                            chk_v,       ld_chk_v,
	                            DEBUG,
	                            stream1);
		}
		
		mem_row = m;
		mem_col = n;
		if (DEBUG) printf("dgemm-before-check-C-row\n");
		abft_checker_rowchk(dC, lddc, mem_row, mem_col, nb,
                            dC_rowchk,   lddc_rowchk,
                            dC_rowchk_r, lddc_rowchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);
	}
				
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t, t1;
    bool DEBUG_GEMM = false;

    if (DEBUG_GEMM) cudaEventRecord(start, stream1->cuda_stream());
	magma_dgemm(transA, transB,
				m, n, k,
				alpha,
				dA, ldda, dB, lddb,
				beta,
				dC, lddc,
				stream1);
	if (DEBUG_GEMM) {
		cudaEventRecord(stop, stream1->cuda_stream());
		cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&t, start, stop);
	    printf("gemm: %f (%f)\n", t, (double)m*n*k*2/t/1e6);
	}

	//bool INJECT = true;

	//if (INJECT) {
	//	magma_dscal( 1, 10, dC, 1, stream1);
	//}  

    if (DEBUG_GEMM) cudaEventRecord(start, stream2->cuda_stream());
	if(COL_FT){	
		if (transA == MagmaNoTrans) {
			magma_dgemm(transA, transB,
					   (m / nb) * 2, n, k,
					   alpha,
					   dA_colchk, ldda_colchk, dB, lddb,
					   beta,
					   dC_colchk, lddc_colchk,
					   stream2);
		} else {
			magma_dgemm(transA, transB,
					   (m / nb) * 2, n, k,
					   alpha,
					   dA_rowchk, ldda_rowchk, dB, lddb,
					   beta,
					   dC_colchk, lddc_colchk,
					   stream2);
		}
	}

	if (DEBUG_GEMM) {
		cudaEventRecord(stop, stream2->cuda_stream());
		cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&t1, start, stop);
	    printf("gemm-col-ft: %f (%f)(%f)\n", t1, t1/t, (double)m/nb*2*n*k*2/t1/1e6);
	}

    if (DEBUG_GEMM) cudaEventRecord(start, stream2->cuda_stream());
	if (ROW_FT) {

		if (transB == MagmaNoTrans) {
			//we can further work on this to support trans A.
			magma_dgemm(transA, transB,
						m , (n / nb) * 2, k,
						alpha,
						dA, ldda,
						dB_rowchk, lddb_rowchk,
						beta,
						dC_rowchk, lddc_rowchk,
						stream2);
		} else {
			//we can further work on this to support trans A.
			// printf( "gemm-A:\n" );
   //      	printMatrix_gpu(dA, ldda, m, k, nb, nb, stream2);
   //      	printf( "gemm-B:\n" );
   //      	printMatrix_gpu(dB, lddb, k, n, nb, nb, stream2);
   //      	printf( "gemm-B-colchk:\n" );
   //      	printMatrix_gpu(dB_colchk, lddb_colchk, (k/nb)*2, n, 2, nb, stream2);
   //      	printf( "gemm-C:\n" );
   //      	printMatrix_gpu(dC, lddc, m, n, nb, nb, stream2);
   //      	printf( "gemm-C-rowchk:\n" );
   //      	printMatrix_gpu(dC_rowchk, lddc_rowchk, m, (n/nb)*2, nb, 2, stream2);


   //      	printf("size = %d %d %d\n", m , (k / nb) * 2, n);
			magma_dgemm(transA, transB,
						m , (n / nb) * 2, k,
						alpha,
						dA, ldda,
						dB_colchk, lddb_colchk,
						beta,
						dC_rowchk, lddc_rowchk,
						stream2);


			// printf( "gemm-C-rowchk:\n" );
   //      	printMatrix_gpu(dC_rowchk, lddc_rowchk, m, (n/nb)*2, nb, 2, stream2);


			// printf( "gemm-C-rowchk-updated:\n" );
   //      	printMatrix_gpu(dC_rowchk, lddc_rowchk, m, (n/nb)*2, nb, 2, stream2);
		}

		if (DEBUG_GEMM) {
			cudaEventRecord(stop, stream2->cuda_stream());
			cudaEventSynchronize(stop);
		    cudaEventElapsedTime(&t1, start, stop);
		    printf("gemm-row-ft: %f (%f)(%f)\n", t1, t1/t, (double)m*(n/nb)*2*k*2/t1/1e6);
		}
		
	}


	if (DEBUG_GEMM) cudaEventRecord(start, stream1->cuda_stream());
	if (COL_FT && CHECK_AFTER) {

		mem_row = m;
		mem_col = n;
		if (DEBUG) printf("dgemm-after-check-C-col\n");
		abft_checker_colchk(dC, lddc, mem_row, mem_col, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);
	}

	if (DEBUG_GEMM) {
		cudaEventRecord(stop, stream1->cuda_stream());
		cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&t1, start, stop);
	    printf("gemm-col-chk: %f (%f) (%f)\n", t1, t1/t, ((double)m/nb)*2*n*nb*2/t1/1e6);
	}

    if (DEBUG_GEMM) cudaEventRecord(start, stream1->cuda_stream());
	if (ROW_FT && CHECK_AFTER) {
		mem_row = m;
		mem_col = n;
		if (DEBUG) printf("dgemm-after-check-C-row\n");
		abft_checker_rowchk(dC, lddc, mem_row, mem_col, nb,
                            dC_rowchk,   lddc_rowchk,
                            dC_rowchk_r, lddc_rowchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);

	}

	if (DEBUG_GEMM) {
		cudaEventRecord(stop, stream1->cuda_stream());
		cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&t1, start, stop);
	    printf("gemm-row-chk: %f (%f)(%f)\n", t1, t1/t, ((double)n/nb)*m*2*nb*2/t1/1e6);
	}

}




size_t abft_dgemm_flops(magma_trans_t transA, magma_trans_t transB,
						int m, int n, int k, int nb,
						bool COL_FT, bool ROW_FT, bool CHECK_BEFORE, bool CHECK_AFTER) {

	size_t flops = 0;
	int mem_row = 0; // number of row and col of B stored in memory(no trans operation)
	int mem_col = 0;
	if (COL_FT && CHECK_BEFORE) {

		// number of row and col of A stored in memory(no trans operation)
		if (transA == MagmaNoTrans) {
			mem_row = m;
			mem_col = k;
			// if (DEBUG) printf("dgemm-before-check-A-col\n");
			// abft_checker_colchk(dA, ldda, mem_row, mem_col, nb,
	  //                           dA_colchk,   ldda_colchk,
	  //                           dA_colchk_r, ldda_colchk_r,
	  //                           chk_v,       ld_chk_v,
	  //                           DEBUG,
	  //                           stream1);
			flops += abft_checker_colchk_flops(mem_row, mem_col, nb);


		} else if (transA == MagmaTrans || transA == MagmaConjTrans) {
			mem_row = k;
			mem_col = m;
			// if (DEBUG) printf("dgemm-before-check-A-row\n");
			// abft_checker_rowchk(dA, ldda, mem_row, mem_col, nb,
	  //                           dA_rowchk,   ldda_rowchk,
	  //                           dA_rowchk_r, ldda_rowchk_r,
	  //                           chk_v,       ld_chk_v,
	  //                           DEBUG,
	  //                           stream1);
			flops += abft_checker_rowchk_flops(mem_row, mem_col, nb);
		}
		mem_row = m;
		mem_col = n;
		// if (DEBUG) printf("dgemm-before-check-C-col\n");
		// abft_checker_colchk(dC, lddc, mem_row, mem_col, nb,
  //                           dC_colchk,   lddc_colchk,
  //                           dC_colchk_r, lddc_colchk_r,
  //                           chk_v,       ld_chk_v,
  //                           DEBUG,
  //                           stream1);
		flops += abft_checker_colchk_flops(mem_row, mem_col, nb);

	}
	
	if (ROW_FT && CHECK_BEFORE) {
		//verify B before use
		if (transB == MagmaNoTrans) {
			mem_row = k;
			mem_col = n;
			// if (DEBUG) printf("dgemm-before-check-B-row\n");
			// abft_checker_rowchk(dB, lddb, mem_row, mem_col, nb,
	  //                           dB_rowchk,   lddb_rowchk,
	  //                           dB_rowchk_r, lddb_rowchk_r,
	  //                           chk_v,       ld_chk_v,
	  //                           DEBUG,
	  //                           stream1);
			flops += abft_checker_rowchk_flops(mem_row, mem_col, nb);

		} else if (transB == MagmaTrans || transB == MagmaConjTrans) {
			mem_row = n;
			mem_col = k;
			// if (DEBUG) printf("dgemm-before-check-B-col\n");
			// abft_checker_colchk(dB, lddb, mem_row, mem_col, nb,
	  //                           dB_colchk,   lddb_colchk,
	  //                           dB_colchk_r, lddb_colchk_r,
	  //                           chk_v,       ld_chk_v,
	  //                           DEBUG,
	  //                           stream1);
			flops += abft_checker_colchk_flops(mem_row, mem_col, nb);
		}
		
		
		mem_row = m;
		mem_col = n;
		// if (DEBUG) printf("dgemm-before-check-C-row\n");
		// abft_checker_rowchk(dC, lddc, mem_row, mem_col, nb,
  //                           dC_rowchk,   lddc_rowchk,
  //                           dC_rowchk_r, lddc_rowchk_r,
  //                           chk_v,       ld_chk_v,
  //                           DEBUG,
  //                           stream1);

		
		flops += abft_checker_rowchk_flops(mem_row, mem_col, nb);
	}
		
	// magma_dgemm(transA, transB,
	// 			m, n, k,
	// 			alpha,
	// 			dA, ldda, dB, lddb,
	// 			beta,
	// 			dC, lddc,
	// 			stream1);

	flops += (size_t)m*n*k*2;

	//bool INJECT = true;

	//if (INJECT) {
	//	magma_dscal( 1, 10, dC, 1, stream1);
	//}  

	if(COL_FT){	
		if (transA == MagmaNoTrans) {
			// magma_dgemm(transA, transB,
			// 		   (m / nb) * 2, n, k,
			// 		   alpha,
			// 		   dA_colchk, ldda_colchk, dB, lddb,
			// 		   beta,
			// 		   dC_colchk, lddc_colchk,
			// 		   stream2);

			flops += (size_t)(m/nb)*2*n*k*2;
		} else {
			// magma_dgemm(transA, transB,
			// 		   (m / nb) *2, n, k,
			// 		   alpha,
			// 		   dA_rowchk, ldda_rowchk, dB, lddb,
			// 		   beta,
			// 		   dC_colchk, lddc_colchk,
			// 		   stream2);
			flops += (size_t)(m/nb)*2*n*k*2;
		}
	}

	if (ROW_FT) {

		if (transB == MagmaNoTrans) {
			//we can further work on this to support trans A.
			// magma_dgemm(transA, transB,
			// 			m , (n / nb) * 2, k,
			// 			alpha,
			// 			dA, ldda,
			// 			dB_rowchk, lddb_rowchk,
			// 			beta,
			// 			dC_rowchk, lddc_rowchk,
			// 			stream2);
			flops += (size_t)m*(n/nb)*2*k*2;
		} else {
			//we can further work on this to support trans A.
			// magma_dgemm(transA, transB,
			// 			m , (n / nb) * 2, k,
			// 			alpha,
			// 			dA, ldda,
			// 			dB_colchk, lddb_colchk,
			// 			beta,
			// 			dC_rowchk, lddc_rowchk,
			// 			stream2);
			flops += (size_t)m*(n/nb)*2*k*2;
		}
		
	}


	if (COL_FT && CHECK_AFTER) {

		mem_row = m;
		mem_col = n;
		// if (DEBUG) printf("dgemm-after-check-C-col\n");
		// abft_checker_colchk(dC, lddc, mem_row, mem_col, nb,
  //                           dC_colchk,   lddc_colchk,
  //                           dC_colchk_r, lddc_colchk_r,
  //                           chk_v,       ld_chk_v,
  //                           DEBUG,
  //                           stream1);
		flops += abft_checker_colchk_flops(mem_row, mem_col, nb);

	}

	if (ROW_FT && CHECK_AFTER) {
		mem_row = m;
		mem_col = n;
		// if (DEBUG) printf("dgemm-after-check-C-row\n");
		// abft_checker_rowchk(dC, lddc, mem_row, mem_col, nb,
  //                           dC_rowchk,   lddc_rowchk,
  //                           dC_rowchk_r, lddc_rowchk_r,
  //                           chk_v,       ld_chk_v,
  //                           DEBUG,
  //                           stream1);
		flops += abft_checker_rowchk_flops(mem_row, mem_col, nb);

	}
}



