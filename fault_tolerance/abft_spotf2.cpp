#include<iostream>
#include "abft_printer.h"
#include "magma_lapack.h"
using namespace std;
//Cholesky Factorization with FT on CPU using ACML
float get(float * matrix, int ld, int n, int i, int j) {
	if (i > ld || j > n)
		cout << "matrix_get_error" << endl;
	return *(matrix + j * ld + i);
}
/**
 * Cholesky factorization with FT on CPU using ACML
 * A: input matrix
 * lda: leading dimension of A
 * n: size of A
 * chksum1: checksum 1
 * inc1: stride between elememts in chksum1
 * chksum2: checksum 2
 * inc2: stride between elememts in chksum2
 */
void abft_spotf2(const char uplo, int n, float * A, int lda, int * info, 
			     int nb, 
			     float * colchk, 	int ld_colchk, 
			     float * rowchk, 	int ld_rowchk, 
			     float * colchk_r, int ld_colchk_r, 
			     float * rowchk_r, int ld_rowchk_r, 
			     float * chk_v, int ld_chk_v, 
			     bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER) {
	
	float one = 1;
	float zero = 0;
	float negone = -1;
	
	if (FT && CHECK_BEFORE) {
		//magma_set_lapack_numthreads(16);
		//verify A before use
		char T = 'T';
		char N = 'N';
		// float * chk1 = new float[n];
		// float * chk2 = new float[n];
		// int chk1_inc = 1;
		// int chk2_inc = 1;
		// blasf77_dgemv(  &T,
		//                 &n, &n,
		//                 &one,
		//                 A, &lda,
		//                 abftEnv->hrz_v, &(abftEnv->hrz_v_ld),
		//                 &zero,
		//                 chk1, &chk1_inc );
		// blasf77_dgemv(  &T,
		// 				&n, &n,
		// 				&one,
		// 				A, &lda,
		// 				abftEnv->hrz_v + 1, &(abftEnv->hrz_v_ld),
		// 				&zero,
		// 				chk2, &chk2_inc ); 
		//handle error 
//		ErrorDetectAndCorrectHost(A, lda, n, n, n,
//								chksum, chksum_ld,
//								chk1, chk1_inc,
//								chk2, chk2_inc);
		//float * recal_colchk = new float[n * 2];
		int num_chk = 2;
		//int ld_recal_colchk = 2;

		blasf77_sgemm(&T, &N,
                      &num_chk, &n, &nb,
                      &one, 
                      chk_v, &ld_chk_v,
                      A, &lda,
                      &zero, 
                      colchk_r, &ld_colchk_r);  
		
		if (DEBUG) {
			cout<<"[DPOTRF-BEFORE]recalcuated checksum on CPU before factorization:"<<endl;
			// printMatrix_host(chk1, 1, 1, n, -1, -1);
			// printMatrix_host(chk2, 1, 1, n, -1, -1);
			printMatrix_host(colchk_r, ld_colchk_r, 2, n, -1, -1);
			cout<<"[DPOTRF-BEFORE]updated checksum on CPU before factorization:"<<endl;
			printMatrix_host(colchk, ld_colchk, 2, n, -1, -1);
		}
	}
	
	//do Choleksy factorization
	//magma_set_lapack_numthreads(1);
	//char uplo = 'L';
	lapackf77_spotrf(&uplo, &n, A, &n, info);


	if (FT) {
		//update checksum1 and checksum2

		//magma_set_lapack_numthreads(64);
		for (int i = 0; i < n; i++) {
			//chksum1[i] = chksum1[i] / get(A, n, n, i, i);
			*(colchk + i*ld_colchk) = *(colchk + i*ld_colchk) / get(A, n, n, i, i);
			//(n-i-1, negone*chksum1[i], A + i*lda + i+1, 1, chksum1 + i+1, 1 );
			int m = n-i-1;
			float alpha = negone * (*(colchk + i * ld_colchk));
			int incx = 1;
			blasf77_saxpy(&m, &alpha, A + i*lda + i+1, &incx, colchk + (i+1) * ld_colchk, &(ld_colchk) );
		}
	
		for (int i = 0; i < n; i++) {
			//chksum2[i] = chksum2[i] / get(A, n, n, i, i);
			*(colchk + i*ld_colchk + 1) = *(colchk + i*ld_colchk + 1) / get(A, n, n, i, i);
			//daxpy(n-i-1, negone*chksum2[i], A + i*lda + i+1, 1, chksum2 + i+1, 1 );
			int m = n-i-1;
			float alpha = negone *  (*(colchk + i * ld_colchk + 1));
			int incx = 1;
			blasf77_saxpy(&m, &alpha, A + i * lda + i+1, &incx, colchk + 1 + (i + 1) * ld_colchk, &(ld_colchk) );
		}	
	}

	if (FT && CHECK_AFTER) {
		//magma_set_lapack_numthreads(16);
		//verify A before use
		char T = 'T';
		char N = 'N';
		// float * chk1 = new float[n];
		// float * chk2 = new float[n];
		// int chk1_inc = 1;
		// int chk2_inc = 1;
		// blasf77_dgemv(  &T,
		//                 &n, &n,
		//                 &one,
		//                 A, &lda,
		//                 abftEnv->hrz_v, &(abftEnv->hrz_v_ld),
		//                 &zero,
		//                 chk1, &chk1_inc );
		// blasf77_dgemv(  &T,
		// 				&n, &n,
		// 				&one,
		// 				A, &lda,
		// 				abftEnv->hrz_v + 1, &(abftEnv->hrz_v_ld),
		// 				&zero,
		// 				chk2, &chk2_inc ); 
		//handle error 
//		ErrorDetectAndCorrectHost(A, lda, n, n, n,
//								chksum, chksum_ld,
//								chk1, chk1_inc,
//								chk2, chk2_inc);
		//float * recal_colchk = new float[n * 2];
		int num_chk = 2;
		//int ld_recal_colchk = 2;

		blasf77_sgemm(&T, &N,
                      &num_chk, &n, &nb,
                      &one, 
                      chk_v, &ld_chk_v,
                      A, &lda,
                      &zero, 
                      colchk_r, &ld_colchk_r);  
		
		if (DEBUG) {
			cout<<"[DPOTRF-AFTER]recalcuated checksum on CPU before factorization:"<<endl;
			// printMatrix_host(chk1, 1, 1, n, -1, -1);
			// printMatrix_host(chk2, 1, 1, n, -1, -1);
			printMatrix_host(colchk_r, ld_colchk_r, 2, n, -1, -1);
			cout<<"[DPOTRF-AFTER]updated checksum on CPU before factorization:"<<endl;
			printMatrix_host(colchk, ld_colchk, 2, n, -1, -1);
		}
    }
}