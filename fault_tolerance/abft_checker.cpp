#include <cstdio>
#include "magma_internal.h"
#undef max
#undef min
#include "abft_encoder.h"
#include "abft_printer.h"
#include "abft_corrector.h"
void abft_checker_colchk(double * dA, int ldda, int m, int n, int nb,
						 double * dA_colchk,    int ldda_colchk,
    					 double * dA_colchk_r,  int ldda_colchk_r,
    					 double * dev_chk_v,    int ld_dev_chk_v,
    					 bool DEBUG,
    					 magma_queue_t stream){
	if (DEBUG) printf("abft_checker_colchk\n");
	col_chk_enc(m, n, nb, 
                dA, ldda,  
                dev_chk_v, ld_dev_chk_v, 
                dA_colchk_r, ldda_colchk_r, 
                stream);
	

	// if (DEBUG) {
	// 		printf( "input matrix:\n" );
 //            printMatrix_gpu(dA, ldda, m, n, nb, nb, stream);
 //            printf( "updated column chk:\n" );
 //            printMatrix_gpu(dA_colchk, ldda_colchk, (m / nb) * 2, n, 2, nb, stream);
 //            printf( "recalculated column chk:\n" );
 //            printMatrix_gpu(dA_colchk_r, ldda_colchk_r, (m / nb) * 2, n, 2, nb, stream);
 //    }
    // colchk_detect_correct(dA, ldda, m, n, nb,
				//           dA_colchk,	ldda_colchk,
				//           dA_colchk_r, 	ldda_colchk_r,
				// 		  stream);
}



void abft_checker_rowchk(double * dA, int ldda, int m, int n, int nb,
						 double * dA_rowchk,    int ldda_rowchk,
    					 double * dA_rowchk_r,  int ldda_rowchk_r,
    					 double * dev_chk_v,    int ld_dev_chk_v,
    					 bool DEBUG,
    					 magma_queue_t stream){
	if (DEBUG) printf("abft_checker_rowchk\n");
	row_chk_enc(m, n, nb, 
                dA, ldda,  
                dev_chk_v, ld_dev_chk_v, 
                dA_rowchk_r, ldda_rowchk_r, 
                stream);

	
	// if (DEBUG) {
	// 		printf( "input matrix:\n" );
 //            printMatrix_gpu(dA, ldda, m, n, nb, nb, stream);
 //            printf( "updated row chk:\n" );
 //            printMatrix_gpu(dA_rowchk, ldda_rowchk, m, (n / nb) * 2, nb, 2, stream);
 //            printf( "recalculated row chk:\n" );
 //            printMatrix_gpu(dA_rowchk_r, ldda_rowchk_r, m, (n / nb) * 2, nb, 2, stream);
 //    }
    rowchk_detect_correct(dA, ldda, m, n, nb,
				          dA_rowchk,	ldda_rowchk,
				          dA_rowchk_r, 	ldda_rowchk_r,
						  stream);
	
}




void abft_checker_fullchk(double * dA, int ldda, int m, int n, int nb,
						  double * dA_colchk,    int ldda_colchk,
    					  double * dA_colchk_r,  int ldda_colchk_r,
    					  double * dA_rowchk,    int ldda_rowchk,
    					  double * dA_rowchk_r,  int ldda_rowchk_r,
    					  double * dev_chk_v,    int ld_dev_chk_v,
    					  bool DEBUG,
    					  magma_queue_t stream){

	abft_checker_colchk(dA, ldda, m, n, nb,
						dA_colchk,		ldda_colchk,
    					dA_colchk_r, 	ldda_colchk_r,
    					dev_chk_v, 		ld_dev_chk_v,
    					DEBUG,
    					stream);

	abft_checker_rowchk(dA, ldda, m, n, nb,
						dA_rowchk,		ldda_rowchk,
    					dA_rowchk_r, 	ldda_rowchk_r,
    					dev_chk_v, 		ld_dev_chk_v,
    					DEBUG,
    					stream);
	
}



void abft_checker_colchk(float * dA, int ldda, int m, int n, int nb,
                         float * dA_colchk,    int ldda_colchk,
                         float * dA_colchk_r,  int ldda_colchk_r,
                         float * dev_chk_v,    int ld_dev_chk_v,
                         bool DEBUG,
                         magma_queue_t stream){
    if (DEBUG) printf("abft_checker_colchk\n");
    col_chk_enc(m, n, nb, 
                dA, ldda,  
                dev_chk_v, ld_dev_chk_v, 
                dA_colchk_r, ldda_colchk_r, 
                stream);
    

    // if (DEBUG) {
    //         printf( "input matrix:\n" );
    //         printMatrix_gpu(dA, ldda, m, n, nb, nb, stream);
    //         printf( "updated column chk:\n" );
    //         printMatrix_gpu(dA_colchk, ldda_colchk, (m / nb) * 2, n, 2, nb, stream);
    //         printf( "recalculated column chk:\n" );
    //         printMatrix_gpu(dA_colchk_r, ldda_colchk_r, (m / nb) * 2, n, 2, nb, stream);
    // }
    colchk_detect_correct(dA, ldda, m, n, nb,
                          dA_colchk,    ldda_colchk,
                          dA_colchk_r,  ldda_colchk_r,
                          stream);
}

void abft_checker_rowchk(float * dA, int ldda, int m, int n, int nb,
                         float * dA_rowchk,    int ldda_rowchk,
                         float * dA_rowchk_r,  int ldda_rowchk_r,
                         float * dev_chk_v,    int ld_dev_chk_v,
                         bool DEBUG,
                         magma_queue_t stream){
    if (DEBUG) printf("abft_checker_rowchk\n");
    row_chk_enc(m, n, nb, 
                dA, ldda,  
                dev_chk_v, ld_dev_chk_v, 
                dA_rowchk_r, ldda_rowchk_r, 
                stream);

    
    // if (DEBUG) {
    //         printf( "input matrix:\n" );
    //         printMatrix_gpu(dA, ldda, m, n, nb, nb, stream);
    //         printf( "updated row chk:\n" );
    //         printMatrix_gpu(dA_rowchk, ldda_rowchk, m, (n / nb) * 2, nb, 2, stream);
    //         printf( "recalculated row chk:\n" );
    //         printMatrix_gpu(dA_rowchk_r, ldda_rowchk_r, m, (n / nb) * 2, nb, 2, stream);
    // }
    rowchk_detect_correct(dA, ldda, m, n, nb,
                          dA_rowchk,    ldda_rowchk,
                          dA_rowchk_r,  ldda_rowchk_r,
                          stream);
    
}

void abft_checker_fullchk(float * dA, int ldda, int m, int n, int nb,
                          float * dA_colchk,    int ldda_colchk,
                          float * dA_colchk_r,  int ldda_colchk_r,
                          float * dA_rowchk,    int ldda_rowchk,
                          float * dA_rowchk_r,  int ldda_rowchk_r,
                          float * dev_chk_v,    int ld_dev_chk_v,
                          bool DEBUG,
                          magma_queue_t stream){

    abft_checker_colchk(dA, ldda, m, n, nb,
                        dA_colchk,      ldda_colchk,
                        dA_colchk_r,    ldda_colchk_r,
                        dev_chk_v,      ld_dev_chk_v,
                        DEBUG,
                        stream);

    abft_checker_rowchk(dA, ldda, m, n, nb,
                        dA_rowchk,      ldda_rowchk,
                        dA_rowchk_r,    ldda_rowchk_r,
                        dev_chk_v,      ld_dev_chk_v,
                        DEBUG,
                        stream);
    
}

size_t abft_checker_colchk_flops(int m, int n, int nb) {
    return col_chk_enc_flops(m, n, nb);//+colchk_detect_correct_flops(m, n, nb);
}

size_t abft_checker_rowchk_flops(int m, int n, int nb) {
    return row_chk_enc_flops(m, n, nb);//+rowchk_detect_correct_flops(m, n, nb);
}