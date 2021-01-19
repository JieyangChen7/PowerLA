#include "magma_internal.h"
#include "../fault_tolerance/abft_printer.h"
void col_chk_enc(int m, int n, int nb, 
                 double * A, int lda,
                 double * chk_v, int ld_chk_v,
                 double * dcolchk, int ld_dcolchk, 
                 magma_queue_t stream) {

    for (int i = 0; i < m; i += nb) {        
        magma_dgemm(MagmaTrans, MagmaNoTrans,
                    2, n, nb,
                    MAGMA_D_ONE, 
                    chk_v, ld_chk_v,
                    A + i, lda,
                    MAGMA_D_ZERO, dcolchk + (i / nb) * 2, ld_dcolchk,
                    stream);           
    }
}

void row_chk_enc(int m, int n, int nb, 
                 double * A, int lda,
                 double * chk_v, int ld_chk_v,
                 double * drowchk, int ld_drowchk, 
                 magma_queue_t stream) {

    for (int i = 0; i < n; i += nb) {     
    
        magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                    m, 2, nb,
                    MAGMA_D_ONE, 
                    A + i * lda, lda,
                    chk_v, ld_chk_v,
                    MAGMA_D_ZERO, drowchk + ((i / nb) * 2) * ld_drowchk, ld_drowchk,
                    stream);           
    }
}


void col_chk_enc_triblock(magma_uplo_t uplo, int nb,
                         double * A, int lda,
                         double * chk_vt, int ld_chk_vt,
                         double * dcolchk, int ld_dcolchk, 
                         magma_queue_t stream) {

    // magma_dcopymatrix(2, nb, chk_vt, ld_chk_vt, dcolchk, ld_dcolchk, stream );
    // magma_dtrmm( MagmaRight, uplo, MagmaNoTrans, MagmaNonUnit,
    //              2, nb,
    //              MAGMA_D_ONE, dcolchk,  ld_dcolchk,
    //              A, lda, stream );

    // printf( "col chk:\n" );
    //     printMatrix_gpu(dcolchk, ld_dcolchk,  
    //                     (nb/nb), (ib / nb) * 2, nb, 2, queues[1]);


    double one = MAGMA_D_ONE;
    cublasStatus_t status = cublasDtrmm(
                stream->cublas_handle(),
                cublas_side_const( MagmaRight ),
                cublas_uplo_const( uplo ),
                cublas_trans_const( MagmaNoTrans ),
                cublas_diag_const( MagmaNonUnit ),
                (nb/nb)*2, nb,
                &one, A, int(lda),
                chk_vt, int(ld_chk_vt),
                dcolchk, int(ld_dcolchk) ); 

    if (CUBLAS_STATUS_SUCCESS != status) {
        printf("cublasDtrmm-col error\n");
    }


}

void row_chk_enc_triblock(magma_uplo_t uplo, int nb,
                         double * A, int lda,
                         double * chk_v, int ld_chk_v,
                         double * drowchk, int ld_drowchk, 
                         magma_queue_t stream) {

    // magma_dcopymatrix(nb, 2, chk_v, ld_chk_v, drowchk, ld_drowchk, stream );
    // magma_dtrmm( MagmaLeft, uplo, MagmaNoTrans, MagmaNonUnit,
    //              nb, 2,
    //              MAGMA_D_ONE, drowchk, ld_drowchk,
    //              A, lda, stream );



    double one = MAGMA_D_ONE;
    cublasStatus_t status = cublasDtrmm(
                stream->cublas_handle(),
                cublas_side_const( MagmaLeft ),
                cublas_uplo_const( uplo ),
                cublas_trans_const( MagmaNoTrans ),
                cublas_diag_const( MagmaNonUnit ),
                nb, (nb/nb)*2,
                &one, A, int(lda),
                chk_v, int(ld_chk_v),
                drowchk, int(ld_drowchk) ); 

     if (CUBLAS_STATUS_SUCCESS != status) {
        printf("cublasDtrmm-row error\n");
    }


}



void col_chk_enc(int m, int n, int nb, 
                 float * A, int lda,
                 float * chk_v, int ld_chk_v,
                 float * dcolchk, int ld_dcolchk, 
                 magma_queue_t stream) {

    for (int i = 0; i < m; i += nb) {        
        magma_sgemm(MagmaTrans, MagmaNoTrans,
                    2, n, nb,
                    MAGMA_D_ONE, 
                    chk_v, ld_chk_v,
                    A + i, lda,
                    MAGMA_D_ZERO, dcolchk + (i / nb) * 2, ld_dcolchk,
                    stream);           
    }
}

void row_chk_enc(int m, int n, int nb, 
                 float * A, int lda,
                 float * chk_v, int ld_chk_v,
                 float * drowchk, int ld_drowchk, 
                 magma_queue_t stream) {

    for (int i = 0; i < n; i += nb) {        
        magma_sgemm(MagmaNoTrans, MagmaNoTrans,
                    m, 2, nb,
                    MAGMA_D_ONE, 
                    A + i * lda, lda,
                    chk_v, ld_chk_v,
                    MAGMA_D_ZERO, drowchk + ((i / nb) * 2) * ld_drowchk, ld_drowchk,
                    stream);           
    }
}


long long col_chk_enc_flops(int m, int n, int nb) {
    return ((size_t)m/nb)*2*n*nb*2;
}
long long row_chk_enc_flops(int m, int n, int nb) {
    return ((size_t)n/nb)*m*2*nb*2;
} 