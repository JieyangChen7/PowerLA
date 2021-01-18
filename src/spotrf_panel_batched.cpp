/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from src/zpotrf_panel_batched.cpp, normal z -> s, Thu Oct  8 23:05:31 2020
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"


/******************************************************************************/
extern "C" magma_int_t
magma_spotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j)    dA_array, i, j
    magma_int_t arginfo = 0;
    if (n < nb) {
        printf("magma_spotrf_panel error n < nb %lld < %lld\n", (long long) n, (long long) nb );
        return -101;
    }

    // panel
    arginfo = magma_spotf2_batched(
                       uplo, nb,
                       dAarray(ai, aj), ldda,
                       info_array, gbstep, 
                       batchCount, queue);

    // trsm
    if ((n-nb) > 0) {
            magmablas_strsm_recursive_batched( 
                    MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                    n-nb, nb, MAGMA_S_ONE, 
                    dAarray(ai   , aj), ldda, 
                    dAarray(ai+nb, aj), ldda, batchCount, queue );
    }
    return arginfo;
#undef dAarray
}


/******************************************************************************/
extern "C" magma_int_t
magma_spotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j)    dA_array, i, j

    magma_int_t arginfo = 0;
    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }
    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable\n");
        arginfo = -1;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }
    if (m < n) {
        printf("error m < n %lld < %lld\n", (long long) m, (long long) n );
        arginfo = -101;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    float alpha = MAGMA_S_NEG_ONE;
    float beta  = MAGMA_S_ONE;
    magma_int_t panel_nb = n;
    if (panel_nb <= min_recpnb) {
        arginfo = magma_spotrf_panel_batched(
                        uplo, m, panel_nb,
                        dAarray(ai,aj), ldda, 
                        info_array, gbstep, batchCount, queue);
    }
    else{
        // split A over two [A1 A2]
        // panel on A1, update on A2 then panel on A1    
        magma_int_t n1 = n/2;
        magma_int_t n2 = n-n1;

        // panel on A1
        arginfo = magma_spotrf_recpanel_batched(
                        uplo, m, n1, min_recpnb, 
                        dAarray(ai,aj), ldda, 
                        info_array, gbstep, 
                        batchCount, queue);

        if (arginfo != 0) {
            return arginfo;
        }

        // update A2
        magma_sgemm_batched_core( 
                        MagmaNoTrans, MagmaConjTrans, m-n1, n2, n1,
                        alpha, dAarray(ai+n1, aj   ), ldda, 
                               dAarray(ai+n1, aj   ), ldda, 
                        beta,  dAarray(ai+n1, aj+n1), ldda, 
                        batchCount, queue );

        // panel on A2
        arginfo = magma_spotrf_recpanel_batched(
                        uplo, m-n1, n2, min_recpnb, 
                        dAarray(ai+n1,aj+n1), ldda, 
                        info_array, gbstep+n1, 
                        batchCount, queue);
    }

    return arginfo;
#undef dAarray
}


/******************************************************************************/
extern "C" magma_int_t
magma_spotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j)    dA_array, i, j

    // Quick return if possible
    if (m == 0 || n == 0) {
        return 1;
    }
    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable\n");
        return -100;
    }
    if (m < n) {
        printf("error m < n %lld < %lld\n", (long long) m, (long long) n );
        return -101;
    }

    float alpha = MAGMA_S_NEG_ONE;
    float beta  = MAGMA_S_ONE;
    magma_int_t panel_nb = n;
    if (panel_nb <= min_recpnb) {
        //  panel factorization
        magma_spotrf_panel_batched(
                           uplo, m, panel_nb,
                           dAarray(ai, aj), ldda,
                           info_array, gbstep,
                           batchCount, queue);
    }
    else {
        // split A over two [A11 A12;  A21 A22; A31 A32]
        // panel on tile A11, 
        // trsm on A21, using A11
        // update on A22 then panel on A22.  
        // finally a trsm on [A31 A32] using the whole [A11 A12; A21 A22]     
        magma_int_t n1 = n/2;
        magma_int_t n2 = n-n1;

        // panel on A11
        magma_spotrf_rectile_batched(
                           uplo, n1, n1, min_recpnb,
                           dAarray(ai, ai), ldda,
                           info_array, gbstep,
                           batchCount, queue);

        // TRSM on A21
        magmablas_strsm_recursive_batched( 
                    MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                    n2, n1, MAGMA_S_ONE, 
                    dAarray(ai   , aj), ldda, 
                    dAarray(ai+n1, aj), ldda, batchCount, queue );

        // update A22
        magma_sgemm_batched_core( MagmaNoTrans, MagmaConjTrans, n2, n2, n1,
                             alpha, dAarray(ai+n1, aj   ), ldda, 
                                    dAarray(ai+n1, aj   ), ldda, 
                             beta,  dAarray(ai+n1, aj+n1), ldda, 
                             batchCount, queue );

        // panel on A22
        magma_spotrf_rectile_batched(
                           uplo, n2, n2, min_recpnb,
                           dAarray(ai+n1, aj+n1), ldda,
                           info_array, gbstep + n1,
                           batchCount, queue);
    }

    // TRSM on A3x
    if (m > n) {
        magmablas_strsm_recursive_batched( 
                    MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                    m-n, n, MAGMA_S_ONE, 
                    dAarray(ai  , aj), ldda, 
                    dAarray(ai+n, aj), ldda, batchCount, queue );
    }

    return 0;

#undef dAarray
}
