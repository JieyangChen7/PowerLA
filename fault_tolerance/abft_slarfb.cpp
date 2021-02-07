/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Stan Tomov
       @author Mark Gates
       
       @generated from src/zlarfb_gpu.cpp, normal z -> d, Thu Oct  8 23:05:25 2020
*/
#include "../power_adjustment/power_adjustment.h"
#include "magma_internal.h"
#include "cuda_runtime.h"
#include "../fault_tolerance/abft_printer.h"
#include "../fault_tolerance/abft_encoder.h"
#include "../fault_tolerance/abft_kernels.h"

/***************************************************************************//**
    Purpose
    -------
    slarfb applies a real block reflector H or its transpose H^H to a
    float PRECISION m by n matrix C, from the left.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply H or H^H from the Left
      -     = MagmaRight:     apply H or H^H from the Right

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    apply H   (No transpose)
      -     = MagmaTrans: apply H^H (Conjugate transpose)

    @param[in]
    direct  magma_direct_t
            Indicates how H is formed from a product of elementary
            reflectors
      -     = MagmaForward:  H = H(1) H(2) . . . H(k) (Forward)
      -     = MagmaBackward: H = H(k) . . . H(2) H(1) (Backward)

    @param[in]
    storev  magma_storev_t
            Indicates how the vectors which define the elementary
            reflectors are stored:
      -     = MagmaColumnwise: Columnwise
      -     = MagmaRowwise:    Rowwise

    @param[in]
    m       INTEGER
            The number of rows of the matrix C.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C.

    @param[in]
    k       INTEGER
            The order of the matrix T (= the number of elementary
            reflectors whose product defines the block reflector).

    @param[in]
    dV      float PRECISION array on the GPU, dimension
                (LDDV,K) if STOREV = MagmaColumnwise
                (LDDV,M) if STOREV = MagmaRowwise and SIDE = MagmaLeft
                (LDDV,N) if STOREV = MagmaRowwise and SIDE = MagmaRight
            The matrix V. See further details.

    @param[in]
    lddv    INTEGER
            The leading dimension of the array V.
            If STOREV = MagmaColumnwise and SIDE = MagmaLeft,  LDDV >= max(1,M);
            if STOREV = MagmaColumnwise and SIDE = MagmaRight, LDDV >= max(1,N);
            if STOREV = MagmaRowwise, LDDV >= K.

    @param[in]
    dT      float PRECISION array on the GPU, dimension (LDDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    @param[in]
    lddt    INTEGER
            The leading dimension of the array T. LDDT >= K.

    @param[in,out]
    dC      float PRECISION array on the GPU, dimension (LDDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C, or H^H*C, or C*H, or C*H^H.

    @param[in]
    lddc    INTEGER
            The leading dimension of the array C. LDDC >= max(1,M).

    @param
    dwork   (workspace) float PRECISION array, dimension (LDWORK,K)

    @param[in]
    ldwork  INTEGER
            The leading dimension of the array WORK.
            If SIDE = MagmaLeft,  LDWORK >= max(1,N);
            if SIDE = MagmaRight, LDWORK >= max(1,M);

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------
    The shape of the matrix V and the storage of the vectors which define
    the H(i) is best illustrated by the following example with n = 5 and
    k = 3.
    All elements including 0's and 1's are stored, unlike LAPACK.

        DIRECT = MagmaForward and         DIRECT = MagmaForward and
        STOREV = MagmaColumnwise:         STOREV = MagmaRowwise:

                 V = (  1  0  0 )                 V = (  1 v1 v1 v1 v1 )
                     ( v1  1  0 )                     (  0  1 v2 v2 v2 )
                     ( v1 v2  1 )                     (  0  0  1 v3 v3 )
                     ( v1 v2 v3 )
                     ( v1 v2 v3 )

        DIRECT = MagmaBackward and        DIRECT = MagmaBackward and
        STOREV = MagmaColumnwise:         STOREV = MagmaRowwise:

                 V = ( v1 v2 v3 )                 V = ( v1 v1  1  0  0 )
                     ( v1 v2 v3 )                     ( v2 v2 v2  1  0 )
                     (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
                     (  0  1 v3 )
                     (  0  0  1 )

    @ingroup magma_larfb
*******************************************************************************/
extern "C" magma_int_t
abft_slarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float * dV,    int lddv,
    float * dT,    int lddt,
    float * dC,    int lddc,
    float * dwork, int ldwork,
    int nb,
    float * dV_colchk, int lddv_colchk,
    float * dV_rowchk,   int lddv_rowchk,
    float * dV_colchk_r, int lddv_colchk_r,
    float * dV_rowchk_r, int lddv_rowchk_r,
    float * dT_colchk, int lddt_colchk,
    float * dT_rowchk,   int lddt_rowchk,
    float * dT_colchk_r, int lddt_colchk_r,
    float * dT_rowchk_r, int lddt_rowchk_r,
    float * dC_colchk, int lddc_colchk,
    float * dC_rowchk,   int lddc_rowchk,
    float * dC_colchk_r, int lddc_colchk_r,
    float * dC_rowchk_r, int lddc_rowchk_r,
    float * dwork_colchk, int lddwork_colchk,
    float * dwork_rowchk,   int lddwork_rowchk,
    float * dwork_colchk_r, int lddwork_colchk_r,
    float * dwork_rowchk_r, int lddwork_rowchk_r,
    float * chk_v, int ld_chk_v, 
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, 
    magma_queue_t stream1, magma_queue_t stream2)
{
    #define dV(i_,j_)  (dV    + (i_) + (j_)*lddv)
    #define dT(i_,j_)  (dT    + (i_) + (j_)*lddt)
    #define dC(i_,j_)  (dC    + (i_) + (j_)*lddc)
    #define dwork(i_)  (dwork + (i_))
    
    /* Constants */
    const float c_zero    = MAGMA_D_ZERO;
    const float c_one     = MAGMA_D_ONE;
    const float c_neg_one = MAGMA_D_NEG_ONE;
    
    /* Check input arguments */
    magma_int_t info = 0;
    if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (k < 0) {
        info = -7;
    } else if ( ((storev == MagmaColumnwise) && (side == MagmaLeft) && lddv < max(1,m)) ||
                ((storev == MagmaColumnwise) && (side == MagmaRight) && lddv < max(1,n)) ||
                ((storev == MagmaRowwise) && lddv < k) ) {
        info = -9;
    } else if (lddt < k) {
        info = -11;
    } else if (lddc < max(1,m)) {
        info = -13;
    } else if ( ((side == MagmaLeft) && ldwork < max(1,n)) ||
                ((side == MagmaRight) && ldwork < max(1,m)) ) {
        info = -15;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Function Body */
    if (m <= 0 || n <= 0) {
        return info;
    }
    
    /* Local variables */
    // opposite of trans
    magma_trans_t transt;
    if (trans == MagmaNoTrans)
        transt = MagmaTrans;
    else
        transt = MagmaNoTrans;
    
    // whether T is upper or lower triangular
    magma_uplo_t uplo;
    if (direct == MagmaForward)
        uplo = MagmaUpper;
    else
        uplo = MagmaLower;
    
    // whether V is stored transposed or not
    magma_trans_t notransV, transV;
    if (storev == MagmaColumnwise) {
        notransV = MagmaNoTrans;
        transV   = MagmaTrans;
    }
    else {
        notransV = MagmaTrans;
        transV   = MagmaNoTrans;
    }

    if ( side == MagmaLeft ) {
        // Form H C or H^H C
        // Comments assume H C.
        // When forming H^H C, T gets transposed via transt.
        
        // W = C^H V
        abft_sgemm( MagmaTrans, notransV,
                     n, k, m,
                     c_one,  dC(0,0),  lddc,
                             dV(0,0),  lddv,
                     c_zero, dwork(0), ldwork,
                     nb,
                     dC_colchk, lddc_colchk,
                     dC_rowchk,   lddc_rowchk,
                     dC_colchk_r, lddc_colchk_r,
                     dC_rowchk_r, lddc_rowchk_r,
                     dV_colchk, lddv_colchk,
                     dV_rowchk,   lddv_rowchk,
                     dV_colchk_r, lddv_colchk_r,
                     dV_rowchk_r, lddv_rowchk_r,
                     dwork_colchk, lddwork_colchk,
                     dwork_rowchk,   lddwork_rowchk,
                     dwork_colchk_r, lddwork_colchk_r,
                     dwork_rowchk_r, lddwork_rowchk_r,
                     chk_v, ld_chk_v,
                     COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                     stream1, stream2);

        // W = W T^H = C^H V T^H
        // magma_strmm( MagmaRight, uplo, transt, MagmaNonUnit,
        //              n, k,
        //              c_one, dT(0,0),  lddt,
        //                     dwork(0), ldwork, stream1 );

        abft_strmm( MagmaRight, uplo, transt, MagmaNonUnit,
                     n, k,
                     c_one, dT(0,0),  lddt,
                            dwork(0), ldwork, 
                     nb,
                     dT_colchk, lddt_colchk,
                     dT_rowchk,   lddt_rowchk,
                     dT_colchk_r, lddt_colchk_r,
                     dT_rowchk_r, lddt_rowchk_r,
                     dwork_colchk, lddwork_colchk,
                     dwork_rowchk,   lddwork_rowchk,
                     dwork_colchk_r, lddwork_colchk_r,
                     dwork_rowchk_r, lddwork_rowchk_r,
                     chk_v, ld_chk_v,
                     COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                     stream1, stream2);

        // C = C - V W^H = C - V T V^H C = (I - V T V^H) C = H C

        // if (DEBUG) {

        //                 printf( "input matrix A:\n" );
        //                 printMatrix_gpu(dC, lddc, m, n, nb, nb, stream2);
        //                 printf( "column chk:\n" );
        //                 printMatrix_gpu(dC_colchk, lddc_colchk, 
        //                                 (m / nb) * 2, n, 2, nb, stream2);
        //                 printf( "row chk:\n" );
        //                 printMatrix_gpu(dC_rowchk, lddc_rowchk,  
        //                                 m, (n / nb) * 2, nb, 2, stream2);
        //             }


        abft_sgemm( notransV, MagmaTrans,
                     m, n, k,
                     c_neg_one, dV(0,0),  lddv,
                                dwork(0), ldwork,
                     c_one,     dC(0,0),  lddc,
                     nb,
                     dV_colchk, lddv_colchk,
                     dV_rowchk,   lddv_rowchk,
                     dV_colchk_r, lddv_colchk_r,
                     dV_rowchk_r, lddv_rowchk_r,
                     dwork_colchk,   lddwork_colchk,
                     dwork_rowchk,   lddwork_rowchk,
                     dwork_colchk_r, lddwork_colchk_r,
                     dwork_rowchk_r, lddwork_rowchk_r,
                     dC_colchk,   lddc_colchk,
                     dC_rowchk,   lddc_rowchk,
                     dC_colchk_r, lddc_colchk_r,
                     dC_rowchk_r, lddc_rowchk_r,
                     chk_v, ld_chk_v,
                     COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                     stream1, stream2);
    }
    else {
        // Form C H or C H^H
        // Comments assume C H.
        // When forming C H^H, T gets transposed via trans.
        
        // W = C V
        abft_sgemm( MagmaNoTrans, notransV,
                     m, k, n,
                     c_one,  dC(0,0),  lddc,
                             dV(0,0),  lddv,
                     c_zero, dwork(0), ldwork, 
                     nb,
                     dC_colchk, lddc_colchk,
                     dC_rowchk,   lddc_rowchk,
                     dC_colchk_r, lddc_colchk_r,
                     dC_rowchk_r, lddc_rowchk_r,
                     dV_colchk, lddv_colchk,
                     dV_rowchk,   lddv_rowchk,
                     dV_colchk_r, lddv_colchk_r,
                     dV_rowchk_r, lddv_rowchk_r,
                     dwork_colchk, lddwork_colchk,
                     dwork_rowchk,   lddwork_rowchk,
                     dwork_colchk_r, lddwork_colchk_r,
                     dwork_rowchk_r, lddwork_rowchk_r,
                     chk_v, ld_chk_v,
                     COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                     stream1, stream2);

        // W = W T = C V T
        // magma_strmm( MagmaRight, uplo, trans, MagmaNonUnit,
        //              m, k,
        //              c_one, dT(0,0),  lddt,
        //                     dwork(0), ldwork, stream1 );

        abft_strmm( MagmaRight, uplo, transt, MagmaNonUnit,
                     n, k,
                     c_one, dT(0,0),  lddt,
                            dwork(0), ldwork, 
                     nb,
                     dT_colchk, lddt_colchk,
                     dT_rowchk,   lddt_rowchk,
                     dT_colchk_r, lddt_colchk_r,
                     dT_rowchk_r, lddt_rowchk_r,
                     dwork_colchk, lddwork_colchk,
                     dwork_rowchk,   lddwork_rowchk,
                     dwork_colchk_r, lddwork_colchk_r,
                     dwork_rowchk_r, lddwork_rowchk_r,
                     chk_v, ld_chk_v,
                     COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                     stream1, stream2);

        // C = C - W V^H = C - C V T V^H = C (I - V T V^H) = C H
        abft_sgemm( MagmaNoTrans, transV,
                     m, n, k,
                     c_neg_one, dwork(0), ldwork,
                                dV(0,0),  lddv,
                     c_one,     dC(0,0),  lddc, 
                     nb,
                     dwork_colchk, lddwork_colchk,
                     dwork_rowchk,   lddwork_rowchk,
                     dwork_colchk_r, lddwork_colchk_r,
                     dwork_rowchk_r, lddwork_rowchk_r,
                     dV_colchk, lddv_colchk,
                     dV_rowchk,   lddv_rowchk,
                     dV_colchk_r, lddv_colchk_r,
                     dV_rowchk_r, lddv_rowchk_r,
                     dC_colchk, lddc_colchk,
                     dC_rowchk,   lddc_rowchk,
                     dC_colchk_r, lddc_colchk_r,
                     dC_rowchk_r, lddc_rowchk_r,
                     chk_v, ld_chk_v,
                     COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                     stream1, stream2);
    }

    return info;
} /* magma_slarfb */


size_t abft_slarfb_flops(magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                            magma_int_t m, magma_int_t n, magma_int_t k, int nb,
                             bool COL_FT, bool ROW_FT,bool CHECK_BEFORE, bool CHECK_AFTER) {
    /* Local variables */
    // opposite of trans
    magma_trans_t transt;
    if (trans == MagmaNoTrans)
        transt = MagmaTrans;
    else
        transt = MagmaNoTrans;
    
    // whether T is upper or lower triangular
    magma_uplo_t uplo;
    if (direct == MagmaForward)
        uplo = MagmaUpper;
    else
        uplo = MagmaLower;
    
    // whether V is stored transposed or not
    magma_trans_t notransV, transV;
    if (storev == MagmaColumnwise) {
        notransV = MagmaNoTrans;
        transV   = MagmaTrans;
    }
    else {
        notransV = MagmaTrans;
        transV   = MagmaNoTrans;
    }

    if ( side == MagmaLeft ) {
        // printf("gemm_flops (%d %d %d)(%f %f %f)\n", m, n, k, (float)abft_sgemm_flops( MagmaNoTrans, notransV, m, k, n, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/1e9 ,
        //         (float)abft_sgemm_flops( MagmaNoTrans, MagmaNoTrans, n, k, k, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/1e9 ,
        //         (float)abft_sgemm_flops( notransV, MagmaTrans,m, n, k,nb,  COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/1e9 );
        return  abft_sgemm_flops( MagmaNoTrans, notransV, m, k, n, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER) +
                abft_sgemm_flops( MagmaNoTrans, MagmaNoTrans, n, k, k, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER) +
                abft_sgemm_flops( notransV, MagmaTrans,m, n, k,nb,  COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER);

        
    } else {
        // printf("gemm_flops (%d %d %d)(%f %f %f)\n", m, n, k, abft_sgemm_flops( MagmaNoTrans, notransV,  m, k, n, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/1e9 ,
        //         (float)abft_sgemm_flops( MagmaNoTrans, MagmaNoTrans, n, k, k, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/1e9 ,
        //         (float)abft_sgemm_flops( MagmaNoTrans, transV, m, n, k, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/1e9 );
        return  abft_sgemm_flops( MagmaNoTrans, notransV,  m, k, n, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER) +
                abft_sgemm_flops( MagmaNoTrans, MagmaNoTrans, n, k, k, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER) +
                abft_sgemm_flops( MagmaNoTrans, transV, m, n, k, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER);

        
    }

}