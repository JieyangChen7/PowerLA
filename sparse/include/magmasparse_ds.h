/*
-- MAGMA (version 2.5.4) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date October 2020

 @generated from sparse/include/magmasparse_zc.h, mixed zc -> ds, Thu Oct  8 23:05:56 2020
 @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_DS_H
#define MAGMASPARSE_DS_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_d


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Matrix Descriptors
*/


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Auxiliary functions
*/
magma_int_t
magma_vector_dlag2s(
    magma_d_matrix x,
    magma_s_matrix *y,
    magma_queue_t queue );

magma_int_t
magma_sparse_matrix_dlag2s(
    magma_d_matrix A,
    magma_s_matrix *B,
    magma_queue_t queue );


magma_int_t
magma_vector_slag2d(
    magma_s_matrix x,
    magma_d_matrix *y,
    magma_queue_t queue );

magma_int_t
magma_sparse_matrix_slag2d(
    magma_s_matrix A,
    magma_d_matrix *B,
    magma_queue_t queue );

void
magmablas_dlag2s_sparse(
    magma_int_t M, 
    magma_int_t N , 
    magmaDouble_const_ptr dA, 
    magma_int_t lda, 
    magmaFloat_ptr dSA, 
    magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info );

void 
magmablas_slag2d_sparse(
    magma_int_t M, 
    magma_int_t N , 
    magmaFloat_const_ptr dSA, 
    magma_int_t ldsa, 
    magmaDouble_ptr dA, 
    magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info );

void 
magma_dlag2s_CSR_DENSE(
    magma_d_matrix A,
    magma_s_matrix *B,
    magma_queue_t queue );

void 
magma_dlag2s_CSR_DENSE_alloc(
    magma_d_matrix A,
    magma_s_matrix *B,
    magma_queue_t queue );

void 
magma_dlag2s_CSR_DENSE_convert(
    magma_d_matrix A,
    magma_s_matrix *B,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on CPU
*/


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on CPU / Multi-GPU
*/

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on GPU
*/

magma_int_t
magma_dsir(
    magma_d_matrix A, 
    magma_d_matrix b, 
    magma_d_matrix *x,
    magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dsgecsrmv_mixed_prec(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr ddiagval,
    magmaFloat_ptr doffdiagval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );



/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/



/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE BLAS function definitions
*/



#ifdef __cplusplus
}
#endif

#undef PRECISION_d
#endif /* MAGMASPARSE_DS_H */
