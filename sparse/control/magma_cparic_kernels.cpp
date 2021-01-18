/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/control/magma_zparic_kernels.cpp, normal z -> c, Thu Oct  8 23:05:50 2020
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define SWAP(a, b)  { val_swap = a; a = b; b = val_swap; }


/***************************************************************************//**
    Purpose
    -------
    This function does one asynchronous ParILU sweep (symmetric case). 
    Input and output array is identical.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                System matrix in COO.

    @param[in]
    L           magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/


extern "C" magma_int_t
magma_cparic_sweep(
    magma_c_matrix A,
    magma_c_matrix *L,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    int i, j;


    magmaFloatComplex zero = MAGMA_C_MAKE(0.0, 0.0);
    int il, iu, jl, ju;

    #pragma omp parallel for
    for (int k=0; k < A.nnz; k++) {
        i = A.rowidx[k];
        j = A.col[k];

        magmaFloatComplex s, sp;
        s =  A.val[k];
        sp = zero;

        il = L->row[i];
        iu = L->row[j];

        while (il < L->row[i+1] && iu < L->row[j+1])
        {
            sp = zero;
            jl = L->col[il];
            ju = L->col[iu];

            // avoid branching
            sp = ( jl == ju ) ? L->val[il] * L->val[iu] : sp;
            s = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
        }
        // undo the last operation (it must be the last)
        s += sp;
        
        if ( i > j )      // modify l entry
            L->val[il-1] =  s / L->val[L->row[j+1]-1];
        else {            // modify u entry
            L->val[iu-1] = MAGMA_C_MAKE( sqrt( fabs( MAGMA_C_REAL(s) )), 0.0 );
        }
    }
    
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function does one synchronized ParILU sweep (symmetric case). 
    Input and output are different arrays.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                System matrix in COO.

    @param[in]
    L           magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/


extern "C" magma_int_t
magma_cparic_sweep_sync(
    magma_c_matrix A,
    magma_c_matrix *L,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    int i, j;
    int il, iu, jl, ju;
    magmaFloatComplex zero = MAGMA_C_MAKE(0.0, 0.0);
    magmaFloatComplex *L_new_val = NULL, *val_swap = NULL;
    CHECK( magma_cmalloc_cpu( &L_new_val, L->nnz ));
    
    #pragma omp parallel for
    for (int k=0; k < A.nnz; k++) {
        i = A.rowidx[k];
        j = A.col[k];
        
        magmaFloatComplex s, sp;
        s =  A.val[k];
        sp = zero;

        il = L->row[i];
        iu = L->row[j];

        while (il < L->row[i+1] && iu < L->row[j+1])
        {
            sp = zero;
            jl = L->col[il];
            ju = L->col[iu];

            // avoid branching
            sp = ( jl == ju ) ? L->val[il] * L->val[iu] : sp;
            s = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
        }
        // undo the last operation (it must be the last)
        s += sp;
        
        if ( i > j )      // modify l entry
            L_new_val[il-1] =  s / L->val[L->row[j+1]-1];
        else {            // modify u entry
            L_new_val[iu-1] = MAGMA_C_MAKE( sqrt( fabs( MAGMA_C_REAL(s) )), 0.0 );
        }
        
    }
    
    // swap old and new values
    SWAP( L_new_val, L->val );
    
cleanup:
    magma_free_cpu( L_new_val );
    
    return info;
}
