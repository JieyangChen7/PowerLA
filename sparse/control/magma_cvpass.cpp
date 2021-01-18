/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/control/magma_zvpass.cpp, normal z -> c, Thu Oct  8 23:05:51 2020
       @author Hartwig Anzt
*/

#include "magmasparse_internal.h"


/**
    Purpose
    -------

    Passes a vector to MAGMA.

    Arguments
    ---------

    @param[in]
    m           magma_int_t
                number of rows

    @param[in]
    n           magma_int_t
                number of columns

    @param[in]
    val         magmaFloatComplex*
                array containing vector entries

    @param[out]
    v           magma_c_matrix*
                magma vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_cvset(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *val,
    magma_c_matrix *v,
    magma_queue_t queue )
{
    // make sure the target structure is empty
    magma_cmfree( v, queue );
    
    v->num_rows = m;
    v->num_cols = n;
    v->nnz = m*n;
    v->memory_location = Magma_CPU;
    v->val = val;
    v->major = MagmaColMajor;
    v->storage_type = Magma_DENSE;
    v->ownership = MagmaFalse;

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Passes a MAGMA vector back. This function requires the array val to be 
    already allocated (of size m x n).

    Arguments
    ---------

    @param[in]
    v           magma_c_matrix
                magma vector

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    val         magmaFloatComplex*
                array of size m x n the vector entries are copied into

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_cvcopy(
    magma_c_matrix v,
    magma_int_t *m, magma_int_t *n,
    magmaFloatComplex *val,
    magma_queue_t queue )
{
    magma_c_matrix v_CPU={Magma_CSR};
    magma_int_t info =0;
    
    if ( v.memory_location == Magma_CPU ) {
        *m = v.num_rows;
        *n = v.num_cols;
        for (magma_int_t i=0; i<v.num_rows*v.num_cols; i++) {
            val[i] = v.val[i];
        }
    } else {
        CHECK( magma_cmtransfer( v, &v_CPU, v.memory_location, Magma_CPU, queue ));
        CHECK( magma_cvcopy( v_CPU, m, n, val, queue ));
    }
    
cleanup:
    return info;
}
