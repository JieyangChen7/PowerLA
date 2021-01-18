/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/control/magma_zmdiagdom.cpp, normal z -> d, Thu Oct  8 23:05:50 2020
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"


/***************************************************************************//**
    Purpose
    -------
    This routine takes a CSR matrix and computes the average diagonal dominance.
    For each row i, it computes the abs(d_ii)/sum_j(abs(a_ij)).
    It returns max, min, and average.

    Arguments
    ---------

    @param[in]
    M           magma_d_matrix
                System matrix.

    @param[out]
    *min_dd     double
                Smallest diagonal dominance.
                
    @param[out]
    *max_dd     double
                Largest diagonal dominance.
               
    @param[out]
    *avg_dd     double
                Average diagonal dominance.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dmdiagdom(
    magma_d_matrix M,
    double *min_dd,
    double *max_dd,
    double *avg_dd,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    *min_dd = 0.0;
    *max_dd = 0.0;
    *avg_dd = 0.0;
    magma_int_t count = 0;
    
    magma_d_matrix x={Magma_CSR};
    magma_d_matrix A={Magma_CSR};
    CHECK( magma_dmtransfer( M, &A, M.memory_location, Magma_CPU, queue ));
    CHECK( magma_dvinit( &x, Magma_CPU, A.num_rows, 1, 0.0, queue ) );
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        double diag = 0.0;
        double offdiag = 0.0;
        for( magma_int_t j=A.row[i]; j<A.row[i+1]; j++ ){
            double val = MAGMA_D_ABS( A.val[j] );
            if( A.col[j] == i ){
                diag += val;
            } else {
                offdiag += val;    
            }
        }
        x.val[i] = offdiag / diag;
    }
    *min_dd = 1e10;
    *max_dd = 0.0;
    *avg_dd =0.0;
    for(magma_int_t i=0; i<A.num_rows; i++ ){
        if( x.val[i] < 0.0 ){
            ;
        } else {
            *min_dd = ( x.val[i] < *min_dd ) ? x.val[i] : *min_dd; 
            *max_dd = ( x.val[i] > *max_dd ) ? x.val[i] : *max_dd; 
            *avg_dd += x.val[i];
            count++;
        }
    }
    
    *avg_dd = *avg_dd / ( (double) count );
    
cleanup:
    magma_dmfree(&x, queue );
    magma_dmfree(&A, queue );
    
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This routine takes a CSR matrix and computes the average block-diagonal 
    dominance.
    For each row i, it computes the abs( D_(i,:) ) / abs( A(i,:) \ D_(i,:) ).
    It returns max, min, and average.
    The input vector bsz contains the blocksizes.

    Arguments
    ---------

    @param[in]
    M           magma_d_matrix
                System matrix.
                
    @param[in]
    blocksizes  magma_d_matrix
                Vector containing blocksizes (as DoubleComplex).

    @param[out]
    *min_dd     double
                Smallest diagonal dominance.
                
    @param[out]
    *max_dd     double
                Largest diagonal dominance.
               
    @param[out]
    *avg_dd     double
                Average diagonal dominance.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dmbdiagdom(
    magma_d_matrix M,
    magma_d_matrix blocksizes,
    double *min_dd,
    double *max_dd,
    double *avg_dd,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t *end=NULL, *start=NULL;
    magma_int_t ii=0;
    magma_int_t count = 0;
    
    magma_int_t rowbsz = 0; //blocksize for this row
    
    *min_dd = 0.0;
    *max_dd = 0.0;
    *avg_dd = 0.0;
    
    magma_d_matrix x={Magma_CSR};
    magma_d_matrix bsz={Magma_CSR};
    magma_d_matrix A={Magma_CSR};
    CHECK( magma_dmtransfer( M, &A, M.memory_location, Magma_CPU, queue ));
    CHECK( magma_dmtransfer( blocksizes, &bsz, blocksizes.memory_location, Magma_CPU, queue ));
    CHECK( magma_dvinit( &x, Magma_CPU, A.num_rows, 1, 0.0, queue ) );
    CHECK( magma_imalloc_cpu( &start, A.num_rows ));
    CHECK( magma_imalloc_cpu( &end,   A.num_rows ));
    for( magma_int_t rowb=0; rowb<bsz.num_rows; rowb++ ){ // block of rows
        rowbsz = (magma_int_t) MAGMA_D_REAL((bsz.val[rowb]));  
        for( magma_int_t j =0; j<rowbsz; j++){
            start[ii] = ii-j;
            end[ii] = ii+rowbsz-j;
            //printf("%d: %d -- %d\n", ii, start[ii], end[ii]);
            ii++;
        }
    }
    #pragma omp parallel for
    for(magma_int_t i=0; i<A.num_rows; i++ ){
        double diag = 0.0;
        double offdiag = 0.0;
        for( magma_int_t j=A.row[i]; j<A.row[i+1]; j++ ){
            double val = MAGMA_D_ABS( A.val[j] );
            if( ((A.col[j] >= start[i]) && (A.col[j]<end[i])) ){
                diag += val;
            } else {
                offdiag += val;    
            }
        }
        x.val[i] = offdiag / diag;
    }
    *min_dd = 1e10;
    *max_dd = 0.0;
    *avg_dd =0.0;
    count = 0;
    for(magma_int_t i=0; i<A.num_rows; i++ ){
        if( x.val[i] < 0.0 ){
            ;
        } else {
            *min_dd = ( x.val[i] < *min_dd ) ? x.val[i] : *min_dd; 
            *max_dd = ( x.val[i] > *max_dd ) ? x.val[i] : *max_dd; 
            *avg_dd += x.val[i];
            count++;
        }
    }
    
    *avg_dd = *avg_dd / ( (double) count );
    
cleanup:
    magma_dmfree(&x, queue );
    magma_dmfree(&bsz, queue );
    magma_dmfree(&A, queue );
    magma_free_cpu( start );
    magma_free_cpu( end );
    
    return info;
}

