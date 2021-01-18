/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/blas/magma_zpreselect.cu, normal z -> s, Thu Oct  8 23:05:47 2020

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256



// kernel copying everything except the last element
__global__ void
magma_spreselect_gpu0( 
    magma_int_t num_rows,
    magmaIndex_ptr row,
    float *val,
    float *valn)
{
    int tidx = threadIdx.x;   
    int bidx = blockIdx.x;
    int gtidx = bidx * blockDim.x + tidx;
    
    if (gtidx < num_rows) {
        for (int i=row[gtidx]; i<row[gtidx+1]-1; i++){
            valn[i-gtidx] = val[i];
        }
    }
}

// kernel copying everything except the first element
__global__ void
magma_spreselect_gpu1( 
    magma_int_t num_rows,
    magmaIndex_ptr row,
    float *val,
    float *valn)
{
    int tidx = threadIdx.x;   
    int bidx = blockIdx.x;
    int gtidx = bidx * blockDim.x + tidx;
    
    if (gtidx < num_rows) {
        for (int i=row[gtidx]+1; i<row[gtidx+1]; i++){
            valn[i-gtidx] = val[i];
        }
    }
}



/***************************************************************************//**
    Purpose
    -------
    This function takes a list of candidates with residuals, 
    and selects the largest in every row. The output matrix only contains these
    largest elements (respectively a zero element if there is no candidate for
    a certain row).

    Arguments
    ---------

    @param[in]
    order       magma_int_t
                order==0 lower triangular
                order==1 upper triangular
                
    @param[in]
    A           magma_s_matrix*
                Matrix where elements are removed.
                
    @param[out]
    oneA        magma_s_matrix*
                Matrix where elements are removed.
                
                

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
*******************************************************************************/

extern "C" magma_int_t
magma_spreselect_gpu(
    magma_int_t order,
    magma_s_matrix *A,
    magma_s_matrix *oneA,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(magma_ceildiv(A->num_rows, BLOCK_SIZE), 1, 1);
    
    oneA->num_rows = A->num_rows;
    oneA->num_cols = A->num_cols;
    oneA->nnz = A->nnz - A->num_rows;
    oneA->storage_type = Magma_CSR;
    oneA->memory_location = Magma_DEV;
    
    CHECK( magma_smalloc( &oneA->dval, oneA->nnz ) );
    
    if( order == 1 ){ // don't copy the first
        magma_spreselect_gpu1<<<grid, block, 0, queue->cuda_stream()>>>
        ( A->num_rows, A->drow, A->dval, oneA->dval );
        // #pragma omp parallel for
        // for( magma_int_t row=0; row<A->num_rows; row++){
        //     for( magma_int_t i=A->row[row]+1; i<A->row[row+1]; i++ ){
        //         oneA->val[ i-row ] = A->val[i];
        //     }
        // }
    } else { // don't copy the last
        magma_spreselect_gpu0<<<grid, block, 0, queue->cuda_stream()>>>
        ( A->num_rows, A->drow,  A->dval, oneA->dval );
        // #pragma omp parallel for
        // for( magma_int_t row=0; row<A->num_rows; row++){
        //     for( magma_int_t i=A->row[row]; i<A->row[row+1]-1; i++ ){
        //         oneA->val[ i-row ] = A->val[i];
        //     }
        // }            
    }
    
cleanup:
    return info;
}
