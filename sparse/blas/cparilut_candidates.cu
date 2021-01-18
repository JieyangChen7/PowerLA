/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/blas/zparilut_candidates.cu, normal z -> c, Thu Oct  8 23:05:49 2020

*/
#include "magmasparse_internal.h"

#define PRECISION_c

__global__ void 
cparilut_candidates_count_1(
    const magma_int_t num_rows,
    const magma_index_t* L0_row,
    const magma_index_t* L0_col,
    const magma_index_t* U0_row,
    const magma_index_t* U0_col,
    const magma_index_t* L_row,
    const magma_index_t* L_col,
    const magma_index_t* U_row,
    const magma_index_t* U_col,
    magma_index_t* L_new_row,
    magma_index_t* U_new_row)    
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    //for(int row=0; row<num_rows; row++){
    if (row < num_rows) {
        int numaddrowL = 0;
        int ilu0 = L0_row[row];
        int ilut = L_row[row];
        int endilu0 = L0_row[ row+1 ];
        int endilut = L_row[ row+1 ]; 
        int ilu0col;
        int ilutcol;
        do{
            ilu0col = L0_col[ ilu0 ];
            ilutcol = L_col[ ilut ];
            if(ilu0col == ilutcol ){
                ilu0++;
                ilut++;
            }
            else if(ilutcol<ilu0col ){
                ilut++;
            }
            else {
                // this element is missing in the current approximation
                // mark it as candidate
                numaddrowL++;
                ilu0++;
            }
        } while (ilut < endilut && ilu0 < endilu0);
        // do the rest if existing
        if(ilu0<endilu0 ){
            do{
                numaddrowL++;
                ilu0++;
            }while(ilu0<endilu0 ); 
        }
        L_new_row[ row ] = L_new_row[ row ]+numaddrowL;
        
        magma_int_t numaddrowU = 0;
        ilu0 = U0_row[row];
        ilut = U_row[row];
        endilu0 = U0_row[ row+1 ];
        endilut = U_row[ row+1 ]; 
        do{
            ilu0col = U0_col[ ilu0 ];
            ilutcol = U_col[ ilut ];
            if(ilu0col == ilutcol ){
                ilu0++;
                ilut++;
            }
            else if(ilutcol<ilu0col ){
                ilut++;
            }
            else {
                // this element is missing in the current approximation
                // mark it as candidate
                numaddrowU++;
                ilu0++;
            }
        }while(ilut<endilut && ilu0<endilu0 );
        if(ilu0<endilu0 ){
            do{
                numaddrowU++;
                ilu0++;
            }while(ilu0<endilu0 ); 
        }
        U_new_row[ row ] = U_new_row[ row ]+numaddrowU;
    }
        
}


__global__ void 
cparilut_candidates_count_2(
    const magma_int_t num_rows,
    const magma_index_t* L0_row,
    const magma_index_t* L0_col,
    const magma_index_t* U0_row,
    const magma_index_t* U0_col,
    const magma_index_t* L_row,
    const magma_index_t* L_col,
    const magma_index_t* U_row,
    const magma_index_t* U_col,
    magma_index_t* L_new_row,
    magma_index_t* U_new_row)    
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    //for(int row=0; row<num_rows; row++){
    if (row < num_rows) {
        // how to determine candidates:
        // for each node i, look at any "intermediate" neighbor nodes numbered
        // less, and then see if this neighbor has another neighbor j numbered
        // more than the intermediate; if so, fill in is (i,j) if it is not
        // already nonzero
        int numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        for(int el1=L_row[row]; el1<L_row[row+1]-1; el1++ ){
            int col1 = L_col[ el1 ];
            // now check the upper triangular
            // second loop first element over row - only for elements larger the intermediate
            for(int el2 = U_row[ col1 ]+1; el2 < U_row[ col1+1 ]; el2++ ){
                int col2 = U_col[ el2 ];
                int cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if(cand_col < row ){
                    // check whether this element already exists in L
                    // int exist = 0;
                    // for(int k=L_row[cand_row]; k<L_row[cand_row+1]; k++ ){
                    //     if(L_col[ k ] == cand_col ){
                    //             exist = 1;
                    //             //break;
                    //     }
                    // }
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    // if(exist == 0 ){
                    numaddrowL++;
                    // }
                } else {
                    // check whether this element already exists in U
                    // int exist = 0;
                    // for(int k=U_row[cand_row]; k<U_row[cand_row+1]; k++ ){
                    //     if(U_col[ k ] == cand_col ){
                    //             exist = 1;
                    //             //break;
                    //     }
                    // }
                    // if(exist == 0 ){
                        //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                    numaddrowU++;
                    // }
                }
            }

        }
        U_new_row[ row ] = U_new_row[ row ]+numaddrowU;
        L_new_row[ row ] = L_new_row[ row ]+numaddrowL;
    }  
}


__global__ void 
cparilut_candidates_insert_1(
    const magma_int_t num_rows,
    const magma_index_t* L0_row,
    const magma_index_t* L0_col,
    const magma_index_t* U0_row,
    const magma_index_t* U0_col,
    const magma_index_t* L_row,
    const magma_index_t* L_col,
    const magma_index_t* U_row,
    const magma_index_t* U_col,
    magma_index_t* L_new_row,
    magma_index_t* L_new_rowidx,
    magma_index_t* L_new_col,
    magmaFloatComplex* L_new_val,
    magma_index_t* insertedL,
    magma_index_t* U_new_row,
    magma_index_t* U_new_rowidx,
    magma_index_t* U_new_col,
    magmaFloatComplex* U_new_val,
    magma_index_t* insertedU)    
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    //for(int row=0; row<num_rows; row++){
    if (row < num_rows) {
        int laddL = 0;
        int offsetL = L_new_row[row];
        int ilu0 = L0_row[row];
        int ilut = L_row[row];
        int endilu0 = L0_row[ row+1 ];
        int endilut = L_row[ row+1 ]; 
        int ilu0col;
        int ilutcol;
        do{
            ilu0col = L0_col[ ilu0 ];
            ilutcol = L_col[ ilut ];
            if(ilu0col == ilutcol ){
                ilu0++;
                ilut++;
            }
            else if(ilutcol<ilu0col ){
                ilut++;
            }
            else {
                // this element is missing in the current approximation
                // mark it as candidate
                L_new_col[ offsetL + laddL ] = ilu0col;
                L_new_rowidx[ offsetL + laddL ] = row;
                L_new_val[ offsetL + laddL ] = MAGMA_C_ONE + MAGMA_C_ONE + MAGMA_C_ONE;
                laddL++;
                ilu0++;
            }
        } while(ilut<endilut && ilu0<endilu0 );
        if (ilu0<endilu0){
            do{
                ilu0col = L0_col[ ilu0 ];
                L_new_col[ offsetL + laddL ] = ilu0col;
                L_new_rowidx[ offsetL + laddL ] = row;
                L_new_val[ offsetL + laddL ] = MAGMA_C_ONE + MAGMA_C_ONE + MAGMA_C_ONE;
                laddL++;
                ilu0++;
            }while(ilu0<endilu0 ); 
        }
        insertedL[row] = laddL;
        
        int laddU = 0;
        int offsetU = U_new_row[row];
        ilu0 = U0_row[row];
        ilut = U_row[row];
        endilu0 = U0_row[ row+1 ];
        endilut = U_row[ row+1 ]; 
        do{
            ilu0col = U0_col[ ilu0 ];
            ilutcol = U_col[ ilut ];
            if(ilu0col == ilutcol ){
                ilu0++;
                ilut++;
            }
            else if(ilutcol<ilu0col ){
                ilut++;
            }
            else {
                // this element is missing in the current approximation
                // mark it as candidate
                U_new_col[ offsetU + laddU ] = ilu0col;
                U_new_rowidx[ offsetU + laddU ] = row;
                U_new_val[ offsetU + laddU ] = MAGMA_C_ONE + MAGMA_C_ONE + MAGMA_C_ONE;
                laddU++;
                ilu0++;
            }
        }while(ilut<endilut && ilu0<endilu0 );
        if(ilu0<endilu0 ){
            do{
                ilu0col = U0_col[ ilu0 ];
                U_new_col[ offsetU + laddU ] = ilu0col;
                U_new_rowidx[ offsetU + laddU ] = row;
                U_new_val[ offsetU + laddU ] = MAGMA_C_ONE + MAGMA_C_ONE + MAGMA_C_ONE;
                laddU++;
                ilu0++;
            }while(ilu0<endilu0 ); 
        }
        insertedU[row] = laddU;
    }
    
}


__global__ void 
cparilut_candidates_insert_2(
    const magma_int_t num_rows,
    const magma_index_t* L0_row,
    const magma_index_t* L0_col,
    const magma_index_t* U0_row,
    const magma_index_t* U0_col,
    const magma_index_t* L_row,
    const magma_index_t* L_col,
    const magma_index_t* U_row,
    const magma_index_t* U_col,
    magma_index_t* L_new_row,
    magma_index_t* L_new_rowidx,
    magma_index_t* L_new_col,
    magmaFloatComplex* L_new_val,
    magma_index_t* insertedL,
    magma_index_t* U_new_row,
    magma_index_t* U_new_rowidx,
    magma_index_t* U_new_col,
    magmaFloatComplex* U_new_val,
    magma_index_t* insertedU)    
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    //for(int row=0; row<num_rows; row++){
    if (row < num_rows) {
        int cand_row = row;
        int laddL = 0;
        int laddU = 0;
        int offsetL = L_new_row[row] + insertedL[row];
        int offsetU = U_new_row[row] + insertedU[row];
        // loop first element over row - only for elements smaller the diagonal
        for(int el1=L_row[row]; el1<L_row[row+1]-1; el1++ ){
            int col1 = L_col[ el1 ];
            // now check the upper triangular
            // second loop first element over row - only for elements larger the intermediate
            for(int el2 = U_row[ col1 ]+1; el2 < U_row[ col1+1 ]; el2++ ){
                int col2 = U_col[ el2 ];
                int cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if(cand_col < row ){
                    int exist = 0;
                    for(int k=L_row[cand_row]; k<L_row[cand_row+1]; k++ ){
                        if(L_col[ k ] == cand_col ){
                                exist = -1;
                                // printf("already exists:(%d,%d\n", row, cand_col);
                                //break;
                        }
                    }
                    for(int k=L_new_row[cand_row]; k<L_new_row[cand_row+1]; k++){
                        if(L_new_col[ k ] == cand_col ){
                            // element included in LU and nonzero
                            // printf("already inserted:(%d,%d\n", row, cand_col);
                            exist = -2;
                            //break;
                        }
                    }
                    L_new_rowidx[ offsetL + laddL ] = cand_row;
                    L_new_col[ offsetL + laddL ] = (exist == 0) ? cand_col : exist;
                    L_new_val[ offsetL + laddL ] = (exist == 0) ? MAGMA_C_ONE : MAGMA_C_ZERO;
                    laddL++;
                } else {
                    // check whether this element already exists in U
                    int exist = 0;
                    for(int k=U_row[cand_row]; k<U_row[cand_row+1]; k++ ){
                        if(U_col[ k ] == cand_col ){
                                // printf("already exists:(%d,%d\n", row, cand_col);
                                exist = -1;
                                //break;
                        }
                    }
                    for(int k=U_new_row[cand_row]; k<U_new_row[cand_row+1]; k++){
                        if(U_new_col[ k ] == cand_col ){
                            // element included in LU and nonzero
                            // printf("already inserted:(%d,%d==%d)  k:%d -> %d -> %d\n", row, cand_col , U_new_col[ k ], U_new_row[cand_row], k, U_new_row[cand_row+1] );
                            exist = -2;
                            //break;
                        }
                    }
                    U_new_rowidx[ offsetU + laddU ] = cand_row;
                    U_new_col[ offsetU + laddU ] = (exist == 0) ? cand_col : exist;
                    U_new_val[ offsetU + laddU ] = (exist == 0) ? MAGMA_C_ONE : MAGMA_C_ZERO;
                    laddU++;
                }
            }
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    This function identifies the locations with a potential nonzero ILU residual
    R = A - L*U where L and U are the current incomplete factors.
    Nonzero ILU residuals are possible
        1 where A is nonzero but L and U have no nonzero entry
        2 where the product L*U has fill-in but the location is not included 
          in L or U
          
    We assume that the incomplete factors are exact fro the elements included in
    the current pattern.
    
    This is the GPU implementation of the candidate search.
    
    2 GPU kernels are used: the first is a dry run assessing the memory need,
    the second then computes the candidate locations, the third eliminates 
    float entries. The fourth kernel ensures the elements in a row are sorted 
    for increasing column index.

    Arguments
    ---------

    @param[in]
    L0          magma_c_matrix
                tril(ILU(0) ) pattern of original system matrix.
                
    @param[in]
    U0          magma_c_matrix
                triu(ILU(0) ) pattern of original system matrix.
                
    @param[in]
    L           magma_c_matrix
                Current lower triangular factor.

    @param[in]
    U           magma_c_matrix
                Current upper triangular factor.

    @param[in,out]
    L_new       magma_c_matrix*
                List of candidates for L in COO format.

    @param[in,out]
    U_new       magma_c_matrix*
                List of candidates for U in COO format.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/

extern "C" magma_int_t
magma_cparilut_candidates_gpu(
    magma_c_matrix L0,
    magma_c_matrix U0,
    magma_c_matrix L,
    magma_c_matrix U,
    magma_c_matrix *L_new,
    magma_c_matrix *U_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    int num_rows = L.num_rows;
    float thrs = 1e-8;
    
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid11 = magma_ceildiv(num_rows, blocksize1 );
    int dimgrid12 = 1;
    int dimgrid13 = 1;
    dim3 grid1(dimgrid11, dimgrid12, dimgrid13 );
    dim3 block1(blocksize1, blocksize2, 1 );
    
    magmaIndex_ptr insertedL = NULL;
    magmaIndex_ptr insertedU = NULL;
    
    magma_cmfree(L_new, queue);
    magma_cmfree(U_new, queue);
        
    CHECK(magma_index_malloc(&insertedL, num_rows));
    CHECK(magma_index_malloc(&insertedU, num_rows));
    CHECK(magma_index_malloc(&L_new->drow, num_rows+1));
    CHECK(magma_index_malloc(&U_new->drow, num_rows+1));    
    CHECK(magma_cindexinit_gpu(num_rows+1, L_new->drow, queue));
    CHECK(magma_cindexinit_gpu(num_rows+1, U_new->drow, queue));
    CHECK(magma_cindexinit_gpu(num_rows, insertedL, queue));
    CHECK(magma_cindexinit_gpu(num_rows, insertedU, queue));
    L_new->num_rows = L.num_rows;
    L_new->num_cols = L.num_cols;
    L_new->storage_type = Magma_CSR;
    L_new->memory_location = Magma_DEV;
    
    U_new->num_rows = L.num_rows;
    U_new->num_cols = L.num_cols;
    U_new->storage_type = Magma_CSR;
    U_new->memory_location = Magma_DEV;
    
    cparilut_candidates_count_1<<<grid1, block1, 0, queue->cuda_stream()>>>(
        L0.num_rows, L0.drow, L0.dcol, U0.drow, U0.dcol,
        L.drow, L.dcol, U.drow, U.dcol,
        insertedL, insertedU);
    cparilut_candidates_count_2<<<grid1, block1, 0, queue->cuda_stream()>>>(
        L0.num_rows, L0.drow, L0.dcol, U0.drow, U0.dcol,
        L.drow, L.dcol, U.drow, U.dcol,
        insertedL, insertedU);
    CHECK(magma_cget_row_ptr(num_rows, &L_new->nnz, insertedL, 
        L_new->drow, queue));
    CHECK(magma_cget_row_ptr(num_rows, &U_new->nnz, insertedU, 
        U_new->drow, queue));
    
    CHECK(magma_cindexinit_gpu(num_rows, insertedL, queue));
    CHECK(magma_cindexinit_gpu(num_rows, insertedU, queue));
    
    CHECK(magma_cmalloc(&L_new->dval, L_new->nnz));
    CHECK(magma_index_malloc(&L_new->drowidx, L_new->nnz));
    CHECK(magma_index_malloc(&L_new->dcol, L_new->nnz));
    CHECK(magma_cmalloc(&U_new->dval, U_new->nnz));
    CHECK(magma_index_malloc(&U_new->drowidx, U_new->nnz));
    CHECK(magma_index_malloc(&U_new->dcol, U_new->nnz));
    CHECK(magma_cvalinit_gpu(L_new->nnz, L_new->dval, queue));
    CHECK(magma_cvalinit_gpu(U_new->nnz, U_new->dval, queue));
    //CHECK(magma_cindexinit_gpu(L_new->nnz, L_new->dcol, queue));
    //CHECK(magma_cindexinit_gpu(U_new->nnz, U_new->dcol, queue));
    //CHECK(magma_cindexinit_gpu(L_new->nnz, L_new->drowidx, queue));
    //CHECK(magma_cindexinit_gpu(U_new->nnz, U_new->drowidx, queue));
    // we don't need to init rowidx and col
    // the uninitilazed values will be removed anyways
    cparilut_candidates_insert_1<<<grid1, block1, 0, queue->cuda_stream()>>>(
        L0.num_rows, L0.drow, L0.dcol, U0.drow, U0.dcol,
        L.drow, L.dcol, U.drow, U.dcol,
        L_new->drow, L_new->drowidx, L_new->dcol, L_new->dval, insertedL,
        U_new->drow, U_new->drowidx, U_new->dcol, U_new->dval, insertedU);
    cparilut_candidates_insert_2<<<grid1, block1, 0, queue->cuda_stream()>>>(
        L0.num_rows, L0.drow, L0.dcol, U0.drow, U0.dcol,
        L.drow, L.dcol, U.drow, U.dcol,
        L_new->drow, L_new->drowidx, L_new->dcol, L_new->dval, insertedL,
        U_new->drow, U_new->drowidx, U_new->dcol, U_new->dval, insertedU);
        
    CHECK(magma_cthrsholdrm_gpu(1, L_new, &thrs, queue));
    CHECK(magma_cthrsholdrm_gpu(1, U_new, &thrs, queue));
    
cleanup:
    magma_free(insertedL);
    magma_free(insertedU);
    return info;
}







