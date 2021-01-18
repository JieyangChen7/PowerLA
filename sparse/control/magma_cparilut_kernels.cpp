/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/control/magma_zparilut_kernels.cpp, normal z -> c, Thu Oct  8 23:05:52 2020
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
    This function does an ParILUT sweep. The difference to the ParILU sweep is
    that the nonzero pattern of A and the incomplete factors L and U can be 
    different. 
    The pattern determing which elements are iterated are hence the pattern 
    of L and U, not A.
    
    This is the CPU version of the asynchronous ParILUT sweep.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix*
                System matrix. The format is sorted CSR.

    @param[in,out]
    L           magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is MAGMA_CSRCOO. This is sorted CSR plus the 
                rowindexes being stored.
                
    @param[in,out]
    U           magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is MAGMA_CSRCOO. This is sorted CSR plus the 
                rowindexes being stored.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/

extern "C" magma_int_t
magma_cparilut_sweep(
    magma_c_matrix *A,
    magma_c_matrix *L,
    magma_c_matrix *U,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    #pragma omp parallel for
    for (magma_int_t e=0; e<L->nnz; e++) {

        magma_int_t i,j,icol,jcol,jold;

        magma_index_t row = L->rowidx[ e ];
        magma_index_t col = L->col[ e ];
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if(col < row) {
            magmaFloatComplex A_e = MAGMA_C_ZERO;
            // check whether A contains element in this location
            for (i = A->row[row]; i<A->row[row+1]; i++) {
                if(A->col[i] == col) {
                    A_e = A->val[i];
                    break;
                }
            }
            //now do the actual iteration
            i = L->row[ row ];
            j = U->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magma_int_t endj = U->row[ col+1 ]; 
            magmaFloatComplex sum = MAGMA_C_ZERO;
            magmaFloatComplex lsum = MAGMA_C_ZERO;
            do{
                lsum = MAGMA_C_ZERO;
                jold = j;
                icol = L->col[i];
                jcol = U->col[j];
                if(icol == jcol) {
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i++;
                    j++;
                }
                else if(icol<jcol) {
                    i++;
                }
                else {
                    j++;
                }
            }while(i<endi && j<endj);
            sum = sum - lsum;

            // write back to location e
            L->val[ e ] =  (A_e - sum)/ U->val[jold];
        } else if(row == col) { // end check whether part of L
            L->val[ e ] = MAGMA_C_ONE; // lower triangular has diagonal equal 1
        }
    }// end omp parallel section

   #pragma omp parallel for
    for (magma_int_t e=0; e<U->nnz; e++) {
        {
            magma_int_t i,j,icol,jcol;
            magma_index_t row = U->col[ e ];
            magma_index_t col = U->rowidx[ e ];
            magmaFloatComplex A_e = MAGMA_C_ZERO;
            // check whether A contains element in this location
            for (i = A->row[row]; i<A->row[row+1]; i++) {
                if(A->col[i] == col) {
                    A_e = A->val[i];
                    break;
                }
            }
            //now do the actual iteration
            i = L->row[ row ];
            j = U->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magma_int_t endj = U->row[ col+1 ];
            magmaFloatComplex sum = MAGMA_C_ZERO;
            magmaFloatComplex lsum = MAGMA_C_ZERO;
            do{
                lsum = MAGMA_C_ZERO;
                icol = L->col[i];
                jcol = U->col[j];
                if(icol == jcol) {
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i++;
                    j++;
                }
                else if(icol<jcol) {
                    i++;
                }
                else {
                    j++;
                }
            }while(i<endi && j<endj);
            sum = sum - lsum;
            // write back to location e
            U->val[ e ] =  (A_e - sum);
        }
    }// end omp parallel section

    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function does an ParILUT sweep. The difference to the ParILU sweep is
    that the nonzero pattern of A and the incomplete factors L and U can be 
    different. 
    The pattern determing which elements are iterated are hence the pattern 
    of L and U, not A.
    
    This is the CPU version of the synchronous ParILUT sweep.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix*
                System matrix. The format is sorted CSR.

    @param[in,out]
    L           magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is MAGMA_CSRCOO. This is sorted CSR plus the 
                rowindexes being stored.
                
    @param[in,out]
    U           magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is MAGMA_CSRCOO. This is sorted CSR plus the 
                rowindexes being stored.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/


extern "C" magma_int_t
magma_cparilut_sweep_sync(
    magma_c_matrix *A,
    magma_c_matrix *L,
    magma_c_matrix *U,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    magmaFloatComplex *L_new_val = NULL, *U_new_val = NULL, *val_swap = NULL;
    CHECK(magma_cmalloc_cpu(&L_new_val, L->nnz));
    CHECK(magma_cmalloc_cpu(&U_new_val, U->nnz));
    
    #pragma omp parallel for
    for (magma_int_t e=0; e<U->nnz; e++) {
        magma_int_t i,j,icol,jcol;

        magma_index_t row = U->col[ e ];
        magma_index_t col = U->rowidx[ e ];
        {   
            magmaFloatComplex A_e = MAGMA_C_ZERO;
            // check whether A contains element in this location
            for (i = A->row[row]; i<A->row[row+1]; i++) {
                if(A->col[i] == col) {
                    A_e = A->val[i];
                    break;
                }
            }
            //now do the actual iteration
            i = L->row[ row ];
            j = U->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magma_int_t endj = U->row[ col+1 ];
            magmaFloatComplex sum = MAGMA_C_ZERO;
            magmaFloatComplex lsum = MAGMA_C_ZERO;
            do{
                lsum = MAGMA_C_ZERO;
                icol = L->col[i];
                jcol = U->col[j];
                if(icol == jcol) {
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i++;
                    j++;
                }
                else if(icol<jcol) {
                    i++;
                }
                else {
                    j++;
                }
            }while(i<endi && j<endj);
            sum = sum - lsum;

            // write back to location e
            U_new_val[ e ] =  (A_e - sum);
        }
    }// end omp parallel section
    
    
    #pragma omp parallel for
    for (magma_int_t e=0; e<L->nnz; e++) {
        magma_int_t i,j,icol,jcol,jold;
        magma_index_t row = L->rowidx[ e ];
        magma_index_t col = L->col[ e ];
        
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if(row == col) { 
            L_new_val[ e ] = MAGMA_C_ONE; // lower triangular has 1-diagonal
        } else {
            magmaFloatComplex A_e = MAGMA_C_ZERO;
            // check whether A contains element in this location
            for (i = A->row[row]; i<A->row[row+1]; i++) {
                if(A->col[i] == col) {
                    A_e = A->val[i];
                    break;
                }
            }
            //now do the actual iteration
            i = L->row[ row ];
            j = U->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magma_int_t endj = U->row[ col+1 ]; 
            magmaFloatComplex sum = MAGMA_C_ZERO;
            magmaFloatComplex lsum = MAGMA_C_ZERO;
            do{
                lsum = MAGMA_C_ZERO;
                jold = j;
                icol = L->col[i];
                jcol = U->col[j];
                
                if(icol == jcol) {
                    lsum = L->val[i] * U_new_val[j];
                    sum = sum + lsum;
                    i++;
                    j++;
                }
                else if(icol<jcol) {
                    i++;
                }
                else {
                    j++;
                }
            }while(i<endi && j<endj);
            sum = sum - lsum;
            // write back to location e
            L_new_val[ e ] =  (A_e - sum)/ U->val[jold];
        }

    }// end omp parallel section

    // swap old and new values
    SWAP(L_new_val, L->val);
    SWAP(U_new_val, U->val);
    
cleanup:
    magma_free_cpu(L_new_val);
    magma_free_cpu(U_new_val);
    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This function computes the ILU residual in the locations included in the 
    sparsity pattern of R.

    Arguments
    ---------


    @param[in]
    A           magma_c_matrix
                System matrix A.

    @param[in]
    L           magma_c_matrix
                Current approximation for the lower triangular factor.
                The format is sorted CSR.
                
    @param[in]
    U           magma_c_matrix
                Current approximation for the upper triangular factor.
                The format is sorted CSR.
                
    @param[in,out]
    R           magma_c_matrix*
                Sparsity pattern on which the ILU residual is computed. 
                R is in COO format. On output, R contains the ILU residual.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/

extern "C" magma_int_t
magma_cparilut_residuals(
    magma_c_matrix A,
    magma_c_matrix L,
    magma_c_matrix U,
    magma_c_matrix *R,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    #pragma omp parallel for
    for (magma_int_t e=0; e<R->nnz; e++) {
        {
            magma_int_t i,j,icol,jcol;
            magma_index_t row = R->rowidx[ e ];
            magma_index_t col = R->col[ e ];
            magmaFloatComplex A_e = MAGMA_C_ZERO;
            for (i = A.row[row]; i<A.row[row+1]; i++) {
                if(A.col[i] == col) {
                    A_e = A.val[i];
                    i = A.row[row+1];
                }
            }
            //now do the actual iteration
            i = L.row[ row ];
            j = U.row[ col ];
            magma_int_t endi = L.row[ row+1 ];
            magma_int_t endj = U.row[ col+1 ];
            magmaFloatComplex sum = MAGMA_C_ZERO;
            magmaFloatComplex lsum = MAGMA_C_ZERO;
            do{
                lsum = MAGMA_C_ZERO;
                icol = L.col[i];
                jcol = U.col[j];
                if(icol == jcol) {
                    lsum = L.val[i] * U.val[j];
                    sum = sum + lsum;
                    i++;
                }
                else if(icol<jcol) {
                    i++;
                }
                else {
                    j++;
                }
            }while(i<endi && j<endj);
            sum = sum - lsum;
            // write back to location e
            R->val[ e ] =  (A_e - sum);
        }
    }// end omp parallel section

    return info;
}
