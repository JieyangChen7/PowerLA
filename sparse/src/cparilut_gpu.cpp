/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Hartwig Anzt

       @generated from sparse/src/zparilut_gpu.cpp, normal z -> c, Thu Oct  8 23:05:55 2020
*/

#include "magmasparse_internal.h"

#define PRECISION_c


/***************************************************************************//**
    Purpose
    -------

    Generates an incomplete threshold LU preconditioner via the ParILUT 
    algorithm. The strategy is to interleave a parallel fixed-point 
    iteration that approximates an incomplete factorization for a given nonzero 
    pattern with a procedure that adaptively changes the pattern. 
    Much of this algorithm has fine-grained parallelism, and can efficiently 
    exploit the compute power of shared memory architectures.

    This is the routine used in the publication by Anzt, Chow, Dongarra:
    ''ParILUT - A new parallel threshold ILU factorization''
    submitted to SIAM SISC in 2017.
    
    This version uses the default setting which adds all candidates to the
    sparsity pattern.

    This function requires OpenMP, and is only available if OpenMP is activated.
    
    The parameter list is:
    
    precond.sweeps : number of ParILUT steps
    precond.atol   : absolute fill ratio (1.0 keeps nnz count constant)


    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A

    @param[in]
    b           magma_c_matrix
                input RHS b

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
*******************************************************************************/
extern "C"
magma_int_t
magma_cparilut_gpu(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue)
{
    magma_int_t info = 0;

    real_Double_t start, end;
    real_Double_t t_rm=0.0, t_add=0.0, t_res=0.0, t_sweep1=0.0, t_sweep2=0.0, 
        t_cand=0.0, t_sort=0.0, t_transpose1=0.0, t_transpose2=0.0, t_selectrm=0.0,
        t_nrm=0.0, t_total = 0.0, accum=0.0;
                    
    float sum, sumL, sumU;

    magma_c_matrix hA={Magma_CSR}, hAT={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR},
        L={Magma_CSR}, U={Magma_CSR}, L_new={Magma_CSR}, U_new={Magma_CSR}, 
        UT={Magma_CSR}, L0={Magma_CSR}, U0={Magma_CSR};
    magma_c_matrix dA={Magma_CSR}, dL={Magma_CSR}, dhL={Magma_CSR}, dU={Magma_CSR}, dUT={Magma_CSR}, dhU={Magma_CSR}, dL0={Magma_CSR}, dU0={Magma_CSR}, dLt={Magma_CSR}, dUt={Magma_CSR} ; 
    magma_int_t num_rmL, num_rmU;
    magma_int_t selecttmp_size = 0;
    magma_ptr selecttmp_ptr = nullptr;
    float thrsL = 0.0;
    float thrsU = 0.0;

    magma_int_t num_threads = 1, timing = 1; // print timing
    magma_int_t L0nnz, U0nnz;
    
    CHECK(magma_cmtransfer(A, &hA, A.memory_location, Magma_CPU, queue));
    
    // in case using fill-in
    if (precond->levels > 0) {
        CHECK(magma_csymbilu(&hA, precond->levels, &hL, &hU , queue));
        magma_cmfree(&hU, queue);
        magma_cmfree(&hL, queue);
    }
    CHECK(magma_cmtransfer(hA, &dA, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_cmatrix_tril(hA, &L0, queue));
    CHECK(magma_cmatrix_triu(hA, &U0, queue));
    CHECK(magma_cmtransfer(L0, &dL0, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_cmtransfer(U0, &dU0, Magma_CPU, Magma_DEV, queue));
    magma_cmfree(&hU, queue);
    magma_cmfree(&hL, queue);
    CHECK(magma_cmatrix_tril(hA, &L, queue));
    CHECK(magma_cmtranspose(hA, &hAT, queue));
    CHECK(magma_cmatrix_tril(hAT, &U, queue));
    CHECK(magma_cmatrix_addrowindex(&L, queue)); 
    CHECK(magma_cmatrix_addrowindex(&U, queue)); 
    L.storage_type = Magma_CSRCOO;
    U.storage_type = Magma_CSRCOO;
    CHECK(magma_cmtransfer(L, &dL, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_cmtransfer(U, &dU, Magma_CPU, Magma_DEV, queue));
    L0nnz=L.nnz;
    U0nnz=U.nnz;

    if (timing == 1) {
        printf("ilut_fill_ratio = %.6f;\n\n", precond->atol);  
        printf("performance_%d = [\n%%iter      L.nnz      U.nnz    ILU-Norm    transp    candidat  resid     sort    transcand    add      sweep1   selectrm    remove    sweep2     total       accum\n", 
            (int) num_threads);
    }

    //##########################################################################

    for (magma_int_t iters =0; iters<precond->sweeps; iters++) {
        t_rm=0.0; t_add=0.0; t_res=0.0; t_sweep1=0.0; t_sweep2=0.0; t_cand=0.0;
        t_transpose1=0.0; t_transpose2=0.0; t_selectrm=0.0; t_sort = 0;
        t_nrm=0.0; t_total = 0.0;
     
        // step 1: transpose U
        start = magma_sync_wtime(queue);
        magma_cmfree(&dUT, queue);
        dU.storage_type = Magma_CSR;
        CHECK(magma_cmtranspose( dU, &dUT, queue));
        end = magma_sync_wtime(queue); t_transpose1+=end-start;
        
        
        // step 2: find candidates
        start = magma_sync_wtime(queue);
        CHECK(magma_cparilut_candidates_gpu(dL0, dU0, dL, dUT, &dhL, &dhU, queue));
        dhL.storage_type = Magma_CSRCOO;
        dhU.storage_type = Magma_CSRCOO;
        end = magma_sync_wtime(queue); t_cand=+end-start;
        
        
        // step 3: compute residuals (optional when adding all candidates)
        start = magma_sync_wtime(queue);
        CHECK(magma_cparilut_residuals_gpu(dA, dL, dU, &dhL, queue));
        CHECK(magma_cparilut_residuals_gpu(dA, dL, dU, &dhU, queue));
        magma_cmfree(&hL, queue);
        magma_cmfree(&hU, queue);
        dhL.storage_type = Magma_CSRCOO;
        dhU.storage_type = Magma_CSRCOO;
        end = magma_sync_wtime(queue); t_res=+end-start;
        start = magma_sync_wtime(queue);
        sumL = magma_scnrm2( dhL.nnz, dhL.dval, 1, queue );
        sumU = magma_scnrm2( dhU.nnz, dhU.dval, 1, queue );
        sum = sumL + sumU;
        end = magma_sync_wtime(queue); t_nrm+=end-start;
        

        // step 4: sort candidates
        start = magma_sync_wtime(queue);
        CHECK(magma_ccsr_sort_gpu( &dhL, queue));
        CHECK(magma_ccsr_sort_gpu( &dhU, queue));
        magma_cmfree(&dLt, queue);
        magma_cmfree(&dUt, queue);
        dhU.storage_type = Magma_CSR;
        end = magma_sync_wtime(queue); t_sort+=end-start;
        
        
        // step 5: transpose candidates
        start = magma_sync_wtime(queue);
        CHECK(magma_cmtranspose( dhU, &dUt, queue));
        dhU.memory_location = Magma_DEV;
        dhL.memory_location = Magma_DEV;
        dUt.memory_location = Magma_DEV;
        dLt.memory_location = Magma_DEV;
        dhL.storage_type = Magma_CSRCOO;
        dhU.storage_type = Magma_CSRCOO;
        dL.storage_type = Magma_CSRCOO;
        dU.storage_type = Magma_CSRCOO;
        dLt.storage_type = Magma_CSRCOO;
        CHECK(magma_cmatrix_swap(&dhL, &dLt, queue));
        magma_cmfree(&dhL, queue);
        magma_cmfree(&dhU, queue);
        end = magma_sync_wtime(queue); t_transpose2+=end-start;
        
        
        // step 6: add candidates
        start = magma_sync_wtime(queue);
        CHECK(magma_cmatrix_cup_gpu(dL, dLt, &dhL, queue));   
        CHECK(magma_cmatrix_cup_gpu(dU, dUt, &dhU, queue));
        dhL.storage_type = Magma_CSRCOO;
        dhU.storage_type = Magma_CSRCOO;
        CHECK(magma_cmatrix_swap(&dhL, &dL, queue));
        CHECK(magma_cmatrix_swap(&dhU, &dU, queue));
        magma_cmfree(&dhL, queue);
        magma_cmfree(&dhU, queue);
        magma_cmfree(&dLt, queue);
        magma_cmfree(&dUt, queue);
        end = magma_sync_wtime(queue); t_add=+end-start;
        
        
        // step 7: sweep
        start = magma_sync_wtime(queue);
        // // GPU kernel
        CHECK(magma_cparilut_sweep_gpu(&dA, &dL, &dU, queue));
        end = magma_sync_wtime(queue); t_sweep1+=end-start;
        
        
        // step 8: select threshold to remove elements
        start = magma_sync_wtime(queue);
        num_rmL = max((dL.nnz-L0nnz*(1+(precond->atol-1.)
            *(iters+1)/precond->sweeps)), 0);
        num_rmU = max((dU.nnz-U0nnz*(1+(precond->atol-1.)
            *(iters+1)/precond->sweeps)), 0);
        // pre-select: ignore the diagonal entries
        if (num_rmL>0) {
            CHECK(magma_csampleselect(dL.nnz, num_rmL, dL.dval, &thrsL, &selecttmp_ptr, &selecttmp_size, queue));
        } else {
            thrsL = 0.0;
        }
        if (num_rmU>0) {
            CHECK(magma_csampleselect(dU.nnz, num_rmU, dU.dval, &thrsU, &selecttmp_ptr, &selecttmp_size, queue));
        } else {
            thrsU = 0.0;
        }
        end = magma_sync_wtime(queue); t_selectrm=end-start;
        
        // step 9: remove elements
        start = magma_sync_wtime(queue);
        // GPU kernel
        CHECK(magma_cthrsholdrm_gpu(1, &dL, &thrsL, queue));
        CHECK(magma_cthrsholdrm_gpu(1, &dU, &thrsU, queue));
        // GPU kernel
        end = magma_sync_wtime(queue); t_rm=end-start;
        
        // step 10: sweep
        start = magma_sync_wtime(queue);
        // GPU kernel
        CHECK(magma_cparilut_sweep_gpu(&dA, &dL, &dU, queue));
        // end GPU kernel
        end = magma_sync_wtime(queue); t_sweep2+=end-start;
        
        if (timing == 1) {
            t_total = t_transpose1+ t_cand+ t_res+ t_sort+ t_transpose2+ t_add+ t_sweep1+ t_selectrm+ t_rm+ t_sweep2;
            accum = accum + t_total;
            printf("%5lld %10lld %10lld  %.4e   %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e      %.2e\n",
                (long long) iters, (long long) dL.nnz, (long long) dU.nnz, 
                (float) sum, 
                t_transpose1, t_cand, t_res, t_sort, t_transpose2, t_add, t_sweep1, t_selectrm, t_rm, t_sweep2, t_total, accum);
            fflush(stdout);
        }
    }
    if (timing == 1) {
        printf("]; \n");
        fflush(stdout);
    }
    //##########################################################################
    
    
    magma_cmfree(&L, queue);
    magma_cmfree(&U, queue);
    CHECK(magma_cmtransfer(dL, &L, Magma_DEV, Magma_CPU, queue));
    CHECK(magma_cmtransfer(dU, &U, Magma_DEV, Magma_CPU, queue));
    magma_cmfree(&dL, queue);
    magma_cmfree(&dU, queue);
    L.storage_type = Magma_CSR;
    U.storage_type = Magma_CSR;
    // for CUSPARSE
    CHECK(magma_cmtransfer(L, &precond->L, Magma_CPU, Magma_DEV , queue));
    CHECK(magma_cmtranspose( U, &UT, queue));
    //CHECK(magma_ccsrcoo_transpose(U, &UT, queue));
    //magma_cmtranspose(U, &UT, queue);
    CHECK(magma_cmtransfer(UT, &precond->U, Magma_CPU, Magma_DEV , queue));
    
    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_ccumilugeneratesolverinfo(precond, queue));
    } else {
        //prepare for iterative solves
        // extract the diagonal of L into precond->d
        CHECK(magma_cjacobisetup_diagscal(precond->L, &precond->d, queue));
        CHECK(magma_cvinit(&precond->work1, Magma_DEV, hA.num_rows, 1, 
            MAGMA_C_ZERO, queue));
        // extract the diagonal of U into precond->d2
        CHECK(magma_cjacobisetup_diagscal(precond->U, &precond->d2, queue));
        CHECK(magma_cvinit(&precond->work2, Magma_DEV, hA.num_rows, 1, 
            MAGMA_C_ZERO, queue));
    }

cleanup:
    magma_cmfree(&hA, queue);
    magma_cmfree(&hAT, queue);
    magma_cmfree(&L, queue);
    magma_cmfree(&U, queue);
    magma_cmfree(&UT, queue);
    magma_cmfree(&L0, queue);
    magma_cmfree(&U0, queue);
    magma_cmfree(&L_new, queue);
    magma_cmfree(&U_new, queue);
    magma_cmfree(&hL, queue);
    magma_cmfree(&hU, queue);
    return info;
}
