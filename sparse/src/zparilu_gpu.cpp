/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Hartwig Anzt

       @precisions normal z -> s d c
*/

#include "magmasparse_internal.h"

#define PRECISION_z


/***************************************************************************//**
    Purpose
    -------

    Generates an ILU(0) preconditer via fixed-point iterations. 
    For reference, see:
    E. Chow and A. Patel: "Fine-grained Parallel Incomplete LU Factorization", 
    SIAM Journal on Scientific Computing, 37, C169-C193 (2015). 
    
    This is the GPU implementation of the ParILU

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
*******************************************************************************/
extern "C"
magma_int_t
magma_zparilu_gpu(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue)
{
    magma_int_t info = 0;

    magma_z_matrix hAT={Magma_CSR}, hA={Magma_CSR}, hAL={Magma_CSR}, 
    hAU={Magma_CSR}, hAUT={Magma_CSR}, hAtmp={Magma_CSR}, hACOO={Magma_CSR},
    dAL={Magma_CSR}, dAU={Magma_CSR}, dAUT={Magma_CSR}, dACOO={Magma_CSR};

    // copy original matrix as COO to device
    if (A.memory_location != Magma_CPU || A.storage_type != Magma_CSR) {
        CHECK(magma_zmtransfer(A, &hAT, A.memory_location, Magma_CPU, queue));
        CHECK(magma_zmconvert(hAT, &hA, hAT.storage_type, Magma_CSR, queue));
        magma_zmfree(&hAT, queue);
    } else {
        CHECK(magma_zmtransfer(A, &hA, A.memory_location, Magma_CPU, queue));
    }

    // in case using fill-in
    if (precond->levels > 0) {
        CHECK(magma_zsymbilu(&hA, precond->levels, &hAL, &hAUT,  queue));
        magma_zmfree(&hAL, queue);
        magma_zmfree(&hAUT, queue);
    }
    CHECK(magma_zmconvert(hA, &hACOO, hA.storage_type, Magma_CSRCOO, queue));
    
    //get L
    magma_zmatrix_tril(hA, &hAL, queue);
    // we need 1 on the main diagonal of L
    #pragma omp parallel for
    for (int k=0; k < hAL.num_rows; k++) {
        hAL.val[hAL.row[k+1]-1] = MAGMA_Z_ONE;
    }
    
    // get U
    magma_zmtranspose(hA, &hAT, queue);
    magma_zmatrix_tril(hAT, &hAU, queue);
    magma_zmfree(&hAT, queue);
    
    CHECK(magma_zmtransfer(hAL, &dAL, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_zmtransfer(hAU, &dAU, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_zmtransfer(hACOO, &dACOO, Magma_CPU, Magma_DEV, queue));
    
    // This is the actual ParILU kernel. 
    // It can be called directly if
    // - the system matrix hACOO is available in COO format on the CPU 
    // - hAL is the lower triangular in CSR on the CPU
    // - hAU is the upper triangular in CSC on the CPU (U transpose in CSR)
    // The kernel is located in sparse/blas/zparilu_kernels.cu
    //
    for (int i=0; i<precond->sweeps; i++) {
        CHECK(magma_zparilu_csr(dACOO, dAL, dAU, queue));
    }
    CHECK(magma_z_cucsrtranspose(dAU, &dAUT, queue));

    CHECK(magma_zmtransfer(dAL, &precond->L, Magma_DEV, Magma_DEV, queue));
    CHECK(magma_zmtransfer(dAUT, &precond->U, Magma_DEV, Magma_DEV, queue));
    
    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_zcumilugeneratesolverinfo(precond, queue));
    } else {
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK(magma_zjacobisetup_diagscal(precond->L, &precond->d, queue));
        CHECK(magma_zvinit(&precond->work1, Magma_DEV, hA.num_rows, 1, 
            MAGMA_Z_ZERO, queue));

        // extract the diagonal of U into precond->d2
        CHECK(magma_zjacobisetup_diagscal(precond->U, &precond->d2, queue));
        CHECK(magma_zvinit(&precond->work2, Magma_DEV, hA.num_rows, 1, 
            MAGMA_Z_ZERO, queue));
    }
    
cleanup:
    magma_zmfree(&dAL, queue);
    magma_zmfree(&dAU, queue);
    magma_zmfree(&dAUT, queue);
    magma_zmfree(&dACOO, queue);
    magma_zmfree(&hAT, queue);
    magma_zmfree(&hA, queue);
    magma_zmfree(&hAL, queue);
    magma_zmfree(&hAU, queue);
    magma_zmfree(&hAUT, queue);
    magma_zmfree(&hAtmp, queue);
    magma_zmfree(&hACOO, queue);

    
    return info;
}

