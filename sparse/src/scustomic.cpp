/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Hartwig Anzt

       @generated from sparse/src/zcustomic.cpp, normal z -> s, Thu Oct  8 23:05:55 2020
*/
#include "magmasparse_internal.h"

#define REAL

// todo: make it spacific
#if CUDA_VERSION >= 11000
#define cusparseCreateSolveAnalysisInfo(info) {;}
#else
#define cusparseCreateSolveAnalysisInfo(info)                                                   \
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( info ))
#endif

// todo: info is passed; buf has to be passed
#if CUDA_VERSION >= 11000
#define cusparseScsrsv_analysis(handle, trans, m, nnz, descr, val, row, col, info)              \
    {                                                                                           \
        csrsv2Info_t linfo = 0;                                                                 \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        cusparseCreateCsrsv2Info(&linfo);                                                       \
        cusparseScsrsv2_bufferSize(handle, trans, m, nnz, descr, val, row, col,                 \
                                   linfo, &bufsize);                                            \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseScsrsv2_analysis(handle, trans, m, nnz, descr, val, row, col, linfo,            \
                                 CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf);                         \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#endif

/**
    Purpose
    -------

    Reads in an Incomplete Cholesky preconditioner.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A
                
    @param[in]
    b           magma_s_matrix
                input RHS b

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_scustomicsetup(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
    
    magma_s_matrix hA={Magma_CSR};
    char preconditionermatrix[255];
    
    snprintf( preconditionermatrix, sizeof(preconditionermatrix),
                "/Users/hanzt0114cl306/work/matrices/matrices/ICT.mtx" );
    
    CHECK( magma_s_csr_mtx( &hA, preconditionermatrix , queue) );
    
    
    // for CUSPARSE
    CHECK( magma_smtransfer( hA, &precond->M, Magma_CPU, Magma_DEV , queue ));

        // copy the matrix to precond->L and (transposed) to precond->U
    CHECK( magma_smtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_smtranspose( precond->L, &(precond->U), queue ));

    // extract the diagonal of L into precond->d
    CHECK( magma_sjacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_svinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_S_ZERO, queue ));

    // extract the diagonal of U into precond->d2
    CHECK( magma_sjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_svinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_S_ZERO, queue ));


    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrL ));
    CHECK_CUSPARSE( cusparseSetMatType( descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrL, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrL, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrL, CUSPARSE_FILL_MODE_LOWER ));
    cusparseCreateSolveAnalysisInfo( &precond->cuinfoL );
    cusparseScsrsv_analysis( cusparseHandle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, precond->M.num_rows,
                             precond->M.nnz, descrL,
                             precond->M.val, precond->M.row, precond->M.col, 
                             precond->cuinfoL );
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrU ));
    CHECK_CUSPARSE( cusparseSetMatType( descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrU, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrU, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrU, CUSPARSE_FILL_MODE_LOWER ));
    cusparseCreateSolveAnalysisInfo( &precond->cuinfoU );
    cusparseScsrsv_analysis( cusparseHandle,
                             CUSPARSE_OPERATION_TRANSPOSE, precond->M.num_rows,
                             precond->M.nnz, descrU,
                             precond->M.val, precond->M.row, precond->M.col, 
                             precond->cuinfoU );

    
    cleanup:
        
    cusparseDestroy( cusparseHandle );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;    
    magma_smfree( &hA, queue );
    
    return info;
}
    
