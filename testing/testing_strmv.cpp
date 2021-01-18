/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from testing/testing_ztrmv.cpp, normal z -> s, Thu Oct  8 23:05:39 2020
       @author Chongxiao Cao
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing strmv
*/
int main( int argc, char** argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dx(i_)      dx, ((i_))
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dx(i_)     (dx + (i_))
    #endif

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, dev_perf, magma_time, dev_time, cpu_perf, cpu_time;
    float          magma_error, dev_error, work[1];
    magma_int_t N;
    magma_int_t Ak;
    magma_int_t sizeA;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    float *hA, *hx, *hxmagma, *hxdev;
    magmaFloat_ptr dA, dx;
    float c_neg_one = MAGMA_S_NEG_ONE;
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    // Allow 3*eps; real needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    float eps = lapackf77_slamch("E");
    float tol = 3 * eps;

    printf("%% If running lapack (option --lapack), MAGMA and %s errors are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to %s result.\n\n",
            g_platform_str, g_platform_str );

    printf("%% uplo = %s, transA = %s, diag = %s \n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA),
           lapack_diag_const(opts.diag) );
    printf("%%   N    MAGMA Gflop/s (ms)    CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error   CUBLAS error\n");
    printf("%%=================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            gflops = FLOPS_STRMM(opts.side, N, 1) / 1e9;

            lda = N;
            Ak = N;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default

            sizeA = lda*Ak;

            TESTING_CHECK( magma_smalloc_cpu( &hA,      lda*Ak ));
            TESTING_CHECK( magma_smalloc_cpu( &hx,      N      ));
            TESTING_CHECK( magma_smalloc_cpu( &hxmagma, N      ));
            TESTING_CHECK( magma_smalloc_cpu( &hxdev,   N      ));

            TESTING_CHECK( magma_smalloc( &dA, ldda*Ak ));
            TESTING_CHECK( magma_smalloc( &dx, N       ));

            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_slarnv( &ione, ISEED, &N, hx );

            // for error checks
            float Anorm = lapackf77_slange( "F", &N, &N,    hA, &lda, work );
            float Xnorm = lapackf77_slange( "F", &N, &ione, hx, &N,   work );

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetmatrix( Ak, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_ssetvector( N, hx, 1, dx(0), 1, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_strmv( opts.uplo, opts.transA, opts.diag,
                         N,
                         dA(0,0), ldda,
                         dx(0),   1, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;

            magma_sgetvector( N, dx(0), 1, hxmagma, 1, opts.queue );

            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_ssetvector( N, hx, 1, dx(0), 1, opts.queue );

            dev_time = magma_sync_wtime( opts.queue );
            magma_strmv( opts.uplo, opts.transA, opts.diag,
                         N,
                         dA(0,0), ldda,
                         dx(0),   1, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;

            magma_sgetvector( N, dx(0), 1, hxdev, 1, opts.queue );

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_strmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               hA, &lda,
                               hx, &ione );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // See testing_sgemm for formula. Here K = N.
                // initial C = 0, alpha = 1
                blasf77_saxpy( &N, &c_neg_one, hx, &ione, hxmagma, &ione );
                dev_error = lapackf77_slange( "F", &N, &ione, hxmagma, &N, work )
                            / (sqrt(float(N+2))*Anorm*Xnorm);

                blasf77_saxpy( &N, &c_neg_one, hx, &ione, hxdev, &ione );
                dev_error = lapackf77_slange( "F", &N, &ione, hxdev, &N, work )
                            / (sqrt(float(N+2))*Anorm*Xnorm);

                bool okay = (magma_error < tol && dev_error < tol);
                status += ! okay;
                printf("%5lld   %7.2f (%7.2f)    %7.2f (%7.2f)     %7.2f (%7.2f)    %8.2e      %8.2e   %s\n",
                       (long long) N,
                       magma_perf, 1000.*magma_time,
                       dev_perf, 1000.*dev_time,
                       cpu_perf, 1000.*cpu_time,
                       magma_error, dev_error, (okay ? "ok" : "failed"));
            }
            else {
                blasf77_saxpy( &N, &c_neg_one, hxdev, &ione, hxmagma, &ione );
                magma_error = lapackf77_slange( "F", &N, &ione, hxmagma, &N, work )
                            / (sqrt(float(N+2))*Anorm*Xnorm);

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%5lld   %7.2f (%7.2f)    %7.2f (%7.2f)       ---   (  ---  )    %8.2e         ---     %s\n",
                       (long long) N,
                       magma_perf, 1000.*magma_time,
                       dev_perf, 1000.*dev_time,
                       magma_error, (okay ? "ok" : "failed"));
            }

            magma_free_cpu( hA );
            magma_free_cpu( hx );
            magma_free_cpu( hxmagma );
            magma_free_cpu( hxdev );

            magma_free( dA );
            magma_free( dx );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
