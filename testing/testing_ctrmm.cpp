/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from testing/testing_ztrmm.cpp, normal z -> c, Thu Oct  8 23:05:39 2020
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
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ctrmm
*/
int main( int argc, char** argv)
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, dev_perf, dev_time, magma_perf=0, magma_time=0, cpu_perf, cpu_time;
    float          dev_error, magma_error=0, work[1];
    magma_int_t M, N;
    magma_int_t Ak;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldb, ldda, lddb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magmaFloatComplex *hA, *hB, *hBdev, *hBmagma;
    magmaFloatComplex_ptr dA, dB;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  0.29, -0.86 );
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    // See testing_cgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;

    #if defined(HAVE_CUBLAS) || defined(HAVE_HIP)
    // for CUDA, we can check MAGMA vs. CUBLAS, without running LAPACK
    printf("%% If running lapack (option --lapack), MAGMA and %s errors are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to %s result.\n\n",
            g_platform_str, g_platform_str );

    printf("%% side = %s, uplo = %s, transA = %s, diag = %s \n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%%   M     N   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)  CPU Gflop/s (ms)   MAGMA error   CUBLAS error\n");
   #else
    // for others, we need LAPACK for check
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    printf("%% side = %s, uplo = %s, transA = %s, diag = %s \n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%%   M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error\n");
    #endif
    printf("%%===================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_CTRMM(opts.side, M, N) / 1e9;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                Ak = M;
            } else {
                lda = N;
                Ak = N;
            }

            ldb = M;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default

            sizeA = lda*Ak;
            sizeB = ldb*N;

            TESTING_CHECK( magma_cmalloc_cpu( &hA,    lda*Ak ));
            TESTING_CHECK( magma_cmalloc_cpu( &hB,    ldb*N  ));
            TESTING_CHECK( magma_cmalloc_cpu( &hBdev, ldb*N  ));
            TESTING_CHECK( magma_cmalloc_cpu( &hBmagma, ldb*N  ));

            TESTING_CHECK( magma_cmalloc( &dA, ldda*Ak ));
            TESTING_CHECK( magma_cmalloc( &dB, lddb*N  ));

            /* Initialize the matrices */
            lapackf77_clarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_clarnv( &ione, ISEED, &sizeB, hB );

            // for error checks
            float Anorm = lapackf77_clantr( "F", lapack_uplo_const(opts.uplo),
                                                  lapack_diag_const(opts.diag),
                                                  &Ak, &Ak, hA, &lda, work );
            float Bnorm = lapackf77_clange( "F", &M,  &N,  hB, &ldb, work );

            /* =====================================================================
               Performs operation using MAGMABLAS ( for CUDA and HIP )
               =================================================================== */
            magma_csetmatrix( Ak, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_csetmatrix( M,  N,  hB, ldb, dB(0,0), lddb, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_ctrmm( opts.side, opts.uplo, opts.transA, opts.diag,
                         M, N,
                         alpha, dA(0,0), ldda,
                                dB(0,0), lddb, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;

            magma_cgetmatrix( M, N, dB(0,0), lddb, hBmagma, ldb, opts.queue );

            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_csetmatrix( Ak, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_csetmatrix( M,  N,  hB, ldb, dB(0,0), lddb, opts.queue );

            #if defined(HAVE_CUBLAS) || defined(HAVE_HIP)
            // note cublas does trmm out-of-place (i.e., adds output matrix C),
            // but allows C=B to do in-place.
            dev_time = magma_sync_wtime( opts.queue );
            magma_ctrmm( opts.side, opts.uplo, opts.transA, opts.diag,
                         M, N,
                         alpha, dA(0,0), ldda,
                                dB(0,0), lddb, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;

            magma_cgetmatrix( M, N, dB(0,0), lddb, hBdev, ldb, opts.queue );
            #endif

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ctrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &alpha, hA, &lda,
                                       hB, &ldb );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // See testing_cgemm for formula. Here K = Ak.
                blasf77_caxpy( &sizeB, &c_neg_one, hB, &ione, hBmagma, &ione );
                magma_error = lapackf77_clange( "M", &M, &N, hBmagma, &ldb, work )
                            / (sqrt(float(Ak+2))*fabs(alpha)*Anorm*Bnorm);

                #if defined(HAVE_CUBLAS) || defined(HAVE_HIP)
                blasf77_caxpy( &sizeB, &c_neg_one, hB, &ione, hBdev, &ione );
                dev_error = lapackf77_clange( "M", &M, &N, hBdev, &ldb, work )
                            / (sqrt(float(Ak+2))*fabs(alpha)*Anorm*Bnorm);

                bool okay = (magma_error < tol && dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e     %8.2e   %s\n",
                       (long long) M, (long long) N,
                       magma_perf, 1000.*magma_time,
                       dev_perf,   1000.*dev_time,
                       cpu_perf,   1000.*cpu_time,
                       magma_error, dev_error, (okay ? "ok" : "failed"));

                #else
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) M, (long long) N,
                       magma_perf, 1000.*magma_time,
                       cpu_perf,   1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
                #endif

            }
            else {
                #if defined(HAVE_CUBLAS) || defined(HAVE_HIP)
                blasf77_caxpy( &sizeB, &c_neg_one, hBdev, &ione, hBmagma, &ione );
                magma_error = lapackf77_clange( "M", &M, &N, hBmagma, &ldb, work )
                            / (sqrt(float(Ak+2))*fabs(alpha)*Anorm*Bnorm);

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)     ---   (  ---  )    %8.2e         ---      %s\n",
                       (long long) M, (long long) N,
                       magma_perf, 1000.*magma_time,
                       dev_perf,   1000.*dev_time,
                       magma_error, (okay ? "ok" : "failed"));
                #else
                printf("%5lld %5lld   %7.2f (%7.2f)   ---   (  ---  )     ---  \n",
                       (long long) M, (long long) N,
                       magma_perf, 1000.*magma_time);
                #endif
            }

            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hBdev );
            magma_free_cpu( hBmagma );

            magma_free( dA );
            magma_free( dB );
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
