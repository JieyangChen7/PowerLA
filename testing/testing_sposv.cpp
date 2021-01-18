/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from testing/testing_zposv.cpp, normal z -> s, Thu Oct  8 23:05:40 2020
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sposv
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    // constants
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t ione = 1;
    
    // locals
    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    float          error, Rnorm, Anorm, Xnorm, *work, *sigma;
    float *h_A, *h_R, *h_B, *h_X;
    magma_int_t N, lda, ldb, info, sizeB;
    int status = 0;
    
    magma_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("%% ngpu = %lld, uplo = %s\n", (long long) opts.ngpu, lapack_uplo_const(opts.uplo) );
    printf("%%   N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = ldb = N;
            gflops = ( FLOPS_SPOTRF( N ) + FLOPS_SPOTRS( N, opts.nrhs ) ) / 1e9;
            
            TESTING_CHECK( magma_smalloc_cpu( &h_A, lda*N         ));
            TESTING_CHECK( magma_smalloc_cpu( &h_R, lda*N         ));
            TESTING_CHECK( magma_smalloc_cpu( &h_B, ldb*opts.nrhs ));
            TESTING_CHECK( magma_smalloc_cpu( &h_X, ldb*opts.nrhs ));
            TESTING_CHECK( magma_smalloc_cpu( &work, N ));
            TESTING_CHECK( magma_smalloc_cpu( &sigma, N ));
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            sizeB = ldb*opts.nrhs;
            magma_generate_matrix( opts, N, N, h_A, lda, sigma );
            lapackf77_slarnv( &ione, opts.iseed, &sizeB, h_B );
            
            // copy A to R and B to X; save A and B for residual
            lapackf77_slacpy( MagmaFullStr, &N, &N,         h_A, &lda, h_R, &lda );
            lapackf77_slacpy( MagmaFullStr, &N, &opts.nrhs, h_B, &ldb, h_X, &ldb );
            
            if (opts.verbose) {
                printf( "A = " ); magma_sprint( N, N, h_A, lda );
                printf( "B = " ); magma_sprint( N, opts.nrhs, h_B, ldb );
                printf( "S = " ); magma_sprint( N, 1, sigma, N );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_sposv( opts.uplo, N, opts.nrhs, h_R, lda, h_X, ldb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_sposv returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            if (opts.verbose) {
                printf( "L = " ); magma_sprint( N, N, h_R, lda );
                printf( "X = " ); magma_sprint( N, opts.nrhs, h_X, ldb );
            }
            
            /* =====================================================================
               Residual
               =================================================================== */
            Anorm = lapackf77_slange("I", &N, &N,         h_A, &lda, work);
            Xnorm = lapackf77_slange("I", &N, &opts.nrhs, h_X, &ldb, work);
            
            blasf77_ssymm( MagmaLeftStr, lapack_uplo_const(opts.uplo), &N, &opts.nrhs,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb );
            
            Rnorm = lapackf77_slange("I", &N, &opts.nrhs, h_B, &ldb, work);
            error = Rnorm/(N*Anorm*Xnorm);
            status += ! (error < tol);
            
            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_sposv( lapack_uplo_const(opts.uplo), &N, &opts.nrhs, h_A, &lda, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sposv returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                if (opts.verbose) {
                    printf( "Lref = " ); magma_sprint( N, N, h_A, lda );
                    printf( "Xref = " ); magma_sprint( N, opts.nrhs, h_B, ldb );
                }
                
                printf( "%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) opts.nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
            }
            else {
                printf( "%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) opts.nrhs, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_R );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( sigma );
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
