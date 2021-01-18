/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from testing/testing_zgeam.cpp, normal z -> c, Thu Oct  8 23:05:40 2020
       @author Stan Tomov
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgeadd
*/
int main( int argc, char** argv)
{
    #define h_A(i_, j_) (h_A + (i_) + (j_)*lda)
    #define h_B(i_, j_) (h_B + (i_) + (j_)*ldb)
    #define h_C(i_, j_) (h_C + (i_) + (j_)*ldc)
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float          error, work[1];
    magmaFloatComplex *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    magmaFloatComplex alpha = MAGMA_C_MAKE( 3.1415, 2.71828 );
    magmaFloatComplex beta  = MAGMA_C_MAKE( 6.0221, 6.67408 );
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    magma_int_t M, N, size;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    float tol = opts.tolerance * lapackf77_slamch("E");

    printf("%% transA = %s, transB = %s\n",
           lapack_trans_const(opts.transA),
           lapack_trans_const(opts.transB) );
    
    printf("%%   M     N      CPU GB/s (ms)      GPU GB/s (ms)    |R|/a|A|+b|B|\n");
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            
            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = N;
            } else {
                lda = Am = N;
                An = M;
            }

            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = M;
                Bn = N;
            } else {
                ldb = Bm = N;
                Bn = M;
            }
            
            ldc = M;

            ldda   = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb   = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc   = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            size  = M*N;

            gbytes = 3.*sizeof(magmaFloatComplex)*M*N / 1e9;
            
            TESTING_CHECK( magma_cmalloc_cpu( &h_A, lda *An ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_B, ldb *Bn ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_C, ldc * N ));
            
            TESTING_CHECK( magma_cmalloc( &d_A, ldda*An ));
            TESTING_CHECK( magma_cmalloc( &d_B, lddb*Bn ));
            TESTING_CHECK( magma_cmalloc( &d_C, lddc*N  ));
            
            lapackf77_clarnv( &ione, ISEED, &size, h_A );
            lapackf77_clarnv( &ione, ISEED, &size, h_B );
            
            // for error checks
            float Anorm = lapackf77_clange( "F", &Am, &An, h_A, &lda, work );
            float Bnorm = lapackf77_clange( "F", &Bm, &Bn, h_B, &ldb, work );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_csetmatrix( Am, An, h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( Bm, Bn, h_B, ldb, d_B, lddb, opts.queue );

            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_cgeam( opts.transA, opts.transB, M, N, 
                             alpha, d_A, ldda, 
                             beta , d_B, lddb, 
                             d_C, lddc, opts.queue );
            /*
            cublasCgeam(opts.handle, 
                        cublas_trans_const(opts.transA), cublas_trans_const(opts.transB), 
                        M, N,
                        &alpha, d_A, ldda,
                        &beta , d_B, lddb,
                        d_C, lddc);
            */
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Performs operation on the CPU
               =================================================================== */
            cpu_time = magma_wtime();
            if ( opts.transA == MagmaNoTrans && opts.transB == MagmaNoTrans )                
                for( int j = 0; j < N; ++j ) {
                    for( int i=0; i < M; ++i ) {
                        *h_C(i,j) = alpha * (*h_A(i,j)) + beta * (*h_B(i,j));
                    }
                }
            else if (opts.transA == MagmaNoTrans)
                for( int j = 0; j < N; ++j ) {
                    for( int i=0; i < M; ++i ) {
                        *h_C(i,j) = alpha * (*h_A(i,j)) + beta * (*h_B(j,i));
                    }
                }
            else if (opts.transB == MagmaNoTrans)
                for( int j = 0; j < N; ++j ) {
                    for( int i=0; i < M; ++i ) {
                        *h_C(i,j) = alpha * (*h_A(j,i)) + beta * (*h_B(i,j));
                    }
                }
            else
                for( int j = 0; j < N; ++j ) {
                    for( int i=0; i < M; ++i ) {
                        *h_C(i,j) = alpha * (*h_A(j,i)) + beta * (*h_B(j,i));
                    }
                }
            
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* =====================================================================
               Check result
               =================================================================== */
            magma_cgetmatrix( M, N, d_C, lddc, h_A, M, opts.queue );
            
            blasf77_caxpy( &size, &c_neg_one, h_C, &ione, h_A, &ione );
            error = lapackf77_clange( "F", &M, &N, h_A, &M, work )/ (fabs(alpha)*Anorm+fabs(beta)*Bnorm);

            
            printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                   (long long) M, (long long) N,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   error, (error < tol ? "ok" : "failed"));
            status += ! (error < tol);
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_C );
            
            magma_free( d_A );
            magma_free( d_B );
            magma_free( d_C );
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
