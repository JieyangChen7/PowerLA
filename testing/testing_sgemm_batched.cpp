/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from testing/testing_zgemm_batched.cpp, normal z -> s, Thu Oct  8 23:05:45 2020
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
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
   -- Testing sgemm_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    float          error, cublas_error, magma_error, normalize, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    float *h_A, *h_B, *h_C, *h_Cmagma, *h_Ccublas;
    float *d_A, *d_B, *d_C;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    float **h_A_array = NULL;
    float **h_B_array = NULL;
    float **h_C_array = NULL;
    float **d_A_array = NULL;
    float **d_B_array = NULL;
    float **d_C_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check; // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;

    float *Anorm, *Bnorm, *Cnorm;
    TESTING_CHECK( magma_smalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Bnorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Cnorm, batchCount ));

    // See testing_sgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;

    printf("%% If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "%% transA = %s, transB = %s\n",
           lapack_trans_const(opts.transA),
           lapack_trans_const(opts.transB));
    printf("%% version = %lld, %s\n", (long long)opts.version, opts.version == 1 ? "regular batch GEMM" : "strided batch GEMM");
    printf("%% BatchCount     M     N     K   MAGMA Gflop/s (ms)   CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error   CUBLAS error\n");
    printf("%%========================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_SGEMM( M, N, K ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            }
            else {
                lda = Am = K;
                An = M;
            }

            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            }
            else {
                ldb = Bm = N;
                Bn = K;
            }
            ldc = M;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*An*batchCount;
            sizeB = ldb*Bn*batchCount;
            sizeC = ldc*N*batchCount;

            TESTING_CHECK( magma_smalloc_cpu( &h_A,  sizeA ));
            TESTING_CHECK( magma_smalloc_cpu( &h_B,  sizeB ));
            TESTING_CHECK( magma_smalloc_cpu( &h_C,  sizeC  ));
            TESTING_CHECK( magma_smalloc_cpu( &h_Cmagma,  sizeC  ));
            TESTING_CHECK( magma_smalloc_cpu( &h_Ccublas, sizeC  ));

            TESTING_CHECK( magma_smalloc( &d_A, ldda*An*batchCount ));
            TESTING_CHECK( magma_smalloc( &d_B, lddb*Bn*batchCount ));
            TESTING_CHECK( magma_smalloc( &d_C, lddc*N*batchCount  ));

            TESTING_CHECK( magma_malloc_cpu( (void**) &h_A_array, batchCount * sizeof(float*) ));
            TESTING_CHECK( magma_malloc_cpu( (void**) &h_B_array, batchCount * sizeof(float*) ));
            TESTING_CHECK( magma_malloc_cpu( (void**) &h_C_array, batchCount * sizeof(float*) ));

            TESTING_CHECK( magma_malloc( (void**) &d_A_array, batchCount * sizeof(float*) ));
            TESTING_CHECK( magma_malloc( (void**) &d_B_array, batchCount * sizeof(float*) ));
            TESTING_CHECK( magma_malloc( (void**) &d_C_array, batchCount * sizeof(float*) ));

            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_slarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_slarnv( &ione, ISEED, &sizeC, h_C );

            // Compute norms for error
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_slange( "F", &Am, &An, &h_A[s*lda*An], &lda, work );
                Bnorm[s] = lapackf77_slange( "F", &Bm, &Bn, &h_B[s*ldb*Bn], &ldb, work );
                Cnorm[s] = lapackf77_slange( "F", &M,  &N,  &h_C[s*ldc*N],  &ldc, work );
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetmatrix( Am, An*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_ssetmatrix( Bm, Bn*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            magma_ssetmatrix( M, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );

            magma_sset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*An, batchCount, opts.queue );
            magma_sset_pointer( d_B_array, d_B, lddb, 0, 0, lddb*Bn, batchCount, opts.queue );
            magma_sset_pointer( d_C_array, d_C, lddc, 0, 0, lddc*N,  batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            if(opts.version == 1){
                magmablas_sgemm_batched(opts.transA, opts.transB, M, N, K,
                                 alpha, d_A_array, ldda,
                                        d_B_array, lddb,
                                 beta,  d_C_array, lddc, batchCount, opts.queue);
            }
            else{
                magmablas_sgemm_batched_strided(opts.transA, opts.transB, M, N, K,
                                 alpha, d_A, ldda, ldda*An,
                                        d_B, lddb, lddb*Bn,
                                 beta,  d_C, lddc, lddc*N, batchCount, opts.queue);
            }
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_sgetmatrix( M, N*batchCount, d_C, lddc, h_Cmagma, ldc, opts.queue );

            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_ssetmatrix( M, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );

            cublas_time = magma_sync_wtime( opts.queue );

            if(opts.version == 1){
                cublasSgemmBatched(opts.handle, cublas_trans_const(opts.transA), cublas_trans_const(opts.transB),
                                   int(M), int(N), int(K),
                                   (const float*)&alpha,
                                   (const float**) d_A_array, int(ldda),
                                   (const float**) d_B_array, int(lddb),
                                   (const float*)&beta,
                                   (float**)d_C_array, int(lddc), int(batchCount) );
            }
            else{
                cublasSgemmStridedBatched(opts.handle, cublas_trans_const(opts.transA), cublas_trans_const(opts.transB),
                                   int(M), int(N), int(K),
                                   (const float*)&alpha,
                                   (const float*) d_A, int(ldda), ldda * An,
                                   (const float*) d_B, int(lddb), lddb * Bn,
                                   (const float*)&beta,
                                   (float*)d_C, int(lddc), lddc*N, int(batchCount) );
            }

            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;

            magma_sgetmatrix( M, N*batchCount, d_C, lddc, h_Ccublas, ldc, opts.queue );

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                // populate pointer arrays on the host
                for(int s = 0; s < batchCount; s++){
                    h_A_array[s] = h_A + s * lda * An;
                    h_B_array[s] = h_B + s * ldb * Bn;
                    h_C_array[s] = h_C + s * ldc * N;
                }
                cpu_time = magma_wtime();
                blas_sgemm_batched( opts.transA, opts.transB,
                       M, N, K,
                       alpha, h_A_array, lda,
                              h_B_array, ldb,
                       beta,  h_C_array, ldc, batchCount );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute error compared lapack
                // error = |dC - C| / (gamma_{k+2}|A||B| + gamma_2|Cin|)
                magma_error = 0;
                cublas_error = 0;

                for (int s=0; s < batchCount; s++) {
                    normalize = sqrt(float(K+2))*Anorm[s]*Bnorm[s] + 2*Cnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Csize = ldc*N;
                    blasf77_saxpy( &Csize, &c_neg_one, &h_C[s*ldc*N], &ione, &h_Cmagma[s*ldc*N], &ione );
                    error = lapackf77_slange( "F", &M, &N, &h_Cmagma[s*ldc*N], &ldc, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );

                    // cublas error
                    blasf77_saxpy( &Csize, &c_neg_one, &h_C[s*ldc*N], &ione, &h_Ccublas[s*ldc*N], &ione );
                    error = lapackf77_slange( "F", &M, &N, &h_Ccublas[s*ldc*N], &ldc, work )
                          / normalize;
                    cublas_error = magma_max_nan( error, cublas_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld %5lld    %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e      %8.2e   %s\n",
                       (long long) batchCount, (long long) M, (long long) N, (long long) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, cublas_error, (okay ? "ok" : "failed") );
            }
            else {
                // compute error compared cublas
                // error = |dC - C| / (gamma_{k+2}|A||B| + gamma_2|Cin|)
                magma_error = 0;

                for (int s=0; s < batchCount; s++) {
                    normalize = sqrt(float(K+2))*Anorm[s]*Bnorm[s] + 2*Cnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Csize = ldc*N;
                    blasf77_saxpy( &Csize, &c_neg_one, &h_Ccublas[s*ldc*N], &ione, &h_Cmagma[s*ldc*N], &ione );
                    error = lapackf77_slange( "F", &M, &N, &h_Cmagma[s*ldc*N], &ldc, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld %5lld    %7.2f (%7.2f)    %7.2f (%7.2f)     ---   (  ---  )   %8.2e        ---      %s\n",
                       (long long) batchCount, (long long) M, (long long) N, (long long) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       magma_error, (okay ? "ok" : "failed") );
            }

            magma_free_cpu( h_A  );
            magma_free_cpu( h_B  );
            magma_free_cpu( h_C  );
            magma_free_cpu( h_Cmagma  );
            magma_free_cpu( h_Ccublas );
            magma_free_cpu( h_A_array );
            magma_free_cpu( h_B_array );
            magma_free_cpu( h_C_array );

            magma_free( d_A );
            magma_free( d_B );
            magma_free( d_C );
            magma_free( d_A_array );
            magma_free( d_B_array );
            magma_free( d_C_array );

            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free_cpu( Anorm );
    magma_free_cpu( Bnorm );
    magma_free_cpu( Cnorm );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
