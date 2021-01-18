/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @generated from testing/testing_zhegvdx_2stage.cpp, normal z -> c, Thu Oct  8 23:05:44 2020

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#include "../control/magma_threadsetting.h"  // internal header

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chegvdx
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    /* Constants */
    const magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    const magma_int_t ione = 1;
    
    /* Local variables */
    real_Double_t gpu_time;

    magmaFloatComplex *h_A, *h_R, *h_B, *h_S, *h_work;

    #ifdef COMPLEX
    float *rwork;
    magma_int_t lrwork;
    #endif

    float *w1, *w2, result[2]={0,0};
    magma_int_t *iwork;
    magma_int_t N, Nfound, n2, info, lda, lwork, liwork;
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol    = opts.tolerance * lapackf77_slamch("E");
    float tolulp = opts.tolerance * lapackf77_slamch("P");

    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% itype = %lld, jobz = %s, uplo = %s, ngpu = %lld\n",
           (long long) opts.itype, lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
           (long long) abs_ngpu);

    if (opts.itype == 1) {
        printf("%%   N Nfound  GPU Time (sec)   |AZ-BZD|   |D - D_magma|\n");
    }                                                   
    else if (opts.itype == 2) {                      
        printf("%%   N Nfound  GPU Time (sec)   |ABZ-ZD|   |D - D_magma|\n");
    }                                                   
    else if (opts.itype == 3) {                      
        printf("%%   N Nfound  GPU Time (sec)   |BAZ-ZD|   |D - D_magma|\n");
    }                                     
        printf("%%======================================================\n");
    magma_int_t threads = magma_get_parallel_numthreads();
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda = N;
            n2  = lda*N;
            
            magma_range_t range;
            magma_int_t il, iu;
            float vl, vu;
            opts.get_range( N, &range, &vl, &vu, &il, &iu );
            
            magma_cheevdx_getworksize(N, threads, (opts.jobz == MagmaVec),
                                     &lwork,
                                     #ifdef COMPLEX
                                     &lrwork,
                                     #endif
                                     &liwork);
            /* Allocate host memory for the matrix */
            TESTING_CHECK( magma_cmalloc_cpu( &h_A,    n2 ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_B,    n2 ));
            TESTING_CHECK( magma_smalloc_cpu( &w1,     N ));
            TESTING_CHECK( magma_smalloc_cpu( &w2,     N ));
            TESTING_CHECK( magma_imalloc_cpu( &iwork,  liwork ));
            
            TESTING_CHECK( magma_cmalloc_pinned( &h_R,    n2 ));
            TESTING_CHECK( magma_cmalloc_pinned( &h_S,    n2 ));
            TESTING_CHECK( magma_cmalloc_pinned( &h_work, max( lwork, N*N ) ));  // check needs N*N
            #ifdef COMPLEX
            TESTING_CHECK( magma_smalloc_pinned( &rwork,  lrwork ));
            #endif

            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda );
            magma_generate_matrix( opts, N, N, h_B, lda );
            magma_cmake_hpd( N, h_B, lda );
            magma_cmake_hermitian( N, h_A, lda );

            lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            lapackf77_clacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );

            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================
            gpu_time = magma_wtime();
            if (opts.ngpu == 1) {
                magma_chegvdx_2stage( opts.itype, opts.jobz, range, opts.uplo,
                                      N, h_R, lda, h_S, lda, vl, vu, il, iu, &Nfound, w1,
                                      h_work, lwork,
                                      #ifdef COMPLEX
                                      rwork, lrwork,
                                      #endif
                                      iwork, liwork,
                                      &info );
            }
            else {
                magma_chegvdx_2stage_m( abs_ngpu, opts.itype, opts.jobz, range, opts.uplo,
                                        N, h_R, lda, h_S, lda, vl, vu, il, iu, &Nfound, w1,
                                        h_work, lwork,
                                        #ifdef COMPLEX
                                        rwork, lrwork,
                                        #endif
                                        iwork, liwork,
                                        &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_chegvdx_2stage returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zc]hegvdx routine.
                   A x = lambda B x is solved
                   and the following 3 tests computed:
                   (1)    | A Z - B Z D | / ( |A| |Z| N )  (itype = 1)
                          | A B Z - Z D | / ( |A| |Z| N )  (itype = 2)
                          | B A Z - Z D | / ( |A| |Z| N )  (itype = 3)
                   (2)    | D(with V, magma) - D(w/o V, lapack) | / | D |
                   =================================================================== */
                #ifdef REAL
                float *rwork = h_work + N*N;
                #endif
                
                if ( opts.jobz != MagmaNoVec ) {
                    result[0] = 1.;
                    result[0] /= safe_lapackf77_clanhe("1", lapack_uplo_const(opts.uplo), &N, h_A, &lda, rwork);
                    result[0] /= lapackf77_clange("1", &N, &Nfound, h_R, &lda, rwork);
                    
                    if (opts.itype == 1) {
                        blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &Nfound, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                        for (int i=0; i < Nfound; ++i)
                            blasf77_csscal(&N, &w1[i], &h_R[i*N], &ione);
                        blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &Nfound, &c_neg_one, h_B, &lda, h_R, &lda, &c_one, h_work, &N);
                        result[0] *= lapackf77_clange("1", &N, &Nfound, h_work, &N, rwork)/N;
                    }
                    else if (opts.itype == 2) {
                        blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &Nfound, &c_one, h_B, &lda, h_R, &lda, &c_zero, h_work, &N);
                        for (int i=0; i < Nfound; ++i)
                            blasf77_csscal(&N, &w1[i], &h_R[i*N], &ione);
                        blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &Nfound, &c_one, h_A, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                        result[0] *= lapackf77_clange("1", &N, &Nfound, h_R, &lda, rwork)/N;
                    }
                    else if (opts.itype == 3) {
                        blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &Nfound, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                        for (int i=0; i < Nfound; ++i)
                            blasf77_csscal(&N, &w1[i], &h_R[i*N], &ione);
                        blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &Nfound, &c_one, h_B, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                        result[0] *= lapackf77_clange("1", &N, &Nfound, h_R, &lda, rwork)/N;
                    }
                }
                
                lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
                lapackf77_clacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );
                
                lapackf77_chegvd( &opts.itype, "N", lapack_uplo_const(opts.uplo), &N,
                                  h_R, &lda, h_S, &lda, w2,
                                  h_work, &lwork,
                                  #ifdef COMPLEX
                                  rwork, &lrwork,
                                  #endif
                                  iwork, &liwork,
                                  &info );
                if (info != 0) {
                    printf("lapackf77_chegvd returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                float maxw=0, diff=0;
                for (int j=0; j < Nfound; j++) {
                    maxw = max(maxw, fabs(w1[j]));
                    maxw = max(maxw, fabs(w2[j]));
                    diff = max(diff, fabs(w1[j] - w2[j]));
                }
                result[1] = diff / (Nfound*maxw);
            }
            
            /* =====================================================================
               Print execution time
               =================================================================== */
            printf("%5lld %5lld   %9.4f     ",
                   (long long) N, (long long) Nfound, gpu_time);
            if ( opts.check ) {
                bool okay = (result[1] < tolulp);
                if ( opts.jobz != MagmaNoVec ) {
                    okay = okay && (result[0] < tol);
                    printf("   %8.2e", result[0] );
                }
                else {
                    printf("     ---   ");
                }
                printf("        %8.2e  %s\n", result[1], (okay ? "ok" : "failed"));
                status += ! okay;
            }
            else {
                printf("     ---\n");
            }
            
            magma_free_cpu( h_A   );
            magma_free_cpu( h_B   );
            magma_free_cpu( w1    );
            magma_free_cpu( w2    );
            magma_free_cpu( iwork );
            
            magma_free_pinned( h_R );
            magma_free_pinned( h_S );
            magma_free_pinned( h_work );
            #ifdef COMPLEX
            magma_free_pinned( rwork );
            #endif
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
