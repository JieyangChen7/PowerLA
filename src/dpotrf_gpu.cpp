/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
       @generated from src/zpotrf_gpu.cpp, normal z -> d, Thu Oct  8 23:05:22 2020
*/
#include "../power_adjustment/power_adjustment.h"
#include "magma_internal.h"
#include "cuda_runtime.h"
#include "../fault_tolerance/abft_printer.h"
#include "../fault_tolerance/abft_encoder.h"
#include "../fault_tolerance/abft_kernels.h"

// === Define what BLAS to use ============================================
    #undef  magma_dtrsm
    #define magma_dtrsm magmablas_dtrsm
// === End defining what BLAS to use =======================================

/***************************************************************************//**
    Purpose
    -------
    DPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix dA.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @param[in]
    mode    magma_mode_t
      -     = MagmaNative:  Factorize dA using GPU only mode (only uplo=MagmaLower is available);
      -     = MagmaHybrid:  Factorize dA using Hybrid (CPU/GPU) mode.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_dpotrf_LL_expert_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info, magma_mode_t mode )
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda + dA_offset)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    /* Define for ABFT */
    #define dA_colchk(i_, j_)   (dA_colchk   + ((i_)/nb)*2 + (j_)*ldda_colchk)
    #define dA_rowchk(i_, j_)   (dA_rowchk   + (i_)        + ((j_)/nb)*2*ldda_rowchk)
    #define dA_colchk_r(i_, j_) (dA_colchk_r + ((i_)/nb)*2 + (j_)*ldda_colchk_r)
    #define dA_rowchk_r(i_, j_) (dA_rowchk_r + (i_)        + ((j_)/nb)*2*ldda_rowchk_r)

    /* Constants */
    const double c_one     = MAGMA_D_ONE;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const double d_one     =  1.0;
    const double d_neg_one = -1.0;
    
    /* Local variables */
    magma_int_t j, jb, nb, recnb;
    double *work;
    magma_int_t *dinfo;

    *info = 0;
    if (uplo != MagmaUpper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    nb = magma_get_dpotrf_nb( n );
    nb = 512;
    recnb = 128;
    //nb = 4;
    if (mode == MagmaHybrid) {
        if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, nb*nb )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }
    }
    else {
        if (MAGMA_SUCCESS != magma_imalloc( &dinfo, 1 ) ) {
            /* alloc failed for workspace */
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }
    }
    
    magma_queue_t queues[3];
    magma_event_t events[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_queue_create( cdev, &queues[2] );
    magma_event_create(&events[0]);
    magma_event_create(&events[1]);
    if (mode == MagmaNative)
        magma_setvector( 1, sizeof(magma_int_t), info, 1, dinfo, 1, queues[0]);
    
    if (uplo == MagmaUpper) {
        //=========================================================
        /* Compute the Cholesky factorization A = U'*U. */
        for (j=0; j < n; j += nb) {
            // apply all previous updates to diagonal block,
            // then transfer it to CPU
            jb = min( nb, n-j );
            magma_dsyrk( MagmaUpper, MagmaConjTrans, jb, j,
                         d_neg_one, dA(0, j), ldda,
                         d_one,     dA(j, j), ldda, queues[1] );
            
            if (mode == MagmaHybrid) {
                magma_queue_sync( queues[1] );
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
            }
            else {
                //Azzam: need to add events to allow overlap
                magma_dpotrf_rectile_native(MagmaUpper, jb, recnb,
                                            dA(j, j), ldda, j,
                                            dinfo, info, queues[1] );
            }

            
            // apply all previous updates to block row right of diagonal block
            if (j+jb < n) {
                magma_dgemm( MagmaConjTrans, MagmaNoTrans,
                             jb, n-j-jb, j,
                             c_neg_one, dA(0, j   ), ldda,
                                        dA(0, j+jb), ldda,
                             c_one,     dA(j, j+jb), ldda, queues[1] );
            }
            
            // simultaneous with above dgemm, transfer diagonal block,
            // factor it on CPU, and test for positive definiteness
            if (mode == MagmaHybrid) {
                magma_queue_sync( queues[0] );
                lapackf77_dpotrf( MagmaUpperStr, &jb, work, &jb, info );
                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[1] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
            }
            
            // apply diagonal block to block row right of diagonal block
            if (j+jb < n) {
                magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                             jb, n-j-jb,
                             c_one, dA(j, j),    ldda,
                                    dA(j, j+jb), ldda, queues[1] );
            }
        }
    }
    else {
        //=========================================================

        if (nvmlInit () != NVML_SUCCESS)
        {
            printf("init error\n");
            return 0;
        }
        int i = 0;
        nvmlReturn_t result;
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        {
          printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
          return 0;
        }

        pm_cpu();
        pm_gpu(device);

        double total_time = 0.0;
        double pd_time_actual = 0.0, tmu_time_actual = 0.0;
        double pd_time_prof = 0.0, tmu_time_prof = 0.0;
        double pd_time_ratio = 1, tmu_time_ratio = 1;
        double pd_time_pred = 0.0, tmu_time_pred = 0.0;
        double pd_avg_error = 0.0, tmu_avg_error = 0.0; 
        double tmu_ft_time_actual = 0.0;

        double slack_time;
        double reclaimnation_ratio = 0.8;

        int tmu_desired_freq;
        int tmu_freq;
        int tmu_curr_freq = 1500;
        int tmu_base_freq = 1500;

        int tmu_desired_offset;
        int tmu_offset;
        int tmu_curr_offset = -500;
        int tmu_base_offset = -500;
        adj_gpu(device, tmu_base_freq, 338000, tmu_base_offset); // lock frequency is necessary for accurate prediction

        int pd_desired_freq;
        int pd_freq;
        int pd_curr_freq = 3500;
        int pd_base_freq = 3500;
        adj_cpu(pd_base_freq);


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEvent_t start_ft, stop_ft;
        cudaEventCreate(&start_ft);
        cudaEventCreate(&stop_ft);

        unsigned long long start_energy;
        int pid;

        bool reclaim_slack = false;
        int profile_interval = 5;
        int last_prof_iter = 1;

        /* flags */
        bool COL_FT = true;
        bool ROW_FT = true;
        bool DEBUG = false;
        bool CHECK_BEFORE = false;
        bool CHECK_AFTER = true;

        /* matrix sizes to be checksumed */
        int cpu_row = nb;
        int cpu_col = nb;
        int gpu_row = n;
        int gpu_col = n;
        
        printf( "initialize checksum vector on CPU\n");
        double * chk_v;
        int ld_chk_v = nb;
        magma_dmalloc_pinned(&chk_v, nb * 2 * sizeof(double));
        for (int i = 0; i < nb; ++i) {
            *(chk_v + i) = 1;
        }
        for (int i = 0; i < nb; ++i) {
            *(chk_v + ld_chk_v + i) = i + 1;
        }

        if (DEBUG) {
            printf("checksum vector on CPU:\n");
            printMatrix_host(chk_v, ld_chk_v, nb, 2, -1, -1);
        }

        printf( "initialize checksum vector on GPUs\n");
        //printf("%d\n", nb);
        double * dev_chk_v;
        size_t pitch_dev_chk_v = magma_roundup(nb * sizeof(double), 32);
        int ld_dev_chk_v;
        
        magma_dmalloc(&dev_chk_v, pitch_dev_chk_v * 2);
        ld_dev_chk_v = pitch_dev_chk_v / sizeof(double);
        magma_dgetmatrix(nb, 2,
                         chk_v, ld_chk_v, 
                         dev_chk_v, ld_dev_chk_v,
                         queues[1]);
        if (DEBUG) {
            printMatrix_gpu(dev_chk_v, ld_dev_chk_v,
                            nb, 2, nb, nb, queues[1]);
        }


        printf( "allocate space for checksum on CPU......\n" );
        double * colchk;
        double * colchk_r;
        magma_dmalloc_pinned(&colchk, (cpu_row / nb) * 2 * cpu_col * sizeof(double));
        int ld_colchk = (cpu_row / nb) * 2;
        magma_dmalloc_pinned(&colchk_r, (cpu_row / nb) * 2 * cpu_col * sizeof(double));
        int ld_colchk_r = (cpu_row / nb) * 2;
        printf( "done.\n" );

        double * rowchk;
        double * rowchk_r;
        magma_dmalloc_pinned(&rowchk, cpu_row * (cpu_col / nb) * 2 * sizeof(double));
        int ld_rowchk = cpu_row;
        magma_dmalloc_pinned(&rowchk_r, cpu_row * (cpu_col / nb) * 2 * sizeof(double));
        int ld_rowchk_r = cpu_row;
        printf( "done.\n" );

        /* allocate space for col checksum on GPU */
        printf( "allocate space for checksums on GPUs......\n" );
        
        double * dA_colchk;
        size_t pitch_dA_colchk = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
        int ldda_colchk = pitch_dA_colchk / sizeof(double);
        magma_dmalloc(&dA_colchk, pitch_dA_colchk * gpu_col);

        double * dA_colchk_r;
        size_t pitch_dA_colchk_r = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
        int ldda_colchk_r = pitch_dA_colchk_r / sizeof(double);
        magma_dmalloc(&dA_colchk_r, pitch_dA_colchk_r * gpu_col);

        double * dA_rowchk;
        size_t pitch_dA_rowchk = magma_roundup(gpu_row * sizeof(double), 32);
        int ldda_rowchk = pitch_dA_rowchk / sizeof(double);
        magma_dmalloc(&dA_rowchk, pitch_dA_rowchk * (gpu_col / nb) * 2);


        double * dA_rowchk_r;
        size_t pitch_dA_rowchk_r = magma_roundup(gpu_row * sizeof(double), 32);
        int ldda_rowchk_r = pitch_dA_rowchk_r / sizeof(double);
        magma_dmalloc(&dA_rowchk_r, pitch_dA_rowchk_r * (gpu_col / nb) * 2);
           
        printf( "done.\n" );

       
        printf( "calculate initial checksum on GPUs......\n" ); 
        col_chk_enc(gpu_row, gpu_col, nb, 
                    dA, ldda,  
                    dev_chk_v, ld_dev_chk_v, 
                    dA_colchk, ldda_colchk, 
                    queues[1]);

        row_chk_enc(gpu_row, gpu_col, nb, 
                    dA, ldda,  
                    dev_chk_v, ld_dev_chk_v, 
                    dA_rowchk, ldda_rowchk, 
                    queues[1]);

        printf( "done.\n" );

        if (DEBUG) {

            printf( "input matrix A:\n" );
            printMatrix_gpu(dA, ldda, gpu_row, gpu_col, nb, nb, queues[1]);
            printf( "column chk:\n" );
            printMatrix_gpu(dA_colchk, ldda_colchk, 
                            (gpu_row / nb) * 2, gpu_col, 2, nb, queues[1]);
            printf( "row chk:\n" );
            printMatrix_gpu(dA_rowchk, ldda_rowchk,  
                            gpu_row, (gpu_col / nb) * 2, nb, 2, queues[1]);
        }
        cudaDeviceSynchronize();
        pid = start_measure_cpu();
        start_energy = start_measure_gpu(device);
        // Compute the Cholesky factorization A = L*L'.
        total_time = magma_wtime();
        for (j=0; j < n; j += nb) {
            jb = min( nb, n-j );
            // apply all previous updates to diagonal block,
            // then transfer it to CPU
            bool profile = false;
            if ((j/jb-last_prof_iter)%profile_interval == 0) profile = true;
            bool predict = false;
            if (j/jb > 1) predict = true;

            //prediction
            if (predict) {
                pd_time_ratio = 1.0;
                // tmu_time_ratio = ((double)(n-j-jb)*jb*j)/((double)(n-jb*last_prof_iter-jb)*jb*jb*last_prof_iter);
                tmu_time_ratio = (double)abft_dgemm_flops(MagmaNoTrans, MagmaConjTrans, n-j-jb, jb, j, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/
                        abft_dgemm_flops(MagmaNoTrans, MagmaConjTrans,n-jb*last_prof_iter-jb, jb, jb*last_prof_iter, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER);


                //printf("%f\n", tmu_time_ratio);

                pd_time_pred = pd_time_prof * pd_time_ratio;
                
                tmu_time_pred = tmu_time_prof * tmu_time_ratio;

                slack_time = tmu_time_pred - pd_time_pred;

                tmu_desired_freq = tmu_time_pred/(tmu_time_pred-(reclaimnation_ratio*slack_time)) * tmu_base_freq;
                tmu_freq = tmu_desired_freq;
                if (tmu_desired_freq > 2000) tmu_freq = 2000;
                if (tmu_desired_freq < 500) tmu_freq = 500;

                tmu_desired_offset = 200;
                tmu_offset = tmu_desired_offset;

                pd_desired_freq = pd_time_pred/(pd_time_pred+((1-reclaimnation_ratio)*slack_time)) * pd_base_freq;
                pd_freq = pd_desired_freq;
                if (pd_desired_freq > 3500) pd_freq = 3500;
                if (pd_desired_freq < 1000) pd_freq = 1000;
            }

            // printf("j = %d\n", j);


            
            // magma_dsyrk( MagmaLower, MagmaNoTrans, jb, j,
            //              d_neg_one, dA(j, 0), ldda,
            //              d_one,     dA(j, j), ldda, queues[0] );
            // printf("calling abft_dsyrk\n");

            //double * t = dA_colchk ; // + ((j)/nb)*2 + (0)*ldda_colchk; //dA_colchk(j, 0);
            
            abft_dsyrk( MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dA(j, 0), ldda,
                            d_one,     dA(j, j), ldda,
                            nb,
                            dA_colchk(j, 0),    ldda_colchk,
                            dA_rowchk(j, 0),    ldda_rowchk,
                            dA_colchk_r(j, 0),  ldda_colchk_r,
                            dA_rowchk_r(j, 0),  ldda_rowchk_r,
                            dA_colchk(j, j),    ldda_colchk,
                            dA_rowchk(j, j),    ldda_rowchk,
                            dA_colchk_r(j, j),  ldda_colchk_r,
                            dA_rowchk_r(j, j),  ldda_rowchk_r,
                            dev_chk_v,          ld_dev_chk_v, 
                            COL_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                            queues[0], queues[0]);


            // Azzam: this section of "ifthenelse" can be moved down to the factorize section and I don't think performane wil be affected.
            if (mode == MagmaHybrid) {
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
                magma_dgetmatrix_async( 2, jb,
                                        dA_colchk(j, j), ldda_colchk,
                                        colchk,     ld_colchk, queues[0] );

                magma_dgetmatrix_async( jb, 2,
                                        dA_rowchk(j, j), ldda_rowchk,
                                        rowchk,     ld_rowchk, queues[0] );
            }
            else {
                magma_dpotrf_rectile_native(MagmaLower, jb, recnb,
                                            dA(j, j), ldda, j,
                                            dinfo, info, queues[0] );
            }
            
            // apply all previous updates to block column below diagonal block
            if (j+jb < n) {
                magma_queue_wait_event(queues[1], events[0]);
                if (reclaim_slack && j > jb) {
                    if (tmu_freq != tmu_curr_freq || tmu_offset != tmu_curr_offset) {
                        adj_gpu(device, tmu_freq, 338000, tmu_offset);
                        //printf("set to %d\n", tmu_freq);
                        tmu_curr_freq = tmu_freq;
                        tmu_curr_offset = tmu_offset;
                    }
                }
                
               // start_energy = start_measure_gpu(device);
               // double tmu_time = magma_wtime();
                cudaEventRecord(start, queues[1]->cuda_stream());
                //cudaEventRecord(start_ft, queues[2]->cuda_stream());
                // magma_dgemm( MagmaNoTrans, MagmaConjTrans,
                //              n-j-jb, jb, j,
                //              c_neg_one, dA(j+jb, 0), ldda,
                //                         dA(j,    0), ldda,
                //              c_one,     dA(j+jb, j), ldda, queues[1] );
                abft_dgemm( MagmaNoTrans, MagmaConjTrans,
                                n-j-jb, jb, j,
                                c_neg_one, dA(j+jb, 0), ldda,
                                           dA(j,    0), ldda,
                                c_one,     dA(j+jb, j), ldda,
                                nb,
                                dA_colchk(j+jb, 0),   ldda_colchk,
                                dA_rowchk(j+jb, 0),   ldda_rowchk,
                                dA_colchk_r(j+jb, 0), ldda_colchk_r,
                                dA_rowchk_r(j+jb, 0), ldda_rowchk_r,

                                dA_colchk(j,    0),   ldda_colchk,
                                dA_rowchk(j,    0),   ldda_rowchk,
                                dA_colchk_r(j,    0), ldda_colchk_r,
                                dA_rowchk_r(j,    0), ldda_rowchk_r,

                                dA_colchk(j+jb, j),   ldda_colchk,
                                dA_rowchk(j+jb, j),   ldda_rowchk,
                                dA_colchk_r(j+jb, j), ldda_colchk_r,
                                dA_rowchk_r(j+jb, j), ldda_rowchk_r,
                                dev_chk_v,          ld_dev_chk_v, 
                                COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                                queues[1], queues[1]);

                cudaEventRecord(stop, queues[1]->cuda_stream());
                //cudaEventRecord(stop_ft, queues[2]->cuda_stream());
                magma_event_record(events[1], queues[1]);

                // for timing only
                // magma_queue_sync( queues[1] );
                // magma_queue_sync( queues[2] );
                // tmu_time = magma_wtime() - tmu_time;
                // printf("%f\n", tmu_time);
                //start_energy = stop_measure_gpu(device, start_energy);

                
            }
            

            // simultaneous with above dgemm, transfer diagonal block,
            // factor it on CPU, and test for positive definiteness
            // Azzam: The above section can be moved here the code will look cleaner.
            if (mode == MagmaHybrid) {
                magma_queue_sync( queues[0] );
                if (reclaim_slack && j > jb) {
                    if (pd_freq != pd_curr_freq) {
                        adj_cpu(pd_freq);
                        pd_curr_freq = pd_freq;
                    }
                }
                
                pd_time_actual = magma_wtime();
                lapackf77_dpotrf( MagmaLowerStr, &jb, work, &jb, info );
                pd_time_actual = magma_wtime() - pd_time_actual;
                pd_time_actual *= 1000;
                if (profile) pd_time_prof = pd_time_actual;
                if (predict) pd_avg_error += fabs(pd_time_actual - pd_time_pred) / pd_time_actual;

                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[0] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
            }
            
            // apply diagonal block to block column below diagonal
            if (j+jb < n) {

                //profile dgemm
                
                // magma_queue_sync( queues[2] );
                cudaEventSynchronize(stop);
                //tmu_time = magma_wtime() - tmu_time;
                float t;
                cudaEventElapsedTime(&t, start, stop);
                tmu_time_actual = t;
                if (profile) tmu_time_prof = tmu_time_actual;
                if (predict) tmu_avg_error += fabs(tmu_time_actual - tmu_time_pred) / tmu_time_actual;

                // cudaEventSynchronize(stop_ft);
                // cudaEventElapsedTime(&t, start_ft, stop_ft);
                // tmu_ft_time_actual = t;
                // printf("%f\n",tmu_ft_time_actual);


                magma_queue_wait_event(queues[0], events[1]);
                // magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                //              n-j-jb, jb,
                //              c_one, dA(j,    j), ldda,
                //                     dA(j+jb, j), ldda, queues[0] );
                abft_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                 n-j-jb, jb,
                                 c_one, dA(j,    j), ldda,
                                        dA(j+jb, j), ldda,
                                nb,
                                dA_colchk(j,    j),   ldda_colchk,
                                dA_rowchk(j,    j),   ldda_rowchk,
                                dA_colchk_r(j,    j), ldda_colchk_r,
                                dA_rowchk_r(j,    j), ldda_rowchk_r,
                                dA_colchk(j+jb, j),   ldda_colchk,
                                dA_rowchk(j+jb, j),   ldda_rowchk,
                                dA_colchk_r(j+jb, j), ldda_colchk_r,
                                dA_rowchk_r(j+jb, j), ldda_rowchk_r,
                                dev_chk_v,          ld_dev_chk_v, 
                                COL_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                                queues[0], queues[0]);
                magma_event_record(events[0], queues[0]);
            }
            if (profile) last_prof_iter = j/jb;
            // printf("%d, pd-tmu, %f, %f, %f, %f, %f, %f, %d, %d, \n", j, pd_time_prof, tmu_time_prof, pd_time_actual, tmu_time_actual, pd_time_pred, tmu_time_pred, tmu_freq, pd_freq);
        
        }
        total_time = magma_wtime() - total_time;
        stop_measure_cpu(pid);
        start_energy = stop_measure_gpu(device, start_energy);
        printf("GPU energy: %llu\n", start_energy);
        printf("Prediction average error: CPU %f, GPU %f\n", pd_avg_error/(n/jb-1), tmu_avg_error/(n/jb-1));
        printf("Total time: %f\n", total_time);
        //reset
        adj_gpu(device, 1500, 338000, -500);
    }
    if (mode == MagmaNative)
        magma_getvector( 1, sizeof(magma_int_t), dinfo, 1, info, 1, queues[0]);

cleanup:
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    if (mode == MagmaHybrid) {
        magma_free_pinned( work );
    }
    else {
        magma_free( dinfo );
    }
    
    return *info;
} /* magma_dpotrf_LL_expert_gpu */

/***************************************************************************//**
    magma_dpotrf_LL_expert_gpu with mode = MagmaHybrid.
    Computation is hybrid, part on CPU (panels), part on GPU (matrix updates).
    @see magma_dpotrf_LL_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_dpotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magma_mode_t mode = MagmaHybrid;
    magma_dpotrf_LL_expert_gpu(uplo, n, dA, ldda, info, mode);
    return *info;
}

/***************************************************************************//**
    magma_dpotrf_LL_expert_gpu with mode = MagmaNative.
    Computation is done only on the GPU, not on the CPU.
    @see magma_dpotrf_LL_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_dpotrf_native(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magma_mode_t mode = MagmaNative;
    magma_dpotrf_LL_expert_gpu(uplo, n, dA, ldda, info, mode);
    return *info;
}
