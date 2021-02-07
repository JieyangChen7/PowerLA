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
#include <chrono>
using namespace std::chrono;
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
    SPOTRF computes the Cholesky factorization of a real symmetric
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
    dA      float PRECISION array on the GPU, dimension (LDDA,N)
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
magma_spotrf_LL_expert_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
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
    const float c_one     = MAGMA_D_ONE;
    const float c_neg_one = MAGMA_D_NEG_ONE;
    const float d_one     =  1.0;
    const float d_neg_one = -1.0;
    
    /* Local variables */
    magma_int_t j, jb, nb, recnb;
    float *work;
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
    
    nb = magma_get_spotrf_nb( n );
    nb = 4096;
    recnb = 128;
    //nb = 4;
    if (mode == MagmaHybrid) {
        if (MAGMA_SUCCESS != magma_smalloc_pinned( &work, nb*nb )) {
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
            magma_ssyrk( MagmaUpper, MagmaConjTrans, jb, j,
                         d_neg_one, dA(0, j), ldda,
                         d_one,     dA(j, j), ldda, queues[1] );
            
            if (mode == MagmaHybrid) {
                magma_queue_sync( queues[1] );
                magma_sgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
            }
            else {
                //Azzam: need to add events to allow overlap
                magma_spotrf_rectile_native(MagmaUpper, jb, recnb,
                                            dA(j, j), ldda, j,
                                            dinfo, info, queues[1] );
            }

            
            // apply all previous updates to block row right of diagonal block
            if (j+jb < n) {
                magma_sgemm( MagmaConjTrans, MagmaNoTrans,
                             jb, n-j-jb, j,
                             c_neg_one, dA(0, j   ), ldda,
                                        dA(0, j+jb), ldda,
                             c_one,     dA(j, j+jb), ldda, queues[1] );
            }
            
            // simultaneous with above dgemm, transfer diagonal block,
            // factor it on CPU, and test for positive definiteness
            if (mode == MagmaHybrid) {
                magma_queue_sync( queues[0] );
                lapackf77_spotrf( MagmaUpperStr, &jb, work, &jb, info );
                magma_ssetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[1] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
            }
            
            // apply diagonal block to block row right of diagonal block
            if (j+jb < n) {
                magma_strsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
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

        high_resolution_clock::time_point t1, t2;

        pm_cpu();
        pm_gpu(device);

        double total_time = 0.0;
        double pd_time_actual = 0.0, tmu_time_actual = 0.0, dt_time_actual = 0.0;;
        double pd_time_prof = 0.0, tmu_time_prof = 0.0, dt_time_prof = 0.0;
        double pd_time_ratio = 1, tmu_time_ratio = 1, dt_time_ratio = 1;
        double pd_time_pred = 0.0, tmu_time_pred = 0.0, dt_time_pred = 0.0;
        double pd_avg_error = 0.0, tmu_avg_error = 0.0, dt_avg_error = 0.0; 
        double tmu_ft_time_actual = 0.0;

        double slack_time_actual_avg = 0;

        double slack_time;
        double reclaimnation_ratio = 0.35;

        int tmu_desired_freq;
        int tmu_freq;
        int tmu_curr_freq = 1300;
        int tmu_base_freq = 1300;

        int tmu_base_offset = 0;
        int tmu_opt_offset = 200;
        adj_gpu(device, tmu_base_freq, 338000); // lock frequency is necessary for accurate prediction
        offset_gpu(tmu_base_offset);

        int pd_desired_freq;
        int pd_freq;
        int pd_curr_freq = 3500;
        int pd_base_freq = 3500;
        adj_cpu(pd_base_freq);
        reset_cpu();


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEvent_t start_ft, stop_ft;
        cudaEventCreate(&start_ft);
        cudaEventCreate(&stop_ft);

        unsigned long long start_energy;
        int pid;

        bool reclaim_slack = true;
        bool overclock = true;
        bool autoboost = false;
        int profile_interval = 1;
        int last_prof_iter = 1;
        int last_prof_freq_tmu = tmu_curr_freq;
        int last_prof_freq_pd = pd_curr_freq;

        /* flags */
        bool COL_FT = false;
        bool ROW_FT = false;
        bool DEBUG = false;
        bool CHECK_BEFORE = false;
        bool CHECK_AFTER = false;

        /* matrix sizes to be checksumed */
        int cpu_row = nb;
        int cpu_col = nb;
        int gpu_row = n;
        int gpu_col = n;
        
        printf( "initialize checksum vector on CPU\n");
        float * chk_v;
        int ld_chk_v = nb;
        magma_smalloc_pinned(&chk_v, nb * 2 * sizeof(float));
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
        float * dev_chk_v;
        size_t pitch_dev_chk_v = magma_roundup(nb * sizeof(float), 32);
        int ld_dev_chk_v;
        
        magma_smalloc(&dev_chk_v, pitch_dev_chk_v * 2);
        ld_dev_chk_v = pitch_dev_chk_v / sizeof(float);
        magma_sgetmatrix(nb, 2,
                         chk_v, ld_chk_v, 
                         dev_chk_v, ld_dev_chk_v,
                         queues[1]);
        if (DEBUG) {
            printMatrix_gpu(dev_chk_v, ld_dev_chk_v,
                            nb, 2, nb, nb, queues[1]);
        }


        printf( "allocate space for checksum on CPU......\n" );
        float * colchk;
        float * colchk_r;
        magma_smalloc_pinned(&colchk, (cpu_row / nb) * 2 * cpu_col * sizeof(float));
        int ld_colchk = (cpu_row / nb) * 2;
        magma_smalloc_pinned(&colchk_r, (cpu_row / nb) * 2 * cpu_col * sizeof(float));
        int ld_colchk_r = (cpu_row / nb) * 2;
        printf( "done.\n" );

        float * rowchk;
        float * rowchk_r;
        magma_smalloc_pinned(&rowchk, cpu_row * (cpu_col / nb) * 2 * sizeof(float));
        int ld_rowchk = cpu_row;
        magma_smalloc_pinned(&rowchk_r, cpu_row * (cpu_col / nb) * 2 * sizeof(float));
        int ld_rowchk_r = cpu_row;
        printf( "done.\n" );

        /* allocate space for col checksum on GPU */
        printf( "allocate space for checksums on GPUs......\n" );
        
        float * dA_colchk;
        size_t pitch_dA_colchk = magma_roundup((gpu_row / nb) * 2 * sizeof(float), 32);
        int ldda_colchk = pitch_dA_colchk / sizeof(float);
        if (COL_FT || ROW_FT) magma_smalloc(&dA_colchk, pitch_dA_colchk * gpu_col);

        float * dA_colchk_r;
        size_t pitch_dA_colchk_r = magma_roundup((gpu_row / nb) * 2 * sizeof(float), 32);
        int ldda_colchk_r = pitch_dA_colchk_r / sizeof(float);
        if (COL_FT || ROW_FT) magma_smalloc(&dA_colchk_r, pitch_dA_colchk_r * gpu_col);

        float * dA_rowchk;
        size_t pitch_dA_rowchk = magma_roundup(gpu_row * sizeof(float), 32);
        int ldda_rowchk = pitch_dA_rowchk / sizeof(float);
        if (COL_FT || ROW_FT) magma_smalloc(&dA_rowchk, pitch_dA_rowchk * (gpu_col / nb) * 2);


        float * dA_rowchk_r;
        size_t pitch_dA_rowchk_r = magma_roundup(gpu_row * sizeof(float), 32);
        int ldda_rowchk_r = pitch_dA_rowchk_r / sizeof(float);
        if (COL_FT || ROW_FT) magma_smalloc(&dA_rowchk_r, pitch_dA_rowchk_r * (gpu_col / nb) * 2);
           
        printf( "done.\n" );

       
        printf( "calculate initial checksum on GPUs......\n" ); 
        if (COL_FT || ROW_FT) {
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
        }

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

        if (overclock) {
            offset_gpu(tmu_opt_offset);
            undervolt_cpu();
        }

        if (autoboost) {
            reset_gpu(device);
        }

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
            if (j/jb > 1 && j + jb < n) predict = true;
            bool reclaim_tmu = reclaim_slack;
            bool reclaim_pd = reclaim_slack;

            pd_time_actual = 0.0;
            dt_time_actual = 0.0;
            tmu_time_actual = 0.0;

            // set default
            pd_freq = pd_base_freq;
            tmu_freq = tmu_base_freq;


            //prediction
            if (predict) {

                // predict execution if it is under base frequency
                float pd_freq_ratio = (float)last_prof_freq_pd/pd_base_freq;
                float tmu_freq_ratio = (float)last_prof_freq_tmu/tmu_base_freq;
                pd_time_ratio = 1.0;
                dt_time_ratio = 1.0;
                tmu_time_ratio = (float)abft_sgemm_flops(MagmaNoTrans, MagmaConjTrans, n-j-jb, jb, j, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/
                        abft_sgemm_flops(MagmaNoTrans, MagmaConjTrans,n-jb*last_prof_iter-jb, jb, jb*last_prof_iter, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER);
                pd_time_pred = pd_time_prof * pd_time_ratio * pd_freq_ratio;
                dt_time_pred = dt_time_prof * dt_time_ratio;
                tmu_time_pred = tmu_time_prof * tmu_time_ratio * tmu_freq_ratio;
                slack_time = tmu_time_pred - pd_time_pred - dt_time_pred;

                printf("j = %d\n", j);
                printf("pd_time_prof: %f, tmu_time_prof: %f\n", pd_time_prof, tmu_time_prof);
                printf("last_prof_freq_pd: %d, last_prof_freq_tmu: %d\n", last_prof_freq_pd, last_prof_freq_tmu);
                printf("pd_time_ratio: %f, tmu_time_ratio: %f\n", pd_time_ratio, tmu_time_ratio);
                printf("pd_freq_ratio: %f, tmu_freq_ratio: %f\n", pd_freq_ratio, tmu_freq_ratio);
                printf("pd_time_pred: %f, tmu_time_pred: %f\n", pd_time_pred, tmu_time_pred);
                // determine frequency
                double gpu_adj_time = 15;
                double cpu_adj_time = 80;

                double tmu_desired_time = tmu_time_pred-(reclaimnation_ratio*slack_time)-gpu_adj_time;
                double pd_desired_time = tmu_desired_time-cpu_adj_time-dt_time_pred;

                if (tmu_desired_time < 0) tmu_desired_time = 1;
                if (pd_desired_time < 0 ) pd_desired_time = 1;
                printf("pd_desired_time: %f, tmu_desired_time: %f\n", pd_desired_time, tmu_desired_time);

                tmu_desired_freq = tmu_time_pred/tmu_desired_time * tmu_base_freq;
                tmu_freq = tmu_desired_freq;
                if (tmu_desired_freq > 2100) tmu_freq = 2100;
                if (tmu_desired_freq < 500) tmu_freq = 500;
                tmu_freq = (int)ceil((double)tmu_freq/100)*100;

                pd_desired_freq = pd_time_pred/pd_desired_time * pd_base_freq;
                pd_freq = pd_desired_freq;
                if (pd_desired_freq > 3500) pd_freq = 3500;
                if (pd_desired_freq < 1000) pd_freq = 1000;
                pd_freq = (int)ceil((double)pd_freq/100)*100;


                
                // performance cannot be worse than baseline
                double max_pd_tmu = max(pd_time_pred+dt_time_pred, tmu_time_pred);

                // projected execution time if we apply frequency
                pd_freq_ratio = (float)pd_base_freq/pd_freq;
                tmu_freq_ratio = (float)tmu_base_freq/tmu_freq;
                printf("pd_freq: %d, tmu_freq: %d\n", pd_freq, tmu_freq);
                printf("pd_time_proj: %f, tmu_time_proj: %f\n", pd_time_pred * pd_freq_ratio, tmu_time_pred * tmu_freq_ratio);

                //if we want to reclaim and there is benefit
                if (reclaim_pd && pd_time_pred * pd_freq_ratio + cpu_adj_time + dt_time_pred <= max_pd_tmu) {
                    printf("pd: plan reclaim %f <  %f\n", pd_time_pred * pd_freq_ratio + cpu_adj_time + dt_time_pred, max_pd_tmu);
                    pd_time_pred = pd_time_pred * pd_freq_ratio;
                } else { //if do not want to reclaim or there is no benefit
                    printf("pd: not worth reclaim %f > %f\n", pd_time_pred * pd_freq_ratio + cpu_adj_time + dt_time_pred, max_pd_tmu);
                    pd_freq_ratio = (float)pd_base_freq/pd_curr_freq;
                    pd_time_pred = pd_time_pred * pd_freq_ratio;
                    reclaim_pd = false;
                }
                // reclaim_pd = false;

                
                //if we want to reclaim and there is benefit
                if (reclaim_tmu && tmu_time_pred * tmu_freq_ratio + gpu_adj_time <= max_pd_tmu) {
                    printf("tmu: plan reclaim %f <  %f\n", tmu_time_pred * tmu_freq_ratio + gpu_adj_time, max_pd_tmu);
                    tmu_time_pred = tmu_time_pred * tmu_freq_ratio;
                } else { //if do not want to reclaim or there is no benefit
                    printf("tmu: not worth reclaim %f >  %f\n", tmu_time_pred * tmu_freq_ratio + gpu_adj_time, max_pd_tmu);
                    tmu_freq_ratio = (float)tmu_base_freq/tmu_curr_freq;
                    tmu_time_pred = tmu_time_pred * tmu_freq_ratio;
                    reclaim_tmu = false; 
                }              
                // reclaim_tmu = false;   
            }

            // printf("j = %d\n", j);


            
            // magma_dsyrk( MagmaLower, MagmaNoTrans, jb, j,
            //              d_neg_one, dA(j, 0), ldda,
            //              d_one,     dA(j, j), ldda, queues[0] );
            // printf("calling abft_dsyrk\n");

            //float * t = dA_colchk ; // + ((j)/nb)*2 + (0)*ldda_colchk; //dA_colchk(j, 0);
            
            abft_ssyrk( MagmaLower, MagmaNoTrans, jb, j,
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

            
            // apply all previous updates to block column below diagonal block
            if (j+jb < n) {
                magma_queue_wait_event(queues[1], events[0]);
                magma_queue_sync( queues[0] );
                magma_queue_sync( queues[1] );
                if (reclaim_tmu && j > jb) {
                    if (tmu_freq != tmu_curr_freq) {
                        // double tt = magma_wtime();
                        adj_gpu(device, tmu_freq, 338000);
                        // tt = magma_wtime() - tt;
                        // printf("adj_gpu: %f\n", tt);
                        //printf("set to %d\n", tmu_freq);
                        tmu_curr_freq = tmu_freq;
                    }
                }
                
               // start_energy = start_measure_gpu(device);
               // float tmu_time = magma_wtime();
                cudaEventRecord(start, queues[1]->cuda_stream());
                //cudaEventRecord(start_ft, queues[2]->cuda_stream());
                // magma_dgemm( MagmaNoTrans, MagmaConjTrans,
                //              n-j-jb, jb, j,
                //              c_neg_one, dA(j+jb, 0), ldda,
                //                         dA(j,    0), ldda,
                //              c_one,     dA(j+jb, j), ldda, queues[1] );
                abft_sgemm( MagmaNoTrans, MagmaConjTrans,
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
            

            // Azzam: this section of "ifthenelse" can be moved down to the factorize section and I don't think performane wil be affected.
            
            if (mode == MagmaHybrid) {

                if (reclaim_pd && j > jb) {
                    if (pd_freq != pd_curr_freq) {
                        double tt = magma_wtime();
                        adj_cpu(pd_freq);
                        // tt = magma_wtime() - tt;
                        // printf("adj_cpu: %f\n", tt);
                        pd_curr_freq = pd_freq;
                    }
                }


                dt_time_actual = magma_wtime();
                magma_sgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
                if (COL_FT || ROW_FT) {
                  magma_sgetmatrix_async( 2, jb,
                                          dA_colchk(j, j), ldda_colchk,
                                          colchk,     ld_colchk, queues[0] );

                  magma_sgetmatrix_async( jb, 2,
                                          dA_rowchk(j, j), ldda_rowchk,
                                          rowchk,     ld_rowchk, queues[0] );
                }
            }
            else {
                magma_spotrf_rectile_native(MagmaLower, jb, recnb,
                                            dA(j, j), ldda, j,
                                            dinfo, info, queues[0] );
            }

            // simultaneous with above dgemm, transfer diagonal block,
            // factor it on CPU, and test for positive definiteness
            // Azzam: The above section can be moved here the code will look cleaner.
            if (mode == MagmaHybrid) {
                magma_queue_sync( queues[0] );
                dt_time_actual = magma_wtime() - dt_time_actual;
                
                
                //t1 = high_resolution_clock::now(); 
                pd_time_actual = magma_wtime();
                lapackf77_spotrf( MagmaLowerStr, &jb, work, &jb, info );
                pd_time_actual = (magma_wtime() - pd_time_actual) * 1000;
                if (profile) pd_time_prof = pd_time_actual;
                if (predict) pd_avg_error += fabs(pd_time_actual - pd_time_pred) / pd_time_actual;
                //t2 = high_resolution_clock::now();
                //pd_time_actual  = duration_cast<milliseconds>(t2 - t1).count();

                dt_time_actual = (-1)*dt_time_actual;
                dt_time_actual += magma_wtime();
                magma_ssetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[0] );

                magma_queue_sync( queues[0] );
                dt_time_actual = magma_wtime() - dt_time_actual;
                dt_time_actual *= 1000;
                if (profile) dt_time_prof = dt_time_actual;
                if (predict) dt_avg_error += fabs(dt_time_actual - dt_time_pred) / dt_time_actual;
                
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
            }


            
            // apply diagonal block to block column below diagonal
            if (j+jb < n) {

                //profile dgemm   
                cudaEventSynchronize(stop);
                float t;
                cudaEventElapsedTime(&t, start, stop);
                tmu_time_actual = t;
                if (profile) tmu_time_prof = tmu_time_actual;
                if (predict) tmu_avg_error += fabs(tmu_time_actual - tmu_time_pred) / tmu_time_actual;

                magma_queue_wait_event(queues[0], events[1]);
                // magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                //              n-j-jb, jb,
                //              c_one, dA(j,    j), ldda,
                //                     dA(j+jb, j), ldda, queues[0] );
                abft_strsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
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
            
            //adj_cpu(pd_base_freq);

            if (profile) {
                last_prof_iter = j/jb;
                last_prof_freq_pd = pd_curr_freq;
                last_prof_freq_tmu = tmu_curr_freq;
            }
            // printf("%d, pd-tmu, %f, %f, %f, %f, %f, %f, %d, %d, \n", j, pd_time_actual, dt_time_actual, tmu_time_actual, pd_time_pred, dt_time_pred, tmu_time_pred, pd_freq, tmu_freq);
            if (j > 0) slack_time_actual_avg += fabs(tmu_time_actual-pd_time_actual-dt_time_actual);

        }
        total_time = magma_wtime() - total_time;
        stop_measure_cpu(pid);
        start_energy = stop_measure_gpu(device, start_energy);
        printf("GPU energy: %llu\n", start_energy);
        printf("Prediction average error: CPU %f, DT, %f, GPU %f\n", pd_avg_error/(n/jb-1), dt_avg_error/(n/jb-1),tmu_avg_error/(n/jb-1));
        printf("Total time: %f\n", total_time);
        printf("Average slack: %f\n", slack_time_actual_avg/(n/jb-1));
        //reset
        adj_gpu(device, 1500, 338000);
        offset_gpu(0);
        reset_cpu();
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
magma_spotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magma_mode_t mode = MagmaHybrid;
    magma_spotrf_LL_expert_gpu(uplo, n, dA, ldda, info, mode);
    return *info;
}

/***************************************************************************//**
    magma_dpotrf_LL_expert_gpu with mode = MagmaNative.
    Computation is done only on the GPU, not on the CPU.
    @see magma_dpotrf_LL_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_spotrf_native(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magma_mode_t mode = MagmaNative;
    magma_spotrf_LL_expert_gpu(uplo, n, dA, ldda, info, mode);
    return *info;
}
