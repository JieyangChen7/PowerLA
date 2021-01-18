/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Stan Tomov
       @author Mark Gates
       
       @generated from src/zgetrf_gpu.cpp, normal z -> d, Thu Oct  8 23:05:24 2020

*/
#include "../power_adjustment/power_adjustment.h"
#include "magma_internal.h"
#include "cuda_runtime.h"
#include "../fault_tolerance/abft_printer.h"
#include "../fault_tolerance/abft_encoder.h"
#include "../fault_tolerance/abft_kernels.h"

/***************************************************************************//**
    Purpose
    -------
    DGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    mode    magma_mode_t
      -     = MagmaNative:  Factorize dA using GPU only mode.
      -     = MagmaHybrid:  Factorize dA using Hybrid (CPU/GPU) mode.

    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_dgetrf_gpu_expert(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info, magma_mode_t mode)
{
    #ifdef HAVE_clBLAS
    #define  dA(i_, j_) dA,  (dA_offset  + (i_)       + (j_)*ldda)
    #define dAT(i_, j_) dAT, (dAT_offset + (i_)*lddat + (j_))
    #define dAP(i_, j_) dAP, (             (i_)          + (j_)*maxm)
    #else
    #define  dA(i_, j_) (dA  + (i_)       + (j_)*ldda)
    #define dAT(i_, j_) (dAT + (i_)*lddat + (j_))
    #define dAP(i_, j_) (dAP + (i_)       + (j_)*maxm)
    #endif

    /* Define for ABFT */
    #define dA_colchk(i_, j_)   (dA_colchk   + ((i_)/nb)*2 + (j_)*ldda_colchk)
    #define dA_rowchk(i_, j_)   (dA_rowchk   + (i_)        + ((j_)/nb)*2*ldda_rowchk)
    #define dA_colchk_r(i_, j_) (dA_colchk_r + ((i_)/nb)*2 + (j_)*ldda_colchk_r)
    #define dA_rowchk_r(i_, j_) (dA_rowchk_r + (i_)        + ((j_)/nb)*2*ldda_rowchk_r)

    #define dAT_colchk(i_, j_)   (dAT_colchk   + (i_)*lddat_colchk + (j_)/nb*2)
    #define dAT_rowchk(i_, j_)   (dAT_rowchk   + ((i_)/nb)*2*lddat_rowchk        + (j_))
    #define dAT_colchk_r(i_, j_) (dAT_colchk_r + (i_)*lddat_colchk_r + (j_)/nb*2)
    #define dAT_rowchk_r(i_, j_) (dAT_rowchk_r + ((i_)/nb)*2*lddat_rowchk_r        + (j_))

    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;

    magma_int_t iinfo, nb;
    magma_int_t maxm, maxn, minmn, liwork;
    magma_int_t i, j, jb, rows, lddat, ldwork;
    magmaDouble_ptr dAT=NULL, dAP=NULL;
    double *work=NULL; // hybrid
    magma_int_t *diwork=NULL, *dipiv=NULL, *dipivinfo=NULL, *dinfo=NULL; // native

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    minmn = min( m, n );
    nb    = (mode == MagmaHybrid) ? magma_get_dgetrf_nb( m, n ) : magma_get_dgetrf_native_nb( m, n );
    nb = 512;
    //nb = 4;
    magma_queue_t queues[2] = { NULL };
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    if (mode == MagmaNative) {
        liwork = m + minmn + 1;
        if (MAGMA_SUCCESS != magma_imalloc(&diwork, liwork)) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }
        else {
            dipivinfo = diwork;     // dipivinfo size = m
            dipiv = dipivinfo + m;  // dipiv size = minmn
            dinfo = dipiv + minmn;  // dinfo size = 1
            cudaMemsetAsync( dinfo, 0, sizeof(magma_int_t), queues[0]->cuda_stream() );
        }
    }
    
    // if (nb <= 1 || nb >= min(m,n) ) {
    //     if (mode == MagmaHybrid) {
    //         /* Use CPU code. */
    //         if ( MAGMA_SUCCESS != magma_dmalloc_cpu( &work, m*n )) {
    //             *info = MAGMA_ERR_HOST_ALLOC;
    //             goto cleanup;
    //         }
    //         magma_dgetmatrix( m, n, dA(0,0), ldda, work, m, queues[0] );
    //         lapackf77_dgetrf( &m, &n, work, &m, ipiv, info );
    //         magma_dsetmatrix( m, n, work, m, dA(0,0), ldda, queues[0] );
    //         magma_free_cpu( work );  work=NULL;
    //     }
    //     else {
    //         /* Use GPU code (native mode). */
    //         magma_dgetrf_recpanel_native( m, n, dA(0,0), ldda, dipiv, dipivinfo, dinfo, 0, queues[0], queues[1]);
    //         magma_igetvector( minmn, dipiv, 1, ipiv, 1, queues[0] );
    //         magma_igetvector( 1, dinfo, 1, info, 1, queues[0] );
    //     }
    // }
    // else 
    {

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

        bool reclaim_slack = true;
        int profile_interval = 1;
        int last_prof_iter = 1;

        /* flags */
        bool COL_FT = false;
        bool ROW_FT = false;
        bool DEBUG = false;
        bool CHECK_BEFORE = false;
        bool CHECK_AFTER = true;

        /* matrix sizes to be checksumed */
        int cpu_row = nb;
        int cpu_col = nb;
        int gpu_row = m;
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
            // printf("checksum vector on CPU:\n");
            // printMatrix_host(chk_v, ld_chk_v, nb, 2, -1, -1);
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
            // printMatrix_gpu(dev_chk_v, ld_dev_chk_v,
            //                 nb, 2, nb, nb, queues[1]);
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

            // printf( "input matrix A:\n" );
            // printMatrix_gpu(dA, ldda, gpu_row, gpu_col, nb, nb, queues[1]);
            // printf( "column chk:\n" );
            // printMatrix_gpu(dA_colchk, ldda_colchk, 
            //                 (gpu_row / nb) * 2, gpu_col, 2, nb, queues[1]);
            // printf( "row chk:\n" );
            // printMatrix_gpu(dA_rowchk, ldda_rowchk,  
            //                 gpu_row, (gpu_col / nb) * 2, nb, 2, queues[1]);
        }

        cudaDeviceSynchronize();

        /* Use blocked code. */
        maxm = magma_roundup( m, 32 );
        maxn = magma_roundup( n, 32 );

        if (MAGMA_SUCCESS != magma_dmalloc( &dAP, nb*maxm )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }

        // square matrices can be done in place;
        // rectangular requires copy to transpose
        if ( m == n ) {
            dAT = dA;
            lddat = ldda;
            magmablas_dtranspose_inplace( m, dAT(0,0), lddat, queues[0] );
        }
        else {
            lddat = maxn;  // N-by-M
            if (MAGMA_SUCCESS != magma_dmalloc( &dAT, lddat*maxm )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                goto cleanup;
            }
            magmablas_dtranspose( m, n, dA(0,0), ldda, dAT(0,0), lddat, queues[0] );
        }

        /* allocate space for col checksum on GPU */
        printf( "allocate space for transposed checksums on GPUs......\n" );
        
        // swap m and n
        gpu_row = n;
        gpu_col = m;
        

        double * dAT_colchk;
        size_t pitch_dAT_colchk = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
        int lddat_colchk = pitch_dAT_colchk / sizeof(double);
        magma_dmalloc(&dAT_colchk, pitch_dAT_colchk * gpu_col);

        double * dAT_colchk_r;
        size_t pitch_dAT_colchk_r = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
        int lddat_colchk_r = pitch_dAT_colchk_r / sizeof(double);
        magma_dmalloc(&dAT_colchk_r, pitch_dAT_colchk_r * gpu_col);

        double * dAT_rowchk;
        size_t pitch_dAT_rowchk = magma_roundup(gpu_row * sizeof(double), 32);
        int lddat_rowchk = pitch_dAT_rowchk / sizeof(double);
        magma_dmalloc(&dAT_rowchk, pitch_dAT_rowchk * (gpu_col / nb) * 2);

        double * dAT_rowchk_r;
        size_t pitch_dAT_rowchk_r = magma_roundup(gpu_row * sizeof(double), 32);
        int lddat_rowchk_r = pitch_dAT_rowchk_r / sizeof(double);
        magma_dmalloc(&dAT_rowchk_r, pitch_dAT_rowchk_r * (gpu_col / nb) * 2);
           
        printf( "done.\n" );

        printf( "Transposing checksums on GPUs......\n" );
        magmablas_dtranspose( (m/nb)*2, n, dA_colchk, ldda_colchk, dAT_rowchk, lddat_rowchk, queues[0] );
        magmablas_dtranspose( m, (n/nb)*2, dA_rowchk, ldda_rowchk, dAT_colchk, lddat_colchk, queues[0] );

        if (DEBUG) {

            // printf( "input matrix AT:\n" );
            // printMatrix_gpu(dAT, lddat, gpu_row, gpu_col, nb, nb, queues[0]);
            // printf( "column chk:\n" );
            // printMatrix_gpu(dAT_colchk, lddat_colchk, 
            //                 (gpu_row / nb) * 2, gpu_col, 2, nb, queues[0]);
            // printf( "row chk:\n" );
            // printMatrix_gpu(dAT_rowchk, lddat_rowchk,  
            //                 gpu_row, (gpu_col / nb) * 2, nb, 2, queues[0]);
        }

        printf( "done.\n" );


        magma_queue_sync( queues[0] );  // finish transpose

        ldwork = maxm;
        if (mode == MagmaHybrid) {
            if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, ldwork*nb )) {
                *info = MAGMA_ERR_HOST_ALLOC;
                goto cleanup;
            }
        }

        pid = start_measure_cpu();
        start_energy = start_measure_gpu(device);
        total_time = magma_wtime();
        for( j=0; j < minmn-nb; j += nb ) {
        //for( j=0; j < nb*4; j += nb ) {
            if (DEBUG) printf("j = %d\n", j);

            bool profile = false;
            if ((j/nb-last_prof_iter)%profile_interval == 0) profile = true;
            bool predict = false;
            if (j/nb > 1) predict = true;

            //prediction
            if (predict) {
                pd_time_ratio = (((double)m-j)*nb*nb/2+nb*nb*nb/6)/(((double)m-(nb*last_prof_iter))*nb*nb/2+nb*nb*nb/6);
                tmu_time_ratio = (double)abft_dgemm_flops(MagmaNoTrans, MagmaNoTrans, n-(j+nb), m-j, nb, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/
                        abft_dgemm_flops(MagmaNoTrans, MagmaNoTrans, n-((nb*last_prof_iter)+nb), m-(nb*last_prof_iter), nb, nb, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER);


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


            // get j-th panel from device
            magmablas_dtranspose( nb, m-j, dAT(j,j), lddat, dAP(0,0), maxm, queues[1] );
            magma_queue_sync( queues[1] );  // wait for transpose
            if (mode == MagmaHybrid) {
                magma_dgetmatrix_async( m-j, nb, dAP(0,0), maxm, work, ldwork, queues[0] );
            }

            if ( j > 0 ) {
                // magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                //              n-(j+nb), nb,
                //              c_one, dAT(j-nb, j-nb), lddat,
                //                     dAT(j-nb, j+nb), lddat, queues[1] );

                if (reclaim_slack && j > nb) {
                    if (tmu_freq != tmu_curr_freq || tmu_offset != tmu_curr_offset) {
                        adj_gpu(device, tmu_freq, 338000, tmu_offset);
                        //printf("set to %d\n", tmu_freq);
                        tmu_curr_freq = tmu_freq;
                        tmu_curr_offset = tmu_offset;
                    }
                }
                 cudaEventRecord(start, queues[1]->cuda_stream());
                 abft_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                    n-(j+nb), nb, c_one,
                    dAT(j-nb, j-nb), lddat,
                    dAT(j-nb, j+nb), lddat,
                    nb,
                    dAT_colchk(j-nb, j-nb),    lddat_colchk,
                    dAT_rowchk(j-nb, j-nb),    lddat_rowchk,
                    dAT_colchk_r(j-nb, j-nb),  lddat_colchk_r,
                    dAT_rowchk_r(j-nb, j-nb),  lddat_rowchk_r,
                    dAT_colchk(j-nb, j+nb),   lddat_colchk,
                    dAT_rowchk(j-nb, j+nb),   lddat_rowchk,
                    dAT_colchk_r(j-nb, j+nb), lddat_colchk_r,
                    dAT_rowchk_r(j-nb, j+nb), lddat_rowchk_r,
                    dev_chk_v, ld_dev_chk_v, 
                    COL_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                    queues[1], queues[1]);


                // magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                //              n-(j+nb), m-j, nb,
                //              c_neg_one, dAT(j-nb, j+nb), lddat,
                //                         dAT(j,    j-nb), lddat,
                //              c_one,     dAT(j,    j+nb), lddat, queues[1] );
                 
                abft_dgemm( MagmaNoTrans, MagmaNoTrans,
                            n-(j+nb), m-j, nb,
                            c_neg_one, dAT(j-nb, j+nb), lddat,
                            dAT(j,    j-nb), lddat,
                            c_one,     dAT(j,    j+nb), lddat,
                            nb,
                            dAT_colchk(j-nb, j+nb),   lddat_colchk,
                            dAT_rowchk(j-nb, j+nb),   lddat_rowchk,
                            dAT_colchk_r(-nb, j+nb), lddat_colchk_r,
                            dAT_rowchk_r(j-nb, j+nb), lddat_rowchk_r,
                            dAT_colchk(j, j-nb),   lddat_colchk,
                            dAT_rowchk(j, j-nb),   lddat_rowchk,
                            dAT_colchk_r(j, j-nb), lddat_colchk_r,
                            dAT_rowchk_r(j, j-nb), lddat_rowchk_r,
                            dAT_colchk(j, j+nb),   lddat_colchk,
                            dAT_rowchk(j, j+nb),   lddat_rowchk,
                            dAT_colchk_r(j, j+nb), lddat_colchk_r,
                            dAT_rowchk_r(j, j+nb), lddat_rowchk_r,
                            dev_chk_v, ld_dev_chk_v, 
                            COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                            queues[1], queues[1]);
                cudaEventRecord(stop, queues[1]->cuda_stream());
            }

            rows = m - j;
            if (mode == MagmaHybrid) {
                // do the cpu part
                magma_queue_sync( queues[0] );  // wait to get work
                if (reclaim_slack && j > nb) {
                    if (pd_freq != pd_curr_freq) {
                        adj_cpu(pd_freq);
                        pd_curr_freq = pd_freq;
                    }
                }
                pd_time_actual = magma_wtime();
                lapackf77_dgetrf( &rows, &nb, work, &ldwork, ipiv+j, &iinfo );
                pd_time_actual = magma_wtime() - pd_time_actual;
                pd_time_actual *= 1000;
                if (profile) pd_time_prof = pd_time_actual;
                if (predict) pd_avg_error += fabs(pd_time_actual - pd_time_pred) / pd_time_actual;

                if ( *info == 0 && iinfo > 0 )
                    *info = iinfo + j;

                // send j-th panel to device
                magma_dsetmatrix_async( m-j, nb, work, ldwork, dAP, maxm, queues[0] );


                magma_queue_sync( queues[1] );
                cudaEventSynchronize(stop);
                float t;
                cudaEventElapsedTime(&t, start, stop);
                tmu_time_actual = t;
                if (profile) tmu_time_prof = tmu_time_actual;
                if (predict) tmu_avg_error += fabs(tmu_time_actual - tmu_time_pred) / tmu_time_actual;

                for( i=j; i < j + nb; ++i ) {
                    ipiv[i] += j;
                }
                magmablas_dlaswp( n, dAT(0,0), lddat, j + 1, j + nb, ipiv, 1, queues[1] );

                if (COL_FT) {
                    magmablas_dlaswp( (n/nb)*2, dAT_colchk(0,0), lddat_colchk, j + 1, j + nb, ipiv, 1, queues[1] );
                }

                if (ROW_FT) {
                    row_chk_enc(n, m, nb, 
                    dAT, lddat,  
                    dev_chk_v, ld_dev_chk_v, 
                    dAT_rowchk, lddat_rowchk, 
                    queues[1]);
                }

                magma_queue_sync( queues[0] );  // wait to set dAP
            }
            else {
                // do the panel on the GPU
                magma_dgetrf_recpanel_native( rows, nb, dAP(0,0), maxm, dipiv+j, dipivinfo, dinfo, j, queues[0], queues[1]);
                adjust_ipiv( dipiv+j, nb, j, queues[0]);
                #ifdef SWP_CHUNK
                magma_igetvector_async( nb, dipiv+j, 1, ipiv+j, 1, queues[0] );
                #endif

                magma_queue_sync( queues[0] );  // wait for the pivot
                #ifdef SWP_CHUNK
                magmablas_dlaswp( n, dAT(0,0), lddat, j + 1, j + nb, ipiv, 1, queues[1] );
                #else
                magma_dlaswp_columnserial(n, dAT(0,0), lddat, j + 1, j + nb, dipiv, queues[1]);
                #endif
            }
            magmablas_dtranspose( m-j, nb, dAP(0,0), maxm, dAT(j,j), lddat, queues[1] );

            if (ROW_FT) {
                    row_chk_enc(nb, m-j, nb, 
                    dAT(j,j), lddat,  
                    dev_chk_v, ld_dev_chk_v, 
                    dAT_rowchk(j,j), lddat_rowchk, 
                    queues[1]);
                // if (DEBUG) {
                //     printf( "Panel AT(%d, %d):\n", j, j);
                //     printMatrix_gpu(dAT, lddat, nb, m-j, nb, nb, queues[1]);
                //     printf( "row chk:\n" );
                //     printMatrix_gpu(dAT_rowchk(j,j), lddat_rowchk,  
                //                     nb, ((m-j) / nb) * 2, nb, 2, queues[1]);
                // }
            }

            // do the small non-parallel computations (next panel update)
            if ( j + nb < minmn - nb ) {
                // magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                //              nb, nb,
                //              c_one, dAT(j, j   ), lddat,
                //                     dAT(j, j+nb), lddat, queues[1] );
                abft_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                    nb, nb, c_one,
                    dAT(j, j   ), lddat,
                    dAT(j, j+nb), lddat,
                    nb,
                    dAT_colchk(j, j),    lddat_colchk,
                    dAT_rowchk(j, j),    lddat_rowchk,
                    dAT_colchk_r(j, j),  lddat_colchk_r,
                    dAT_rowchk_r(j, j),  lddat_rowchk_r,
                    dAT_colchk(j, j+nb),   lddat_colchk,
                    dAT_rowchk(j, j+nb),   lddat_rowchk,
                    dAT_colchk_r(j, j+nb), lddat_colchk_r,
                    dAT_rowchk_r(j, j+nb), lddat_rowchk_r,
                    dev_chk_v, ld_dev_chk_v, 
                    false, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                    queues[1], queues[1]);

                // magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                //              nb, m-(j+nb), nb,
                //              c_neg_one, dAT(j,    j+nb), lddat,
                //                         dAT(j+nb, j   ), lddat,
                //              c_one,     dAT(j+nb, j+nb), lddat, queues[1] );

                abft_dgemm( MagmaNoTrans, MagmaNoTrans,
                            nb, m-(j+nb), nb,
                            c_neg_one, dAT(j,    j+nb), lddat,
                            dAT(j+nb, j   ), lddat,
                            c_one,     dAT(j+nb, j+nb), lddat,
                            nb,
                            dAT_colchk(j, j+nb),   lddat_colchk,
                            dAT_rowchk(j, j+nb),   lddat_rowchk,
                            dAT_colchk_r(j, j+nb), lddat_colchk_r,
                            dAT_rowchk_r(j, j+nb), lddat_rowchk_r,
                            dAT_colchk(j+nb, j),   lddat_colchk,
                            dAT_rowchk(j+nb, j),   lddat_rowchk,
                            dAT_colchk_r(j+nb, j), lddat_colchk_r,
                            dAT_rowchk_r(j+nb, j), lddat_rowchk_r,
                            dAT_colchk(j+nb, j+nb),   lddat_colchk,
                            dAT_rowchk(j+nb, j+nb),   lddat_rowchk,
                            dAT_colchk_r(j+nb, j+nb), lddat_colchk_r,
                            dAT_rowchk_r(j+nb, j+nb), lddat_rowchk_r,
                            dev_chk_v, ld_dev_chk_v, 
                            false, false, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                            queues[1], queues[1]);




            }
            else {
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-(j+nb), nb,
                             c_one, dAT(j, j   ), lddat,
                                    dAT(j, j+nb), lddat, queues[1] );
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+nb), m-(j+nb), nb,
                             c_neg_one, dAT(j,    j+nb), lddat,
                                        dAT(j+nb, j   ), lddat,
                             c_one,     dAT(j+nb, j+nb), lddat, queues[1] );
            }
            if (profile) last_prof_iter = j/nb;
            printf("%d, pd-tmu, %f, %f, %f, %f, %f, %f, %d, %d, \n", j, pd_time_prof, tmu_time_prof, pd_time_actual, tmu_time_actual, pd_time_pred, tmu_time_pred, tmu_freq, pd_freq);

        }
        total_time = magma_wtime() - total_time;
        stop_measure_cpu(pid);
        start_energy = stop_measure_gpu(device, start_energy);
        printf("GPU energy: %llu\n", start_energy);
        printf("Prediction average error: CPU %f, GPU %f\n", pd_avg_error/(n/nb-1), tmu_avg_error/(n/nb-1));
        printf("Total time: %f\n", total_time);

        jb = min( m-j, n-j );
        if ( jb > 0 ) {
            rows = m - j;
            
            magmablas_dtranspose( jb, rows, dAT(j,j), lddat, dAP(0,0), maxm, queues[1] );
            if (mode == MagmaHybrid) {
                magma_dgetmatrix( rows, jb, dAP(0,0), maxm, work, ldwork, queues[1] );
            
                // do the cpu part
                lapackf77_dgetrf( &rows, &jb, work, &ldwork, ipiv+j, &iinfo );
                if ( *info == 0 && iinfo > 0 )
                    *info = iinfo + j;
            
                for( i=j; i < j + jb; ++i ) {
                    ipiv[i] += j;
                }
                magmablas_dlaswp( n, dAT(0,0), lddat, j + 1, j + jb, ipiv, 1, queues[1] );
            
                // send j-th panel to device
                magma_dsetmatrix( rows, jb, work, ldwork, dAP(0,0), maxm, queues[1] );
            }
            else {
                magma_dgetrf_recpanel_native( rows, jb, dAP(0,0), maxm, dipiv+j, dipivinfo, dinfo, j, queues[1], queues[0]);
                adjust_ipiv( dipiv+j, jb, j, queues[1]);
                #ifdef SWP_CHUNK
                magma_igetvector( jb, dipiv+j, 1, ipiv+j, 1, queues[1] );
                magmablas_dlaswp( n, dAT(0,0), lddat, j + 1, j + jb, ipiv, 1, queues[1] );
                #else
                magma_dlaswp_columnserial(n, dAT(0,0), lddat, j + 1, j + jb, dipiv, queues[1]);
                #endif
            }

            magmablas_dtranspose( rows, jb, dAP(0,0), maxm, dAT(j,j), lddat, queues[1] );
            
            magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-j-jb, jb,
                         c_one, dAT(j,j),    lddat,
                                dAT(j,j+jb), lddat, queues[1] );
        }
        
        if (mode == MagmaNative) {
            // copy the pivot vector to the CPU
            #ifndef SWP_CHUNK
            magma_igetvector(minmn, dipiv, 1, ipiv, 1, queues[1] );
            #endif
        }

        // undo transpose
        if ( m == n ) {
            magmablas_dtranspose_inplace( m, dAT(0,0), lddat, queues[1] );
        }
        else {
            magmablas_dtranspose( n, m, dAT(0,0), lddat, dA(0,0), ldda, queues[1] );
        }
    }
    
cleanup:
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free( dAP );
    if (m != n) {
        magma_free( dAT );
    }

    if (mode == MagmaHybrid) {
        magma_free_pinned( work );
    }
    else {
        magma_free( diwork );
    }

    return *info;
} /* magma_dgetrf_gpu */

/***************************************************************************//**
    magma_dgetrf_gpu_expert with mode = MagmaHybrid.
    Computation is hybrid, part on CPU (panels), part on GPU (matrix updates).
    @see magma_dgetrf_gpu_expert
    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_dgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info )
{
    magma_dgetrf_gpu_expert(m, n, dA, ldda, ipiv, info, MagmaHybrid);
    return *info;
} /* magma_dgetrf_gpu */

/***************************************************************************//**
    magma_dgetrf_gpu_expert with mode = MagmaNative.
    Computation is done only on the GPU, not on the CPU.
    @see magma_dgetrf_gpu_expert
    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_dgetrf_native(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info )
{
    magma_dgetrf_gpu_expert(m, n, dA, ldda, ipiv, info, MagmaNative);
    return *info;
} /* magma_dgetrf_native */