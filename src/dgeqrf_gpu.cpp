/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Stan Tomov
       @author Mark Gates

       @generated from src/zgeqrf_gpu.cpp, normal z -> d, Thu Oct  8 23:05:24 2020
*/
#include <chrono>
using namespace std::chrono;
#include "../power_adjustment/power_adjustment.h"
#include "magma_internal.h"
#include "cuda_runtime.h"
#include "../fault_tolerance/abft_printer.h"
#include "../fault_tolerance/abft_encoder.h"
#include "../fault_tolerance/abft_kernels.h"

/***************************************************************************//**
    Auxiliary function: "A" is pointer to the current panel holding the
    Householder vectors for the QR factorization of the panel. This routine
    puts ones on the diagonal and zeros in the upper triangular part of "A".
    The upper triangular values are stored in work.
    
    Then, the inverse is calculated in place in work, so as a final result,
    work holds the inverse of the upper triangular diagonal block.
*******************************************************************************/
void dsplit_diag_block_invert(
    magma_int_t ib, double *A, magma_int_t lda,
    double *work )
{
    const double c_zero = MAGMA_D_ZERO;
    const double c_one  = MAGMA_D_ONE;
    
    magma_int_t i, j, info;
    double *cola, *colw;

    for (i=0; i < ib; i++) {
        cola = A    + i*lda;
        colw = work + i*ib;
        for (j=0; j < i; j++) {
            colw[j] = cola[j];
            cola[j] = c_zero;
        }
        colw[i] = cola[i];
        cola[i] = c_one;
    }
    lapackf77_dtrtri( MagmaUpperStr, MagmaNonUnitStr, &ib, work, &ib, &info );
}


/***************************************************************************//**
    Purpose
    -------
    DGEQRF computes a QR factorization of a real M-by-N matrix A:
    A = Q * R.
    
    This version stores the triangular dT matrices used in
    the block QR factorization so that they can be applied directly (i.e.,
    without being recomputed) later. As a result, the application
    of Q is much faster. Also, the upper triangular matrices for V have 0s
    in them. The corresponding parts of the upper triangular R are inverted and
    stored separately in dT.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    tau     DOUBLE PRECISION array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    dT      (workspace) DOUBLE PRECISION array on the GPU,
            dimension (2*MIN(M, N) + ceil(N/32)*32 )*NB,
            where NB can be obtained through magma_get_dgeqrf_nb( M, N ).
            It starts with a MIN(M,N)*NB block that stores the triangular T
            matrices, followed by a MIN(M,N)*NB block that stores inverses of
            the diagonal blocks of the R matrix.
            The rest of the array is used as workspace.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v^H

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf
*******************************************************************************/
extern "C" magma_int_t
magma_dgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT,
    magma_int_t *info )
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, (dA_offset + (i_) + (j_)*(ldda))
    #define dT(i_)      dT, (dT_offset + (i_)*nb)
    #define dR(i_)      dT, (dT_offset + (  minmn + (i_))*nb)
    #define dwork(i_)   dT, (dT_offset + (2*minmn + (i_))*nb)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*(ldda))
    #define dT(i_)     (dT + (i_)*nb)
    #define dR(i_)     (dT + (  minmn + (i_))*nb)
    #define dwork(i_)  (dT + (2*minmn + (i_))*nb)
    #endif

    /* Define for ABFT */
    #define dA_colchk(i_, j_)   (dA_colchk   + ((i_)/nb)*2 + (j_)*ldda_colchk)
    #define dA_rowchk(i_, j_)   (dA_rowchk   + (i_)        + ((j_)/nb)*2*ldda_rowchk)
    #define dA_colchk_r(i_, j_) (dA_colchk_r + ((i_)/nb)*2 + (j_)*ldda_colchk_r)
    #define dA_rowchk_r(i_, j_) (dA_rowchk_r + (i_)        + ((j_)/nb)*2*ldda_rowchk_r)

    #define dT_colchk(i_)   (dT_colchk   + ((i_)/nb)*2)
    #define dT_rowchk(i_)   (dT_rowchk   + (i_))
    #define dT_colchk_r(i_) (dT_colchk_r + ((i_)/nb)*2)
    #define dT_rowchk_r(i_) (dT_rowchk_r + (i_))

    #define dR_colchk(i_)   (dT_colchk   + ((minmn+(i_))/nb)*2)
    #define dR_rowchk(i_)   (dT_rowchk   + (minmn+(i_)))
    #define dR_colchk_r(i_) (dT_colchk_r + ((minmn+(i_))/nb)*2)
    #define dR_rowchk_r(i_) (dT_rowchk_r + (minmn+(i_)))

    #define dwork_colchk(i_)   (dT_colchk   + ((minmn*2+(i_))/nb)*2)
    #define dwork_rowchk(i_)   (dT_rowchk   + (minmn*2+(i_)))
    #define dwork_colchk_r(i_) (dT_colchk_r + ((minmn*2+(i_))/nb)*2)
    #define dwork_rowchk_r(i_) (dT_rowchk_r + (minmn*2+(i_)))
    
    double *work, *hwork, *R;
    magma_int_t cols, i, ib, ldwork, lddwork, lhwork, lwork, minmn, nb, old_i, old_ib, rows;
    
    // check arguments
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    minmn = min( m, n );
    if (minmn == 0)
        return *info;
    
    // TODO: use min(m,n), but that affects dT
    nb = magma_get_dgeqrf_nb( m, n );
    //nb = 2;
    // dT contains 3 blocks:
    // dT    is minmn*nb
    // dR    is minmn*nb
    // dwork is n*nb
    lddwork = n;
    
    // work  is m*nb for panel
    // hwork is n*nb, and at least nb*nb for T in larft
    // R     is nb*nb
    ldwork = m;
    lhwork = max( n*nb, nb*nb );
    lwork  = ldwork*nb + lhwork + nb*nb;
    // last block needs rows*cols for matrix and prefers cols*nb for work
    // worst case is n > m*nb, m a small multiple of nb:
    // needs n*nb + n > (m+n)*nb
    // prefers 2*n*nb, about twice above (m+n)*nb.
    i = ((minmn-1)/nb)*nb;
    lwork = max( lwork, (m-i)*(n-i) + (n-i)*nb );
    
    if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, lwork )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    hwork = work + ldwork*nb;
    R     = work + ldwork*nb + lhwork;
    memset( R, 0, nb*nb*sizeof(double) );
    
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    if (nvmlInit () != NVML_SUCCESS)
    {
        printf("init error\n");
        return 0;
    }
    int d = 0;
    nvmlReturn_t result;
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(d, &device);
    if (NVML_SUCCESS != result)
    {
      printf("Failed to get handle for device %d: %s\n", d, nvmlErrorString(result));
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
    double slack_time_actual_avg = 0.0;

    double slack_time;
    double reclaimnation_ratio = 0.0;

    int tmu_desired_freq;
    int tmu_freq;
    int tmu_curr_freq = 1800;
    int tmu_base_freq = 1800;

    int tmu_base_offset = 0;
    int tmu_opt_offset = 200;
    adj_gpu(device, tmu_base_freq, 338000); // lock frequency is necessary for accurate prediction
    offset_gpu(tmu_base_offset);

    int pd_desired_freq;
    int pd_freq;
    int pd_curr_freq = 4500;
    int pd_base_freq = 4500;
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

    bool reclaim_slack = false;
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


    printf( "initialize transposed checksum vector on CPU\n");
    double * chk_vt;
    int ld_chk_vt = 2;
    magma_dmalloc_pinned(&chk_vt, 2 * nb * sizeof(double));
    for (int i = 0; i < nb; ++i) {
        *(chk_vt + i * ld_chk_vt) = 1;
    }
    for (int i = 0; i < nb; ++i) {
        *(chk_vt + i * ld_chk_vt + 1) = i + 1;
    }

    if (DEBUG) {
        printf("checksum vector on CPU:\n");
        printMatrix_host(chk_vt, ld_chk_vt, 2, nb, -1, -1);
    }

    printf( "initialize tranposed checksum vector on GPUs\n");
    //printf("%d\n", nb);
    double * dev_chk_vt;
    size_t pitch_dev_chk_vt = magma_roundup(2 * sizeof(double), 32);
    int ld_dev_chk_vt;
    
    magma_dmalloc(&dev_chk_vt, pitch_dev_chk_vt * nb);
    ld_dev_chk_vt = pitch_dev_chk_vt / sizeof(double);
    magma_dgetmatrix(2, nb,
                     chk_vt, ld_chk_vt, 
                     dev_chk_vt, ld_dev_chk_vt,
                     queues[1]);
    if (DEBUG) {
        printMatrix_gpu(dev_chk_vt, ld_dev_chk_vt,
                        2, nb, nb, nb, queues[1]);
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
    printf( "allocate space for checksums of dA on GPUs......\n" );
    
    double * dA_colchk;
    size_t pitch_dA_colchk = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
    int ldda_colchk = pitch_dA_colchk / sizeof(double);
    if (COL_FT || ROW_FT) magma_dmalloc(&dA_colchk, pitch_dA_colchk * gpu_col);

    double * dA_colchk_r;
    size_t pitch_dA_colchk_r = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
    int ldda_colchk_r = pitch_dA_colchk_r / sizeof(double);
    if (COL_FT || ROW_FT) magma_dmalloc(&dA_colchk_r, pitch_dA_colchk_r * gpu_col);

    double * dA_rowchk;
    size_t pitch_dA_rowchk = magma_roundup(gpu_row * sizeof(double), 32);
    int ldda_rowchk = pitch_dA_rowchk / sizeof(double);
    if (COL_FT || ROW_FT) magma_dmalloc(&dA_rowchk, pitch_dA_rowchk * (gpu_col / nb) * 2);


    double * dA_rowchk_r;
    size_t pitch_dA_rowchk_r = magma_roundup(gpu_row * sizeof(double), 32);
    int ldda_rowchk_r = pitch_dA_rowchk_r / sizeof(double);
    if (COL_FT || ROW_FT) magma_dmalloc(&dA_rowchk_r, pitch_dA_rowchk_r * (gpu_col / nb) * 2);
       
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

    printf( "allocate space for checksums of dT on GPUs......\n" );
    
    gpu_row = minmn + minmn + n;
    gpu_col = nb;

    double * dT_colchk;
    size_t pitch_dT_colchk = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
    int lddt_colchk = pitch_dT_colchk / sizeof(double);
    if (COL_FT || ROW_FT) magma_dmalloc(&dT_colchk, pitch_dT_colchk * gpu_col);
    if (COL_FT || ROW_FT) cudaMemset(dT_colchk, 0, pitch_dT_colchk * gpu_col);

    double * dT_colchk_r;
    size_t pitch_dT_colchk_r = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
    int lddt_colchk_r = pitch_dT_colchk_r / sizeof(double);
    if (COL_FT || ROW_FT) magma_dmalloc(&dT_colchk_r, pitch_dT_colchk_r * gpu_col);

    double * dT_rowchk;
    size_t pitch_dT_rowchk = magma_roundup(gpu_row * sizeof(double), 32);
    int lddt_rowchk = pitch_dT_rowchk / sizeof(double);
    if (COL_FT || ROW_FT) magma_dmalloc(&dT_rowchk, pitch_dT_rowchk * (gpu_col / nb) * 2);
    if (COL_FT || ROW_FT) cudaMemset(dT_rowchk, 0, pitch_dT_rowchk * (gpu_col / nb) * 2);


    double * dT_rowchk_r;
    size_t pitch_dT_rowchk_r = magma_roundup(gpu_row * sizeof(double), 32);
    int lddt_rowchk_r = pitch_dT_rowchk_r / sizeof(double);
    if (COL_FT || ROW_FT) magma_dmalloc(&dT_rowchk_r, pitch_dT_rowchk_r * (gpu_col / nb) * 2);
       
    printf( "done.\n" );



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
    total_time = magma_wtime();
        
    if ( nb > 1 && nb < minmn ) {
        // need nb*nb for T in larft
        assert( lhwork >= nb*nb );
        
        // Use blocked code initially
        old_i = 0; old_ib = nb;
        for (i = 0; i < minmn-nb; i += nb) {
            ib = min( minmn-i, nb );
            rows = m - i;
            
            bool profile = false;
            if ((i/ib-last_prof_iter)%profile_interval == 0) profile = true;
            bool predict = false;
            if (i/ib > 1) predict = true;

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
                double pd_freq_ratio = (double)last_prof_freq_pd/pd_base_freq;
                double tmu_freq_ratio = (double)last_prof_freq_tmu/tmu_base_freq;
                pd_time_ratio = (2*((double)m-i)*ib*ib-2*ib*ib*ib/3)/(2*((double)m-last_prof_iter*ib)*ib*ib-2*ib*ib*ib/3);
                dt_time_ratio = (((double)m-i)*nb)/(((double)m-nb*last_prof_iter)*nb);
                // tmu_time_ratio = ((double)(n-j-jb)*jb*j)/((double)(n-jb*last_prof_iter-jb)*jb*jb*last_prof_iter);
                tmu_time_ratio = (double)abft_dlarfb_flops(MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                                           m-(i-ib), n-(i-ib)-2*ib, ib, ib, COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER)/
                                        abft_dlarfb_flops(MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                                           m-(last_prof_iter*ib-ib), n-(last_prof_iter*ib-ib)-2*ib, ib, ib, 
                                                           COL_FT, ROW_FT, CHECK_BEFORE, CHECK_AFTER);


                pd_time_pred = pd_time_prof * pd_time_ratio * pd_freq_ratio;
                dt_time_pred = dt_time_prof * dt_time_ratio;
                tmu_time_pred = tmu_time_prof * tmu_time_ratio * tmu_freq_ratio;
                slack_time = tmu_time_pred - pd_time_pred - dt_time_pred;

                printf("i = %d\n", i);
                printf("[Last Iter] profiled time  PD: %6.2f, TMU: %6.2f, DT: %6.2f\n", pd_time_prof, tmu_time_prof, dt_time_prof);
                printf("[Last Iter] used freq.     PD: %6d, TMU: %6d\n", last_prof_freq_pd, last_prof_freq_tmu);
                printf("[Curr Iter] workload ratio PD: %6.2f, TMU: %6.2f, DT: %6.2f\n", pd_time_ratio, tmu_time_ratio, dt_time_ratio);
                printf("[Curr Iter] freq ratio     PD: %6.2f, TMU: %6.2f\n", pd_freq_ratio, tmu_freq_ratio);
                printf("[Curr Iter] predicted time PD: %6.2f, TMU: %6.2f, DT: %6.2f\n", pd_time_pred, tmu_time_pred, dt_time_pred);

                // determine frequency
                if (reclaim_slack) {
                    double gpu_adj_time = 15;
                    double cpu_adj_time = 80;

                    double tmu_desired_time = tmu_time_pred-(reclaimnation_ratio*slack_time)-gpu_adj_time;
                    double pd_desired_time = tmu_desired_time-cpu_adj_time-dt_time_pred;

                    if (tmu_desired_time < 0) tmu_desired_time = 1;
                    if (pd_desired_time < 0 ) pd_desired_time = 1;
                    printf("[Curr Iter] desired time   PD: %6.2f, TMU: %6.2f\n", pd_desired_time, tmu_desired_time);

                    tmu_desired_freq = tmu_time_pred/tmu_desired_time * tmu_base_freq;
                    tmu_freq = tmu_desired_freq;
                    if (tmu_desired_freq > 2000) tmu_freq = 2000;
                    if (tmu_desired_freq < 500) tmu_freq = 500;
                    tmu_freq = (int)ceil((double)tmu_freq/100)*100;

                    pd_desired_freq = pd_time_pred/pd_desired_time * pd_base_freq;
                    pd_freq = pd_desired_freq;
                    if (pd_desired_freq > 4500) pd_freq = 4500;
                    if (pd_desired_freq < 1000) pd_freq = 1000;
                    pd_freq = (int)ceil((double)pd_freq/100)*100;
                    
                    // performance cannot be worse than baseline
                    double max_pd_tmu = max(pd_time_pred+dt_time_pred, tmu_time_pred);

                    // projected execution time if we apply frequency
                    pd_freq_ratio = (float)pd_base_freq/pd_freq;
                    tmu_freq_ratio = (float)tmu_base_freq/tmu_freq;
                    printf("[Curr Iter] planned freq.  PD: %6d, TMU: %6d\n", pd_freq, tmu_freq);
                    printf("[Curr Iter] projected time PD: %6.2f, TMU: %6.2f\n", pd_time_pred * pd_freq_ratio, tmu_time_pred * tmu_freq_ratio);

                    // determine if we need to use ABFT
                    // CHECK_AFTER = tmu_freq > 1900;
                    printf("[Curr Iter] ABFT enabled: %d\n", CHECK_AFTER);


                    //if we want to reclaim and there is benefit
                    if (reclaim_pd && pd_time_pred * pd_freq_ratio + cpu_adj_time + dt_time_pred <= max_pd_tmu) {
                        printf("[Curr Iter] PD: plan to reclaim %.2f (achievable) <  %.2f (baseline)\n", pd_time_pred * pd_freq_ratio + cpu_adj_time + dt_time_pred, max_pd_tmu);
                        pd_time_pred = pd_time_pred * pd_freq_ratio;
                    } else { //if do not want to reclaim or there is no benefit
                        printf("[Curr Iter] PD: not worth to reclaim %.2f (achievable) > %.2f (baseline)\n", pd_time_pred * pd_freq_ratio + cpu_adj_time + dt_time_pred, max_pd_tmu);
                        pd_freq_ratio = (float)pd_base_freq/pd_curr_freq;
                        pd_time_pred = pd_time_pred * pd_freq_ratio;
                        reclaim_pd = false;
                    }

                    
                    //if we want to reclaim and there is benefit
                    if (reclaim_tmu && tmu_time_pred * tmu_freq_ratio + gpu_adj_time <= max_pd_tmu) {
                        printf("[Curr Iter] TMU: plan to reclaim %.2f (achievable) <  %.2f (baseline)\n", tmu_time_pred * tmu_freq_ratio + gpu_adj_time, max_pd_tmu);
                        tmu_time_pred = tmu_time_pred * tmu_freq_ratio;
                    } else { //if do not want to reclaim or there is no benefit
                        printf("[Curr Iter] TMU: not worth to reclaim %.2f (achievable) <  %.2f (baseline)\n", tmu_time_pred * tmu_freq_ratio + gpu_adj_time, max_pd_tmu);
                        tmu_freq_ratio = (float)tmu_base_freq/tmu_curr_freq;
                        tmu_time_pred = tmu_time_pred * tmu_freq_ratio;
                        reclaim_tmu = false; 
                    }                
                }
            }

            magma_queue_sync( queues[0] );
            magma_queue_sync( queues[1] );

            // magma_dgetmatrix_async( rows, ib,
            //                         dA(i,i), ldda,
            //                         work,    ldwork, queues[1] );


            if (i > 0) {
                // Apply H^H to A(i:m,i+2*ib:n) from the left
                cols = n - old_i - 2*old_ib;
                // magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                //                   m-old_i, cols, old_ib,
                //                   dA(old_i, old_i         ), ldda, dT(old_i), nb,
                //                   dA(old_i, old_i+2*old_ib), ldda, dwork(0),  lddwork, queues[0] );

                if (reclaim_tmu && i > ib) {
                    if (tmu_freq != tmu_curr_freq) {
                        adj_gpu(device, tmu_freq, 338000);
                        //printf("set to %d\n", tmu_freq);
                        tmu_curr_freq = tmu_freq;
                    }
                }
                cudaEventRecord(start, queues[0]->cuda_stream());
                // printf("[i = %d]abft_dlarfb_gpu(%d %d %d)\n",i,  m-old_i, cols, old_ib);
                abft_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                 m-old_i, cols, old_ib,
                                  dA(old_i, old_i         ), ldda, dT(old_i), nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork(0),  lddwork, 
                                  nb,
                                  dA_colchk(old_i, old_i),   ldda_colchk,
                                  dA_rowchk(old_i, old_i),   ldda_rowchk,
                                  dA_colchk_r(old_i, old_i), ldda_colchk_r,
                                  dA_rowchk_r(old_i, old_i), ldda_rowchk_r,
                                  dT_colchk(old_i),   lddt_colchk,
                                  dT_rowchk(old_i),   lddt_rowchk,
                                  dT_colchk_r(old_i), lddt_colchk_r,
                                  dT_rowchk_r(old_i), lddt_rowchk_r,
                                  dA_colchk(old_i, old_i+2*old_ib),   ldda_colchk,
                                  dA_rowchk(old_i, old_i+2*old_ib),   ldda_rowchk,
                                  dA_colchk_r(old_i, old_i+2*old_ib), ldda_colchk_r,
                                  dA_rowchk_r(old_i, old_i+2*old_ib), ldda_rowchk_r,
                                  dwork_colchk(0),   lddt_colchk,
                                  dwork_rowchk(0),   lddt_rowchk,
                                  dwork_colchk_r(0), lddt_colchk_r,
                                  dwork_rowchk_r(0), lddt_rowchk_r,
                                  dev_chk_v, ld_dev_chk_v,
                                  COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER, 
                                  queues[0], queues[0]);
                cudaEventRecord(stop, queues[0]->cuda_stream());
                
                // Fix the diagonal block
                magma_dsetmatrix_async( old_ib, old_ib,
                                        R,         old_ib,
                                        dR(old_i), old_ib, queues[0] );
            }


             
            
            // magma_queue_sync( queues[1] );  // wait to get work(i)
            if (reclaim_pd && i > ib) {
                if (pd_freq != pd_curr_freq) {
                    adj_cpu(pd_freq);
                    pd_curr_freq = pd_freq;
                }
            }
            
            // get i-th panel from device
            dt_time_actual = magma_wtime();
            magma_dgetmatrix_async( rows, ib,
                                    dA(i,i), ldda,
                                    work,    ldwork, queues[1] );
            magma_queue_sync( queues[1] );  // wait to get work(i)
            dt_time_actual = magma_wtime() - dt_time_actual;
       
            pd_time_actual = magma_wtime();

            

            lapackf77_dgeqrf( &rows, &ib, work, &ldwork, &tau[i], hwork, &lhwork, info );
            // Form the triangular factor of the block reflector in hwork
            // H = H(i) H(i+1) . . . H(i+ib-1)
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              work, &ldwork, &tau[i], hwork, &ib );



            // if (i == ib) {
            
            //     double * correct_result = new double[rows*ib];
            //     for (int i = 0; i < rows; i++) {
            //         for (int j = 0; j < ib; j++) {
            //             correct_result[i+j*rows] = work[i+j*ldwork];
            //         }
            //     }

            //     int R = 100;
            //     double ** tmp = new double*[R];
            //     for (int r = 0; r < R; r++) tmp[r] = new double[rows*ib];


            //     for (int f = 1000; f <= 4500; f += 500) {
            //         for (int r = 0; r < R; r++) {
            //             magma_dgetmatrix_async( rows, ib,
            //                             dA(i,i), ldda,
            //                             tmp[r],    rows, queues[1] );
            //         }
            //         magma_queue_sync( queues[1] );  // wait to get work(i)

            //         printf("frequency = %d\n", f);
            //         adj_cpu(f);
            //         pid = start_measure_cpu();
            //         for (int r = 0; r < R; r++) {
            //             lapackf77_dgeqrf( &rows, &ib, tmp[r],    &rows, &tau[i], hwork, &lhwork, info );
            //             // Form the triangular factor of the block reflector in hwork
            //             // H = H(i) H(i+1) . . . H(i+ib-1)
            //             lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
            //                               &rows, &ib,
            //                              tmp[r],    &rows, &tau[i], hwork, &ib );
            //         }
            //         stop_measure_cpu(pid);

            //         for (int r = 0; r < R; r++) {
            //             for (int i = 0; i < rows; i++) {
            //                 for (int j = 0; j < ib; j++) {
            //                     if (abs(correct_result[i+j*rows] - tmp[r][i+j*rows])> 0.001) {
            //                         printf("error: %d %d %d\n", r, i, j);
            //                     }
            //                 }
            //             }
            //         }

            //     }
            // }

            pd_time_actual = magma_wtime() - pd_time_actual;
            pd_time_actual *= 1000;
            if (profile) pd_time_prof = pd_time_actual;
            if (predict) pd_avg_error += fabs(pd_time_actual - pd_time_pred) / pd_time_actual;

            
            // send i-th V matrix to device (this cannot be here)
            // dt_time_actual = (-1)*dt_time_actual;
            // dt_time_actual += magma_wtime();
            // magma_dsetmatrix( rows, ib,
            //                   work, ldwork,
            //                   dA(i,i), ldda, queues[1] );
            // magma_queue_sync( queues[1] ); 
            // dt_time_actual = magma_wtime() - dt_time_actual;
            // dt_time_actual *= 1000;
            if (profile) dt_time_prof = dt_time_actual;
            if (predict) dt_avg_error += fabs(dt_time_actual - dt_time_pred) / dt_time_actual;


            // wait for previous trailing matrix update (above) to finish with R
            magma_queue_sync( queues[0] );
            cudaEventSynchronize(stop);
            //tmu_time = magma_wtime() - tmu_time;
            float t;
            cudaEventElapsedTime(&t, start, stop);
            tmu_time_actual = t;
            if (profile) tmu_time_prof = tmu_time_actual;
            if (predict) tmu_avg_error += fabs(tmu_time_actual - tmu_time_pred) / tmu_time_actual;

            // copy the upper triangle of panel to R and invert it, and
            // set  the upper triangle of panel (V) to identity
            dsplit_diag_block_invert( ib, work, ldwork, R );
            

            magma_dsetmatrix( rows, ib,
                              work, ldwork,
                              dA(i,i), ldda, queues[1] );
            
            if (COL_FT || ROW_FT) {
                col_chk_enc(rows, ib, nb, 
                            dA(i,i), ldda,  
                            dev_chk_v, ld_dev_chk_v, 
                            dA_colchk(i,i), ldda_colchk, 
                            queues[1]);
                row_chk_enc(rows, ib, nb, 
                            dA(i,i), ldda,  
                            dev_chk_v, ld_dev_chk_v, 
                            dA_rowchk(i,i), ldda_rowchk, 
                            queues[1]);
                // if (DEBUG) {
                //     printf( "Panel A (%d * %d):\n", rows, ib);
                //     printMatrix_gpu(dA(i,i), ldda, rows, ib,  nb, nb, queues[1]);
                //     printf( "column chk:\n" );
                //     printMatrix_gpu(dA_colchk(i,i), ldda_colchk, 
                //                     (rows / nb) * 2, ib, 2, nb, queues[1]);
                //     printf( "row chk:\n" );
                //     printMatrix_gpu(dA_rowchk(i,i), ldda_rowchk,  
                //                     rows, (ib / nb) * 2, nb, 2, queues[1]);
                // }
            }
            
            if (i + ib < n) {
                // send T matrix to device
                magma_dsetmatrix( ib, ib,
                                  hwork, ib,
                                  dT(i), nb, queues[1] );

                if (COL_FT || ROW_FT) {
                    col_chk_enc_triblock(MagmaUpper, ib,
                                        dT(i), nb,
                                        dev_chk_vt, ld_dev_chk_vt,
                                        dT_colchk(i), lddt_colchk, 
                                        queues[1]);
                    row_chk_enc_triblock(MagmaUpper, ib,
                                        dT(i), nb,
                                        dev_chk_v, ld_dev_chk_v,
                                        dT_rowchk(i), lddt_rowchk, 
                                        queues[1]);
                }

                // if (DEBUG) {
                //     printf( "T (%d * %d):\n", ib, ib);
                //     printMatrix_gpu(dT(i), nb, ib, ib,  nb, nb, queues[1]);
                //     printf( "column chk:\n" );
                //     printMatrix_gpu(dT_colchk(i), lddt_colchk, 
                //                     2, ib, 2, nb, queues[1]);
                //     printf( "row chk:\n" );
                //     printMatrix_gpu(dT_rowchk(i), lddt_rowchk,  
                //                     ib, 2, nb, 2, queues[1]);

                // }
                
                if (i+nb < minmn-nb) {
                    // Apply H^H to A(i:m,i+ib:i+2*ib) from the left
                    // magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                    //                   rows, ib, ib,
                    //                   dA(i, i   ), ldda, dT(i),  nb,
                    //                   dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );

                    //  if (DEBUG) {
                    //     gpu_row = rows;
                    //     gpu_col = ib;
                    //     printf( "input matrix A:\n" );
                    //     printMatrix_gpu(dA(i, i+ib), ldda, gpu_row, gpu_col, nb, nb, queues[1]);
                    //     printf( "column chk:\n" );
                    //     printMatrix_gpu(dA_colchk(i, i+ib), ldda_colchk, 
                    //                     (gpu_row / nb) * 2, gpu_col, 2, nb, queues[1]);
                    //     printf( "row chk:\n" );
                    //     printMatrix_gpu(dA_rowchk(i, i+ib), ldda_rowchk,  
                    //                     gpu_row, (gpu_col / nb) * 2, nb, 2, queues[1]);
                    // }


                    abft_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                     rows, ib, ib,
                                     dA(i, i   ), ldda, dT(i),  nb,
                                     dA(i, i+ib), ldda, dwork(0), lddwork,
                                     nb,
                                     dA_colchk(i, i),   ldda_colchk,
                                     dA_rowchk(i, i),   ldda_rowchk,
                                     dA_colchk_r(i, i), ldda_colchk_r,
                                     dA_rowchk_r(i, i), ldda_rowchk_r,
                                     dT_colchk(i),   lddt_colchk,
                                     dT_rowchk(i),   lddt_rowchk,
                                     dT_colchk_r(i), lddt_colchk_r,
                                     dT_rowchk_r(i), lddt_rowchk_r,
                                     dA_colchk(i, i+ib),   ldda_colchk,
                                     dA_rowchk(i, i+ib),   ldda_rowchk,
                                     dA_colchk_r(i, i+ib), ldda_colchk_r,
                                     dA_rowchk_r(i, i+ib), ldda_rowchk_r,
                                     dwork_colchk(0),   lddt_colchk,
                                     dwork_rowchk(0),   lddt_rowchk,
                                     dwork_colchk_r(0), lddt_colchk_r,
                                     dwork_rowchk_r(0), lddt_rowchk_r,
                                     dev_chk_v, ld_dev_chk_v,
                                     COL_FT, ROW_FT, DEBUG, CHECK_BEFORE, CHECK_AFTER, 
                                     queues[1], queues[1]);
                    // wait for larfb to finish with dwork before larfb in next iteration starts
                    magma_queue_sync( queues[1] );
                }
                else {
                    // Apply H^H to A(i:m,i+ib:n) from the left
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    // Fix the diagonal block
                    magma_dsetmatrix( ib, ib,
                                      R,     ib,
                                      dR(i), ib, queues[1] );
                }
                old_i  = i;
                old_ib = ib;

            }
            if (profile) {
                last_prof_iter = i/ib;
                last_prof_freq_pd = pd_curr_freq;
                last_prof_freq_tmu = tmu_curr_freq;
            }
            // printf("%d, pd-tmu, %f, %f, %f, %f, %f, %f, %d, %d, \n", j, pd_time_actual, dt_time_actual, tmu_time_actual, pd_time_pred, dt_time_pred, tmu_time_pred, pd_freq, tmu_freq);
            if (i > 0) slack_time_actual_avg += fabs(tmu_time_actual-pd_time_actual);
        
        }
        total_time = magma_wtime() - total_time;
        stop_measure_cpu(pid);
        start_energy = stop_measure_gpu(device, start_energy);
        printf("GPU energy: %f Joule\n", (float)start_energy/1000.0);
        printf("Prediction average error: CPU %f, DT, %f, GPU %f\n", pd_avg_error/(n/ib-1), dt_avg_error/(n/ib-1),tmu_avg_error/(n/ib-1));
        printf("Total time: %f\n", total_time);
        printf("Average slack: %f\n", slack_time_actual_avg/(n/ib-1));
        //reset
        adj_gpu(device, 1500, 338000);
        offset_gpu(0);
        // reset_cpu();

    } else {
        i = 0;
    }
    
    // Use unblocked code to factor the last or only block.
    if (i < minmn) {
        rows = m-i;
        cols = n-i;
        magma_dgetmatrix( rows, cols, dA(i, i), ldda, work, rows, queues[1] );
        // see comments for lwork above
        lhwork = lwork - rows*cols;
        lapackf77_dgeqrf( &rows, &cols, work, &rows, &tau[i], &work[rows*cols], &lhwork, info );
        magma_dsetmatrix( rows, cols, work, rows, dA(i, i), ldda, queues[1] );
    }
        
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free_pinned( work );
    
    return *info;
} // magma_dgeqrf_gpu
