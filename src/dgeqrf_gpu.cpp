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
    nb = 4;
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
    bool DEBUG = true;
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

    printf( "allocate space for checksums of dT on GPUs......\n" );
    
    gpu_row = nb;
    gpu_col = minmn;

    double * dT_colchk;
    size_t pitch_dT_colchk = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
    int lddt_colchk = pitch_dT_colchk / sizeof(double);
    magma_dmalloc(&dT_colchk, pitch_dT_colchk * gpu_col);

    double * dT_colchk_r;
    size_t pitch_dT_colchk_r = magma_roundup((gpu_row / nb) * 2 * sizeof(double), 32);
    int lddt_colchk_r = pitch_dT_colchk_r / sizeof(double);
    magma_dmalloc(&dT_colchk_r, pitch_dT_colchk_r * gpu_col);

    double * dT_rowchk;
    size_t pitch_dT_rowchk = magma_roundup(gpu_row * sizeof(double), 32);
    int lddt_rowchk = pitch_dT_rowchk / sizeof(double);
    magma_dmalloc(&dT_rowchk, pitch_dT_rowchk * (gpu_col / nb) * 2);


    double * dT_rowchk_r;
    size_t pitch_dT_rowchk_r = magma_roundup(gpu_row * sizeof(double), 32);
    int lddt_rowchk_r = pitch_dT_rowchk_r / sizeof(double);
    magma_dmalloc(&dT_rowchk_r, pitch_dT_rowchk_r * (gpu_col / nb) * 2);
       
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
    total_time = magma_wtime();
        
    if ( nb > 1 && nb < minmn ) {
        // need nb*nb for T in larft
        assert( lhwork >= nb*nb );
        
        // Use blocked code initially
        old_i = 0; old_ib = nb;
        for (i = 0; i < minmn-nb; i += nb) {
            ib = min( minmn-i, nb );
            rows = m - i;
            
            // get i-th panel from device
            magma_dgetmatrix_async( rows, ib,
                                    dA(i,i), ldda,
                                    work,    ldwork, queues[1] );
            if (i > 0) {
                // Apply H^H to A(i:m,i+2*ib:n) from the left
                cols = n - old_i - 2*old_ib;
                magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, cols, old_ib,
                                  dA(old_i, old_i         ), ldda, dT(old_i), nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork(0),  lddwork, queues[0] );
                
                // Fix the diagonal block
                magma_dsetmatrix_async( old_ib, old_ib,
                                        R,         old_ib,
                                        dR(old_i), old_ib, queues[0] );
            }
            
            magma_queue_sync( queues[1] );  // wait to get work(i)
            lapackf77_dgeqrf( &rows, &ib, work, &ldwork, &tau[i], hwork, &lhwork, info );
            // Form the triangular factor of the block reflector in hwork
            // H = H(i) H(i+1) . . . H(i+ib-1)
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              work, &ldwork, &tau[i], hwork, &ib );
            
            // wait for previous trailing matrix update (above) to finish with R
            magma_queue_sync( queues[0] );
            
            // copy the upper triangle of panel to R and invert it, and
            // set  the upper triangle of panel (V) to identity
            dsplit_diag_block_invert( ib, work, ldwork, R );
            
            // send i-th V matrix to device
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
                if (DEBUG) {
                    printf( "Panel A (%d * %d):\n", rows, ib);
                    printMatrix_gpu(dA(i,i), ldda, rows, ib,  nb, nb, queues[1]);
                    printf( "column chk:\n" );
                    printMatrix_gpu(dA_colchk(i,i), ldda_colchk, 
                                    (rows / nb) * 2, ib, 2, nb, queues[1]);
                    printf( "row chk:\n" );
                    printMatrix_gpu(dA_rowchk(i,i), ldda_rowchk,  
                                    rows, (ib / nb) * 2, nb, 2, queues[1]);
                }
            }
            
            if (i + ib < n) {
                // send T matrix to device
                magma_dsetmatrix( ib, ib,
                                  hwork, ib,
                                  dT(i), nb, queues[1] );
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
                
                if (DEBUG) {
                    printf( "T (%d * %d):\n", ib, ib);
                    printMatrix_gpu(dT(i), nb, ib, ib,  nb, nb, queues[1]);
                    printf( "column chk:\n" );
                    printMatrix_gpu(dT_colchk(i), lddt_colchk, 
                                    2, ib, 2, nb, queues[1]);
                    printf( "row chk:\n" );
                    printMatrix_gpu(dT_rowchk(i), lddt_rowchk,  
                                    ib, 2, nb, 2, queues[1]);

                }
                
                if (i+nb < minmn-nb) {
                    // Apply H^H to A(i:m,i+ib:i+2*ib) from the left
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
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
        }
        total_time = magma_wtime() - total_time;
        stop_measure_cpu(pid);
        start_energy = stop_measure_gpu(device, start_energy);
        printf("GPU energy: %llu\n", start_energy);
        printf("Prediction average error: CPU %f, GPU %f\n", pd_avg_error/(n/nb-1), tmu_avg_error/(n/nb-1));
        printf("Total time: %f\n", total_time);
        //reset
        adj_gpu(device, 1500, 338000, -500);

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
