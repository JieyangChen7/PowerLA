/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @generated from magmablas/zherk_small_reduce.cu, normal z -> c, Thu Oct  8 23:05:33 2020
*/

#include "magma_internal.h"
#include "atomics.cuh"
#include "batched_kernel_param.h"

#define PRECISION_c
#define SLDA(N)    ( (N==15||N==23||N==31)? N : (N+1) )

extern __shared__ magmaFloatComplex zdata[];
//-----------------------------------------------------------------------------
__global__ void
cherk_small_reduce_scale_beta_kernel(magma_uplo_t uplo, int N, magmaFloatComplex beta, magmaFloatComplex* dC, int lddc)
{
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gty = blockIdx.y * blockDim.y + threadIdx.y;

    magma_int_t lo = (uplo == MagmaLower) ? gty : gtx;
    magma_int_t hi = (uplo == MagmaLower) ? gtx : gty;
    
    if(gtx < N && gty < N && lo <= hi) {
        // ignore the imaginary part of C for complex precisions, as per the definition of HERK
        magmaFloatComplex rC = dC[gty * lddc + gtx];
        #if defined(PRECISION_z) || defined(PRECISION_c)
        if(gtx == gty) rC = MAGMA_C_MAKE( MAGMA_C_REAL(rC), MAGMA_D_ZERO );
        #endif
        dC[gty * lddc + gtx] = beta * rC;
    }
}

//-----------------------------------------------------------------------------
template<int N>
__global__ void
cherk_small_reduce_kernel(
        magma_uplo_t uplo, magma_trans_t trans, int k, 
        const magmaFloatComplex alpha, magmaFloatComplex *dA, const int ldda, 
        magmaFloatComplex *dC, const int lddc, const int nthread_blocks)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int bx = blockIdx.x * blockDim.z + tz;
    const int slda = SLDA(N);
    magmaFloatComplex rTmp = MAGMA_C_ZERO;

    magmaFloatComplex* sA = (magmaFloatComplex*)(zdata);
    sA += tz * slda * N;

    // make sure only nthread_blocks blocks are used
    const int max_nblocks = magma_ceildiv(k, N);
    const int nblocks = min( nthread_blocks, max_nblocks );
    if(bx >= nblocks) return;

    // determine your share of k
    const int segment = magma_roundup(k, nblocks) / nblocks; 
    const int myk = min(segment, k - bx * segment);

    // advance dA
    dA += ( trans == MagmaNoTrans ) ? bx * segment * ldda : bx * segment;

    // main loop
    int kk = 0;
    for(kk = 0; kk < myk-N; kk += N) {
        // read A
        sA[ty * slda + tx] = dA[ty * ldda + tx];
        __syncthreads();
        
        // multiply A x A^T or A^T x A
        if(trans == MagmaNoTrans) {
            #pragma unroll
            for(int j = 0; j < N; j++){
                rTmp += sA[j * slda + tx] * MAGMA_C_CONJ( sA[j * slda + ty] );
            }
        }
        else {
            #pragma unroll
            for(int j = 0; j < N; j++){
                rTmp += MAGMA_C_CONJ(sA[tx * slda + j]) * sA[ty * slda + j];
            }
        }
        __syncthreads();

        // advance A
        dA += ( trans == MagmaNoTrans ) ? N * ldda : N;
    }
    
    // txy is used for last partial block
    const int txy = (trans == MagmaNoTrans) ? ty : tx;
    if(txy < myk-kk) {
        sA[ty * slda + tx] = dA[ty * ldda + tx];
    }
    else {
        sA[ty * slda + tx] = MAGMA_C_ZERO;
    }
    __syncthreads();
    // multiply A x A^T or A^T x A
    if(trans == MagmaNoTrans) {
        #pragma unroll
        for(int j = 0; j < N; j++){
            rTmp += sA[j * slda + tx] * MAGMA_C_CONJ( sA[j * slda + ty] );
        }
    }
    else {
        #pragma unroll
        for(int j = 0; j < N; j++){
            rTmp += MAGMA_C_CONJ(sA[tx * slda + j]) * sA[ty * slda + j];
        }
    }

    // write through atomics
    magma_int_t tlo = (uplo == MagmaLower) ? ty : tx;
    magma_int_t thi = (uplo == MagmaLower) ? tx : ty;
    if(tlo <= thi)
        magmablas_catomic_add(dC + ty*lddc + tx, alpha * rTmp);
}

/***************************************************************************//**
    Purpose
    -------
    CHERK performs one of the Hermitian rank k operations

    C := alpha*A*A**H + beta*C,

    or

    C := alpha*A**H*A + beta*C,

    where alpha and beta are real scalars, C is an n by n Hermitian
    matrix and A is an n by k matrix in the first case and a k by n
    matrix in the second case.

    This is a special routine that supports n up to 32 only. It assumes that 
    k is very large so that the computation of the small matrix C is distributed 
    across many thread blocks. The number of thread blocks can be defined by the 
    user through the interface. However, the kernel can work with a maximum of 
    ceil(k / n) thread blocks. Extra thread blocks, if any, are ignored by the kernel. 
    Reduction across thread blocks is performed using atomics. 

    Parameters
    ----------

    @param[in]
    uplo    magma_uplo_t.
           On entry, uplo specifies whether the upper or lower
           triangular part of the array C is to be referenced as
           follows:

           uplo = MagmaUpper Only the upper triangular part of C
           is to be referenced.

           uplo = MagmaLower Only the lower triangular part of C
           is to be referenced.

    @param[in]
    trans   magma_trans_t.
            On entry, trans specifies the operation to be performed as
            follows:

            trans = MagmaNoTrans,   C := alpha*A*A**H + beta*C.

            trans = MagmaConjTrans, C := alpha*A**H*A + beta*C.

    @param[in]
    n       INTEGER.
            On entry,  specifies the order of the matrix C. N must be
            at least zero, and at most 32.

    @param[in]
    k       INTEGER.
            On entry with trans = MagmaNoTrans, k specifies the number
            of columns of the matrix A, and on entry with
            trans = MagmaConjTrans, k specifies the number of rows of the
            matrix A. K must be at least zero.

    @param[in]
    alpha   REAL
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA       A COMPLEX array DIMENSION ( ldda, ka ), where ka is
             k  when  trans = MagmaNoTrans,  and is  n  otherwise.
             Before entry with  trans = MagmaNoTrans,  the leading  n by k
             part of the array A must contain the matrix A, otherwise
             the leading  k by n  part of the array A must contain  the
             matrix A.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of A as declared
            in the calling (sub) program. When  trans = MagmaNoTrans then
            ldda must be at least  max( 1, n ), otherwise  ldda must be at
            least  max( 1, k ).
    
    @param[in]
    beta    REAL.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.

    @param[in,out]
    dC       A COMPLEX array of DIMENSION ( lddc, n ).
             Before entry with uplo = MagmaUpper, the leading n by n
             upper triangular part of the array C must contain the upper
             triangular part of the Hermitian matrix and the strictly
             lower triangular part of C is not referenced. On exit, the
             upper triangular part of the array C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry with uplo = MagmaLower, the leading n by n
             lower triangular part of the array C must contain the lower
             triangular part of the Hermitian matrix and the strictly
             upper triangular part of C is not referenced. On exit, the
             lower triangular part of the array C is overwritten by the
             lower triangular part of the updated matrix.
             Note that the imaginary parts of the diagonal elements need
             not be set, they are assumed to be zero, and on exit they
             are set to zero.

    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of C as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, n ).
    
    @param[in]
    nthread_blocks  INTEGER
                    The number of thread blocks used to update C.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_herk
*******************************************************************************/
extern "C" void 
magmablas_cherk_small_reduce( 
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k, 
    float alpha, magmaFloatComplex* dA, magma_int_t ldda,
    float beta,  magmaFloatComplex* dC, magma_int_t lddc, 
    magma_int_t nthread_blocks, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    #if defined(PRECISION_c) || defined(PRECISION_z)
    else if ( trans != MagmaNoTrans && trans != MagmaConjTrans )
    #else
    else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
    #endif
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 )
        info = -4;
    else if ( trans == MagmaNoTrans ? ldda < n : ldda < k )
        info = -7;
    else if ( lddc < n )
        info = -10;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;   // info
    }

    magmaFloatComplex z_alpha = MAGMA_C_MAKE(alpha, 0.0);
    magmaFloatComplex z_beta = MAGMA_C_MAKE(beta, 0.0);

    // This routine supports output matrix size up to 32x32 only
    assert(n <= 32);

    // first, scale by beta
    dim3 scale_block(16, 16, 1);
    dim3 scale_grid( magma_ceildiv(n, scale_block.x), magma_ceildiv(n, scale_block.y), 1);
    cherk_small_reduce_scale_beta_kernel<<<scale_grid, scale_block, 0, queue->cuda_stream()>>>
    (uplo, n, z_beta, dC, lddc); 

    // second, alpha A x A^T or alpha A^T x A
    magma_int_t slda = SLDA(n);
    magma_int_t shmem = slda * n * sizeof(magmaFloatComplex);
    
    // check num threads and shmem
    assert(n * n <= MAX_NTHREADS);
    assert(shmem <= (47 * 1024)); // 47 KB max per thread block

    dim3 grid(nthread_blocks, 1, 1);
    dim3 threads(n, n, 1);
    
    switch(n){
        case  1: cherk_small_reduce_kernel< 1><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case  2: cherk_small_reduce_kernel< 2><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case  3: cherk_small_reduce_kernel< 3><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case  4: cherk_small_reduce_kernel< 4><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case  5: cherk_small_reduce_kernel< 5><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case  6: cherk_small_reduce_kernel< 6><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case  7: cherk_small_reduce_kernel< 7><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case  8: cherk_small_reduce_kernel< 8><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case  9: cherk_small_reduce_kernel< 9><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 10: cherk_small_reduce_kernel<10><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 11: cherk_small_reduce_kernel<11><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 12: cherk_small_reduce_kernel<12><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 13: cherk_small_reduce_kernel<13><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 14: cherk_small_reduce_kernel<14><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 15: cherk_small_reduce_kernel<15><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 16: cherk_small_reduce_kernel<16><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 17: cherk_small_reduce_kernel<17><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 18: cherk_small_reduce_kernel<18><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 19: cherk_small_reduce_kernel<19><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 20: cherk_small_reduce_kernel<20><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 21: cherk_small_reduce_kernel<21><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 22: cherk_small_reduce_kernel<22><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 23: cherk_small_reduce_kernel<23><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 24: cherk_small_reduce_kernel<24><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 25: cherk_small_reduce_kernel<25><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 26: cherk_small_reduce_kernel<26><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 27: cherk_small_reduce_kernel<27><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 28: cherk_small_reduce_kernel<28><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 29: cherk_small_reduce_kernel<29><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 30: cherk_small_reduce_kernel<30><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 31: cherk_small_reduce_kernel<31><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        case 32: cherk_small_reduce_kernel<32><<<grid, threads, shmem, queue->cuda_stream()>>>(uplo, trans, k, z_alpha, dA, ldda, dC, lddc, nthread_blocks); break;
        default: {printf("N = %lld is not supported\n", (long long)n);}
    }
}


