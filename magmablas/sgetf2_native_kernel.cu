/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @generated from magmablas/zgetf2_native_kernel.cu, normal z -> s, Thu Oct  8 23:05:37 2020
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "shuffle.cuh"
#include "sync.cuh"
#include "atomics.cuh"
#include "batched_kernel_param.h"

#define PRECISION_s

/**
    Purpose
    -------
    LU factorization of m-by-n matrix ( m >= n ).
    Each thread block caches an entire column in register.
    Thread blocks communicate and synchronize through global memory.
    Assumptions:
        1. dA is of size MxN such that N <= M.
        2. Thread block must be 1D, with TX multiple of 32 (warp size)
        3. TX must be >= n
        4. n must be less than the number of SMs on the GPU
**/

// =============================================================================
// init kernel
__global__ void
sgetf2_native_init_kernel( int n, int npages, magma_int_t *ipiv, int* update_flags)
{
    const int tx = threadIdx.x;
    if( tx < n){
        ipiv[ tx ] = 0;
    }
    if( tx < max(n,npages) ){
        update_flags[ tx ] = 0;
    }
}

// =============================================================================
// the main kernel
template<int TX, int NPAGES>
__global__ void
sgetf2_native_kernel( int m, int n,
                      magmaFloat_ptr dA, int ldda,
                      volatile magma_int_t *ipiv, int gbstep,
                      volatile int* update_flag,
                      volatile magma_int_t *info)
{
#ifdef HAVE_CUBLAS
    const int tx  = threadIdx.x;
    const int bx = blockIdx.x;
    float rA[NPAGES] = {MAGMA_S_ZERO};
    float rx, rx_max;
    magmaFloat_ptr da = dA;
    int rx_id, max_id, flag = 0;
    float  rx_abs = 0.0, rx_abs_max = 0.0;
    const int m_ = m-(NPAGES-1)*TX;
    if( bx >= n ) return;

    __shared__ float sx[ TX ];
    __shared__ float sabs[ TX ];
    __shared__ int smax_id[ TX ];
    __shared__ float sreg;

    // read
    dA += bx * ldda + tx;
    #pragma unroll
    for(int i = 0; i < NPAGES-1; i++){
        rA[i] = dA[ i * TX ];
    }
    if( tx <  m_){
        rA[NPAGES-1] = dA[ (NPAGES-1) * TX ];
    }

    // main loop
    #pragma unroll
    for(int i = 0; i < n; i++){
        // isamax and write pivot for the ith thread block
        if(bx == i){
            rx_max     = rx     = (tx < i) ? MAGMA_S_ZERO : rA[0];
            rx_abs_max = rx_abs = fabs(MAGMA_S_REAL(rx)) + fabs(MAGMA_S_IMAG(rx));
            max_id = rx_id = tx;
            #pragma unroll
            for(int j = 1; j < NPAGES; j++){
                rx = rA[j];
                rx_abs = fabs(MAGMA_S_REAL(rx)) + fabs(MAGMA_S_IMAG(rx));
                if ( rx_abs  > rx_abs_max ){
                    rx_max = rx;
                    rx_abs_max = rx_abs;
                    max_id = j * TX + tx;
                }
            }
            sx[ tx ] = rx_max;
            sabs[ tx ] = rx_abs_max;
            smax_id[ tx ] = max_id;
            __syncthreads();

            // let the first warp do the final reduction step
            if(tx < 32){
                #pragma unroll
                for(int j = 0; j < TX; j+= 32){
                    rx     = sx[ j + tx ];
                    rx_abs = sabs[ j + tx ];
                    rx_id  = smax_id[ j + tx ];
                    if ( rx_abs  > rx_abs_max ){
                        rx_max = rx;
                        rx_abs_max = rx_abs;
                        max_id = rx_id;
                    }
                }
                magmablas_syncwarp();
                sx[ tx ] = rx_max;
                sabs[ tx ] = rx_abs_max;
                smax_id[ tx ] = max_id;
                magmablas_syncwarp();
                #pragma unroll
                for(int j = 0; j < 32; j++){
                    rx     = sx[j];
                    rx_abs = sabs[j];
                    rx_id  = smax_id[j];
                    if ( rx_abs  > rx_abs_max ){
                        rx_abs_max = rx_abs;
                        rx_max = rx;
                        max_id = rx_id;
                    }
                }
            }

            if(tx == 0){
                sx[ 0 ] = rx_max;
                sabs[ 0 ] = rx_abs_max;
                smax_id[ 0 ] = max_id;
            }
            __syncthreads();
            rx_max = sx[ 0 ];
            rx_abs_max = sabs[ 0 ];
            max_id = smax_id[ 0 ];
            __syncthreads();

            // now every thread in the i^th block has the maximum
            if( tx == 0){
                if( rx_abs_max == MAGMA_D_ZERO){
                    magmablas_iatomic_exchange( (magma_int_t*)info, (magma_int_t)(max_id + gbstep + 1) );
                }
                magmablas_iatomic_exchange((magma_int_t*)&ipiv[i], (magma_int_t)(max_id+1) ); // fortran indexing
            }
            __syncthreads();
            if( rx_abs_max == MAGMA_D_ZERO )return;
        }
        else{ // other thread blocks are waiting
            if(tx == 0){
                max_id = 0;
                while( max_id == 0 ){
                    max_id = ipiv[i];
                };
                smax_id[ 0 ] = max_id;
            }
            __syncthreads();
            max_id = smax_id[ 0 ];
            max_id -= 1; // revert fortran indexing
            __syncthreads();
            if( (*info) != 0 ) return;
        }

        // swap
        // swap always happens between page 0 and page x
        // to avoid spilling rA to local memory, we use shared memory
        if( max_id != i){
            // all blocks swap in registers
            // for bx < i, the column is already written in memory,
            // but we have a copy in reg., so continue to swap in reg.,
            // and do one final write to memory
            #pragma unroll
            for(int j = 0; j < NPAGES; j++){
                if( j == (max_id/TX) ){
                    sx[ tx ] = rA[j];
                    __syncthreads();
                    if( tx == i ){
                        float tmp    = sx[ max_id%TX ];
                        sx[ max_id%TX ] = rA[0];
                        rA[0] = tmp;
                    }
                    __syncthreads();
                    if( tx == max_id%TX ){
                        rA[j] = sx[ tx ];
                    }
                    __syncthreads();
                }
            }
            //__syncthreads();
        }

        // the ith block does scal
        if(bx == i){
            float reg = MAGMA_S_DIV(MAGMA_S_ONE, rx_max );
            // scal
            if( tx > i ){
                rA[0] *= reg;
            }
            #pragma unroll
            for(int j = 1; j < NPAGES; j++){
                rA[j] *= reg;
            }
            // write column i to global memory
            #pragma unroll
            for(int j = 0; j < NPAGES-1; j++){
                dA[ j * TX ] = rA[j];
            }
            if( tx <  m_){
                dA[ (NPAGES-1) * TX ] = rA[NPAGES-1];
            }
            __threadfence(); __syncthreads(); // after cuda 9.0, both are needed, not sure why
            if(tx == 0) magmablas_iatomic_exchange( (int *)&update_flag[ i ], 1);
        }

        // thread blocks with ID larger than i perform ger
        if(bx > i){
            if( tx == i ){
                sreg = rA[0];
            }
            // wait for scal
            if( tx == 0){
                flag = 0;
                while( flag == 0 ){
                    flag = update_flag[ i ];
                };
            }
            __syncthreads();

            float reg = sreg;
            if( NPAGES == 1){
                if(tx > i && tx < m_){
                    rA[0] -= da[ i * ldda + tx ] * reg;
                }
            }else{
                if(tx > i){
                    rA[0] -= da[ i * ldda + tx ] * reg;
                }
            }
            #pragma unroll
            for(int j = 1; j < NPAGES-1; j++){
                rA[j] -= da[ i * ldda + j * TX + tx ] * reg;
            }
            if( NPAGES > 1){
                if( tx < m_ ){
                    rA[ NPAGES-1 ] -= da[ i * ldda + (NPAGES-1)*TX + tx ] * reg;
                }
            }
        }
    }

    // all blocks write their columns again except the last one
    if( bx < n-1 ){
        #pragma unroll
        for(int i = 0; i < NPAGES-1; i++){
            dA[ i * TX ] = rA[i];
        }
        if( tx <  m_){
            dA[ (NPAGES-1) * TX ] = rA[NPAGES-1];
        }
    }

#endif    // HAVE_CUBLAS
}

// =============================================================================
extern "C" magma_int_t
magma_sgetf2_native_fused(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t gbstep,
    magma_int_t *flags,
    magma_int_t *info, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    const magma_int_t ntx   = SGETF2_FUSED_NTH;

    if( m < n || m > SGETF2_FUSED_MAX_M ){
        arginfo = -1;
    }
    else if( n > magma_getdevice_multiprocessor_count() ){
        arginfo = -2;
    }
    else if( ldda < max(1, m) ){
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t arch = magma_getdevice_arch();

    dim3 grid(n, 1, 1);
    dim3 threads(ntx, 1, 1);
    const magma_int_t npages = magma_ceildiv(m, ntx);
    // the kernel uses communication among thread blocks
    // as a safeguard, force one thread block per multiprocessor
    // by allocating more than half the shared memory
    magma_int_t shmem = magma_getdevice_shmem_block();
    shmem = (shmem / 2);
    int *update_flag = (int*) flags;    // update_flag is an int, not magma_int_t
    sgetf2_native_init_kernel<<< 1, max(n,npages), 0, queue->cuda_stream() >>>( n, npages, ipiv, update_flag);
    // The case statement should cover up to ( xGETF2_CHAIN_MAX_M / ntx )
    switch(npages){
        case  1: sgetf2_native_kernel< ntx,  1><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  2: sgetf2_native_kernel< ntx,  2><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  3: sgetf2_native_kernel< ntx,  3><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  4: sgetf2_native_kernel< ntx,  4><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  5: sgetf2_native_kernel< ntx,  5><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  6: sgetf2_native_kernel< ntx,  6><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  7: sgetf2_native_kernel< ntx,  7><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  8: sgetf2_native_kernel< ntx,  8><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  9: sgetf2_native_kernel< ntx,  9><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 10: sgetf2_native_kernel< ntx, 10><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 11: sgetf2_native_kernel< ntx, 11><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 12: sgetf2_native_kernel< ntx, 12><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 13: sgetf2_native_kernel< ntx, 13><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 14: sgetf2_native_kernel< ntx, 14><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 15: sgetf2_native_kernel< ntx, 15><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 16: sgetf2_native_kernel< ntx, 16><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 17: sgetf2_native_kernel< ntx, 17><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 18: sgetf2_native_kernel< ntx, 18><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 19: sgetf2_native_kernel< ntx, 19><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 20: sgetf2_native_kernel< ntx, 20><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        #if defined(PRECISION_s) || defined(PRECISION_d)
        case 21: sgetf2_native_kernel< ntx, 21><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 22: sgetf2_native_kernel< ntx, 22><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 23: sgetf2_native_kernel< ntx, 23><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 24: sgetf2_native_kernel< ntx, 24><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 25: sgetf2_native_kernel< ntx, 25><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 26: sgetf2_native_kernel< ntx, 26><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 27: sgetf2_native_kernel< ntx, 27><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 28: sgetf2_native_kernel< ntx, 28><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 29: sgetf2_native_kernel< ntx, 29><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 30: sgetf2_native_kernel< ntx, 30><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 31: sgetf2_native_kernel< ntx, 31><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 32: sgetf2_native_kernel< ntx, 32><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 33: sgetf2_native_kernel< ntx, 33><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 34: sgetf2_native_kernel< ntx, 34><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 35: sgetf2_native_kernel< ntx, 35><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 36: sgetf2_native_kernel< ntx, 36><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 37: sgetf2_native_kernel< ntx, 37><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 38: sgetf2_native_kernel< ntx, 38><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 39: sgetf2_native_kernel< ntx, 39><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 40: sgetf2_native_kernel< ntx, 40><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 41: sgetf2_native_kernel< ntx, 41><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 42: sgetf2_native_kernel< ntx, 42><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 43: sgetf2_native_kernel< ntx, 43><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 44: sgetf2_native_kernel< ntx, 44><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 45: sgetf2_native_kernel< ntx, 45><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 46: sgetf2_native_kernel< ntx, 46><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        #endif // defined(PRECISION_s) || defined(PRECISION_d)
        #if defined(PRECISION_s)
        case 47: sgetf2_native_kernel< ntx, 47><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 48: sgetf2_native_kernel< ntx, 48><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 49: sgetf2_native_kernel< ntx, 49><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 50: sgetf2_native_kernel< ntx, 50><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 51: sgetf2_native_kernel< ntx, 51><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 52: sgetf2_native_kernel< ntx, 52><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 53: sgetf2_native_kernel< ntx, 53><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 54: sgetf2_native_kernel< ntx, 54><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 55: sgetf2_native_kernel< ntx, 55><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 56: sgetf2_native_kernel< ntx, 56><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 57: sgetf2_native_kernel< ntx, 57><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 58: sgetf2_native_kernel< ntx, 58><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 59: sgetf2_native_kernel< ntx, 59><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 60: sgetf2_native_kernel< ntx, 60><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 61: sgetf2_native_kernel< ntx, 61><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 62: sgetf2_native_kernel< ntx, 62><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 63: sgetf2_native_kernel< ntx, 63><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 64: sgetf2_native_kernel< ntx, 64><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 65: sgetf2_native_kernel< ntx, 65><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 66: sgetf2_native_kernel< ntx, 66><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 67: sgetf2_native_kernel< ntx, 67><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 68: sgetf2_native_kernel< ntx, 68><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 69: sgetf2_native_kernel< ntx, 69><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 70: sgetf2_native_kernel< ntx, 70><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 71: sgetf2_native_kernel< ntx, 71><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 72: sgetf2_native_kernel< ntx, 72><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 73: sgetf2_native_kernel< ntx, 73><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 74: sgetf2_native_kernel< ntx, 74><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 75: sgetf2_native_kernel< ntx, 75><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 76: sgetf2_native_kernel< ntx, 76><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 77: sgetf2_native_kernel< ntx, 77><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 78: sgetf2_native_kernel< ntx, 78><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 79: sgetf2_native_kernel< ntx, 79><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 80: sgetf2_native_kernel< ntx, 80><<<grid, threads, shmem, queue->cuda_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        #endif // defined(PRECISION_s)
        default: printf("size not supported \n");
    }
    return 0;
}
