
/*
   -- MAGMA (version 2.5.4) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date October 2020

   @author Azzam Haidar
   @author Ahmad Ahmad

   @generated from magmablas/zpotf2_devicesfunc.cuh, normal z -> s, Thu Oct  8 23:05:56 2020
 */


#ifndef MAGMABLAS_SPOTF2_DEVICES_Z_H
#define MAGMABLAS_SPOTF2_DEVICES_Z_H


extern __shared__ float shared_data[];

/******************************************************************************/
static inline __device__ void spotf2_sminout_anywidth_device(const int m, const int n, float *A, const int lda, int* info)
{
    const int tx = threadIdx.x;
    float factor;
    int linfo = 0;

    #pragma unroll
    for (int iter=0; iter < n; iter++)
    {
        //sqrt(diag) and sscal
        #ifdef ENABLE_COND1
        if ( tx >= iter && tx < m )
        {
        #endif
            float xreal = MAGMA_S_REAL(A[iter + iter * lda]);
            linfo = ( linfo == 0 && (xreal <= MAGMA_D_ZERO) ) ? (iter+1) : linfo;
            xreal = sqrt(xreal);
            factor = MAGMA_S_MAKE(1.0/xreal, 0.0);
        #ifdef ENABLE_COND1
        }
        #endif
        __syncthreads(); // must sync to make sure that A[iter + iter * lda] is read by all threads before modifying it
        #ifdef ENABLE_COND1
        if ( tx >= iter && tx < m )
        {
        #endif
            A[ tx + iter * lda ] *= factor; // or use the next line and remove the sync above
            //A[ tx + iter * lda ]  = tx == iter ? MAGMA_S_MAKE(xreal, 0.0) : A[ tx + iter * lda ] * factor;
        #ifdef ENABLE_COND1
        }
        #endif
        __syncthreads();


        // slacgv: TODO, ssyrk
        #ifdef ENABLE_COND1
        if ( tx > iter && tx < m )
        {
        #endif
            #pragma unroll 
            for (int j=iter+1; j < n; j++)
            {
                A [tx + j * lda] -= A[tx + iter * lda]  *  MAGMA_S_CONJ(A[iter * lda + j]);
            }   
        #ifdef ENABLE_COND1
        }
        #endif
        __syncthreads();
    }// end of iter
    // ENABLE_COND1 must be disabled, which the default config., so that the right info is returned 
    if(tx == 0) *info = linfo;
    __syncthreads();
}


/******************************************************************************/
static inline __device__ void spotf2_sminout_fixsize_device(const int m, float *A, const int lda, int* info)
{
    const int tx = threadIdx.x;
    float factor;
    int linfo = 0;

    #pragma unroll
    for (int iter=0; iter < POTF2_NB; iter++)
    {
        //sqrt(diag) and sscal
        #ifdef ENABLE_COND2
        if ( tx >= iter && tx < m )
        {
        #endif
            float xreal = MAGMA_S_REAL(A[iter + iter * lda]);
            linfo = ( linfo == 0 && (xreal <= MAGMA_D_ZERO || isnan(xreal)) ) ? (iter+1) : linfo;
            xreal = sqrt(xreal);
            factor = MAGMA_S_MAKE(1.0/xreal, 0.0);
        #ifdef ENABLE_COND2
        }
        #endif
        __syncthreads();
        #ifdef ENABLE_COND2
        if ( tx >= iter && tx < m )
        {
        #endif
            A[ tx + iter * lda ] *= factor;

            //A[ tx + iter * lda ]  = tx == iter ? MAGMA_S_MAKE(xreal, 0.0) : A[ tx + iter * lda ] * factor;
            //if (tx < POTF2_NB) row[ tx ] = MAGMA_S_CONJ( A[ tx + iter * lda ] );
            //if (tx < POTF2_NB) A[ iter + tx * lda ] = MAGMA_S_CONJ( A[ tx + iter * lda ] );
        #ifdef ENABLE_COND2
        }
        #endif

        __syncthreads();


        // ssyrk
        #ifdef ENABLE_COND2
        if ( tx > iter && tx < m )
        {
        #endif
            #pragma unroll
            for (int j=iter+1; j < POTF2_NB; j++)
            {
                A [tx + j * lda] -= A[tx + iter * lda]  *  MAGMA_S_CONJ(A[iter * lda + j]);
                //A [tx + j * lda] -= A[tx + iter * lda]  *  row[j];
                //A [tx + j * lda] -= A[tx + iter * lda]  *  A[iter +lda * j];
            }   
        #ifdef ENABLE_COND2
        }
        #endif
        __syncthreads();
    }// end of iter
    // ENABLE_COND1 must be disabled, which the default config., so that the right info is returned
    if(tx == 0) *info = linfo;
    __syncthreads();
}


/******************************************************************************/
static inline __device__ void sgemm_v20_1_fixsize_device(int m, int k,
        const float* __restrict__ A0, const int lda,
        float *sC, float  *sB)
{
    const int tx = threadIdx.x;
    float rC[POTF2_NB];
    float rA[POTF2_NB]; 
    float rp[POTF2_NB]; 

    // prefetch next block. 
    #ifdef ENABLE_COND4
    if (tx < m) 
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rp[i] = A0[tx + i * lda];
            rC[i] = MAGMA_S_ZERO;
        }
    #ifdef ENABLE_COND4
    }
    #endif

    __syncthreads();



    // accumulate 
    #pragma unroll
    for (int iter=0; iter < k; iter += POTF2_NB)
    {
        #ifdef ENABLE_COND4
        if (tx < m) 
        {
        #endif
            // rp to rA
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rA[i] = rp[i];
            }
        #ifdef ENABLE_COND4
        }
        #endif

        // rA to sB
        if (tx < POTF2_NB) 
        {      
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                sB[tx + i * POTF2_NB] = MAGMA_S_CONJ(rp[i]);
            }
        }

        __syncthreads();

        // prefetch next block. Azzam
        #ifdef ENABLE_COND4
        if (tx < m )  
        {      
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rp[i] = A0[tx + (i+(iter+POTF2_NB)) * lda];
            }
        #ifdef ENABLE_COND4
        }
        #endif
        //__syncthreads();

        // multiply current block
        #ifdef ENABLE_COND4
        if (tx < m) 
        {
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                #pragma unroll
                for (int col=0; col < POTF2_NB; col++)
                {
                    // A0 is multiplied by POTF2_NB times
                    rC[col] +=  rA[i] * sB[col + i * POTF2_NB];
                }
            }
        #ifdef ENABLE_COND4
        }
        #endif
        __syncthreads();
    }//end of accumulation

    // finalyzing gemm.
    #ifdef ENABLE_COND4
    if (tx < m) 
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            sC[tx + i *m] = rp[i] - rC[i];
        }
    #ifdef ENABLE_COND4
    }
    #endif
    __syncthreads();
}


/******************************************************************************/
static inline __device__ void sgemm_v20_1_anywidth_device(int m, int n, int k,
        const float* __restrict__ A0, int lda,
        float *sC, float  *sB)
{
    const int tx = threadIdx.x;
    float rC[POTF2_NB];
    float rA[POTF2_NB]; 
    float rp[POTF2_NB]; 

    const int bound_A = lda*(k+n-1)+m;

    // prefetch next block. 
    #ifdef ENABLE_COND5
    if (tx < m) 
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rp[i] = A0[min(bound_A, tx + i * lda)];
            rC[i] = MAGMA_S_ZERO;
        }
    #ifdef ENABLE_COND5
    }
    #endif

    __syncthreads();



    // accumulate 
    #pragma unroll
    for (int iter=0; iter < k; iter += POTF2_NB)
    {
        #ifdef ENABLE_COND5
        if (tx < m) 
        {
        #endif
            // rp to rA
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rA[i] = rp[i];
            }
        #ifdef ENABLE_COND5
        }
        #endif

        // rA to sB
        if (tx < POTF2_NB) 
        {      
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                sB[tx + i * POTF2_NB] = MAGMA_S_CONJ(rp[i]);
            }
        }

        __syncthreads();

        // prefetch next block. Azzam
        #ifdef ENABLE_COND5
        if (tx < m )  
        {      
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rp[i] = A0[min(bound_A, tx + (i+(iter+POTF2_NB)) * lda)]; // min(bound,xxx) is to avoid reading out of bound
            }
        #ifdef ENABLE_COND5
        }
        #endif
        //__syncthreads();

        // multiply current block
        #ifdef ENABLE_COND5
        if (tx < m) 
        {
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                #pragma unroll
                for (int col=0; col < POTF2_NB; col++)
                {
                    // A0 is multiplied by POTF2_NB times
                    rC[col] +=  rA[i] * sB[col + i * POTF2_NB];
                }
            }
        #ifdef ENABLE_COND5
        }
        #endif
        __syncthreads();
    }//end of accumulation

    // finalyzing gemm.
    #ifdef ENABLE_COND5
    if (tx < m) 
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            sC[tx + i *m] = rp[i] - rC[i];
        }
    #ifdef ENABLE_COND5
    }
    #endif
    __syncthreads();
}


/******************************************************************************/
static inline __device__ void spotf2_smlpout_fixwidth_device(const int m,  
        float *A0, float *A, int lda,
        const int localstep, const int gbstep,
        magma_int_t *info)
{
    // checkinfo to avoid computation of the singular matrix
    #ifndef BATCH_DISABLE_CHECKING
    if (*info != 0 ) return;
    #endif

    const int orginfo = (*info);
    int panel_info = 0, newinfo = 0;
    const int tx = threadIdx.x;
    float *sdata_A = shared_data + threadIdx.y * (m+POTF2_NB)*POTF2_NB;
    float *sdata_B = sdata_A + m * POTF2_NB;


    #if 1
    sgemm_v20_1_fixsize_device(m, localstep, 
                       A0, lda, sdata_A, sdata_B);
    #else
    sgemm_v20_1_anywidth_device(m, POTF2_NB, localstep, 
                       A0, lda, sdata_A, sdata_B);
    #endif

    // panel fact. in shared memory
    spotf2_sminout_fixsize_device(m, sdata_A, m, &panel_info);
    //----------------------------------------------------
    // Check for not SPD generating info
    #ifndef BATCH_DISABLE_CHECKING
    if(tx == 0) {
        newinfo = ( orginfo == 0 && panel_info != 0 ) ? panel_info + localstep + gbstep : orginfo;
        (*info) = newinfo;
    }
    __syncthreads();
    #endif
    //----------------------------------------------------

    //copy sdata_A to A
    #ifdef ENABLE_COND6
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {  
            #ifdef BATCH_DISABLE_CLEANUP
            A[tx + i * lda] = sdata_A[tx + i * m];
            #else
            if (tx >= i) A[tx + i * lda] = sdata_A[tx + i * m];
            #endif
        }
    #ifdef ENABLE_COND6
    }
    __syncthreads();
    #endif
}


/******************************************************************************/
static inline __device__ void spotf2_smlpout_anywidth_device(const int m, const int n,
        float *A0, float *A, int lda,
        const int localstep, const int gbstep,
        magma_int_t *info)
{
    // checkinfo to avoid computation of the singular matrix
    #ifndef BATCH_DISABLE_CHECKING
    if (*info != 0 ) return;
    #endif
    
    const int orginfo = (*info);
    int panel_info = 0, newinfo = 0;
    const int tx = threadIdx.x;
    float *sdata_A = shared_data + threadIdx.y * (m+POTF2_NB)*POTF2_NB;
    float *sdata_B = sdata_A + m * POTF2_NB;

    #if 0
    sgemm_v20_1_fixsize_device(m, localstep, 
                       A0, lda, sdata_A, sdata_B);
    spotf2_sminout_fixsize_device(m, sdata_A, m);
    #else
    sgemm_v20_1_anywidth_device(m, n, localstep, 
                       A0, lda, sdata_A, sdata_B);
    #endif

    spotf2_sminout_anywidth_device(m, n, sdata_A, m, &panel_info);
    //----------------------------------------------------
    // Check for not SPD generating info
    #ifndef BATCH_DISABLE_CHECKING
    if(tx == 0) {
        newinfo = ( orginfo == 0 && panel_info != 0 ) ? panel_info + localstep + gbstep : orginfo;
        (*info) = newinfo;
    }
    __syncthreads();
    #endif
    //----------------------------------------------------


    //copy sdata_A to A
    #ifdef ENABLE_COND6
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < n; i++)
        {  
            #ifdef BATCH_DISABLE_CLEANUP
            A[tx + i * lda] = sdata_A[tx + i * m];
            #else
            if (tx >= i) A[tx + i * lda] = sdata_A[tx + i * m];
            #endif
        }
    #ifdef ENABLE_COND6
    }
    __syncthreads();
    #endif
}

#endif // MAGMABLAS_SPOTF2_DEVICES_Z_H
