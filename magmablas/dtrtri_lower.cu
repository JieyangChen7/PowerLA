/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from magmablas/ztrtri_lower.cu, normal z -> d, Thu Oct  8 23:05:35 2020

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       This file implements lower case, and is called by dtrtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

#define TRTRI_NONBATCHED
#include "dtrtri.cuh"
#include "dtrtri_lower_device.cuh"


/******************************************************************************/
__global__ void
dtrtri_diag_lower_kernel(
    magma_diag_t diag, int n, const double *A, int lda, double *d_dinvA)
{
    dtrtri_diag_lower_device(diag, n, A, lda, d_dinvA);
}


/******************************************************************************/
__global__ void
triple_dgemm16_part1_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm16_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm16_part2_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm16_part2_lower_device( n,  Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm32_part1_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm32_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm32_part2_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm32_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm64_part1_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm64_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm64_part2_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm64_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part1_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm_above64_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part2_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm_above64_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part3_lower_kernel(
    int n, const double *Ain, int lda, double *d_dinvA, int jb, int npages)
{
    triple_dgemm_above64_part3_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}
