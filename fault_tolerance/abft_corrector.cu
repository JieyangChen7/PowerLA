/*
    Enhanced Online ABFT
    UC Riverside
    Jieyang Chen
*/
#include <stdlib.h>
#include <stdio.h>
#include "magma_internal.h"
#undef max
#undef min

__global__ void
colchk_detect_correct_kernel(double * dA, int ldda, int nb, double E,
						     double * dA_colchk, 	int ldda_colchk,
						     double * dA_colchk_r, int ldda_colchk_r)
{
    //determin the block to process
    dA = dA + blockIdx.x * nb + blockIdx.y * nb * ldda;
	
    dA_colchk   = dA_colchk   + blockIdx.x * 2  + blockIdx.y * nb * ldda_colchk;
    dA_colchk_r = dA_colchk_r + blockIdx.x * 2  + blockIdx.y * nb * ldda_colchk_r;
    
    //determine the specific colum to process
    dA = dA + threadIdx.x * ldda;
    dA_colchk   = dA_colchk   + threadIdx.x * ldda_colchk;
    dA_colchk_r = dA_colchk_r + threadIdx.x * ldda_colchk_r;
	
    double d1 = (*dA_colchk)       - (*dA_colchk_r);
    double d2 = (*(dA_colchk + 1)) - (*(dA_colchk_r + 1));
	
    //error detected
    if(fabs(d1) > E) {
    	//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[col check]error detected (d1 = %f, d2 = %f, loc = %d) \n",d1, d2, loc);
			
		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for (int i = 0; i < nb; i++) {
			if (i != loc) {
				sum +=	*(dA + i); 
			}
		}
		//correct the error
		*(dA + loc) = *dA_colchk - sum;
    }
}


__global__ void
rowchk_detect_correct_kernel(double * dA, int ldda, int nb, double E,
							 double * dA_rowchk, 	int ldda_rowchk,
							 double * dA_rowchk_r, 	int ldda_rowchk_r)
{
    //determin the block to process
    dA = dA + blockIdx.x * nb + blockIdx.y * nb * ldda;
	
    dA_rowchk   = dA_rowchk   + blockIdx.x * nb + blockIdx.y * 2 * ldda_rowchk;
    dA_rowchk_r = dA_rowchk_r + blockIdx.x * nb + blockIdx.y * 2 * ldda_rowchk_r;
        
    //determine the specific colum to process
    dA = dA + threadIdx.x;
    dA_rowchk   = dA_rowchk   + threadIdx.x;
    dA_rowchk_r = dA_rowchk_r + threadIdx.x;
	
    double d1 = (*dA_rowchk)                 - (*dA_rowchk_r);
    double d2 = (*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r));
	
    //error detected
    if(fabs(d1) > E) {
		//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[row check]error detected (d1 = %f, d2 = %f, loc = %d) \n",d1, d2, loc);
			
		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for (int i = 0; i < nb; i++) {
		    if (i != loc) {
			sum +=	*(dA + i * ldda); 
		    }
		}
		//correct the error
		*(dA + loc * ldda) = *dA_rowchk - sum;
     }
}

/*
 * B: block size
 * m: # of row
 * n: # of column
 */
void colchk_detect_correct(double * dA, int ldda, int m, int n, int nb,
				           double * dA_colchk,		int ldda_colchk,
				           double * dA_colchk_r, 	int ldda_colchk_r,
						   magma_queue_t stream) 
{
	//printf("col_detect_correct called \n");
	//error threshold 
	double E = 1e-3;
	
	colchk_detect_correct_kernel<<<dim3(m/nb, n/nb), dim3(nb), 0, stream->cuda_stream()>>>(dA, ldda, nb, E,
																		                   dA_colchk,		ldda_colchk,
																		                   dA_colchk_r, 	ldda_colchk_r);
}




/*
 * B: block size
 * m: # of row
 * n: # of column
 */
void rowchk_detect_correct(double * dA, int ldda, int m, int n, int nb,
					 	   double * dA_rowchk,		int ldda_rowchk,
						   double * dA_rowchk_r,	int ldda_rowchk_r,
						   magma_queue_t stream) 
{
	//printf("row_detect_correct called \n");
	//error threshold 
	
	double E = 1e-3;
	
	rowchk_detect_correct_kernel<<<dim3(m/nb, n/nb), dim3(nb), 0, stream->cuda_stream()>>>(dA, ldda, nb, E,
																		                   dA_rowchk, ldda_rowchk,
																		                   dA_rowchk_r, ldda_rowchk_r);
					
}








__global__ void
colchk_detect_correct_kernel(float * dA, int ldda, int nb, float E,
                             float * dA_colchk,    int ldda_colchk,
                             float * dA_colchk_r, int ldda_colchk_r)
{
    //determin the block to process
    dA = dA + blockIdx.x * nb + blockIdx.y * nb * ldda;
    
    dA_colchk   = dA_colchk   + blockIdx.x * 2  + blockIdx.y * nb * ldda_colchk;
    dA_colchk_r = dA_colchk_r + blockIdx.x * 2  + blockIdx.y * nb * ldda_colchk_r;
    
    //determine the specific colum to process
    dA = dA + threadIdx.x * ldda;
    dA_colchk   = dA_colchk   + threadIdx.x * ldda_colchk;
    dA_colchk_r = dA_colchk_r + threadIdx.x * ldda_colchk_r;
    
    float d1 = (*dA_colchk)       - (*dA_colchk_r);
    float d2 = (*(dA_colchk + 1)) - (*(dA_colchk_r + 1));
    
    //error detected
    if(fabs(d1) > E) {
        //locate the error
        int loc = round(d2 / d1) - 1;
        printf("[col check]error detected (d1 = %f, d2 = %f, loc = %d) \n",d1, d2, loc);
            
        //the sum of the rest correct number except the error one
        float sum = 0.0;
        for (int i = 0; i < nb; i++) {
            if (i != loc) {
                sum +=  *(dA + i); 
            }
        }
        //correct the error
        *(dA + loc) = *dA_colchk - sum;
    }
}


__global__ void
rowchk_detect_correct_kernel(float * dA, int ldda, int nb, float E,
                             float * dA_rowchk,    int ldda_rowchk,
                             float * dA_rowchk_r,  int ldda_rowchk_r)
{
    //determin the block to process
    dA = dA + blockIdx.x * nb + blockIdx.y * nb * ldda;
    
    dA_rowchk   = dA_rowchk   + blockIdx.x * nb + blockIdx.y * 2 * ldda_rowchk;
    dA_rowchk_r = dA_rowchk_r + blockIdx.x * nb + blockIdx.y * 2 * ldda_rowchk_r;
        
    //determine the specific colum to process
    dA = dA + threadIdx.x;
    dA_rowchk   = dA_rowchk   + threadIdx.x;
    dA_rowchk_r = dA_rowchk_r + threadIdx.x;
    
    float d1 = (*dA_rowchk)                 - (*dA_rowchk_r);
    float d2 = (*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r));
    
    //error detected
    if(fabs(d1) > E) {
        //locate the error
        int loc = round(d2 / d1) - 1;
        printf("[row check]error detected (d1 = %f, d2 = %f, loc = %d) \n",d1, d2, loc);
            
        //the sum of the rest correct number except the error one
        float sum = 0.0;
        for (int i = 0; i < nb; i++) {
            if (i != loc) {
            sum +=  *(dA + i * ldda); 
            }
        }
        //correct the error
        *(dA + loc * ldda) = *dA_rowchk - sum;
     }
}

/*
 * B: block size
 * m: # of row
 * n: # of column
 */
void colchk_detect_correct(float * dA, int ldda, int m, int n, int nb,
                           float * dA_colchk,      int ldda_colchk,
                           float * dA_colchk_r,    int ldda_colchk_r,
                           magma_queue_t stream) 
{
    //printf("col_detect_correct called \n");
    //error threshold 
    float E = 1e-1;
    
    colchk_detect_correct_kernel<<<dim3(m/nb, n/nb), dim3(nb), 0, stream->cuda_stream()>>>(dA, ldda, nb, E,
                                                                                           dA_colchk,       ldda_colchk,
                                                                                           dA_colchk_r,     ldda_colchk_r);
}


/*
 * B: block size
 * m: # of row
 * n: # of column
 */
void rowchk_detect_correct(float * dA, int ldda, int m, int n, int nb,
                           float * dA_rowchk,      int ldda_rowchk,
                           float * dA_rowchk_r,    int ldda_rowchk_r,
                           magma_queue_t stream) 
{
    //printf("row_detect_correct called \n");
    //error threshold 
    
    float E = 1e-1;
    
    rowchk_detect_correct_kernel<<<dim3(m/nb, n/nb), dim3(nb), 0, stream->cuda_stream()>>>(dA, ldda, nb, E,
                                                                                           dA_rowchk, ldda_rowchk,
                                                                                           dA_rowchk_r, ldda_rowchk_r);
                    
}



size_t colchk_detect_correct_flops(int m, int n, int nb) {
    return (size_t)(m/nb)*(n/nb)*nb*2;
}
size_t rowchk_detect_correct_flops(int m, int n, int nb) {
    return (size_t)(m/nb)*(n/nb)*nb*2;
}




