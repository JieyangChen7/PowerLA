#include "magma_internal.h"
#undef max
#undef min
#include "abft_printer.h"
//#include "magma.h"
#include <iostream>
#include <iomanip>
using namespace std;
//printing tools

/*
 * row_block and col_block control the display block, -1 represents no block
 */
void printMatrix_host(double * matrix_host, int ld,  int M, int N, int row_block, int col_block) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout.width(10);
			cout.setf(ios::left);
			cout << setprecision(5) << matrix_host[j * ld + i];
			if (col_block != -1 && (j + 1) % col_block == 0) {
				cout << "	";
			}
		}
		cout << endl;
		if (row_block != -1 && (i + 1) % row_block == 0) {
			cout << endl;
		}
	}
	cout << endl;
}


/*
 * row_block and col_block control the display block, -1 represents no block
 */
void printMatrix_host(float * matrix_host, int ld,  int M, int N, int row_block, int col_block) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout.width(10);
			cout.setf(ios::left);
			cout << setprecision(5) << matrix_host[j * ld + i];
			if (col_block != -1 && (j + 1) % col_block == 0) {
				cout << "	";
			}
		}
		cout << endl;
		if (row_block != -1 && (i + 1) % row_block == 0) {
			cout << endl;
		}
	}
	cout << endl;
}

void printMatrix_host_int(int * matrix_host, int ld,  int M, int N, int row_block, int col_block) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout.width(10);
			cout.setf(ios::left);
			cout << setprecision(5) << matrix_host[j * ld + i];
			if (col_block != -1 && (j + 1) % col_block == 0) {
				cout << "	";
			}
		}
		cout << endl;
		if (row_block != -1 && (i + 1) % row_block == 0) {
			cout << endl;
		}
	}
	cout << endl;
}


/**
 * M: number of row
 * N: number of col
 */
void printMatrix_gpu(double * matrix_device, int matrix_ld, 
					 int M, int N, int row_block, int col_block, 
					 magma_queue_t stream) {
	double * matrix_host = new double[M * N]();
//	cudaMemcpy2D(matrix_host, M * sizeof(double), matrix_device, matrix_pitch,
//			M * sizeof(double), N, cudaMemcpyDeviceToHost);
	magma_dgetmatrix(M, N, matrix_device, matrix_ld, matrix_host, M, stream);
	magma_queue_sync(stream);
	printMatrix_host(matrix_host, M, M, N, row_block, col_block);
	delete[] matrix_host;
}


/**
 * M: number of row
 * N: number of col
 */
void printMatrix_gpu(float * matrix_device, int matrix_ld, 
					 int M, int N, int row_block, int col_block, 
					 magma_queue_t stream) {
	float * matrix_host = new float[M * N]();
//	cudaMemcpy2D(matrix_host, M * sizeof(double), matrix_device, matrix_pitch,
//			M * sizeof(double), N, cudaMemcpyDeviceToHost);
	magma_sgetmatrix(M, N, matrix_device, matrix_ld, matrix_host, M, stream);
	magma_queue_sync(stream);
	printMatrix_host(matrix_host, M, M, N, row_block, col_block);
	delete[] matrix_host;
}

