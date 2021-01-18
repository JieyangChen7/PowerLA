#include "magma_internal.h"
#undef max
#undef min
#include "abft_io.h"
#include <string>
#include <fstream> 
#include <iostream>
#include <cstdlib>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include <vector>
#include <nvml.h>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>




void store_matrix(int m, int n, double * dA, int ldda, magma_queue_t stream, std::string file_prefix){
	double * A = new double[m * n];
	int lda = m;
	magma_dgetmatrix_async( m, n,
                            dA, ldda,
                            A,  lda, stream);
	magma_queue_sync( stream );
	std::string filename = file_prefix + ".csv";
	std::ofstream myfile;
    myfile.open (filename);
    myfile << std::fixed << std::setprecision(16);
    for (int i = 0; i < m; i++) {
    	std::string tmp = "";
    	for (int j = 0; j < n; j++) {
    		myfile << A[i + j * lda] << ",";
    	}
    	myfile << std::endl;
    }
    myfile.close();
    delete[] A;
}

void load_matrix(int m, int n, double * A, int lda, std::string file_prefix){
	std::string filename = file_prefix + ".csv";
	std::ifstream myfile;
	std::string line;
    myfile.open (filename);
    int i = 0; 
    int j = 0;
    while(std::getline(myfile, line)) {
    	j = 0;
    	std::stringstream lineStream(line);
        std::string cell;
        while(std::getline(lineStream,cell,',')){
        	A[i + j * lda] = atof(cell.c_str());
        	j++;
        }
        i++;
    }
    myfile.close();
}


void load_matrix_to_dev(int m, int n, double * dA, int ldda, magma_queue_t stream, std::string file_prefix){
	double * A = new double[m * n];
	int lda = m;
	load_matrix(m, n, A, lda, file_prefix);
	magma_dsetmatrix_async( m, n,
							A,  lda, 
                            dA, ldda, stream);
	magma_queue_sync( stream );

	delete[] A;
}




void compare_matrices(int m, int n, std::string file_prefix1, std::string file_prefix2, std::string output_prefix) {
	double * A1 = new double[m * n];
	double * A2 = new double[m * n];
	int lda = m;
	load_matrix(m, n, A1, lda, file_prefix1);
	load_matrix(m, n, A2, lda, file_prefix2);
	//double E = 1e-10;

	//std::unordered_set<int> row_set;
	//std::unordered_set<int> col_set;

	std::string filename = output_prefix + ".csv";
	std::ofstream myfile;
    myfile.open (filename);
    myfile << std::fixed << std::setprecision(16);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			// if (fabs(A1[i + j * lda] - A2[i + j * lda]) > E) {
			// 	row_set.insert(i);
			// 	col_set.insert(j);
			// }
			myfile << fabs(A1[i + j * lda] - A2[i + j * lda]) << ",";
		}
		myfile << std::endl;

	}
	myfile.close();
	delete[] A1;
	delete[] A2;
// 	if (row_set.size() == 0 && col_set.size() == 0) {
// 		return 0;
// 	} else if (row_set.size() == 1 && col_set.size() == 1) {
// 		return 1;
// 	} else if (row_set.size() == 1 || col_set.size() == 1) {
// 		return 2;
// 	} else {
// 		return 3;
// 	}
}


void store_matrix(int m, int n, float * dA, int ldda, magma_queue_t stream, std::string file_prefix){
    float * A = new float[m * n];
    int lda = m;
    magma_sgetmatrix_async( m, n,
                            dA, ldda,
                            A,  lda, stream);
    magma_queue_sync( stream );
    std::string filename = file_prefix + ".csv";
    std::ofstream myfile;
    myfile.open (filename);
    myfile << std::fixed << std::setprecision(16);
    for (int i = 0; i < m; i++) {
        std::string tmp = "";
        for (int j = 0; j < n; j++) {
            myfile << A[i + j * lda] << ",";
        }
        myfile << std::endl;
    }
    myfile.close();
    delete[] A;
}

void load_matrix(int m, int n, float * A, int lda, std::string file_prefix){
    std::string filename = file_prefix + ".csv";
    std::ifstream myfile;
    std::string line;
    myfile.open (filename);
    int i = 0; 
    int j = 0;
    while(std::getline(myfile, line)) {
        j = 0;
        std::stringstream lineStream(line);
        std::string cell;
        while(std::getline(lineStream,cell,',')){
            A[i + j * lda] = atof(cell.c_str());
            j++;
        }
        i++;
    }
    myfile.close();
}

void load_matrix_to_dev(int m, int n, float * dA, int ldda, magma_queue_t stream, std::string file_prefix){
	float * A = new float[m * n];
	int lda = m;
	load_matrix(m, n, A, lda, file_prefix);
	magma_ssetmatrix_async( m, n,
							A,  lda, 
                            dA, ldda, stream);
	magma_queue_sync( stream );

	delete[] A;
}


void compare_matrices_float(int m, int n, std::string file_prefix1, std::string file_prefix2, std::string output_prefix) {
    float * A1 = new float[m * n];
    float * A2 = new float[m * n];
    int lda = m;
    load_matrix(m, n, A1, lda, file_prefix1);
    load_matrix(m, n, A2, lda, file_prefix2);
    //float E = 1e-10;

    //std::unordered_set<int> row_set;
    //std::unordered_set<int> col_set;

    std::string filename = output_prefix + ".csv";
    std::ofstream myfile;
    myfile.open (filename);
    myfile << std::fixed << std::setprecision(16);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // if (fabs(A1[i + j * lda] - A2[i + j * lda]) > E) {
            //  row_set.insert(i);
            //  col_set.insert(j);
            // }
            myfile << fabs(A1[i + j * lda] - A2[i + j * lda]) << ",";
        }
        myfile << std::endl;

    }
    myfile.close();
    delete[] A1;
    delete[] A2;
//  if (row_set.size() == 0 && col_set.size() == 0) {
//      return 0;
//  } else if (row_set.size() == 1 && col_set.size() == 1) {
//      return 1;
//  } else if (row_set.size() == 1 || col_set.size() == 1) {
//      return 2;
//  } else {
//      return 3;
//  }
}

void record_time_and_energy_start(double * time, magma_queue_t stream, nvmlDevice_t device) {
    magma_queue_sync( stream );

    // nvmlReturn_t result;
    // result = nvmlDeviceSetPowerManagementLimit (device, POWER_LIMIT);
    // if (NVML_SUCCESS != result)
    // {
    //   printf("Failed to set power limit of device: %s\n", nvmlErrorString(result));
    //   return;
    // }

    // result = nvmlDeviceSetApplicationsClocks ( device, MEMORY_CLOCK, GRAPHICS_CLOCK );
    // if (NVML_SUCCESS != result)
    // {
    //   printf("Failed to set clock of device: %s\n", nvmlErrorString(result));
    //   return;
    // }
    // nvmlEnableState_t set = NVML_FEATURE_ENABLED;
    // result = nvmlDeviceSetAutoBoostedClocksEnabled(device, set);
    // if (NVML_SUCCESS != result)
    // {
    //   printf("Failed to disable autoboost of device: %s\n", nvmlErrorString(result));
    //   return;
    // }

    * time = magma_wtime();
}

void record_time_and_energy_end(double * time, magma_queue_t stream, std::string output_prefix, nvmlDevice_t device) {
    
    unsigned int p = 0;
    int sample_freq = 3000;
    std::vector<int> power_samples;
    while (cudaStreamQuery(stream->cuda_stream()) != cudaSuccess) {
        nvmlDeviceGetPowerUsage(device, &p);
        power_samples.push_back(p);
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000/sample_freq));
        //std::this_thread::sleep_for(std::chrono::microseconds(1000000/sample_freq));
    }
    * time = magma_wtime() - (*time);
    double energy = 0.0;
    for (unsigned int j = 0; j < power_samples.size(); j++) {
        energy += (power_samples[j]/100.0) * ((*time)/power_samples.size());
    }

    std::string filename = output_prefix + ".csv";
    std::ofstream myfile;
    myfile.open (filename);
    myfile << *time << "," << energy << ","<< power_samples.size() << std::endl;
    myfile.close();
}
