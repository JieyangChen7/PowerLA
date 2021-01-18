void colchk_detect_correct(double * dA, int ldda, int m, int n, int nb,
				           double * dA_colchk, 		int ldda_colchk,
				           double * dA_colchk_r, 	int ldda_colchk_r,
						   magma_queue_t stream);

void rowchk_detect_correct(double * dA, int ldda, int m, int n, int nb,
					 	   double * dA_rowchk, 		int ldda_rowchk,
						   double * dA_rowchk_r, 	int ldda_rowchk_r,
						   magma_queue_t stream);

void colchk_detect_correct(float * dA, int ldda, int m, int n, int nb,
                           float * dA_colchk,      int ldda_colchk,
                           float * dA_colchk_r,    int ldda_colchk_r,
                           magma_queue_t stream);

void rowchk_detect_correct(float * dA, int ldda, int m, int n, int nb,
                           float * dA_rowchk,      int ldda_rowchk,
                           float * dA_rowchk_r,    int ldda_rowchk_r,
                           magma_queue_t stream);

size_t colchk_detect_correct_flops(int m, int n, int nb);
size_t rowchk_detect_correct_flops(int m, int n, int nb);