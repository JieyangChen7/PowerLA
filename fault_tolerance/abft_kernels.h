void abft_dpotf2(const char uplo, int n, double * A, int lda, int * info, 
           int nb, 
           double * colchk,   int ld_colchk, 
           double * rowchk,   int ld_rowchk, 
           double * colchk_r, int ld_colchk_r, 
           double * rowchk_r, int ld_rowchk_r, 
           double * chk_v, int ld_chk_v, 
           bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER);

void abft_dtrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    int m, int n,
    double alpha,
    double * dA, int ldda,
    double *       dB, int lddb,
    double *       dX, int lddx,
    int flag,
    double * d_dinvA, int dinvA_length,
    int nb,
    double * dA_colchk,    int ldda_colchk,
    double * dA_rowchk,    int ldda_rowchk,
    double * dA_colchk_r,  int ldda_colchk_r,
    double * dA_rowchk_r,  int ldda_rowchk_r,
    double * dB_colchk,    int lddb_colchk,
    double * dB_rowchk,    int lddb_rowchk,
    double * dB_colchk_r,  int lddb_colchk_r,
    double * dB_rowchk_r,  int lddb_rowchk_r,
    double * chk_v,        int ld_chk_v, 
    bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    magma_queue_t stream1, magma_queue_t stream2);


void abft_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    int m, int n,
    double alpha,
    double * dA, int ldda,
    double * dB, int lddb,
    int nb,
    double * dA_colchk,    int ldda_colchk,
    double * dA_rowchk,    int ldda_rowchk,
    double * dA_colchk_r,  int ldda_colchk_r,
    double * dA_rowchk_r,  int ldda_rowchk_r,
    double * dB_colchk,    int lddb_colchk,
    double * dB_rowchk,    int lddb_rowchk,
    double * dB_colchk_r,  int lddb_colchk_r,
    double * dB_rowchk_r,  int lddb_rowchk_r,
    double * chk_v,        int ld_chk_v, 
    bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    magma_queue_t stream1, magma_queue_t stream2);

void abft_dsyrk(magma_uplo_t uplo, magma_trans_t trans,
                 int n, int k, 
                 double alpha,
                 double * dA, int ldda,
                 double beta,
                 double * dC, int lddc,
                 int nb,
                 double * dA_colchk,    int ldda_colchk,
                 double * dA_rowchk,    int ldda_rowchk,
                 double * dA_colchk_r,  int ldda_colchk_r,
                 double * dA_rowchk_r,  int ldda_rowchk_r,
                 double * dC_colchk,    int lddc_colchk,
                 double * dC_rowchk,    int lddc_rowchk,
                 double * dC_colchk_r,  int lddc_colchk_r,
                 double * dC_rowchk_r,  int lddc_rowchk_r,
                 double * chk_v,        int ld_chk_v, 
                 bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
                 magma_queue_t stream1, magma_queue_t stream2
                 );

void abft_dgemm( magma_trans_t transA, magma_trans_t transB,
              int m, int n, int k, 
              double alpha, 
              double * dA, int ldda,
              double * dB, int lddb, 
              double beta, 
              double * dC, int lddc,
              int nb,
              double * dA_colchk,   int ldda_colchk,
              double * dA_rowchk,   int ldda_rowchk,
              double * dA_colchk_r, int ldda_colchk_r,
              double * dA_rowchk_r, int ldda_rowchk_r,
              double * dB_colchk,   int lddb_colchk,
              double * dB_rowchk,   int lddb_rowchk,
              double * dB_colchk_r, int lddb_colchk_r,
              double * dB_rowchk_r, int lddb_rowchk_r,
              double * dC_colchk,   int lddc_colchk,
              double * dC_rowchk,   int lddc_rowchk,
              double * dC_colchk_r, int lddc_colchk_r,
              double * dC_rowchk_r, int lddc_rowchk_r,
              double * chk_v, int ld_chk_v, 
              bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
              magma_queue_t stream1, magma_queue_t stream2);

extern "C" void
abft_dtrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n, 
    double alpha,
    double * dA, int ldda,
    double *       dB, int lddb,
    magma_int_t nb,
    double * dA_colchk,   int ldda_colchk,
    double * dA_rowchk,   int ldda_rowchk,
    double * dA_colchk_r, int ldda_colchk_r,
    double * dA_rowchk_r, int ldda_rowchk_r,
    double * dB_colchk,   int lddb_colchk,
    double * dB_rowchk,   int lddb_rowchk,
    double * dB_colchk_r, int lddb_colchk_r,
    double * dB_rowchk_r, int lddb_rowchk_r,
    double * chk_v, int ld_chk_v, 
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    magma_queue_t stream1, magma_queue_t stream2);


extern "C" magma_int_t
abft_dlarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double * dV,    int lddv,
    double * dT,    int lddt,
    double * dC,    int lddc,
    double * dwork, int ldwork,
    int nb,
    double * dV_colchk, int lddv_colchk,
    double * dV_rowchk,   int lddv_rowchk,
    double * dV_colchk_r, int lddv_colchk_r,
    double * dV_rowchk_r, int lddv_rowchk_r,
    double * dT_colchk, int lddt_colchk,
    double * dT_rowchk,   int lddt_rowchk,
    double * dT_colchk_r, int lddt_colchk_r,
    double * dT_rowchk_r, int lddt_rowchk_r,
    double * dC_colchk, int lddc_colchk,
    double * dC_rowchk,   int lddc_rowchk,
    double * dC_colchk_r, int lddc_colchk_r,
    double * dC_rowchk_r, int lddc_rowchk_r,
    double * dwork_colchk, int lddwork_colchk,
    double * dwork_rowchk,   int lddwork_rowchk,
    double * dwork_colchk_r, int lddwork_colchk_r,
    double * dwork_rowchk_r, int lddwork_rowchk_r,
    double * chk_v, int ld_chk_v, 
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, 
    magma_queue_t stream1, magma_queue_t stream2);




void abft_spotf2(const char uplo, int n, float * A, int lda, int * info, 
           int nb, 
           float * colchk,   int ld_colchk, 
           float * rowchk,   int ld_rowchk, 
           float * colchk_r, int ld_colchk_r, 
           float * rowchk_r, int ld_rowchk_r, 
           float * chk_v, int ld_chk_v, 
           bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER);




// void abft_strsm_work(
//     magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
//     int m, int n,
//     float alpha,
//     float * dA, int ldda,
//     float *       dB, int lddb,
//     float *       dX, int lddx,
//     int flag,
//     float * d_dinvA, int dinvA_length,
//     int nb,
//     float * dA_colchk,    int ldda_colchk,
//     float * dA_rowchk,    int ldda_rowchk,
//     float * dA_colchk_r,  int ldda_colchk_r,
//     float * dA_rowchk_r,  int ldda_rowchk_r,
//     float * dB_colchk,    int lddb_colchk,
//     float * dB_rowchk,    int lddb_rowchk,
//     float * dB_colchk_r,  int lddb_colchk_r,
//     float * dB_rowchk_r,  int lddb_rowchk_r,
//     float * chk_v,        int ld_chk_v, 
//     bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
//     magma_queue_t stream1, magma_queue_t stream2);


void abft_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    int m, int n,
    float alpha,
    float * dA, int ldda,
    float * dB, int lddb,
    int nb,
    float * dA_colchk,    int ldda_colchk,
    float * dA_rowchk,    int ldda_rowchk,
    float * dA_colchk_r,  int ldda_colchk_r,
    float * dA_rowchk_r,  int ldda_rowchk_r,
    float * dB_colchk,    int lddb_colchk,
    float * dB_rowchk,    int lddb_rowchk,
    float * dB_colchk_r,  int lddb_colchk_r,
    float * dB_rowchk_r,  int lddb_rowchk_r,
    float * chk_v,        int ld_chk_v, 
    bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    magma_queue_t stream1, magma_queue_t stream2);

void abft_ssyrk(magma_uplo_t uplo, magma_trans_t trans,
                 int n, int k, 
                 float alpha,
                 float * dA, int ldda,
                 float beta,
                 float * dC, int lddc,
                 int nb,
                 float * dA_colchk,    int ldda_colchk,
                 float * dA_rowchk,    int ldda_rowchk,
                 float * dA_colchk_r,  int ldda_colchk_r,
                 float * dA_rowchk_r,  int ldda_rowchk_r,
                 float * dC_colchk,    int lddc_colchk,
                 float * dC_rowchk,    int lddc_rowchk,
                 float * dC_colchk_r,  int lddc_colchk_r,
                 float * dC_rowchk_r,  int lddc_rowchk_r,
                 float * chk_v,        int ld_chk_v, 
                 bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
                 magma_queue_t stream1, magma_queue_t stream2);

void abft_sgemm( magma_trans_t transA, magma_trans_t transB,
              int m, int n, int k, 
              float alpha, 
              float * dA, int ldda,
              float * dB, int lddb, 
              float beta, 
              float * dC, int lddc,
              int nb,
              float * dA_colchk,   int ldda_colchk,
              float * dA_rowchk,   int ldda_rowchk,
              float * dA_colchk_r, int ldda_colchk_r,
              float * dA_rowchk_r, int ldda_rowchk_r,
              float * dB_colchk,   int lddb_colchk,
              float * dB_rowchk,   int lddb_rowchk,
              float * dB_colchk_r, int lddb_colchk_r,
              float * dB_rowchk_r, int lddb_rowchk_r,
              float * dC_colchk,   int lddc_colchk,
              float * dC_rowchk,   int lddc_rowchk,
              float * dC_colchk_r, int lddc_colchk_r,
              float * dC_rowchk_r, int lddc_rowchk_r,
              float * chk_v, int ld_chk_v, 
              bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
              magma_queue_t stream1, magma_queue_t stream2);



extern "C" void
abft_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n, 
    float alpha,
    float * dA, int ldda,
    float *       dB, int lddb,
    magma_int_t nb,
    float * dA_colchk,   int ldda_colchk,
    float * dA_rowchk,   int ldda_rowchk,
    float * dA_colchk_r, int ldda_colchk_r,
    float * dA_rowchk_r, int ldda_rowchk_r,
    float * dB_colchk,   int lddb_colchk,
    float * dB_rowchk,   int lddb_rowchk,
    float * dB_colchk_r, int lddb_colchk_r,
    float * dB_rowchk_r, int lddb_rowchk_r,
    float * chk_v, int ld_chk_v, 
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    magma_queue_t stream1, magma_queue_t stream2);


extern "C" magma_int_t
abft_slarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float * dV,    int lddv,
    float * dT,    int lddt,
    float * dC,    int lddc,
    float * dwork, int ldwork,
    int nb,
    float * dV_colchk, int lddv_colchk,
    float * dV_rowchk,   int lddv_rowchk,
    float * dV_colchk_r, int lddv_colchk_r,
    float * dV_rowchk_r, int lddv_rowchk_r,
    float * dT_colchk, int lddt_colchk,
    float * dT_rowchk,   int lddt_rowchk,
    float * dT_colchk_r, int lddt_colchk_r,
    float * dT_rowchk_r, int lddt_rowchk_r,
    float * dC_colchk, int lddc_colchk,
    float * dC_rowchk,   int lddc_rowchk,
    float * dC_colchk_r, int lddc_colchk_r,
    float * dC_rowchk_r, int lddc_rowchk_r,
    float * dwork_colchk, int lddwork_colchk,
    float * dwork_rowchk,   int lddwork_rowchk,
    float * dwork_colchk_r, int lddwork_colchk_r,
    float * dwork_rowchk_r, int lddwork_rowchk_r,
    float * chk_v, int ld_chk_v, 
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, 
    magma_queue_t stream1, magma_queue_t stream2);



size_t abft_dgemm_flops(magma_trans_t transA, magma_trans_t transB,
            int m, int n, int k, int nb,
            bool COL_FT, bool ROW_FT, bool CHECK_BEFORE, bool CHECK_AFTER);

size_t abft_dlarfb_flops(magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                            magma_int_t m, magma_int_t n, magma_int_t k, int nb,
                             bool COL_FT, bool ROW_FT, bool CHECK_BEFORE, bool CHECK_AFTER);


size_t abft_sgemm_flops(magma_trans_t transA, magma_trans_t transB,
            int m, int n, int k, int nb,
            bool COL_FT, bool ROW_FT, bool CHECK_BEFORE, bool CHECK_AFTER);

size_t abft_slarfb_flops(magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                            magma_int_t m, magma_int_t n, magma_int_t k, int nb,
                             bool COL_FT, bool ROW_FT, bool CHECK_BEFORE, bool CHECK_AFTER);