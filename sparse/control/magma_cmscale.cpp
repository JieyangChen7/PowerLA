/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/control/magma_zmscale.cpp, normal z -> c, Thu Oct  8 23:05:52 2020
       @author Hartwig Anzt
       @author Stephen Wood

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Scales a matrix.

    Arguments
    ---------

    @param[in,out]
    A           magma_c_matrix*
                input/output matrix

    @param[in]
    scaling     magma_scale_t
                scaling type (unit rownorm / unit diagonal)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmscale(
    magma_c_matrix *A,
    magma_scale_t scaling,
    magma_queue_t queue ){
    magma_int_t info = 0;
    
    magmaFloatComplex *tmp=NULL;
    
    magma_c_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
    
    if( A->num_rows != A->num_cols && scaling != Magma_NOSCALE ){
        printf("%% warning: non-square matrix.\n");
        printf("%% Fallback: no scaling.\n");
        scaling = Magma_NOSCALE;
    } 
        
   
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        if ( scaling == Magma_NOSCALE ) {
            // no scale
            ;
        }
        else if( A->num_rows == A->num_cols ){
            if ( scaling == Magma_UNITROW ) {
                // scale to unit rownorm
                CHECK( magma_cmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                        s+= MAGMA_C_REAL(A->val[f])*MAGMA_C_REAL(A->val[f]);
                    tmp[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );
                }        
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
            }
            else if (scaling == Magma_UNITDIAG ) {
                // scale to unit diagonal
                CHECK( magma_cmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                        if ( A->col[f]== z ) {
                            // add some identity matrix
                            //A->val[f] = A->val[f] +  MAGMA_C_MAKE( 100000.0, 0.0 );
                            s = A->val[f];
                        }
                    }
                    if ( s == MAGMA_C_MAKE( 0.0, 0.0 ) ){
                        printf("%%error: zero diagonal element.\n");
                        info = MAGMA_ERR;
                    }
                    tmp[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );
                }
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
            }
            else {
                printf( "%%error: scaling not supported.\n" );
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
        else {
            printf( "%%error: scaling not supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_cmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_cmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));

        CHECK( magma_cmscale( &CSRA, scaling, queue ));

        magma_cmfree( &hA, queue );
        magma_cmfree( A, queue );
        CHECK( magma_cmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
        CHECK( magma_cmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
cleanup:
    magma_free_cpu( tmp );
    magma_cmfree( &hA, queue );
    magma_cmfree( &CSRA, queue );
    return info;
}

/**
    Purpose
    -------

    Scales a matrix and a right hand side vector of a Ax = b system.

    Arguments
    ---------

    @param[in,out]
    A           magma_c_matrix*
                input/output matrix
                
    @param[in,out]
    b           magma_c_matrix*
                input/output right hand side vector    
                
    @param[out]
    scaling_factors   magma_c_matrix*
                output scaling factors vector   

    @param[in]
    scaling     magma_scale_t
                scaling type (unit rownorm / unit diagonal)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmscale_matrix_rhs(
    magma_c_matrix *A,
    magma_c_matrix *b,
    magma_c_matrix *scaling_factors,
    magma_scale_t scaling,
    magma_queue_t queue ) {
    magma_int_t info = 0;
    
    magmaFloatComplex *tmp=NULL;
    
    magma_c_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
    
    // printf("%% scaling = %d\n", scaling);
    
    if( A->num_rows != A->num_cols && scaling != Magma_NOSCALE ){
        printf("%% warning: non-square matrix.\n");
        printf("%% Fallback: no scaling.\n");
        scaling = Magma_NOSCALE;
    } 
        
   
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        if ( scaling == Magma_NOSCALE ) {
            // no scale
            ;
        }
        else if( A->num_rows == A->num_cols ){
            if ( scaling == Magma_UNITROW ) {
                // scale to unit rownorm by rows
                CHECK( magma_cmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                        s+= MAGMA_C_MAKE( MAGMA_C_REAL(A->val[f])*MAGMA_C_REAL(A->val[f]), 0.0 );
                    tmp[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );
                }        
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->rowidx[z]];
                }
                for ( int i=0; i<A->num_rows; i++ ) {
                  b->val[i] = b->val[i] * tmp[i];
                }
            }
            else if (scaling == Magma_UNITDIAG ) {
                // scale to unit diagonal by rows
                CHECK( magma_cmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                        if ( A->col[f]== z ) {
                            s = A->val[f];
                        }
                    }
                    if ( s == MAGMA_C_MAKE( 0.0, 0.0 ) ){
                        printf("%%error: zero diagonal element.\n");
                        info = MAGMA_ERR;
                    }
                    tmp[z] = MAGMA_C_MAKE( 1.0/MAGMA_C_REAL( s ), 0.0 );
                }
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->rowidx[z]];
                }
                for ( int i=0; i<A->num_rows; i++ ) {
                  b->val[i] = b->val[i] * tmp[i];
                }
            }
            else if ( scaling == Magma_UNITROWCOL ) {
                // scale to unit rownorm by rows and columns
                CHECK( magma_cmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                        s+= MAGMA_C_MAKE( MAGMA_C_REAL(A->val[f])*MAGMA_C_REAL(A->val[f]), 0.0 );
                    tmp[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );
                }        
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
                scaling_factors->num_rows = A->num_rows;
                scaling_factors->num_cols = 1;
                scaling_factors->ld = 1;
                scaling_factors->nnz = A->num_rows;
                scaling_factors->val = NULL;
                CHECK( magma_cmalloc_cpu( &scaling_factors->val, A->num_rows ));
                for ( int i=0; i<A->num_rows; i++ ) {
                  scaling_factors->val[i] = tmp[i];
                  b->val[i] = b->val[i] * tmp[i];
                }
            }
            else if (scaling == Magma_UNITDIAGCOL ) {
                // scale to unit diagonal by rows and columns
                CHECK( magma_cmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                        if ( A->col[f]== z ) {
                            s = A->val[f];
                        }
                    }
                    if ( s == MAGMA_C_MAKE( 0.0, 0.0 ) ){
                        printf("%%error: zero diagonal element.\n");
                        info = MAGMA_ERR;
                    }
                    tmp[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );
                }
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
                scaling_factors->num_rows = A->num_rows;
                scaling_factors->num_cols = 1;
                scaling_factors->ld = 1;
                scaling_factors->nnz = A->num_rows;
                scaling_factors->val = NULL;
                CHECK( magma_cmalloc_cpu( &scaling_factors->val, A->num_rows ));
                for ( int i=0; i<A->num_rows; i++ ) {
                  scaling_factors->val[i] = tmp[i];
                  b->val[i] = b->val[i] * tmp[i];
                }
            }
            else {
                printf( "%%error: scaling %d not supported line = %d.\n", 
                  scaling, __LINE__ );
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
        else {
            printf( "%%error: scaling %d not supported line = %d.\n", 
                  scaling, __LINE__ );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_cmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_cmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));

        CHECK( magma_cmscale_matrix_rhs( &CSRA, b, scaling_factors, scaling, queue ));

        magma_cmfree( &hA, queue );
        magma_cmfree( A, queue );
        CHECK( magma_cmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
        CHECK( magma_cmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
cleanup:
    magma_free_cpu( tmp );
    magma_cmfree( &hA, queue );
    magma_cmfree( &CSRA, queue );
    return info;
}

/**
    Purpose
    -------

    Adds a multiple of the Identity matrix to a matrix: A = A+add * I

    Arguments
    ---------

    @param[in,out]
    A           magma_c_matrix*
                input/output matrix

    @param[in]
    add         magmaFloatComplex
                scaling for the identity matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmdiagadd(
    magma_c_matrix *A,
    magmaFloatComplex add,
    magma_queue_t queue ){
    magma_int_t info = 0;
    
    magma_c_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
    
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        for( magma_int_t z=0; z<A->nnz; z++ ) {
            if ( A->col[z]== A->rowidx[z] ) {
                // add some identity matrix
                A->val[z] = A->val[z] +  add;
            }
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_cmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_cmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));

        CHECK( magma_cmdiagadd( &CSRA, add, queue ));

        magma_cmfree( &hA, queue );
        magma_cmfree( A, queue );
        CHECK( magma_cmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
        CHECK( magma_cmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
cleanup:
    magma_cmfree( &hA, queue );
    magma_cmfree( &CSRA, queue );
    return info;
}

/**
    Purpose
    -------

    Generates n vectors of scaling factors from the A matrix 
    and stores them in the factors matrix as column vectors in 
    column major ordering.

    Arguments
    ---------

    @param[in]
    n           magma_int_t
                number of diagonal scaling matrices 
                
    @param[in]
    scaling     magma_scale_t*
                array of scaling specifiers 
                
    @param[in]
    side        magma_side_t*
                array of side specifiers 
                
    @param[in]
    A           magma_c_matrix*
                input matrix
                
    @param[out]
    scaling_factors  magma_c_matrix*
                array of diagonal matrices
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.


    @ingroup magmasparse_caux
    ********************************************************************/
    
extern "C" magma_int_t
magma_cmscale_generate( 
      magma_int_t n, 
      magma_scale_t* scaling, 
      magma_side_t* side, 
      magma_c_matrix* A, 
      magma_c_matrix* scaling_factors,
    magma_queue_t queue  ){
    magma_int_t info = 0;
    
    magmaFloatComplex *tmp=NULL;
    
    magma_c_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
    
    
    if( A->num_rows != A->num_cols && scaling[0] != Magma_NOSCALE ) {
        printf("%% warning: non-square matrix.\n");
        printf("%% Fallback: no scaling.\n");
        scaling[0] = Magma_NOSCALE;
    } 
        
   
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        for ( magma_int_t j=0; j<n; j++ ) {
        // printf("%% scaling[%d] = %d\n", j, scaling[j]);
            if ( scaling[j] == Magma_NOSCALE ) {
                // no scale
            
            }
            else if( A->num_rows == A->num_cols ) {
                if ( scaling[j] == Magma_UNITROW && side[j] != MagmaBothSides ) {
                // scale to unit rownorm
                    for( magma_int_t z=0; z<A->num_rows; z++ ) {
                        magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                        for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                            s+= MAGMA_C_MAKE( MAGMA_C_REAL(A->val[f])*MAGMA_C_REAL(A->val[f]), 0.0 );
                        scaling_factors[j].val[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );
                    }        
                }
                else if (scaling[j] == Magma_UNITDIAG && side[j] != MagmaBothSides ) {
                // scale to unit diagonal
                    for( magma_int_t z=0; z<A->num_rows; z++ ) {
                        magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                        for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                            if ( A->col[f]== z ) {
                                s = A->val[f];
                            }
                        }
                        if ( s == MAGMA_C_MAKE( 0.0, 0.0 ) ){
                            printf("%%error: zero diagonal element.\n");
                            info = MAGMA_ERR;
                        }
                        scaling_factors[j].val[z] = MAGMA_C_MAKE( 1.0/MAGMA_C_REAL( s ), 0.0 );
                    }
                }
                else if ( scaling[j] == Magma_UNITCOL && side[j] != MagmaBothSides ) {
                // scale to unit column norm
                    CHECK( magma_cmtranspose( *A, &CSRA, queue ) );
                    magma_scale_t tscale = Magma_UNITROW;
                    magma_cmscale_generate( 1, &tscale, &side[j], &CSRA, 
                            &scaling_factors[j], queue );
                }
                else if ( scaling[j] == Magma_UNITROW && side[j] == MagmaBothSides ) {
                    // scale to unit rownorm by rows and columns
                    for( magma_int_t z=0; z<A->num_rows; z++ ) {
                        magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                        for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                            s+= MAGMA_C_MAKE( MAGMA_C_REAL(A->val[f])*MAGMA_C_REAL(A->val[f]), 0.0 );
                        scaling_factors[j].val[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );
                    } 
                }
                else if (scaling[j] == Magma_UNITDIAG && side[j] == MagmaBothSides ) {
                    // scale to unit diagonal by rows and columns
                    for( magma_int_t z=0; z<A->num_rows; z++ ) {
                        magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                        for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                            if ( A->col[f]== z ) {
                                s = A->val[f];
                            }
                        }
                        if ( s == MAGMA_C_MAKE( 0.0, 0.0 ) ){
                            printf("%%error: zero diagonal element.\n");
                            info = MAGMA_ERR;
                        }
                        scaling_factors[j].val[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );
                    }
                }
                else if ( scaling[j] == Magma_UNITCOL && side[j] == MagmaBothSides ) {
                    // scale to unit column norm
                    CHECK( magma_cmtranspose( *A, &CSRA, queue ) );
                    magma_scale_t tscale = Magma_UNITROW;
                    magma_cmscale_generate( 1, &tscale, &side[j], &CSRA, 
                      &scaling_factors[j], queue );
                }
                else {
                    printf( "%%error: scaling %d not supported line = %d.\n", 
                      scaling[j], __LINE__ );
                    info = MAGMA_ERR_NOT_SUPPORTED;
                }
            }
            else {
                printf( "%%error: scaling of non-square matrices %d not supported line = %d.\n", 
                      scaling[0], __LINE__ );
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_cmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_cmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));

        CHECK( magma_cmscale_generate( n, scaling, side, &CSRA, scaling_factors, queue ));

        magma_cmfree( &hA, queue );
        magma_cmfree( A, queue );
        CHECK( magma_cmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
        CHECK( magma_cmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
    
cleanup:
    magma_free_cpu( tmp );
    magma_cmfree( &hA, queue );
    magma_cmfree( &CSRA, queue );
    return info;
}


/**
    Purpose
    -------

    Applies n diagonal scaling matrices to a matrix A; 
    n=[1,2], factor[i] is applied to side[i] of the matrix.

    Arguments
    ---------

    @param[in]
    n           magma_int_t
                number of diagonal scaling matrices 
                
    @param[in]
    side        magma_side_t*
                array of side specifiers 
                
    @param[in]
    scaling_factors  magma_c_matrix*
                array of diagonal matrices
                
    @param[in,out]
    A           magma_c_matrix*
                input/output matrix
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/
    
extern "C" magma_int_t
magma_cmscale_apply( 
      magma_int_t n,  
      magma_side_t* side, 
      magma_c_matrix* scaling_factors, 
      magma_c_matrix* A,
      magma_queue_t queue ){
    magma_int_t info = 0;
      
    magmaFloatComplex *tmp=NULL;
    
    magma_c_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
    
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        for ( magma_int_t j=0; j<n; j++ ) {
            
            if( A->num_rows == A->num_cols ) {
                if ( side[j] == MagmaLeft ) {
                    // scale by rows       
                    for( magma_int_t z=0; z<A->nnz; z++ ) {
                        A->val[z] = A->val[z] * scaling_factors[j].val[A->rowidx[z]];
                    }
                }
                else if ( side[j] == MagmaBothSides ) {
                    // scale by rows and columns       
                    for( magma_int_t z=0; z<A->nnz; z++ ) {
                        A->val[z] = A->val[z] 
                            * scaling_factors[j].val[A->col[z]] 
                            * scaling_factors[j].val[A->rowidx[z]];
                    }
                }
                else if ( side[j] == MagmaRight ) {
                    // scale by columns
                    for( magma_int_t z=0; z<A->nnz; z++ ) {
                        A->val[z] = A->val[z] * scaling_factors[j].val[A->rowidx[z]];
                    }
                    
                }
            }
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_cmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_cmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));

        CHECK( magma_cmscale_apply( n, side, scaling_factors, &CSRA, queue ));

        magma_cmfree( &hA, queue );
        magma_cmfree( A, queue );
        CHECK( magma_cmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
        CHECK( magma_cmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
    
cleanup:
    magma_free_cpu( tmp );
    magma_cmfree( &hA, queue );
    magma_cmfree( &CSRA, queue );
  
  
    return info;
}


/**
    Purpose
    -------

    Multiplies a diagonal matrix (vecA) and a vector (vecB).

    Arguments
    ---------

    @param[in]
    vecA        magma_c_matrix*
                input matrix
                
    @param[in,out]
    vecB        magma_c_matrix*
                input/output matrix
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/
    
extern "C" magma_int_t
magma_cdimv( 
    magma_c_matrix* vecA, 
    magma_c_matrix* vecB,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_c_matrix hA={Magma_CSR}, hB={Magma_CSR};
    
    if ( vecA->memory_location == Magma_DEV && vecB->memory_location == Magma_DEV)  {
        //printf("%% magma_cdimv scaling\n");
        
        magmablas_clascl2( 
            vecB->fill_mode, vecB->num_rows, vecB->num_cols,
            (magmaFloat_ptr) vecA->val,
            vecB->val, vecB->ld,
            queue,
            &info );
    }
    else {
        //printf("%% magma_cdimv transfering vectors to device\n");
        
        magma_location_t vecA_location = vecA->memory_location;
        CHECK( magma_cmtransfer( *vecA, &hA, vecA->memory_location, Magma_DEV, queue ));
        magma_location_t vecB_location = vecB->memory_location;
        CHECK( magma_cmtransfer( *vecB, &hB, vecB->memory_location, Magma_DEV, queue ));
        
        //printf("%% vecA->memory_location=%d vecB->memory_location=%d\n", 
        //  vecA->memory_location, vecB->memory_location );
       
        magma_cdimv( &hA, &hB, queue );
        
        magma_cmfree( vecA, queue );
        magma_cmfree( vecB, queue );
        CHECK( magma_cmtransfer( hA, vecA, Magma_DEV, vecA_location, queue ));
        CHECK( magma_cmtransfer( hB, vecB, Magma_DEV, vecB_location, queue ));
    }
    
cleanup:
        magma_cmfree( &hA, queue );
        magma_cmfree( &hB, queue );
    
    return info;
}