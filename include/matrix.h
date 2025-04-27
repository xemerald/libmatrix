/**
 * @file matrix.h
 * @author Benjamin Ming Yang @ Department of Geoscience, National Taiwan University (b98204032@gmail.com)
 * @brief
 * @date 2018-03-05
 *
 * @copyright Copyright (c) 2018
 *
 */
#pragma once

#include <stdio.h>  /* Need for the FILE* */
#include <stddef.h> /* Need for the size_t */

/* Library version */
#define LIBMATRIX_VERSION "1.1.0"
/* Library release date */
#define LIBMATRIX_RELEASE "2025.04.27"

/**
 * @brief
 *
 */
typedef struct _matrix_t {
	size_t  i;
	size_t  j;
	size_t  total;
	double *element;
} matrix_t;

/**
 * @brief
 *
 */
#define matrix_at( _MATRIX, _ROW, _COL ) \
		((_MATRIX)->element[(_ROW - 1) * (_MATRIX)->j + (_COL - 1)])

/**
 * @name
 *
 */
void matrix_print( const matrix_t *, FILE *, const char * );

/**
 * @name
 *
 */
matrix_t *matrix_new( const size_t, const size_t );
matrix_t *matrix_idt( const size_t );
matrix_t *matrix_dup( const matrix_t * );
matrix_t *matrix_tps( const matrix_t * );
matrix_t *matrix_inv( const matrix_t * );

/**
 * @name
 *
 */
matrix_t *matrix_add( const matrix_t *, const matrix_t * );
matrix_t *matrix_sub( const matrix_t *, const matrix_t * );
matrix_t *matrix_mul( const matrix_t *, const matrix_t * );
matrix_t *matrix_div( const matrix_t *, const matrix_t * );
matrix_t *matrix_wls( const matrix_t *, const matrix_t *, const matrix_t *, const matrix_t * );

/**
 * @name
 *
 */
matrix_t *matrix_scalar_add( matrix_t *, const double );
matrix_t *matrix_scalar_sub( matrix_t *, const double );
matrix_t *matrix_scalar_mul( matrix_t *, const double );
matrix_t *matrix_scalar_div( matrix_t *, const double );

/**
 * @name
 *
 */
matrix_t *matrix_lup_dec( const matrix_t *, matrix_t **, matrix_t **, matrix_t ** );

/**
 * @name
 *
 */
matrix_t *matrix_set( matrix_t *, const double, const size_t, const size_t );
matrix_t *matrix_set_all( matrix_t *, const double *, const size_t );
matrix_t *matrix_set_row( matrix_t *, const double *, const size_t, const size_t );
matrix_t *matrix_set_col( matrix_t *, const double *, const size_t, const size_t );
matrix_t *matrix_set_diag( matrix_t *, const double *, const size_t );
matrix_t *matrix_set_va( matrix_t *, size_t, ... );

/**
 * @name
 *
 */
matrix_t *matrix_get( const matrix_t *, double *, const size_t, const size_t );
matrix_t *matrix_get_all( const matrix_t *, double *, const size_t );
matrix_t *matrix_get_row( const matrix_t *, double *, const size_t, const size_t );
matrix_t *matrix_get_col( const matrix_t *, double *, const size_t, const size_t );
matrix_t *matrix_get_diag( const matrix_t *, double *, const size_t );

/**
 * @name
 *
 */
matrix_t *matrix_apply( matrix_t *, double (*)( double, void * ), void *, const size_t, const size_t );
matrix_t *matrix_apply_all( matrix_t *, double (*)( double, void * ), void * );
matrix_t *matrix_apply_row( matrix_t *, double (*)( double, void * ), void *, const size_t );
matrix_t *matrix_apply_col( matrix_t *, double (*)( double, void * ), void *, const size_t );
matrix_t *matrix_apply_diag( matrix_t *, double (*)( double, void * ), void * );

/**
 * @name
 *
 */
size_t matrix_rk( const matrix_t * );
double matrix_tr( const matrix_t * );
double matrix_det( const matrix_t * );

/**
 * @name
 *
 */
int  matrix_cmp( const matrix_t *, const matrix_t *, const double );
void matrix_free( matrix_t * );
