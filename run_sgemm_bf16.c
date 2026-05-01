/*
  Minimal driver to exercise the bf16 sandbox path.

  Build after configuring BLIS with: ./configure -s bf16 auto && make -j
*/

#include "blis.h"

#include <stdio.h>
#include <stdlib.h>

static void fill_rand_s( float* x, dim_t m, dim_t n, inc_t rs, inc_t cs )
{
	for ( dim_t j = 0; j < n; ++j )
	for ( dim_t i = 0; i < m; ++i )
		x[i*rs + j*cs] = (float)rand() / (float)RAND_MAX - 0.5f;
}

static float max_abs_diff( const float* a, const float* b, dim_t m, dim_t n, inc_t rs, inc_t cs )
{
	float md = 0.0f;
	for ( dim_t j = 0; j < n; ++j )
	for ( dim_t i = 0; i < m; ++i )
	{
		float d = a[i*rs + j*cs] - b[i*rs + j*cs];
		if ( d < 0.0f ) d = -d;
		if ( d > md ) md = d;
	}
	return md;
}

int main( int argc, char** argv )
{
	dim_t m = 128, n = 128, k = 128;
	if ( argc == 4 )
	{
		m = (dim_t)atoi( argv[1] );
		n = (dim_t)atoi( argv[2] );
		k = (dim_t)atoi( argv[3] );
	}

	const inc_t rs_a = 1, cs_a = m;
	const inc_t rs_b = 1, cs_b = k;
	const inc_t rs_c = 1, cs_c = m;

	float* A  = (float*)malloc( (size_t)m * (size_t)k * sizeof(float) );
	float* B  = (float*)malloc( (size_t)k * (size_t)n * sizeof(float) );
	float* C0 = (float*)malloc( (size_t)m * (size_t)n * sizeof(float) );
	float* C1 = (float*)malloc( (size_t)m * (size_t)n * sizeof(float) );

	if ( !A || !B || !C0 || !C1 )
	{
		fprintf( stderr, "malloc failed\n" );
		return 1;
	}

	srand( 1 );
	fill_rand_s( A,  m, k, rs_a, cs_a );
	fill_rand_s( B,  k, n, rs_b, cs_b );
	fill_rand_s( C0, m, n, rs_c, cs_c );
	for ( dim_t j = 0; j < n; ++j )
	for ( dim_t i = 0; i < m; ++i )
		C1[i*rs_c + j*cs_c] = C0[i*rs_c + j*cs_c];

	bli_init();

	const float alpha = 1.0f;
	const float beta  = 1.0f;

	obj_t a, b, c0, c1;
	bli_obj_create_with_attached_buffer( BLIS_FLOAT, m, k, A,  rs_a, cs_a, &a  );
	bli_obj_create_with_attached_buffer( BLIS_FLOAT, k, n, B,  rs_b, cs_b, &b  );
	bli_obj_create_with_attached_buffer( BLIS_FLOAT, m, n, C0, rs_c, cs_c, &c0 );
	bli_obj_create_with_attached_buffer( BLIS_FLOAT, m, n, C1, rs_c, cs_c, &c1 );

	obj_t alpha_o, beta_o;
	bli_obj_create_1x1_with_attached_buffer( BLIS_FLOAT, (void*)&alpha, &alpha_o );
	bli_obj_create_1x1_with_attached_buffer( BLIS_FLOAT, (void*)&beta,  &beta_o  );

	// Reference: normal BLIS sgemm.
	unsetenv( "BLIS_SANDBOX_BF16" );
	bli_gemm( &alpha_o, &a, &b, &beta_o, &c0 );

	// Sandbox path: sgemm replaced by bf16 microkernel path.
	setenv( "BLIS_SANDBOX_BF16", "1", 1 );
	bli_gemm( &alpha_o, &a, &b, &beta_o, &c1 );

	printf( "m=%ld n=%ld k=%ld  max_abs_diff=%g\n", (long)m, (long)n, (long)k,
	        (double)max_abs_diff( C0, C1, m, n, rs_c, cs_c ) );

	bli_finalize();

	free( A );
	free( B );
	free( C0 );
	free( C1 );

	return 0;
}

