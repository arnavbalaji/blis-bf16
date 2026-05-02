#include "blis.h"
#include <stdlib.h>

//Helper implemented in bls_sbgemm.c
void bls_sgemm_via_bf16
     (
       dim_t m, dim_t n, dim_t k, const float* alpha,
       const float* a, inc_t rs_a, inc_t cs_a, trans_t transa,
       const float* b, inc_t rs_b, inc_t cs_b, trans_t transb,
       const float* beta, float* c, inc_t rs_c, inc_t cs_c
     );

static bool bls_env_enabled_bf16(void)
{
	const char* env = getenv("BLIS_SANDBOX_BF16");
	if (env == NULL) 
	{
		return FALSE;
	}

	//Accept a few common "truthy" values
	if (env[0] == '1' || env[0] == 'y' || env[0] == 'Y' ||
	     env[0] == 't' || env[0] == 'T') 
		 {
			return TRUE;
		 }
	return FALSE;
}

void bli_gemm_ex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	const bool enable_bf16 = bls_env_enabled_bf16();

	if (enable_bf16 && bli_obj_is_float(a) && bli_obj_is_float(b) && bli_obj_is_float(c))
	{
		const dim_t m = bli_obj_length(c);
		const dim_t n = bli_obj_width(c);
		const dim_t k = bli_obj_width_after_trans(a);

		const float* alpha_s = (const float*) bli_obj_buffer_for_const(BLIS_FLOAT, alpha);
		const float* beta_s  = (const float*) bli_obj_buffer_for_const(BLIS_FLOAT, beta);

		const float* a_buf = (const float*) bli_obj_buffer_at_off(a);
		const float* b_buf = (const float*) bli_obj_buffer_at_off(b);
		float* c_buf = (float*) bli_obj_buffer_at_off(c);

		const inc_t rs_a = bli_obj_row_stride(a);
		const inc_t cs_a = bli_obj_col_stride(a);
		const inc_t rs_b = bli_obj_row_stride(b);
		const inc_t cs_b = bli_obj_col_stride(b);
		const inc_t rs_c = bli_obj_row_stride(c);
		const inc_t cs_c = bli_obj_col_stride(c);

		const trans_t transa = bli_obj_onlytrans_status(a);
		const trans_t transb = bli_obj_onlytrans_status(b);

		bls_sgemm_via_bf16
		(
		  m, n, k,
		  alpha_s,
		  a_buf, rs_a, cs_a, transa,
		  b_buf, rs_b, cs_b, transb,
		  beta_s,
		  c_buf, rs_c, cs_c
		);
		return;
	}

	bli_gemm_def_ex
	(
	  (obj_t*)alpha, (obj_t*)a, (obj_t*)b, (obj_t*)beta, (obj_t*)c,
	  (cntx_t*)cntx, (rntm_t*)rntm
	);
}

