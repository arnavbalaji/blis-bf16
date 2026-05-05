/*
   BLIS sandbox: replace float32 GEMM with a BF16 microkernel path.

   - BF16 (uint16) 16x4 microkernel that accumulates into float32
   - float32 -> BF16 packing routines that produce microkernel's expected
     layout
   - A small blocked driver (`bls_sgemm_via_bf16`) that:
       * packs float32 A/B to BF16
       * calls microkernel over full m x n result
*/

#include "blis.h"

#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define BLS_BF16_AVX2 1
#else
#define BLS_BF16_AVX2 0
#endif


#ifndef BLS_BF16_MC
#define BLS_BF16_MC 224
#endif
#ifndef BLS_BF16_NC
#define BLS_BF16_NC 2304
#endif
#ifndef BLS_BF16_KC
#define BLS_BF16_KC 192
#endif

#define BLS_BF16_MR 16
#define BLS_BF16_NR 4

static inline float bls_bf16_to_f32(uint16_t bf16)
{
	uint32_t bits = ((uint32_t)bf16) << 16;
	float out;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static inline uint16_t bls_f32_to_bf16_trunc(float x)
{
	uint32_t bits;
	memcpy(&bits, &x, sizeof(bits));
	return (uint16_t)(bits >> 16);
}

// ---- Microkernel: C := beta*C + alpha*A*B (A,B packed) ----------------------

#if BLS_BF16_AVX2

static inline __m256 bls_vec_bf16_to_f32_8(__m128i bf16_bits)
{
	__m256i fp32_bits = _mm256_cvtepu16_epi32(bf16_bits);
	fp32_bits = _mm256_slli_epi32(fp32_bits, 16);
	return _mm256_castsi256_ps(fp32_bits);
}

static void bls_microkernel_sbgemm_16x4
     (
       int k,
       const uint16_t* a_packed,
       const uint16_t* b_packed,
       const float* alpha,
       const float* beta,
       float* c, inc_t rs_c, inc_t cs_c
     )
{
	const float alpha_s = *alpha;
	const float beta_s  = *beta;

	__m256 gamma0[BLS_BF16_NR], gamma1[BLS_BF16_NR];

	//Initialize accumulators
	if (beta_s == 0.0f)
	{
		for (int j = 0; j < BLS_BF16_NR; ++j)
		{
			gamma0[j] = _mm256_setzero_ps();
			gamma1[j] = _mm256_setzero_ps();
		}
	}
	else
	{
		for (int j = 0; j < BLS_BF16_NR; ++j)
		{
			float* c_col = c + j * cs_c;

			if (rs_c == 1)
			{
				gamma0[j] = _mm256_mul_ps(_mm256_loadu_ps(c_col + 0), _mm256_set1_ps(beta_s));
				gamma1[j] = _mm256_mul_ps(_mm256_loadu_ps(c_col + 8), _mm256_set1_ps(beta_s));
			}
			else
			{
				float tmp0[8], tmp1[8];
				for (int i = 0; i < 8; ++i) 
				{
					tmp0[i] = c_col[(0 + i) * rs_c];
				}
				for (int i = 0; i < 8; ++i) 
				{
					tmp1[i] = c_col[(8 + i) * rs_c];
				}

				gamma0[j] = _mm256_mul_ps(_mm256_loadu_ps(tmp0), _mm256_set1_ps(beta_s));
				gamma1[j] = _mm256_mul_ps(_mm256_loadu_ps(tmp1), _mm256_set1_ps(beta_s));
			}
		}
	}

	int p = 0;
	while (p + 1 < k)
	{
		__m128i a00 = _mm_loadu_si128((const __m128i*)(a_packed + (p + 0) * BLS_BF16_MR + 0));
		__m128i a01 = _mm_loadu_si128((const __m128i*)(a_packed + (p + 0) * BLS_BF16_MR + 8));
		__m256 alpha0_0 = bls_vec_bf16_to_f32_8(a00);
		__m256 alpha1_0 = bls_vec_bf16_to_f32_8(a01);

		__m128i a10 = _mm_loadu_si128((const __m128i*)(a_packed + (p + 1) * BLS_BF16_MR + 0));
		__m128i a11 = _mm_loadu_si128((const __m128i*)(a_packed + (p + 1) * BLS_BF16_MR + 8));
		__m256 alpha0_1 = bls_vec_bf16_to_f32_8(a10);
		__m256 alpha1_1 = bls_vec_bf16_to_f32_8(a11);

		for (int j = 0; j < BLS_BF16_NR; ++j)
		{
			uint32_t bj0 = ((uint32_t)b_packed[(p + 0) * BLS_BF16_NR + j]) << 16;
			uint32_t bj1 = ((uint32_t)b_packed[(p + 1) * BLS_BF16_NR + j]) << 16;

			float fbj0, fbj1;
			memcpy(&fbj0, &bj0, sizeof(fbj0));
			memcpy(&fbj1, &bj1, sizeof(fbj1));

			__m256 b0 = _mm256_broadcast_ss(&fbj0);
			__m256 b1 = _mm256_broadcast_ss(&fbj1);

			//Fold alpha into B broadcasts
			if (alpha_s != 1.0f)
			{
				const __m256 aalpha = _mm256_set1_ps(alpha_s);
				b0 = _mm256_mul_ps(b0, aalpha);
				b1 = _mm256_mul_ps(b1, aalpha);
			}

			gamma0[j] = _mm256_fmadd_ps(alpha0_0, b0, gamma0[j]);
			gamma1[j] = _mm256_fmadd_ps(alpha1_0, b0, gamma1[j]);
			gamma0[j] = _mm256_fmadd_ps(alpha0_1, b1, gamma0[j]);
			gamma1[j] = _mm256_fmadd_ps(alpha1_1, b1, gamma1[j]);
		}

		p += 2;
	}

	while (p < k)
	{
		__m128i a0 = _mm_loadu_si128((const __m128i*)(a_packed + p * BLS_BF16_MR + 0));
		__m128i a1 = _mm_loadu_si128((const __m128i*)(a_packed + p * BLS_BF16_MR + 8));
		__m256 alpha0 = bls_vec_bf16_to_f32_8(a0);
		__m256 alpha1 = bls_vec_bf16_to_f32_8(a1);

		for (int j = 0; j < BLS_BF16_NR; ++j)
		{
			uint32_t bj = ((uint32_t)b_packed[p * BLS_BF16_NR + j]) << 16;
			float fbj;
			memcpy( &fbj, &bj, sizeof(fbj));
			__m256 bvec = _mm256_broadcast_ss(&fbj);
			if (alpha_s != 1.0f) 
			{
				bvec = _mm256_mul_ps(bvec, _mm256_set1_ps(alpha_s));
			}
			gamma0[j] = _mm256_fmadd_ps(alpha0, bvec, gamma0[j]);
			gamma1[j] = _mm256_fmadd_ps(alpha1, bvec, gamma1[j]);
		}
		++p;
	}

	//Store back to C
	for (int j = 0; j < BLS_BF16_NR; ++j)
	{
		float* c_col = c + j * cs_c;
		if (rs_c == 1)
		{
			_mm256_storeu_ps(c_col + 0, gamma0[j]);
			_mm256_storeu_ps(c_col + 8, gamma1[j]);
		}
		else
		{
			float tmp0[8], tmp1[8];
			_mm256_storeu_ps(tmp0, gamma0[j]);
			_mm256_storeu_ps(tmp1, gamma1[j]);

			for (int i = 0; i < 8; ++i) 
			{
				c_col[(0+i)*rs_c] = tmp0[i];
			}
			for (int i = 0; i < 8; ++i) 
			{
				c_col[(8+i)*rs_c] = tmp1[i];
			}
		}
	}
}

#endif

static void bls_pack_a_16xk_from_f32
     (
       dim_t m, dim_t k,
       const float* a, inc_t rs_a, inc_t cs_a, trans_t transa,
       uint16_t* a_packed
     )
{
	dim_t idx = 0;

	if (transa == BLIS_NO_TRANSPOSE)
	{
		for (dim_t p = 0; p < k; ++p) 
		{
			for (dim_t i = 0; i < BLS_BF16_MR; ++i) 
			{
				a_packed[idx] = (i < m ? bls_f32_to_bf16_trunc(a[i * rs_a + p * cs_a]) : 0);
				idx++;
			}
		}
	}
	else
	{
		//A is logically transposed: A_op(i,p) = A(p,i)
		for (dim_t p = 0; p < k; ++p) 
		{
			for (dim_t i = 0; i < BLS_BF16_MR; ++i) 
			{
				a_packed[idx] = ( i < m ? bls_f32_to_bf16_trunc(a[p * rs_a + i * cs_a] ) : 0);
				idx++;
			}
		}
	}
}

static void bls_pack_b_kx4_from_f32
     (
       dim_t        k, dim_t n,
       const float* b, inc_t rs_b, inc_t cs_b, trans_t transb,
             uint16_t* b_packed
     )
{
	dim_t idx = 0;

	if (transb == BLIS_NO_TRANSPOSE)
	{
		for (dim_t p = 0; p < k; ++p) 
		{
			for (dim_t j = 0; j < BLS_BF16_NR; ++j) 
			{
				b_packed[idx] = (j < n ? bls_f32_to_bf16_trunc(b[p * rs_b + j * cs_b]) : 0);
				idx++;
			}
		}
	}
	else
	{
		//B is logically transposed: B_op(p,j) = B(j,p)
		for (dim_t p = 0; p < k; ++p) 
		{
			for (dim_t j = 0; j < BLS_BF16_NR; ++j) 
			{
				b_packed[idx] = (j < n ? bls_f32_to_bf16_trunc(b[j * rs_b + p * cs_b]) : 0);
				idx++;
			}
		}
	}
}

//Replace sgemm with BF16 microkernel
void bls_sgemm_via_bf16
     (
       dim_t m,
       dim_t n,
       dim_t k,
       const float* alpha,
       const float* a, inc_t rs_a, inc_t cs_a, trans_t transa,
       const float* b, inc_t rs_b, inc_t cs_b, trans_t transb,
       const float* beta,
       float* c, inc_t rs_c, inc_t cs_c
     )
{
	if  (m == 0 || n == 0) 
	{
		return;
	}

	const float alpha_s = alpha ? *alpha : 1.0f;
	const float beta_s = beta ? *beta : 1.0f;

	if (k == 0) 
	{
		if (beta_s == 1.0f)
		{
			return;
		}
		for (dim_t j = 0; j < n; ++j)
		{
			for (dim_t i = 0; i < m; ++i)
			{
				c[i * rs_c + j * cs_c] *= beta_s;
			}
		}
		return;
	}

	const dim_t MC = BLS_BF16_MC;
	const dim_t NC = BLS_BF16_NC;
	const dim_t KC = BLS_BF16_KC;

	const dim_t MC_pad = ((MC + BLS_BF16_MR - 1) / BLS_BF16_MR) * BLS_BF16_MR;
	const dim_t NC_pad = ((NC + BLS_BF16_NR - 1) / BLS_BF16_NR) * BLS_BF16_NR;

	uint16_t* a_packed = (uint16_t*)malloc((size_t)MC_pad * (size_t)KC * sizeof(uint16_t));
	uint16_t* b_packed = (uint16_t*)malloc((size_t)KC * (size_t)NC_pad * sizeof(uint16_t));

	if (a_packed == NULL || b_packed == NULL)
	{
	//Catch / handle pack failure
		for (dim_t jj = 0; jj < n; ++jj)
		{
			for (dim_t ii = 0; ii < m; ++ii)
			{
				float acc = (beta_s == 0.0f ? 0.0f : beta_s * c[ii * rs_c + jj * cs_c]);
				for (dim_t pp = 0; pp < k; ++pp)
				{
					const float av = (transa == BLIS_NO_TRANSPOSE ? 
						a[ii * rs_a + pp * cs_a] : a[pp * rs_a + ii * cs_a]);
					const float bv = ( transb == BLIS_NO_TRANSPOSE ? 
						b[pp * rs_b + jj * cs_b]: b[jj * rs_b + pp * cs_b]);
					acc += alpha_s * av * bv;
				}
				c[ii * rs_c + jj * cs_c] = acc;
			}
		}

		free(a_packed);
		free(b_packed);
		return;
	}

	for (dim_t jc = 0; jc < n; jc += NC)
	{
		const dim_t nc = (n - jc < NC ? n - jc : NC);

		for (dim_t pc = 0; pc < k; pc += KC)
		{
			const dim_t kc = (k - pc < KC ? k - pc : KC);

			//Pack B
			dim_t bidx = 0;
			for (dim_t jr = 0; jr < nc; jr += BLS_BF16_NR)
			{
				const dim_t nr = (nc - jr < BLS_BF16_NR ? nc - jr : BLS_BF16_NR);
				bls_pack_b_kx4_from_f32
				(
				  kc, nr,
				  b + (transb == BLIS_NO_TRANSPOSE
				          ? (pc * rs_b + (jc + jr) * cs_b)
				          : ((jc + jr) * rs_b + pc * cs_b)),
				  rs_b, cs_b, transb,
				  b_packed + bidx
				);
				bidx += kc * BLS_BF16_NR;
			}

			for (dim_t ic = 0; ic < m; ic += MC)
			{
				const dim_t mc = (m - ic < MC ? m - ic : MC);

				//Pack A
				dim_t aidx = 0;
				for (dim_t ir = 0; ir < mc; ir += BLS_BF16_MR)
				{
					const dim_t mr = (mc - ir < BLS_BF16_MR ? mc - ir : BLS_BF16_MR);
					bls_pack_a_16xk_from_f32
					(
					  mr, kc,
					  a + ( transa == BLIS_NO_TRANSPOSE
					          ? ((ic + ir) * rs_a + pc * cs_a)
					          : (pc * rs_a + (ic + ir) * cs_a)),
					  rs_a, cs_a, transa,
					  a_packed + aidx
					);
					aidx += kc * BLS_BF16_MR;
				}

				//Compute microtiles
				for (dim_t jr = 0; jr < nc; jr += BLS_BF16_NR)
				{
					const dim_t nr = nc - jr < BLS_BF16_NR ? nc - jr : BLS_BF16_NR;

					for (dim_t ir = 0; ir < mc; ir += BLS_BF16_MR)
					{
						const dim_t mr = mc - ir < BLS_BF16_MR ? mc - ir : BLS_BF16_MR;

						const uint16_t* a_ker = a_packed + ((ir / BLS_BF16_MR) * kc * BLS_BF16_MR);
						const uint16_t* b_ker = b_packed + ((jr / BLS_BF16_NR) * kc * BLS_BF16_NR);

						float* c_tile = c + (ic + ir) * rs_c + (jc + jr) * cs_c;

						const float beta_eff = (pc == 0 ? beta_s : 1.0f);
						const float alpha_eff = alpha_s;

						if (mr == BLS_BF16_MR && nr == BLS_BF16_NR)
						{
#if BLS_BF16_AVX2
							bls_microkernel_sbgemm_16x4
							(
							  (int)kc, a_ker, b_ker,
							  &alpha_eff, &beta_eff,
							  c_tile, rs_c, cs_c
							);
#else
							(void)alpha_eff; 
							(void)beta_eff;

							for (dim_t jj = 0; jj < nr; ++jj)
							{
								for (dim_t ii = 0; ii < mr; ++ii)
								{
									float acc = (beta_eff == 0.0f ? 0.0f : beta_eff * c_tile[ii * rs_c + jj * cs_c]);
									for (dim_t pp = 0; pp < kc; ++pp)
									{
										const float av = bls_bf16_to_f32(a_ker[pp*BLS_BF16_MR + ii]);
										const float bv = bls_bf16_to_f32(b_ker[pp*BLS_BF16_NR + jj]);
										acc += alpha_eff * av * bv;
									}
									c_tile[ii * rs_c + jj * cs_c] = acc;
								}
							}
#endif
						}
						else
						{
							float tmp[BLS_BF16_MR * BLS_BF16_NR];
							for (dim_t jj = 0; jj < BLS_BF16_NR; ++jj)
							{
								for (dim_t ii = 0; ii < BLS_BF16_MR; ++ii)
								{
									tmp[ii + jj * BLS_BF16_MR] = 0.0f;
								}
							}

							for (dim_t jj = 0; jj < nr; ++jj)
							{
								for (dim_t ii = 0; ii < mr; ++ii)
								{
									tmp[ii + jj * BLS_BF16_MR] = (beta_eff == 0.0f ? 
										0.0f : beta_eff * c_tile[ii * rs_c + jj * cs_c]);
								}
							}

							for (dim_t pp = 0; pp < kc; ++pp)
							{
								for (dim_t jj = 0; jj < nr; ++jj)
								{
									const float bv = bls_bf16_to_f32(b_ker[pp * BLS_BF16_NR + jj]);
									for (dim_t ii = 0; ii < mr; ++ii)
									{
										const float av = bls_bf16_to_f32(a_ker[pp * BLS_BF16_MR + ii]);
										tmp[ii + jj * BLS_BF16_MR] += alpha_eff * av * bv;
									}
								}
							}

							for (dim_t jj = 0; jj < nr; ++jj)
							{
								for (dim_t ii = 0; ii < mr; ++ii)
								{
									c_tile[ii * rs_c + jj * cs_c] = tmp[ii + jj * BLS_BF16_MR];
								}
							}
						}
					}
				}
			}
		}
	}

	free(a_packed);
	free(b_packed);
}

