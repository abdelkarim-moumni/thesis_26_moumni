#ifndef ENCRYPT_HPP
#define ENCRYPT_HPP

#include <vector>
#include <random>

#include <immintrin.h>

#include <wmmintrin.h>


namespace encryption {

using encr_key_t = unsigned int;

void encrypt_int_sum_naive(unsigned int *encr_sbuf, const unsigned int *sbuf, int count, int rank,
			   std::vector<unsigned int> &k_s, unsigned int k_n, bool is_edge);
void decrypt_int_sum_naive(unsigned int *rbuf, int count,
			   std::vector<unsigned int> &k_s, unsigned int k_n);
void encrypt_int_sum_sha1sse2(unsigned int *encr_sbuf, const unsigned int *sbuf, int count, int rank,
			      std::vector<unsigned int> &k_s, unsigned int k_n, bool is_edge);
void decrypt_int_sum_sha1sse2(unsigned int *rbuf, int count, std::vector<unsigned int> &k_s, unsigned int k_n);
void encrypt_int_sum_sha1avx2(unsigned int *encr_sbuf, const unsigned int *sbuf, int count, int rank,
			      std::vector<unsigned int> &k_s, unsigned int k_n, bool is_edge);
void decrypt_int_sum_sha1avx2(unsigned int *rbuf, int count, std::vector<unsigned int> &k_s, unsigned int k_n);
void encrypt_int_prod_naive(unsigned int *encr_sbuf, const unsigned int *sbuf, int count, int rank,
			   std::vector<unsigned int> &k_s, unsigned int k_n, bool is_edge);
void decrypt_int_prod_naive(unsigned int *rbuf, int count,
			    std::vector<unsigned int> &k_s, unsigned int k_n);
void encrypt_float_sum_naive(float *encr_sbuf, const float *sbuf, int count, int rank,
			     std::vector<unsigned int> &k_s, unsigned int k_n);
void decrypt_float_sum_naive(float *rbuf, int count, std::vector<unsigned int> &k_s, unsigned int k_n);

void aesni128_encrypt_m128i(char *plain_text, char *cipher_text);


unsigned int aesni128_prng(unsigned int);
void aesni128_load_key(char *enc_key);

void encrypt_int_sum_aesni128(unsigned int *encr_sbuf, const unsigned int *sbuf, int count, int rank,
			      std::vector<unsigned int> &k_s, unsigned int k_n, bool is_edge);
void decrypt_int_sum_aesni128(unsigned int *rbuf, int count, std::vector<unsigned int> &k_s, unsigned int k_n);
void encrypt_int_sum_aesni128_unroll(unsigned int *encr_sbuf, const unsigned int *sbuf, int count, int rank,
				     std::vector<unsigned int> &k_s, unsigned int k_n, bool is_edge);
void decrypt_int_sum_aesni128_unroll(unsigned int *rbuf, int count, std::vector<unsigned int> &k_s, unsigned int k_n);
void encrypt_float_sum_aesni128_unroll(float *encr_sbuf, const float *sbuf, int count, int rank,
				       std::vector<unsigned int> &k_s, unsigned int k_n);
void decrypt_float_sum_aesni128_unroll(float *rbuf, int count, std::vector<unsigned int> &k_s, unsigned int k_n);


}

#endif
