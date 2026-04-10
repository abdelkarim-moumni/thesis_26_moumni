#include "kernels.hpp"
#include "aes_gnulib.hpp"
#include "float.hpp"

__device__ inline uint4 VecAdd(uint4 a, uint4 b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

__device__ inline uint4 VecSub(uint4 a, uint4 b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

__device__ inline uint4 VecScalarMul(uint4 a, uint32_t scalar) {
    return { a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar };
}

namespace cuhear {
    __global__ void IntraNodeSum(int size, int num_followers, uint32_t* leader_buf, uint32_t** follower_bufs) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            uint32_t sum = leader_buf[idx];
            for (int i = 0; i < num_followers; i++) {
                sum += follower_bufs[i][idx];
            }
            leader_buf[idx] = sum;
        }
    }
}


namespace cuhear::kernels::int_sum {

    __global__ void encrypt(cuhear::KeyStorage *keys, cuhear::rng::AesContext *aes, void *raw_out, void *raw_in, size_t count, bool isLast) {
        __shared__ uint4 rkS[AES_ROUND_KEY_COUNT];

        if (threadIdx.x < AES_ROUND_KEY_COUNT) {
            rkS[threadIdx.x] = aes->roundKeys[threadIdx.x];
        }
        __syncthreads();

        uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= count) {
            return;
        }
        index *= 4;

        uint32_t *out = (uint32_t *) raw_out, *in = (uint32_t *) raw_in;

        cuhear::Key tmp1 = keys->communicatorKey + keys->ownKey;
        cuhear::Key tmp2 = keys->communicatorKey + keys->nextKey;
        uint4 data { in[index], in[index + 1], in[index + 2], in[index + 3] };

        uint4 ind1 { 3 + tmp1, 2 + tmp1, 1 + tmp1, tmp1 };
        ind1 = VecAdd(ind1, {index, index, index, index});
        uint4 ind2 { 3 + tmp2, 2 + tmp2, 1 + tmp2, tmp2 };
        ind2 = VecAdd(ind2, {index, index, index, index});

        uint4 noise1 = aes->GenNoiseB(ind1, rkS);
        uint4 noise2 = aes->GenNoiseB(ind2, rkS);

        uint4 encrypted = !isLast ? VecSub(VecAdd(data, noise1), noise2) : VecAdd(data, noise1);
        out[index] = encrypted.x;
        out[index + 1] = encrypted.y;
        out[index + 2] = encrypted.z;
        out[index + 3] = encrypted.w;
    }

    __global__ void decrypt(cuhear::KeyStorage *keys, cuhear::rng::AesContext *aes, void *raw_inout, size_t count) {
        __shared__ uint4 rkS[AES_ROUND_KEY_COUNT];

        if (threadIdx.x < AES_ROUND_KEY_COUNT) {
            rkS[threadIdx.x] = aes->roundKeys[threadIdx.x];
        }
        __syncthreads();

        uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= count) {
            return;
        }
        index *= 4;

        uint32_t *inout = (uint32_t *) raw_inout;

        cuhear::Key tmp = keys->communicatorKey + keys->rootKey;
        uint4 data { inout[index], inout[index + 1], inout[index + 2], inout[index + 3] };

        uint4 ind { 3 + tmp, 2 + tmp, 1 + tmp, tmp };
        ind = VecAdd(ind, {index, index, index, index});

        uint4 noise = aes->GenNoiseB(ind, rkS);

        uint4 decrypted = VecSub(data, noise);
        inout[index] = decrypted.x;
        inout[index + 1] = decrypted.y;
        inout[index + 2] = decrypted.z;
        inout[index + 3] = decrypted.w;
    }

}

namespace cuhear::kernels::float_sum {

    __global__ void encrypt(cuhear::KeyStorage *keys, cuhear::rng::AesContext *aes, void *raw_out, void *raw_in, size_t count, bool isLast) {
        __shared__ uint4 rkS[AES_ROUND_KEY_COUNT];

        if (threadIdx.x < AES_ROUND_KEY_COUNT) {
            rkS[threadIdx.x] = aes->roundKeys[threadIdx.x];
        }
        __syncthreads();

        uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= count) {
            return;
        }
        index *= 4;

        float *out = (float *) raw_out, *in = (float *) raw_in;

        cuhear::Key tmp = keys->communicatorKey;
        uint4 ind { tmp + 1, tmp + 2, tmp + 3, tmp + 4 };
        ind = VecAdd(ind, {index, index, index, index});
        uint4 tmpNoise = aes->GenNoiseB(ind, rkS);
        cuhear::floats::FloatBits *noise = reinterpret_cast<cuhear::floats::FloatBits *>(&tmpNoise);

        for (int i = 0; i < 4; i++) {
            int32_t exponent = noise[i].crypto.exponent;
            noise[i].ieee_float.ieee.exponent = IEEE754_FLOAT_BIAS;
            noise[i].ieee_float.ieee.mantissa <<= SHIFT;
            noise[i].native_float *= in[index + i];
            noise[i].crypto_simplified.remainder >>= SHIFT;
            noise[i].crypto.exponent += exponent - IEEE754_FLOAT_BIAS;
            out[index + i] = noise[i].native_float;
        }
    }

    __global__ void decrypt(cuhear::KeyStorage *keys, cuhear::rng::AesContext *aes, void *raw_inout, size_t count) {
        __shared__ uint4 rkS[AES_ROUND_KEY_COUNT];

        if (threadIdx.x < AES_ROUND_KEY_COUNT) {
            rkS[threadIdx.x] = aes->roundKeys[threadIdx.x];
        }
        __syncthreads();

        uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= count) {
            return;
        }
        index *= 4;

        float *inout = (float *) raw_inout;

        cuhear::Key tmp = keys->communicatorKey;
        uint4 ind { tmp + 1, tmp + 2, tmp + 3, tmp + 4 };
        ind = VecAdd(ind, {index, index, index, index});
        uint4 tmpNoise = aes->GenNoiseB(ind, rkS);
        cuhear::floats::FloatBits *noise = reinterpret_cast<cuhear::floats::FloatBits *>(&tmpNoise);

        for (int i = 0; i < 4; i++) {
            cuhear::floats::FloatBits hnum = reinterpret_cast<cuhear::floats::FloatBits &>(inout[index + i]);
            hnum.crypto.exponent = (hnum.crypto.exponent - noise[i].crypto.exponent + IEEE754_FLOAT_BIAS) << SHIFT;
            hnum.ieee_float.ieee.mantissa <<= SHIFT;
            noise[i].ieee_float.ieee.exponent = IEEE754_FLOAT_BIAS;
            noise[i].ieee_float.ieee.mantissa <<= SHIFT;
            inout[index + i] = hnum.native_float / noise[i].native_float;
        }
    }

}

namespace cuhear::kernels::benchmark {

    __global__ void aes(cuhear::rng::AesContext *ctx, uint4 *buf, uint4 *outBuf, size_t len) {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < len) {
            uint4 res = ctx->GenNoise(buf[index]);
            outBuf[index] = res;
        }
    }

}

namespace cuhear::kernels::crypto {

    __global__ void rotate_key(cuhear::rng::AesContext *ctx, cuhear::KeyStorage *keys) {
        keys->communicatorKey = ctx->GenNoise({ keys->communicatorKey, 0, 0, 0 }).x;
    }

}

