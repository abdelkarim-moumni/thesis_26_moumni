#pragma once

#include "cuhear.hpp"
#include <cstdint>

// Integer sum
namespace cuhear::kernels::int_sum {

    __global__ void encrypt(cuhear::KeyStorage *keys, cuhear::rng::AesContext *aes, void *out, void *in, size_t count, bool isLast);
    __global__ void decrypt(cuhear::KeyStorage *keys, cuhear::rng::AesContext *aes, void *inout, size_t count);

}

// Float sum
namespace cuhear::kernels::float_sum {

    __global__ void encrypt(cuhear::KeyStorage *keys, cuhear::rng::AesContext *aes, void *out, void *in, size_t count, bool isLast);
    __global__ void decrypt(cuhear::KeyStorage *keys, cuhear::rng::AesContext *aes, void *inout, size_t count);

}

// General purpose kernels for testing/benchmarks
namespace cuhear::kernels::benchmark {

    __global__ void aes(cuhear::rng::AesContext *ctx, uint4 *buf, uint4 *outBuf, size_t len);

}

namespace cuhear::kernels::crypto {

    __global__ void rotate_key(cuhear::rng::AesContext *ctx, cuhear::KeyStorage *keys);

}