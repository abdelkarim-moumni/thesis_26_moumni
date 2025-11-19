#pragma once

#include <cstdint>

namespace aes {
    const int NUM_ROUNDS = 10;
    typedef uint4 u128;
}

namespace cuhear::rng {

    class AesContext {
        private:
            __host__ __device__ AesContext(aes::u128 *roundKeys);

        public:
            aes::u128 roundKeys[aes::NUM_ROUNDS + 1];
            __device__ AesContext() {}
            
            __device__ static AesContext ExpandOnDevice(aes::u128 key);
            __host__ static AesContext ExpandOnHost(uint8_t *key);

            __device__ aes::u128 GenNoise(aes::u128 block);
            __device__ aes::u128 GenNoiseB(aes::u128 block, aes::u128 *rk);
    };

}