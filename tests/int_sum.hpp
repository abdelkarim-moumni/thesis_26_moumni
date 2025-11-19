#pragma once

#include <iostream>
#include <cstring>

#include <cuhear.hpp>
#include "utils.hpp"
#include "time.hpp"

namespace tests::int_sum {

    __global__ static void vectorSum(uint32_t *out, uint32_t *in, size_t count) {
        size_t index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < count) {
            out[index] += in[index];
        }
    }

    static void SecretMessageEq(cuhear::rng::AesContext *d_aes) {
        // Simulate an allreduce ring with two ranks:
        // Rank 0: (d_keys0, d_secretMessageEnc0)
        // Rank 1: (d_keys1, d_secretMessageEnc1)
        //
        // Rank 0 has the payload "This is a secret message!"
        // Rank 1 has the payload comprised of all zeroes
        // Operation is sum allreduce
        //
        // Rank 0 encrypts its payload and passes it on
        // Rank 1 encrypts its payload and passes it on
        // Rank 0 sums the two encrypted payloads
        // Rank 0 decrypts the payload with its own key
        // If everything went well, Rank 0 should still read "This is a secret message!"

        cuhear::KeyStorage *d_keys0, *d_keys1;
        cuhear::KeyStorage keys0 {
            .communicatorKey = 0xDEADBEEF,
            .ownKey = 0xCAFECAFE,
            .nextKey = 0xDEADDEAD,
            .rootKey = 0xCAFECAFE,
        };
        cuhear::KeyStorage keys1 {
            .communicatorKey = 0xDEADBEEF,
            .ownKey = 0xDEADDEAD,
            .nextKey = 0,
            .rootKey = 0xCAFECAFE,
        };

        CHECK_CUDA_CALL(cudaMalloc(&d_keys0, sizeof(keys0)));
        CHECK_CUDA_CALL(cudaMalloc(&d_keys1, sizeof(keys1)));
        CHECK_CUDA_CALL(cudaMemcpy(d_keys0, &keys0, sizeof(keys0), cudaMemcpyHostToDevice));
        CHECK_CUDA_CALL(cudaMemcpy(d_keys1, &keys1, sizeof(keys1), cudaMemcpyHostToDevice));

        int blockSize = 32;

        uint32_t *d_secretMessage, *d_secretMessageEnc0, *d_secretMessageEnc1;
        alignas(4) char secretMessage[512] = "This is a secret message!";
        size_t secretMessageSize = (strlen(secretMessage) + 1 + 3) & ~3;
        size_t count = secretMessageSize / sizeof(uint32_t);

        CHECK_CUDA_CALL(cudaMalloc(&d_secretMessage, secretMessageSize));
        CHECK_CUDA_CALL(cudaMalloc(&d_secretMessageEnc0, secretMessageSize));
        CHECK_CUDA_CALL(cudaMalloc(&d_secretMessageEnc1, secretMessageSize));
        CHECK_CUDA_CALL(cudaMemset(d_secretMessage, 0, secretMessageSize));
        CHECK_CUDA_CALL(cudaMemcpy(d_secretMessage, (uint32_t *) secretMessage, secretMessageSize, cudaMemcpyHostToDevice));

        // Encrypt on rank 0
        Timer timer {"Rank 0 encrypt"};
        cuhear::kernels::int_sum::encrypt<<<(count + blockSize - 1)/blockSize, blockSize>>>(d_keys0, d_aes, d_secretMessageEnc0, d_secretMessage, count, false);
        CHECK_CUDA_LAST();
        CHECK_CUDA_CALL(cudaDeviceSynchronize());
        timer.Lap();
        // Encrypt all zeroes on rank 1
        CHECK_CUDA_CALL(cudaMemset(d_secretMessage, 0, secretMessageSize));
        cuhear::kernels::int_sum::encrypt<<<(count + blockSize - 1)/blockSize, blockSize>>>(d_keys1, d_aes, d_secretMessageEnc1, d_secretMessage, count, true);
        CHECK_CUDA_LAST();
        // Sum on rank 0
        vectorSum<<<(count + blockSize - 1)/blockSize, blockSize>>>(d_secretMessageEnc0, d_secretMessageEnc1, count);
        CHECK_CUDA_LAST();
        // Decrypt on rank 0
        cuhear::kernels::int_sum::decrypt<<<(count + blockSize - 1)/blockSize, blockSize>>>(d_keys0, d_aes, d_secretMessageEnc0, count);
        CHECK_CUDA_LAST();

        CHECK_CUDA_CALL(cudaMemcpy((uint32_t*) secretMessage, d_secretMessageEnc0, secretMessageSize, cudaMemcpyDeviceToHost));

        std::cout << "Result: \"" << secretMessage << "\"" << std::endl;

        cudaFree(d_secretMessage);
        cudaFree(d_secretMessageEnc0);
        cudaFree(d_secretMessageEnc1);
        cudaFree(d_keys0);
        cudaFree(d_keys1);
    }

    void Run(cuhear::rng::AesContext *d_aes) {
        SecretMessageEq(d_aes);
    }

}