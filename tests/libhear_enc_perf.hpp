#pragma once

#include <vector>
#include <chrono>
#include <iostream>
#include <random>
#include <functional>
#include <cstring>

using namespace std::chrono;

std::mt19937 random_gen;

const size_t warmup_niters = 25;

template<typename T>
void randomize_vector(T *input, std::size_t count)
{
    for (auto i = 0; i < count; i++) {
        input[i] = static_cast<T>(random_gen());
    }
}

template<typename T>
void print_vector(T *input, std::size_t count)
{
    for (auto i = 0; i < count; i++) {
	std::cout << input[i] << std::endl;
    }
}

int libhear_main(int argc, char **argv)
{
    /* ./encr_perf_test <niters> <nitems> <nranks> <dtype> <op> <func>*/
    if (argc != 7) {
        std::cerr << "Bad arguments!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::size_t niters = std::atoi(argv[1]);
    std::size_t nitems = std::atoi(argv[2]);
    std::size_t bufsize = nitems * sizeof(uint32_t);
    bufsize = ((bufsize + 127) / 128) * 128;
    std::size_t nranks = std::atoi(argv[3]);
    size_t blocksize = 128;
    
    uint32_t *d_buf, *d_encBuf;

    CHECK_CUDA_CALL(cudaMalloc(&d_buf, bufsize));
    CHECK_CUDA_CALL(cudaMalloc(&d_encBuf, bufsize));
    CHECK_CUDA_CALL(cudaMemset(d_buf, 0, bufsize));

    uint8_t key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    cuhear::rng::AesContext aesContext = cuhear::rng::AesContext::ExpandOnHost(key);
    cuhear::rng::AesContext *d_context;
    CHECK_CUDA_CALL(cudaMalloc(&d_context, sizeof(aesContext)));
    CHECK_CUDA_CALL(cudaMemcpy(d_context, &aesContext, sizeof(aesContext), cudaMemcpyHostToDevice));

    cuhear::KeyStorage keyStorage {
        .communicatorKey = random_gen(),
        .ownKey = random_gen(),
        .nextKey = random_gen(),
        .rootKey = random_gen()
    };
    cuhear::KeyStoragePtr keyPtr(keyStorage);

    std::cout << "Buffer size: " << bufsize << " Bytes" << std::endl;

    for (auto i = 0; i < warmup_niters; i++) {
	    cuhear::kernels::int_sum::encrypt<<<(nitems/4 + blocksize - 1)/blocksize,blocksize>>>(keyPtr.d_keys, d_context, d_encBuf, d_buf, nitems/4, false);
        CHECK_CUDA_LAST();
	}
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t1_encr = high_resolution_clock::now();

    for (auto i = 0; i < niters; i++) {
        cuhear::kernels::int_sum::encrypt<<<(nitems/4 + blocksize - 1)/blocksize,blocksize>>>(keyPtr.d_keys, d_context, d_encBuf, d_buf, nitems/4, false);
        CHECK_CUDA_LAST();
	}
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t2_encr = high_resolution_clock::now();

    for (auto i = 0; i < warmup_niters; i++) {
	    cuhear::kernels::int_sum::decrypt<<<(nitems/4 + blocksize - 1)/blocksize,blocksize>>>(keyPtr.d_keys, d_context, d_encBuf, nitems/4);
        CHECK_CUDA_LAST();
	}
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t1_decr = high_resolution_clock::now();

    for (auto i = 0; i < niters; i++) {
	    cuhear::kernels::int_sum::decrypt<<<(nitems/4 + blocksize - 1)/blocksize,blocksize>>>(keyPtr.d_keys, d_context, d_encBuf, nitems/4);
        CHECK_CUDA_LAST();
	}
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t2_decr = high_resolution_clock::now();

    uint8_t *buf = new uint8_t[bufsize];
    CHECK_CUDA_CALL(cudaMemcpy(d_encBuf, buf, bufsize, cudaMemcpyDeviceToHost));
    delete[] buf;

    duration<double> encr_diff = duration_cast<duration<double>>(t2_encr - t1_encr);
    auto encr_latency = encr_diff.count() / niters;
    auto encr_tput = static_cast<double>(bufsize) / (1e+9) / encr_latency;
    duration<double> decr_diff = duration_cast<duration<double>>(t2_decr - t1_decr);
    auto decr_latency = decr_diff.count() / niters;
    auto decr_tput = static_cast<double>(bufsize) / (1e+9) / decr_latency;
    std::cout << "Avg encryption time: " << encr_latency << " sec." << std::endl;
    std::cout << "Avg encryption throughput: " << encr_tput << " Gbytes/sec." << std::endl;
    std::cout << "Avg decryption time: " << decr_latency << " sec." << std::endl;
    std::cout << "Avg decryption throughput: " << decr_tput << " Gbytes/sec." << std::endl;

    /*
    delete[] sbuf;
    delete[] encr_sbuf;
    */

    return 0;
}
