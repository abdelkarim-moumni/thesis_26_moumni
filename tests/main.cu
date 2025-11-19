#include <aes.hpp>
#include <kernels.hpp>
#include <cuhear.hpp>

#include <cstdio>
#include <cstdint>
#include <iostream>
#include <random>

#include "time.hpp"
#include "hear_aes.hpp"
#include "int_sum.hpp"
#include "utils.hpp"
#include "libhear_enc_perf.hpp"
#include "mpi_perf.hpp"

__constant__ cuhear::rng::AesContext AES_CONTEXT;

#include "omp.h"

int main(int argc, char** argv) {
    CHECK_CUDA_CALL(cudaSetDevice(0));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    int kib = 1024;
    int mib = 1024 * 1024;
    int sizes[] = {
        32,
        128,
 
    };

    if (true) {
        MPI_Init(&argc, &argv);
        for (int size : sizes) {
            std::cout << "[TEST] " << size << " Bytes" << std::endl;
            if (argc == 1) {
                mpi_perf<uint32_t>(size / 4, argc, argv, 1, MPI_INT);
            } else {
                mpi_perf<float>(size / 4, argc, argv, 4.0, MPI_FLOAT);
            }
        }
        MPI_Finalize();
        return 0;
    }

    std::cout << "Generating random data" << std::endl;
    Timer timer("Buffer allocation");
    int size = 250 * 1000 * 1000;
    uint32_t *buf = new uint32_t[size];
    uint32_t *d_buf;
    for (int i = 0; i < size; i++) {
        buf[i] = std::rand();
    }
    timer.Lap();

    std::cout << "Generated data" << std::endl;
    std::cout << "Before: " << std::hex << buf[0] << " " << buf[1] << " " << buf[2] << " " << buf[3] << std::endl;


    timer = {"Buffer copy"};
    CHECK_CUDA_CALL(cudaMalloc(&d_buf, size * sizeof(uint32_t)));
    CHECK_CUDA_CALL(cudaMemcpy(d_buf, buf, size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    timer.Lap();

    uint8_t key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    cuhear::rng::AesContext aesContext = cuhear::rng::AesContext::ExpandOnHost(key);
    cuhear::rng::AesContext *d_context;
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(AES_CONTEXT, &aesContext, sizeof(aesContext)));
    CHECK_CUDA_CALL(cudaMalloc(&d_context, sizeof(aesContext)));
    CHECK_CUDA_CALL(cudaMemcpy(d_context, &aesContext, sizeof(aesContext), cudaMemcpyHostToDevice));

    encryption::aesni128_load_key((char*) key);
    uint32_t *in_buf = new uint32_t[size];
    for (int i = 0; i < size; i++) in_buf[i] = buf[i];
    timer = {"HEAR CPU"};
    #pragma omp parallel for
    for (int i = 0; i < size; i += 4) {
        encryption::aesni128_encrypt_m128i((char*) &in_buf[i], (char*) &buf[i]);
    }
    timer.Lap();
    std::cout << "After [CPU]: " << std::hex << buf[0] << " " << buf[1] << " " << buf[2] << " " << buf[3] << std::endl;

    timer = {"Calculation"};
    cuhear::kernels::benchmark::aes<<<size/4/128,128>>>(d_context, (uint4*) d_buf, (uint4*) d_buf, size/4);
    CHECK_CUDA_LAST();
    cudaDeviceSynchronize();
    timer.Lap();

    timer = {"Copy back"};
    CHECK_CUDA_CALL(cudaMemcpy(buf, (uint32_t*) d_buf, size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    timer.Lap();

    std::cout << "Done!" << std::endl;
    std::cout << "After [GPU]: " << std::hex << buf[0] << " " << buf[1] << " " << buf[2] << " " << buf[3] << std::endl;

    delete[] in_buf;
    delete[] buf;

    std::cout << std::endl << " New tests " << std::endl;
    tests::int_sum::Run(d_context);

    return 0;
}