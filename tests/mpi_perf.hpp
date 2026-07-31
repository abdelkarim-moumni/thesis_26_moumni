#pragma once

#include <mpi.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>

extern "C" void cuhear_setup_intra_node(size_t num_items, MPI_Datatype mpi_type);
extern "C" void cuhear_cleanup_intra_node();

template <typename I>
int mpi_perf(size_t num_items, int argc, char *argv[], I unit, MPI_Datatype mpi_type, bool random_data) {
  static std::mt19937 random_gen;
  int num_iters = 10;
  bool copy = std::getenv("CUHEAR_COPY");
  bool cpuonly = std::getenv("CUHEAR_NOGPU");

  int rank, numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double start, end;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  // DEBUG DATATYPE VALUE
  if (rank == 0) {
      std::string typeName = "";
      if (mpi_type == MPI_INT) typeName = "INT (int32_t)";
      else if (mpi_type == MPI_FLOAT) typeName = "FLOAT (float32)";

      std::cout << "\n========================================" << std::endl;
      std::cout << "[TEST CONFIG] Type: " << typeName << std::endl;
      std::cout << "[TEST CONFIG] Items: " << num_items << " (" << (num_items * sizeof(I)) / 1024.0 << " KB)" << std::endl;
      std::cout << "[TEST CONFIG] Mode: " << (cpuonly ? "CPU-ONLY" : "CUDA-GPU") << std::endl;
      std::cout << "========================================\n" << std::endl;
  }

  I* d_buf, *hostBuf, *testBuf = nullptr;
  if (!cpuonly) {
    CHECK_CUDA_CALL(cudaMalloc(&d_buf, sizeof(I) * num_items));
    cuhear_setup_intra_node(num_items, mpi_type);
  }

  hostBuf = new I[num_items];
  
  if (copy) {
    testBuf = hostBuf;
  } else {
    testBuf = d_buf;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  std::vector<double> times(num_iters + 1, 0);

  for (int i = 0; i < num_iters; i++) {
    for (int j = 0; j < num_items; j++) {
      hostBuf[j] = random_data ? random_gen() : unit;
    }
    if (!cpuonly) {
      CHECK_CUDA_CALL(cudaMemcpy(d_buf, hostBuf, sizeof(I) * num_items, cudaMemcpyHostToDevice));
      CHECK_CUDA_CALL(cudaDeviceSynchronize());
    }

    double in_start = MPI_Wtime();
    int before;
    if (copy) {
      before = testBuf[0];
      if (!cpuonly) {
        CHECK_CUDA_CALL(cudaMemcpy(testBuf, d_buf, sizeof(I) * num_items, cudaMemcpyDeviceToHost));
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, testBuf, num_items, mpi_type, MPI_SUM, MPI_COMM_WORLD);
    // if (copy) {
    //   for (int j = 0; j < num_items; j++) {
    //     assert(unit * numProcs == testBuf[j]);
    //   }
    //   if (!cpuonly) {
    //     CHECK_CUDA_CALL(cudaMemcpy(d_buf, testBuf, sizeof(I) * num_items, cudaMemcpyHostToDevice));
    //   }
    // }
    if (i == 0) {
        I* checkBuf = new I[num_items];
        if (!cpuonly && !copy) {
            cudaMemcpy(checkBuf, testBuf, num_items * sizeof(I), cudaMemcpyDeviceToHost);
        } else {
            checkBuf = (I*)testBuf;
        }

        if (rank == 0) {
            bool correct = true;
            I expected = unit * (I)numProcs;
            for (size_t j = 0; j < std::min(num_items, (size_t)10); j++) {
                if (checkBuf[j] != expected) {
                    std::cout << "ERROR!!! Element " << j << " was " << checkBuf[j] 
                              << " but expected " << expected << std::endl;
                    correct = false;
                    break;
                }
            }
            if (correct) {
                std::cout << "Verification of the first 10 elements: OK (Value: " << expected << ")" << std::endl;
            }
        }
        if (!cpuonly && !copy) delete[] checkBuf;
    }

    // CHECK_CUDA_CALL(cudaMemcpy(hostBuf, d_buf, sizeof(I) * num_items, cudaMemcpyDeviceToHost));
    // std::cout << hostBuf[0] << std::endl;

    times[i] = MPI_Wtime() - in_start;
  }

  end = MPI_Wtime();

  times[num_iters] = end - start;

  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : times.data(), times.data(), num_iters + 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < num_iters; i++) {
      std::cout << "Run " << i + 1 << ": " << times[i] << std::endl;
    }
    std::cout << "Total: " << times[num_iters] << std::endl;
  }

  if (!cpuonly) {
    cudaFree(d_buf);
    cuhear_cleanup_intra_node();
  }
  delete[] hostBuf;

  return 0;
}
