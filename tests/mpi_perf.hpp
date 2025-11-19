#pragma once

#include <mpi.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>

template <typename I>
int mpi_perf(size_t num_items, int argc, char *argv[], I unit, MPI_Datatype mpi_type) {
  int num_iters = 10;
  bool copy = std::getenv("CUHEAR_COPY");
  bool cpuonly = std::getenv("CUHEAR_NOGPU");

  int rank, numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double start, end;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  I* d_buf, *hostBuf, *testBuf = nullptr;
  if (!cpuonly) {
    CHECK_CUDA_CALL(cudaMalloc(&d_buf, sizeof(I) * num_items));
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
      hostBuf[j] = unit;
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
    if (copy) {
      int after = testBuf[0];
      assert(before * numProcs == after);
      if (!cpuonly) {
        CHECK_CUDA_CALL(cudaMemcpy(d_buf, testBuf, sizeof(I) * num_items, cudaMemcpyHostToDevice));
      }
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
  }
  delete[] hostBuf;

  return 0;
}
