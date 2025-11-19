#include "cuhear.hpp"
#include "utils.hpp"
#include "kernels.hpp"

#include <iostream>
#include <vector>
#include <random>

#include <mpi-ext.h>

static cuhear::CuHearState *cuhearState = nullptr;
const int root_rank = 0;
static std::mt19937 keyGenerator;

static bool IsMPICudaAware() {
    #if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        return MPIX_Query_cuda_support() != 0;
    #else
        return false;
    #endif
}

static void InitState() {
    if (cuhearState != nullptr) {
        std::cerr << "MPI_Init called multiple times before MPI_Finalize" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK_EXIT(cudaDeviceSynchronize());

    uint8_t key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    cuhear::rng::AesContext aesContext = cuhear::rng::AesContext::ExpandOnHost(key);

    cuhearState = new cuhear::CuHearState();

    cudaDeviceProp props;
    CUDA_CHECK_EXIT(cudaGetDeviceProperties(&props, 0));
    cuhearState->cudaUnifiedAddressing = props.unifiedAddressing != 0;
    cuhearState->mpiCudaAware = IsMPICudaAware();

    std::cout << "cuhear: Unified Addressing " << cuhearState->cudaUnifiedAddressing << std::endl;
    std::cout << "cuhear: CUDA-Aware MPI " << cuhearState->mpiCudaAware << std::endl;

    CUDA_CHECK_EXIT(cudaMalloc(&cuhearState->d_aesContext, sizeof(aesContext)));
    CUDA_CHECK_EXIT(cudaMemcpy(cuhearState->d_aesContext, &aesContext, sizeof(aesContext), cudaMemcpyHostToDevice));
}

static int NewComm(MPI_Comm comm, bool init_state) {
    int comm_size;
    int my_rank;

    // Multi GPU support - get node-based rank number and assign GPU device accordingly
    MPI_Comm node_comm;
    int node_rank;
    int gpu_count;
    PMPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    CUDA_CHECK_EXIT(cudaGetDeviceCount(&gpu_count));
    CUDA_CHECK_EXIT(cudaSetDevice(node_rank % gpu_count));

    if (init_state) {
        InitState();
    }

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &my_rank);

    std::vector<uint32_t> keys(comm_size, my_rank);
    keys[my_rank] = keyGenerator();

    // Each rank gets a copy of everyone's keys
    int ret = PMPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED, keys.data(), 1, MPI_UNSIGNED, comm);
    if (ret != MPI_SUCCESS) {
        return ret;
    }

    // Root rank generates and broadcasts shared communicator key
    uint32_t communicatorKey = my_rank == root_rank ? keyGenerator() : 0;
    ret = PMPI_Bcast(&communicatorKey, 1, MPI_UNSIGNED, root_rank, comm);
    if (ret != MPI_SUCCESS) {
        return ret;
    }

    cuhear::KeyStorage keyStorage {
        .communicatorKey = communicatorKey,
        .ownKey = keys[my_rank],
        .nextKey = my_rank != comm_size - 1 ? keys[my_rank + 1] : 0,
        .rootKey = root_rank < comm_size ? keys[root_rank] : 0
    };

    cuhear::KeyStoragePtr keyPtr(keyStorage);
    cuhearState->commKeys.emplace(comm, std::move(keyPtr));
    
    return MPI_SUCCESS;
}

typedef void (*ENC_K)(cuhear::KeyStorage*, cuhear::rng::AesContext*, void*, void*, size_t, bool);
typedef void (*DEC_K)(cuhear::KeyStorage*, cuhear::rng::AesContext*, void*, size_t);

template <ENC_K encrypt_fn, DEC_K decrypt_fn>
static inline int AllReduceImpl(const void *sendbuf, void *recvbuf, int count, MPI_Datatype aggregateType, MPI_Op aggregateOp, MPI_Comm comm) {
    cudaPointerAttributes ptrAttribs;
    int ret = 0;

    if (sendbuf == MPI_IN_PLACE) {
        sendbuf = recvbuf;
    }

    uint32_t *d_send = (uint32_t *) sendbuf;
    uint32_t *d_recv = (uint32_t *) recvbuf;
    uint32_t *d_encSend;

    bool ownSend = false, ownRecv = false;

    int dataSize;
    MPI_Type_size(aggregateType, &dataSize);

    size_t bufSize = dataSize * count;
    size_t blockSize = 48;

    // std::cout << "Sending " << *(uint32_t*) sendbuf << std::endl;

    if (cuhearState->cudaUnifiedAddressing) {
        CUDA_CHECK_EXIT(cudaPointerGetAttributes(&ptrAttribs, d_recv));
        if (ptrAttribs.type != cudaMemoryTypeDevice) {
            CUDA_CHECK_EXIT(cudaMalloc(&d_recv, bufSize));
            CUDA_CHECK_EXIT(cudaMemcpy(d_recv, recvbuf, bufSize, cudaMemcpyHostToDevice));
            ownRecv = true;
        }
        CUDA_CHECK_EXIT(cudaPointerGetAttributes(&ptrAttribs, d_send));
        if (ptrAttribs.type != cudaMemoryTypeDevice) {
            CUDA_CHECK_EXIT(cudaMalloc(&d_send, bufSize));
            CUDA_CHECK_EXIT(cudaMemcpy(d_send, sendbuf, bufSize, cudaMemcpyHostToDevice));
            ownSend = true;
        }
    }
    
    CUDA_CHECK_EXIT(cudaMalloc(&d_encSend, bufSize));

    int rank, csz;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &csz);

    // Communicator key is rotated globally at the start of every allreduce
    cuhear::kernels::crypto::rotate_key<<<1, 1>>>(cuhearState->d_aesContext, cuhearState->commKeys.at(comm).d_keys);
    CUDA_LAST_EXIT();

    int aes_blocks = (bufSize + 15) / 16;

    encrypt_fn<<<(aes_blocks + blockSize - 1) / blockSize, blockSize>>>(
        cuhearState->commKeys.at(comm).d_keys,
        cuhearState->d_aesContext,
        d_encSend,
        d_send,
        aes_blocks,
        rank == csz - 1
    );
    CUDA_LAST_EXIT();
    CUDA_CHECK_EXIT(cudaDeviceSynchronize());
    
    if (cuhearState->mpiCudaAware) {
        // CUDA-Aware MPI, can use GPU buffers directly
        ret = PMPI_Allreduce(d_encSend, d_recv, count, aggregateType, aggregateOp, comm);
    } else {
        // No GPU support, need to copy buffer to host
        uint32_t *encSend = new uint32_t[(bufSize + sizeof(uint32_t) - 1) / sizeof(uint32_t)];
        CUDA_CHECK_EXIT(cudaMemcpy(encSend, d_encSend, bufSize, cudaMemcpyDeviceToHost));
        ret = PMPI_Allreduce(MPI_IN_PLACE, encSend, count, aggregateType, aggregateOp, comm);
        if (ret == MPI_SUCCESS) {
            CUDA_CHECK_EXIT(cudaMemcpy(d_recv, encSend, bufSize, cudaMemcpyHostToDevice));
        }
        delete[] encSend;
    }

    if (ret != MPI_SUCCESS) {
        std::cerr << " MPI error " << ret << std::endl;
        std::exit(EXIT_FAILURE);
    }
    decrypt_fn<<<(aes_blocks + blockSize - 1) / blockSize, blockSize>>>(
        cuhearState->commKeys.at(comm).d_keys,
        cuhearState->d_aesContext,
        d_recv,
        aes_blocks
    );
    CUDA_LAST_EXIT();

    if (ownRecv) {
        CUDA_CHECK_EXIT(cudaMemcpy(recvbuf, d_recv, bufSize, cudaMemcpyDeviceToHost));
        // std::cout << "Received " << *(uint32_t*) recvbuf << std::endl;
    }

    CUDA_CHECK_EXIT(cudaFree(d_encSend));
    if (ownSend) CUDA_CHECK_EXIT(cudaFree(d_send));
    if (ownRecv) CUDA_CHECK_EXIT(cudaFree(d_recv));

    CUDA_CHECK_EXIT(cudaDeviceSynchronize());

    return ret;
}

int MPI_Init(int *argc, char ***argv) {
    int orig = PMPI_Init(argc, argv);
    if (orig == MPI_SUCCESS) {
        orig = NewComm(MPI_COMM_WORLD, true);
    }
    return orig;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int orig = PMPI_Init_thread(argc, argv, required, provided);
    if (orig == MPI_SUCCESS) {
        orig = NewComm(MPI_COMM_WORLD, true);
    }
    return orig;
}

int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) {
    int orig = PMPI_Comm_create(comm, group, newcomm);
    if (orig == MPI_SUCCESS) {
        orig = NewComm(*newcomm, false);
    }
    return orig;
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
    int orig = PMPI_Comm_split(comm, color, key, newcomm);
    if (orig == MPI_SUCCESS) {
        orig = NewComm(*newcomm, false);
    }
    return orig;
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
    int orig = PMPI_Comm_dup(comm, newcomm);
    if (orig == MPI_SUCCESS) {
        orig = NewComm(*newcomm, false);
    }
    return orig;
}

int MPI_Finalize() {
    if (cuhearState != nullptr) {
        delete cuhearState;
        cuhearState = nullptr;
    }
    return PMPI_Finalize();
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

    if (datatype == MPI_INT && op == MPI_SUM) {
        return AllReduceImpl<cuhear::kernels::int_sum::encrypt, cuhear::kernels::int_sum::decrypt>(sendbuf, recvbuf, count, datatype, op, comm);
    }

    if (datatype == MPI_FLOAT && op == MPI_SUM) {
        return AllReduceImpl<cuhear::kernels::float_sum::encrypt, cuhear::kernels::float_sum::decrypt>(sendbuf, recvbuf, count, datatype, op, comm);
    }

    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}