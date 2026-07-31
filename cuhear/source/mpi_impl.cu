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
    int comm_size, my_rank;
    MPI_Comm node_comm, inter_node_comm;
    int node_rank, gpu_count;

    // node-level split and GPU assignment
    PMPI_Comm_size(comm, &comm_size);
    PMPI_Comm_rank(comm, &my_rank);
    
    PMPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    
    CUDA_CHECK_EXIT(cudaGetDeviceCount(&gpu_count));
    CUDA_CHECK_EXIT(cudaSetDevice(node_rank % gpu_count));

    if (init_state) {
        InitState();
    }

    cuhearState->node_comm = node_comm;
    cuhearState->node_rank = node_rank;
    MPI_Comm_size(node_comm, &cuhearState->node_size);

    // node_rank is used as the split colour: all Rank 0 processes from each node
    // form one communicator, all Rank 1 form another, and so on.
    PMPI_Comm_split(comm, node_rank, my_rank, &inter_node_comm);
    cuhearState->inter_node_comm = inter_node_comm;

    int inter_rank, inter_size;
    MPI_Comm_rank(inter_node_comm, &inter_rank);
    MPI_Comm_size(inter_node_comm, &inter_size);

    uint32_t communicatorKey = (my_rank == root_rank) ? keyGenerator() : 0;
    PMPI_Bcast(&communicatorKey, 1, MPI_UNSIGNED, root_rank, comm);

    uint32_t my_key = keyGenerator();
    std::vector<uint32_t> peer_keys(inter_size);

    PMPI_Allgather(&my_key, 1, MPI_UNSIGNED, 
                   peer_keys.data(), 1, MPI_UNSIGNED, 
                   inter_node_comm);

    cuhear::KeyStorage keyStorage {
        .communicatorKey = communicatorKey,
        .ownKey = my_key,
        .nextKey = (inter_rank < inter_size - 1) ? peer_keys[inter_rank + 1] : 0,
        .rootKey = peer_keys[0] 
    };

    cuhear::KeyStoragePtr keyPtr(keyStorage);
    cuhearState->commKeys.emplace(comm, std::move(keyPtr));

    // pipeline streams initialization
    CUDA_CHECK_EXIT(cudaStreamCreate(&cuhearState->s_compute));
    CUDA_CHECK_EXIT(cudaStreamCreate(&cuhearState->s_p2p));

    return MPI_SUCCESS;
}

extern "C" void cuhear_cleanup_intra_node() {
    if (cuhearState == nullptr) return;

    for (size_t i = 0; i < cuhearState->h_peer_inbox_ptrs.size(); i++) {
        if (i != (size_t)cuhearState->node_rank && cuhearState->h_peer_inbox_ptrs[i] != nullptr) {
            cudaIpcCloseMemHandle(cuhearState->h_peer_inbox_ptrs[i]);
        }
    }
    cuhearState->h_peer_inbox_ptrs.clear();

    if (cuhearState->d_peer_inbox_ptrs) {
        cudaFree(cuhearState->d_peer_inbox_ptrs);
        cuhearState->d_peer_inbox_ptrs = nullptr;
    }

    if (cuhearState->d_inbox_buffer) {
        cudaFree(cuhearState->d_inbox_buffer);
        cuhearState->d_inbox_buffer = nullptr;
    }

    cuhearState->chunk_items = 0;
}

extern "C" void cuhear_setup_intra_node(size_t num_items, MPI_Datatype mpi_type) {
    int node_rank = cuhearState->node_rank;
    int node_size = cuhearState->node_size;
    MPI_Comm node_comm = cuhearState->node_comm;

    int type_size;
    MPI_Type_size(mpi_type, &type_size);
    cuhearState->type_size = type_size;

    cuhearState->chunk_items = num_items / node_size;
    size_t chunk_bytes = cuhearState->chunk_items * type_size;
    
    size_t inbox_size = (node_size > 1) ? (node_size - 1) * chunk_bytes : 0;

    int my_gpu_id;
    cudaGetDevice(&my_gpu_id);

    std::vector<int> all_gpu_ids(node_size);
    MPI_Allgather(&my_gpu_id, 1, MPI_INT, all_gpu_ids.data(), 1, MPI_INT, node_comm);

    cudaIpcMemHandle_t my_handle;
    if (node_size > 1) {
        CUDA_CHECK_EXIT(cudaMalloc(&cuhearState->d_inbox_buffer, inbox_size));
        cudaMemset(cuhearState->d_inbox_buffer, 0, inbox_size);

        CUDA_CHECK_EXIT(cudaIpcGetMemHandle(&my_handle, cuhearState->d_inbox_buffer));
    }

    std::vector<cudaIpcMemHandle_t> all_handles(node_size);
    MPI_Allgather(&my_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
                  all_handles.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE, node_comm);

    cuhearState->h_peer_inbox_ptrs.assign(node_size, nullptr);

    if (node_size > 1) {
        for (int i = 0; i < node_size; i++) {
            if (i == node_rank) {
                cuhearState->h_peer_inbox_ptrs[i] = cuhearState->d_inbox_buffer;
                continue;
            }

            int target_gpu_id = all_gpu_ids[i];

            int can_access;
            cudaDeviceCanAccessPeer(&can_access, my_gpu_id, target_gpu_id);
            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(target_gpu_id, 0);
            }

            CUDA_CHECK_EXIT(cudaIpcOpenMemHandle((void**)&cuhearState->h_peer_inbox_ptrs[i], 
                                                 all_handles[i], 
                                                 cudaIpcMemLazyEnablePeerAccess));
        }

        CUDA_CHECK_EXIT(cudaMalloc(&cuhearState->d_peer_inbox_ptrs, node_size * sizeof(uint32_t*)));
        CUDA_CHECK_EXIT(cudaMemcpy(cuhearState->d_peer_inbox_ptrs, 
                                   cuhearState->h_peer_inbox_ptrs.data(), 
                                   node_size * sizeof(uint32_t*), 
                                   cudaMemcpyHostToDevice));
    }

    MPI_Barrier(node_comm);
}

typedef void (*ENC_K)(cuhear::KeyStorage*, cuhear::rng::AesContext*, void*, void*, size_t, bool);
typedef void (*DEC_K)(cuhear::KeyStorage*, cuhear::rng::AesContext*, void*, size_t);

template <ENC_K encrypt_fn, DEC_K decrypt_fn>
static inline int AllReduceImpl(const void *sendbuf, void *recvbuf, int count, MPI_Datatype aggregateType, MPI_Op aggregateOp, MPI_Comm comm) {
    int node_rank = cuhearState->node_rank;
    int node_size = cuhearState->node_size;
    int type_size = cuhearState->type_size;
    
    // each slice is 1/N of the total vector
    size_t slice_items = cuhearState->chunk_items; 
    size_t slice_bytes = slice_items * type_size;

    int num_chunks = 1; 
    if (slice_bytes > cuhearState->segment_size && cuhearState->segment_size > 0) {
        // Compute how many whole chunks of 'segment_size' fit in the slice
        num_chunks = slice_bytes / cuhearState->segment_size;
        
        // Ensure at least 1 chunk if rounding yielded zero
        if (num_chunks == 0) num_chunks = 1;
    }
    
    // Elements per pipeline chunk
    size_t pipeline_chunk_items = slice_items / num_chunks;
    // Edge case: if pipeline_chunk_items is 0, force to 1 to keep geometry valid
    if (pipeline_chunk_items == 0) {
        pipeline_chunk_items = 1;
        num_chunks = slice_items;
    }

    if (sendbuf == MPI_IN_PLACE) {
        sendbuf = recvbuf;
    }

    uint32_t *d_send = (uint32_t *) sendbuf;
    uint32_t *d_recv = (uint32_t *) recvbuf;

    uint32_t *d_encChunk;
    CUDA_CHECK_EXIT(cudaMalloc(&d_encChunk, slice_bytes));

    int inter_rank, inter_size;
    PMPI_Comm_rank(cuhearState->inter_node_comm, &inter_rank);
    PMPI_Comm_size(cuhearState->inter_node_comm, &inter_size);
    bool isLast = (inter_rank == inter_size - 1);

    std::vector<MPI_Request> reqs(num_chunks);

    cuhear::kernels::crypto::rotate_key<<<1, 1, 0, cuhearState->s_compute>>>(
        cuhearState->d_aesContext, 
        cuhearState->commKeys.at(comm).d_keys
    );

    // distributed pipeline loop
    for (int i = 0; i < num_chunks; i++) {
        // Geometric offset calculation within the current slice
        size_t chunk_offset = i * pipeline_chunk_items;
        size_t current_chunk_items = (i == num_chunks - 1) ? (slice_items - chunk_offset) : pipeline_chunk_items;
        size_t current_chunk_bytes = current_chunk_items * type_size;

        // intra-node chunked reduce-scatter
        if (node_size > 1) {
            for (int target_peer = 0; target_peer < node_size; target_peer++) {
                if (target_peer == node_rank) continue;

                uint32_t* my_chunk_for_peer = d_send + (target_peer * slice_items) + chunk_offset;
                
                int slot = (node_rank < target_peer) ? node_rank : node_rank - 1;
                uint32_t* target_inbox_slot = cuhearState->h_peer_inbox_ptrs[target_peer] + (slot * slice_items) + chunk_offset;

                cudaMemcpyAsync(target_inbox_slot, my_chunk_for_peer, current_chunk_bytes, cudaMemcpyDeviceToDevice, cuhearState->s_p2p);
            }

            cudaStreamSynchronize(cuhearState->s_p2p);

            PMPI_Barrier(cuhearState->node_comm);

            int threads = 256;
            int blocks = (current_chunk_items + threads - 1) / threads;
            cuhear::IntraNodeChunkSum<<<blocks, threads, 0, cuhearState->s_compute>>>(
                current_chunk_items, 
                node_size, 
                slice_items, 
                chunk_offset,
                d_send + (node_rank * slice_items) + chunk_offset, 
                cuhearState->d_inbox_buffer
            );
        }

        // encryption of the current chunk
        int aes_blocks = (current_chunk_bytes + 15) / 16;
        encrypt_fn<<<(aes_blocks + 255) / 256, 256, 0, cuhearState->s_compute>>>(
            cuhearState->commKeys.at(comm).d_keys,
            cuhearState->d_aesContext,
            d_encChunk + chunk_offset,
            d_send + (node_rank * slice_items) + chunk_offset,
            aes_blocks,
            isLast
        );

        cudaStreamSynchronize(cuhearState->s_compute);

        // asynchronous inter-node allreduce
        if (inter_size == 1) {
            cudaMemcpyAsync(
                d_recv + (node_rank * slice_items) + chunk_offset,
                d_encChunk + chunk_offset,
                current_chunk_bytes,
                cudaMemcpyDeviceToDevice,
                cuhearState->s_compute
            );
            cudaStreamSynchronize(cuhearState->s_compute);
            reqs[i] = MPI_REQUEST_NULL;
        } else {
            PMPI_Iallreduce(
                d_encChunk + chunk_offset, 
                d_recv + (node_rank * slice_items) + chunk_offset, 
                current_chunk_items, 
                aggregateType, 
                aggregateOp, 
                cuhearState->inter_node_comm, 
                &reqs[i]
            );
        }
    }

    // post-pipelined
    PMPI_Waitall(num_chunks, reqs.data(), MPI_STATUSES_IGNORE);

    // Final decryption of the entire accumulated slice
    int total_aes_blocks = (slice_bytes + 15) / 16;
    decrypt_fn<<<(total_aes_blocks + 255) / 256, 256>>>(
        cuhearState->commKeys.at(comm).d_keys,
        cuhearState->d_aesContext,
        d_recv + (node_rank * slice_items),
        total_aes_blocks
    );
    cudaDeviceSynchronize();
    cudaFree(d_encChunk);

    if (node_size > 1) {
        // Reconstruct the global vector on each GPU's d_recv via intra-node Allgather
        PMPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                       d_recv, slice_items, aggregateType, cuhearState->node_comm);
    }

    return MPI_SUCCESS;
}

int MPI_Init(int *argc, char ***argv) {
    int orig = PMPI_Init(argc, argv);
    if (orig == MPI_SUCCESS) {
        orig = NewComm(MPI_COMM_WORLD, true);
        
        if (argc != nullptr && argv != nullptr && *argv != nullptr) {
            int nargs = *argc;
            char** args = *argv;
            for (int i = 1; i < nargs - 1; i++) {
                if (std::string(args[i]) == "--segment_size") {
                    cuhearState->segment_size = std::stoull(args[i + 1]);
                    break;
                }
            }
            if (cuhearState->myRank == 0) {
                std::cout << "[CUHEAR INIT] segment_size set to: " 
                          << cuhearState->segment_size << " bytes (" 
                          << cuhearState->segment_size / 1024.0 / 1024.0 << " MiB)" << std::endl;
            }
        }
    }
    return orig;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int orig = PMPI_Init_thread(argc, argv, required, provided);
    if (orig == MPI_SUCCESS) {
        orig = NewComm(MPI_COMM_WORLD, true);
        
        if (argc != nullptr && argv != nullptr && *argv != nullptr) {
            int nargs = *argc;
            char** args = *argv;
            for (int i = 1; i < nargs - 1; i++) {
                if (std::string(args[i]) == "--segment_size") {
                    cuhearState->segment_size = std::stoull(args[i + 1]);
                    break;
                }
            }
        }
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
