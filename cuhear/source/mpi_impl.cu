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
    MPI_Comm node_comm;
    int node_rank, gpu_count;

    // Split communicator by node and assing GPUs
    PMPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    CUDA_CHECK_EXIT(cudaGetDeviceCount(&gpu_count));
    CUDA_CHECK_EXIT(cudaSetDevice(node_rank % gpu_count));

    if (init_state) {
        InitState();
    }

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &my_rank);

    cuhearState->node_comm = node_comm;
    cuhearState->node_rank = node_rank;

    int current_device;
    cudaGetDevice(&current_device);

    int leader_device = current_device;

    if (node_rank == 0) {
        cuhearState->leader_device = current_device;
    }

    // Intra-node broadcast of the leader device id
    MPI_Bcast(&cuhearState->leader_device, 1, MPI_INT, 0, node_comm);

    // Leader communicator creation
    MPI_Comm leader_comm;
    int color = (node_rank == 0) ? 1 : MPI_UNDEFINED;
    PMPI_Comm_split(comm, color, my_rank, &leader_comm);
    cuhearState->leader_comm = leader_comm;

    // Global broadcast of the communicatorKey for synchronization
    uint32_t communicatorKey = (my_rank == root_rank) ? keyGenerator() : 0;
    int ret = PMPI_Bcast(&communicatorKey, 1, MPI_UNSIGNED, root_rank, comm);
    if (ret != MPI_SUCCESS) return ret;

    if (node_rank == 0) {
        int leader_rank, leader_size;
        MPI_Comm_rank(cuhearState->leader_comm, &leader_rank);
        MPI_Comm_size(cuhearState->leader_comm, &leader_size);

        // Key exchange between node leaders
        std::vector<uint32_t> leader_keys(leader_size);
        uint32_t my_key = keyGenerator(); 
        
        PMPI_Allgather(&my_key, 1, MPI_UNSIGNED, leader_keys.data(), 1, MPI_UNSIGNED, cuhearState->leader_comm);

        cuhear::KeyStorage leaderKeyStorage {
            .communicatorKey = communicatorKey,
            .ownKey = my_key,
            .nextKey = (leader_rank < leader_size - 1) ? leader_keys[leader_rank + 1] : 0,
            .rootKey = leader_keys[0] // The decryption root is the first leader (leader_rank 0)
        };

        cuhear::KeyStoragePtr keyPtr(leaderKeyStorage);
        cuhearState->commKeys.emplace(comm, std::move(keyPtr));
    }

    return MPI_SUCCESS;
}

extern "C" void cuhear_cleanup_intra_node() {
    // Leader-specific cleanup (Rank 0 of the node)
    if (cuhearState->node_rank == 0) {
        // Free all remote follower buffers stored in the vector
        for (uint32_t* ptr : cuhearState->h_follower_bufs) {
            if (ptr) cudaFree(ptr);
        }
        cuhearState->h_follower_bufs.clear();
        
        if (cuhearState->d_follower_bufs_ptrs) {
            cudaFree(cuhearState->d_follower_bufs_ptrs);
            cuhearState->d_follower_bufs_ptrs = nullptr;
        }
    }
    // Follower-specific cleanup 
    else {
        // Close the CUDA IPC memory handle to the leader's buffer
        if (cuhearState->d_ptr_to_leader) {
            cudaIpcCloseMemHandle(cuhearState->d_ptr_to_leader);
            cuhearState->d_ptr_to_leader = nullptr;
        }
    }
}

extern "C" void cuhear_setup_intra_node(size_t num_items, MPI_Datatype mpi_type) {
    int node_rank = cuhearState->node_rank;
    MPI_Comm node_comm = cuhearState->node_comm;
    
    int node_size;
    MPI_Comm_size(node_comm, &node_size);
    cuhearState->node_size = node_size;
    
    int num_followers = node_size - 1;

    int type_size;
    MPI_Type_size(mpi_type, &type_size);
    size_t bufSize = num_items * type_size;

    // Allocate memory handles for all potential followers within the node
    std::vector<cudaIpcMemHandle_t> handles(num_followers > 0 ? num_followers : 1);
    int current_device;
    cudaGetDevice(&current_device);

    // LEADER (Rank 0): Allocate buffers and generate IPC handles
    if (node_rank == 0) {
        cuhearState->leader_device = current_device;

        cuhear_cleanup_intra_node(); 

        if (num_followers > 0) {
            cuhearState->h_follower_bufs.resize(num_followers);
            for (int i = 0; i < num_followers; i++) {
                // Allocate and initialize buffers for each follower
                CUDA_CHECK_EXIT(cudaMalloc(&cuhearState->h_follower_bufs[i], bufSize));
                cudaMemset(cuhearState->h_follower_bufs[i], 0, bufSize);
                // Generate the IPC handle for the allocated buffer
                cudaIpcGetMemHandle(&handles[i], cuhearState->h_follower_bufs[i]);
            }

            // Transfer follower buffer pointers to the device for GPU-side access
            CUDA_CHECK_EXIT(cudaMalloc(&cuhearState->d_follower_bufs_ptrs, num_followers * sizeof(uint32_t*)));
            CUDA_CHECK_EXIT(cudaMemcpy(cuhearState->d_follower_bufs_ptrs, 
                                       cuhearState->h_follower_bufs.data(), 
                                       num_followers * sizeof(uint32_t*), 
                                       cudaMemcpyHostToDevice));
        }
        cudaDeviceSynchronize();
    }

    // Exchange leader device info and IPC handles across the node
    MPI_Bcast(&cuhearState->leader_device, 1, MPI_INT, 0, node_comm);
    MPI_Barrier(node_comm);
    if (num_followers > 0) {
        MPI_Bcast(handles.data(), num_followers * sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, node_comm);
    }
    MPI_Barrier(node_comm);

    // FOLLOWER (Rank > 0): Enable Peer-to-Peer access and open IPC handles
    if (node_rank > 0) {
        int target_device = cuhearState->leader_device;

        // Enable peer-to-peer access to the leader's device
        if (current_device != target_device) {
            int can_access;
            cudaDeviceCanAccessPeer(&can_access, current_device, target_device);
            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(target_device, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    printf("[RANK %d] Errore P2P Enable: %s\n", node_rank, cudaGetErrorString(err));
                }
            } else {
                printf("[RANK %d] CRITICO: Hardware non supporta P2P tra %d e %d\n", 
                        node_rank, current_device, target_device);
            }
        }

        // Close any existing handle before opening a new one
        if (cuhearState->d_ptr_to_leader) 
            cudaIpcCloseMemHandle(cuhearState->d_ptr_to_leader);
        
        // Map the specific IPC handle assigned to this follower (index = node_rank - 1)
        cudaIpcOpenMemHandle((void**)&cuhearState->d_ptr_to_leader, 
                             handles[node_rank - 1], 
                             cudaIpcMemLazyEnablePeerAccess);
    }
    MPI_Barrier(node_comm);
}

typedef void (*ENC_K)(cuhear::KeyStorage*, cuhear::rng::AesContext*, void*, void*, size_t, bool);
typedef void (*DEC_K)(cuhear::KeyStorage*, cuhear::rng::AesContext*, void*, size_t);

template <ENC_K encrypt_fn, DEC_K decrypt_fn>
static inline int AllReduceImpl(const void *sendbuf, void *recvbuf, int count, MPI_Datatype aggregateType, MPI_Op aggregateOp, MPI_Comm comm) {
    int gpu_count;
    CUDA_CHECK_EXIT(cudaGetDeviceCount(&gpu_count));
    int my_gpu = cuhearState->node_rank % gpu_count; 
    cudaSetDevice(my_gpu);
    
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
    size_t blockSize = 128;

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

    // 1. Intra-node section: followers send their data to the node leader
    int node_rank = cuhearState->node_rank;
    MPI_Comm node_comm = cuhearState->node_comm;
    // 1.2 Each follower copies its local data to the Leader's allocated buffer via IPC
    if (node_rank > 0) {
        // Execute asynchronous IPC device-to-device copy
        cudaMemcpyAsync(cuhearState->d_ptr_to_leader, d_send, bufSize, cudaMemcpyDeviceToDevice, 0);
        cudaDeviceSynchronize();
    }
    // 1.3 Synchronization barrier: ensure all followers have completed their data transfers
    // cudaDeviceSynchronize();
    MPI_Barrier(node_comm);
    // cudaDeviceSynchronize();
    // 1.4 Local sum: only the Node Leader (rank 0) performs the reduction
    if (node_rank == 0) {
        // DYNAMIC PRE-SUM DEBUG
        // if (node_rank == 0) {
        //     uint32_t v_send[3];
        //     cudaMemcpy(v_send, d_send, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //     std::cout << "DEBUG PRE-SOMMA:\n  d_send: " << v_send[0] << " " << v_send[1] << " " << v_send[2] << std::endl;

        //     for (int i = 0; i < cuhearState->h_follower_bufs.size(); ++i) {
        //         uint32_t v_f[3];
        //         cudaMemcpy(v_f, cuhearState->h_follower_bufs[i], 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //         std::cout << "  f" << i+1 << ":     " << v_f[0] << " " << v_f[1] << " " << v_f[2] << std::endl;
        //     }
        // }

        int threads_per_block = 256;
        int blocks = (count + threads_per_block - 1) / threads_per_block;
        int num_followers = cuhearState->node_size - 1;
        
        if (num_followers > 0) {
            // Launch the kernel to sum up data from all followers on the same node
            cuhear::IntraNodeSum<<<blocks, threads_per_block>>>(
                count, 
                num_followers,
                d_send, 
                cuhearState->d_follower_bufs_ptrs
            );
        }
        cudaDeviceSynchronize();
        // At this point, d_send on the Leader contains the sum for the entire node

        // DEBUG POST-SUM
        // uint32_t debug_res[3];
        // cudaMemcpy(debug_res, d_send, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // std::cout << "RISULTATO SOMMA d_send: " << debug_res[0] << " " << debug_res[1] << " " << debug_res[2] << std::endl;
    }

    // 2. Inter-node section: communication between node leaders
    if (node_rank == 0) {
        int l_rank, l_size;
        MPI_Comm_rank(cuhearState->leader_comm, &l_rank);
        MPI_Comm_size(cuhearState->leader_comm, &l_size);

        bool l_isLast = (l_rank == l_size - 1);
        
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
            l_isLast
        );
        CUDA_LAST_EXIT();
        CUDA_CHECK_EXIT(cudaDeviceSynchronize());
        
        if (cuhearState->mpiCudaAware) {
            // CUDA-Aware MPI, can use GPU buffers directly
            ret = PMPI_Allreduce(d_encSend, d_recv, count, aggregateType, aggregateOp, cuhearState->leader_comm);
        } else {
            // No GPU support, need to copy buffer to host
            uint32_t *encSend = new uint32_t[(bufSize + sizeof(uint32_t) - 1) / sizeof(uint32_t)];
            CUDA_CHECK_EXIT(cudaMemcpy(encSend, d_encSend, bufSize, cudaMemcpyDeviceToHost));
            ret = PMPI_Allreduce(MPI_IN_PLACE, encSend, count, aggregateType, aggregateOp, cuhearState->leader_comm);
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

        // uint32_t debug_res[3];
        // cudaMemcpy(debug_res, d_recv, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // std::cout << "\033[1;32mRISULTATO recv: " << debug_res[0] << " " << debug_res[1] << " " << debug_res[2] << "\033[0m" << std::endl;

        CUDA_LAST_EXIT();        
    }

    // 3. Final intra-node distribution: leader broadcasts the decrypted buffer to all node processes decrypted buffer to the node
    MPI_Bcast(d_recv, count, aggregateType, 0, cuhearState->node_comm);
    

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
