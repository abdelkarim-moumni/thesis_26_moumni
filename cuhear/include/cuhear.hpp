#pragma once

#include <cstdint>
#include <unordered_map>
#include <mpi.h>
#include <vector>

#include "aes.hpp"

// MPI interface
extern "C" {
    void cuhear_setup_intra_node(size_t num_items, MPI_Datatype mpi_type);
    void cuhear_cleanup_intra_node();

    int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    int MPI_Init(int *argc, char ***argv);
    int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
    int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
    int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
    int MPI_Finalize();
}

namespace cuhear {
    __global__ void IntraNodeSum(int size, int num_followers, uint32_t* leader_buf, uint32_t** follower_bufs);

    typedef uint32_t Key;

    struct KeyStorage {
        // K_c (aka K_n)
        Key communicatorKey;
        // K_s[rank]
        Key ownKey;
        // K_s[rank + 1]
        Key nextKey;
        // K_s[0]
        Key rootKey;
    };

    // Smart pointer for device-allocated keys
    struct KeyStoragePtr {
        KeyStorage *d_keys;
        
        KeyStoragePtr(KeyStorage& keys);
        KeyStoragePtr(KeyStoragePtr&& p);
        KeyStoragePtr(KeyStoragePtr& p) = delete;
        ~KeyStoragePtr();
    };

    struct CuHearState {
        uint32_t myRank;
        // AES context, stored on device
        rng::AesContext *d_aesContext;
        // Map (stored on the host) for (MPI_Comm, KeyStorage*).
        // Communicator keys are stored on device
        std::unordered_map<MPI_Comm, KeyStoragePtr> commKeys;
        bool cudaUnifiedAddressing;
        bool mpiCudaAware;

        int node_rank; // local rank at the node  
        int node_size;
        MPI_Comm node_comm; // node communicator
        MPI_Comm leader_comm; // leader communicator

        // Leader's pointers
        uint32_t** d_follower_bufs_ptrs = nullptr;
        std::vector<uint32_t*> h_follower_bufs;

        // Follower's pointer (P2P access)
        uint32_t *d_ptr_to_leader = nullptr;

        int leader_device;
    };

}
