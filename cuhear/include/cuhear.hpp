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
    __global__ void IntraNodeChunkSum(int pipeline_chunk_items, int node_size, size_t slice_items, size_t chunk_offset, uint32_t* my_chunk, uint32_t* inbox);

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
        rng::AesContext *d_aesContext;
        std::unordered_map<MPI_Comm, KeyStoragePtr> commKeys;
        bool cudaUnifiedAddressing;
        bool mpiCudaAware;

        int node_rank; 
        int node_size;
        MPI_Comm node_comm; 

        // Vertical communicator for the inter-node phase
        MPI_Comm inter_node_comm;

        uint32_t* d_inbox_buffer = nullptr;
        std::vector<uint32_t*> h_peer_inbox_ptrs;
        uint32_t** d_peer_inbox_ptrs = nullptr;
        std::vector<cudaIpcMemHandle_t> peer_ipc_handles;

        size_t chunk_items;
        int type_size;
        std::vector<int> node_gpu_ids;

        size_t segment_size = 1024 * 1024; // Default: 1 MiB in bytes (overridden from command line)
        cudaStream_t s_compute;   // Stream for local reduction (kernel) and encryption
        cudaStream_t s_p2p;       // Stream for peer to peer inboxes
    };

}
