#pragma once

#include <cstdint>
#include <unordered_map>
#include <mpi.h>

#include "aes.hpp"

// MPI interface
extern "C" {
    int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    int MPI_Init(int *argc, char ***argv);
    int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
    int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
    int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
    int MPI_Finalize();
}

namespace cuhear {

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
    };

}
