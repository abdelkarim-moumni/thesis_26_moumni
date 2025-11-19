#include "cuhear.hpp"
#include "utils.hpp"

namespace cuhear {

    KeyStoragePtr::KeyStoragePtr(KeyStorage& keys) {
        CUDA_CHECK_EXIT(cudaMalloc(&d_keys, sizeof(KeyStorage)));
        CUDA_CHECK_EXIT(cudaMemcpy(d_keys, &keys, sizeof(KeyStorage), cudaMemcpyHostToDevice));
    }

    KeyStoragePtr::KeyStoragePtr(KeyStoragePtr&& p) {
        d_keys = p.d_keys;
        p.d_keys = nullptr;
    }

    KeyStoragePtr::~KeyStoragePtr() {
        if (d_keys != nullptr) {
            CUDA_CHECK_EXIT(cudaFree(d_keys));
        }
    }

}