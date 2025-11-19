# cuHEAR: GPU-Accelerated Encryption for Allreduce Collectives

cuHEAR is an implementation of [HEAR](https://github.com/spcl/libhear) for GPUs that support the NVIDIA CUDA architecture.

## Building & usage

cuHEAR uses CMake:

```
cmake -Bbuild -D CMAKE_BUILD_TYPE=RelWithDebInfo
cd build
make
```

The shared library can then be found in `build/cuhear`. To instrument a program to use it, you can:
```
LD_PRELOAD=./libcuhear.so ./my_program
```

## License

cuHEAR is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.