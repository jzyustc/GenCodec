// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

#ifndef NO_CUDA_PINNED_MEMORY
    #include <cuda_runtime.h>
    #include <string>
#endif

template <typename T>
class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(const size_t size = 0)
    {
        if (size == 0) {
            return;
        }

#ifndef NO_CUDA_PINNED_MEMORY
        cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&m_data), size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMallocHost failed for pinned host buffer: "
                                     + std::string(cudaGetErrorString(err)));
        }
#else
        m_data = static_cast<T*>(std::malloc(size * sizeof(T)));
        if (m_data == nullptr) {
            throw std::runtime_error("malloc failed for host buffer");
        }
#endif
        m_size = size;
    }

    ~PinnedHostBuffer()
    {
        if (m_data != nullptr) {
#ifndef NO_CUDA_PINNED_MEMORY
            cudaFreeHost(m_data);
#else
            std::free(m_data);
#endif
            m_data = nullptr;
        }
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(PinnedHostBuffer&&) = delete;

    size_t size() const { return m_size; }

    T* data() { return m_data; }

    const T* data() const { return m_data; }

private:
    T* m_data{ nullptr };
    size_t m_size{ 0 };
};
