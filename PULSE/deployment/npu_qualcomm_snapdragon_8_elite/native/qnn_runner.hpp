#pragma once

#include <cstddef>
#include <memory>
#include <string>

struct AHardwareBuffer;

namespace pulse_mobile {

class QnnRunner {
public:
    QnnRunner(
        const std::string& native_library_dir,
        const std::string& context_binary_path,
        bool prefer_hardware_output = true);
    ~QnnRunner();

    QnnRunner(const QnnRunner&) = delete;
    QnnRunner& operator=(const QnnRunner&) = delete;

    void* input_data(const std::string& name);
    size_t input_bytes(const std::string& name) const;
    const void* output_data(const std::string& name) const;
    size_t output_bytes(const std::string& name) const;
    const void* output_data() const;
    size_t output_bytes() const;
    size_t output_allocation_bytes() const;
    int output_fd() const;
    AHardwareBuffer* output_hardware_buffer() const;
    const std::string& graph_name() const;

    double execute();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pulse_mobile
