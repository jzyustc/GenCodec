#include "qnn_runner.hpp"

#include <QnnBackend.h>
#include <QnnContext.h>
#include <QnnDevice.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnMem.h>
#include <QnnTensor.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>

#include <android/hardware_buffer.h>
#include <android/log.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "QnnTypeMacros.hpp"

namespace pulse_mobile {
namespace {

constexpr const char* kLogTag = "PulseQnn";
constexpr int kRpcMemHeapSystem = 25;
constexpr uint32_t kRpcMemDefaultFlags = 1;

using QnnGetProviders = Qnn_ErrorHandle_t (*)(
    const QnnInterface_t***,
    uint32_t*);
using QnnSystemGetProviders = Qnn_ErrorHandle_t (*)(
    const QnnSystemInterface_t***,
    uint32_t*);
using RpcMemAlloc = void* (*)(int, uint32_t, int);
using RpcMemFree = void (*)(void*);
using RpcMemToFd = int (*)(void*);
using AHardwareBufferGetNativeHandle = const void* (*)(
    const AHardwareBuffer*);

struct NativeHandleHeader {
    int version;
    int num_fds;
    int num_ints;
};

void log_info(const std::string& message)
{
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "%s", message.c_str());
}

std::runtime_error failure(const std::string& message)
{
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s", message.c_str());
    return std::runtime_error(message);
}

void* open_library(const std::string& path, const int flags)
{
    void* handle = dlopen(path.c_str(), flags);
    if (handle == nullptr) {
        throw failure("dlopen failed for " + path + ": " + dlerror());
    }
    return handle;
}

template <typename T>
T load_symbol(void* handle, const char* name)
{
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    if (symbol == nullptr || error != nullptr) {
        throw failure(
            std::string("dlsym failed for ") + name + ": "
            + (error ? error : "null"));
    }
    return reinterpret_cast<T>(symbol);
}

std::vector<uint8_t> read_file(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw failure("cannot open " + path);
    }
    return {
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()};
}

size_t data_type_bytes(const Qnn_DataType_t type)
{
    switch (type) {
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
        return 1;
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
        return 2;
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
        return 4;
    case QNN_DATATYPE_FLOAT_64:
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
        return 8;
    default:
        throw failure("unsupported QNN tensor data type");
    }
}

size_t tensor_bytes(const Qnn_Tensor_t& tensor)
{
    size_t elements = 1;
    const uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    const uint32_t* dimensions = QNN_TENSOR_GET_DIMENSIONS(tensor);
    for (uint32_t index = 0; index < rank; ++index) {
        elements *= dimensions[index];
    }
    return elements * data_type_bytes(QNN_TENSOR_GET_DATA_TYPE(tensor));
}

struct GraphMetadata {
    const char* name{};
    uint32_t input_count{};
    Qnn_Tensor_t* inputs{};
    uint32_t output_count{};
    Qnn_Tensor_t* outputs{};
};

GraphMetadata first_graph(const QnnSystemContext_BinaryInfo_t* binary)
{
    if (binary == nullptr) {
        throw failure("QNN context metadata is null");
    }
    uint32_t count = 0;
    QnnSystemContext_GraphInfo_t* graphs = nullptr;
    switch (binary->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
        count = binary->contextBinaryInfoV1.numGraphs;
        graphs = binary->contextBinaryInfoV1.graphs;
        break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
        count = binary->contextBinaryInfoV2.numGraphs;
        graphs = binary->contextBinaryInfoV2.graphs;
        break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
        count = binary->contextBinaryInfoV3.numGraphs;
        graphs = binary->contextBinaryInfoV3.graphs;
        break;
    default:
        throw failure("unsupported QNN binary metadata version");
    }
    if (count != 1 || graphs == nullptr) {
        throw failure("expected exactly one graph in the context");
    }
    const QnnSystemContext_GraphInfo_t& graph = graphs[0];
    switch (graph.version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
        return {
            graph.graphInfoV1.graphName,
            graph.graphInfoV1.numGraphInputs,
            graph.graphInfoV1.graphInputs,
            graph.graphInfoV1.numGraphOutputs,
            graph.graphInfoV1.graphOutputs};
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
        return {
            graph.graphInfoV2.graphName,
            graph.graphInfoV2.numGraphInputs,
            graph.graphInfoV2.graphInputs,
            graph.graphInfoV2.numGraphOutputs,
            graph.graphInfoV2.graphOutputs};
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
        return {
            graph.graphInfoV3.graphName,
            graph.graphInfoV3.numGraphInputs,
            graph.graphInfoV3.graphInputs,
            graph.graphInfoV3.numGraphOutputs,
            graph.graphInfoV3.graphOutputs};
    default:
        throw failure("unsupported QNN graph metadata version");
    }
}

}  // namespace

class QnnRunner::Impl {
public:
    Impl(
        const std::string& native_library_dir,
        const std::string& context_binary_path,
        const bool prefer_hardware_output)
        : context_binary_(read_file(context_binary_path)),
          prefer_hardware_output_(prefer_hardware_output)
    {
        backend_library_ = open_library(
            native_library_dir + "/libQnnHtp.so",
            RTLD_NOW | RTLD_GLOBAL);
        system_library_ = open_library(
            native_library_dir + "/libQnnSystem.so",
            RTLD_NOW | RTLD_LOCAL);
        rpc_library_ = open_library("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
        rpc_alloc_ = load_symbol<RpcMemAlloc>(rpc_library_, "rpcmem_alloc");
        rpc_free_ = load_symbol<RpcMemFree>(rpc_library_, "rpcmem_free");
        rpc_to_fd_ = load_symbol<RpcMemToFd>(rpc_library_, "rpcmem_to_fd");

        load_qnn_interfaces();
        check(qnn_.backendCreate(nullptr, nullptr, &backend_), "backendCreate");
        if (qnn_.deviceCreate != nullptr) {
            const Qnn_ErrorHandle_t status =
                qnn_.deviceCreate(nullptr, nullptr, &device_);
            if (status != QNN_SUCCESS) {
                device_ = nullptr;
                log_info("QNN deviceCreate unavailable; using default device");
            }
        }
        configure_burst_mode();
        parse_metadata();
        check(
            qnn_.contextCreateFromBinary(
                backend_,
                device_,
                nullptr,
                context_binary_.data(),
                context_binary_.size(),
                &context_,
                nullptr),
            "contextCreateFromBinary");
        check(
            qnn_.graphRetrieve(context_, graph_name_.c_str(), &graph_),
            "graphRetrieve");
        allocate_tensors();
        log_info(
            "loaded graph " + graph_name_ + " with "
            + std::to_string(inputs_.size()) + " inputs");
    }

    ~Impl()
    {
        release_tensors();
        if (context_ != nullptr && qnn_.contextFree != nullptr) {
            qnn_.contextFree(context_, nullptr);
        }
        if (system_context_ != nullptr
            && qnn_system_.systemContextFree != nullptr) {
            qnn_system_.systemContextFree(system_context_);
        }
        if (power_config_active_
            && perf_infrastructure_.destroyPowerConfigId != nullptr) {
            const Qnn_ErrorHandle_t status =
                perf_infrastructure_.destroyPowerConfigId(power_config_id_);
            if (status != QNN_SUCCESS) {
                log_info(
                    "destroyPowerConfigId returned "
                    + std::to_string(status));
            }
            power_config_active_ = false;
        }
        if (device_ != nullptr && qnn_.deviceFree != nullptr) {
            qnn_.deviceFree(device_);
        }
        if (backend_ != nullptr && qnn_.backendFree != nullptr) {
            qnn_.backendFree(backend_);
        }
        if (rpc_library_ != nullptr) {
            dlclose(rpc_library_);
        }
        if (system_library_ != nullptr) {
            dlclose(system_library_);
        }
        if (backend_library_ != nullptr) {
            dlclose(backend_library_);
        }
    }

    void* input_data(const std::string& name)
    {
        const auto found = input_by_name_.find(name);
        if (found == input_by_name_.end()) {
            throw failure("unknown QNN input " + name);
        }
        return inputs_.at(found->second).data;
    }

    size_t input_bytes(const std::string& name) const
    {
        const auto found = input_by_name_.find(name);
        if (found == input_by_name_.end()) {
            throw failure("unknown QNN input " + name);
        }
        return inputs_.at(found->second).bytes;
    }

    const void* output_data(const std::string& name) const
    {
        return output(name).data;
    }

    size_t output_bytes(const std::string& name) const
    {
        return output(name).bytes;
    }

    double execute()
    {
        const auto begin = std::chrono::steady_clock::now();
        check(
            qnn_.graphExecute(
                graph_,
                input_tensors_.data(),
                static_cast<uint32_t>(input_tensors_.size()),
                output_tensors_.data(),
                static_cast<uint32_t>(output_tensors_.size()),
                nullptr,
                nullptr),
            "graphExecute");
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(
            end - begin).count();
    }

    struct Buffer {
        void* data{};
        int fd{-1};
        size_t bytes{};
        size_t allocation_bytes{};
        Qnn_MemHandle_t handle{};
        AHardwareBuffer* hardware_buffer{};
        bool rpc_owned{};
        bool mmap_owned{};
        bool fd_owned{};
    };

    const Buffer& output() const
    {
        if (outputs_.size() != 1) {
            throw failure("expected one QNN output");
        }
        return outputs_[0];
    }

    const Buffer& output(const std::string& name) const
    {
        const auto found = output_by_name_.find(name);
        if (found == output_by_name_.end()) {
            throw failure("unknown QNN output " + name);
        }
        return outputs_.at(found->second);
    }

    const std::string& graph_name() const
    {
        return graph_name_;
    }

private:
    void check(const Qnn_ErrorHandle_t status, const char* operation)
    {
        if (status == QNN_SUCCESS) {
            return;
        }
        const char* text = nullptr;
        if (qnn_.errorGetMessage != nullptr) {
            qnn_.errorGetMessage(status, &text);
        }
        throw failure(
            std::string(operation) + " failed: "
            + std::to_string(status)
            + (text ? std::string(" ") + text : ""));
    }

    void load_qnn_interfaces()
    {
        const auto get_qnn = load_symbol<QnnGetProviders>(
            backend_library_,
            "QnnInterface_getProviders");
        const QnnInterface_t** providers = nullptr;
        uint32_t count = 0;
        check(get_qnn(&providers, &count), "QnnInterface_getProviders");
        bool found = false;
        for (uint32_t index = 0; index < count; ++index) {
            if (providers[index]->apiVersion.coreApiVersion.major
                    == QNN_API_VERSION_MAJOR
                && providers[index]->apiVersion.coreApiVersion.minor
                    >= QNN_API_VERSION_MINOR) {
                qnn_ = providers[index]->QNN_INTERFACE_VER_NAME;
                found = true;
                break;
            }
        }
        if (!found) {
            throw failure("no compatible QNN backend interface");
        }

        const auto get_system = load_symbol<QnnSystemGetProviders>(
            system_library_,
            "QnnSystemInterface_getProviders");
        const QnnSystemInterface_t** system_providers = nullptr;
        count = 0;
        check(
            get_system(&system_providers, &count),
            "QnnSystemInterface_getProviders");
        found = false;
        for (uint32_t index = 0; index < count; ++index) {
            if (system_providers[index]->systemApiVersion.major
                    == QNN_SYSTEM_API_VERSION_MAJOR
                && system_providers[index]->systemApiVersion.minor
                    >= QNN_SYSTEM_API_VERSION_MINOR) {
                qnn_system_ =
                    system_providers[index]->QNN_SYSTEM_INTERFACE_VER_NAME;
                found = true;
                break;
            }
        }
        if (!found) {
            throw failure("no compatible QNN system interface");
        }
    }

    void configure_burst_mode()
    {
        if (qnn_.deviceGetInfrastructure == nullptr) {
            log_info("HTP performance infrastructure is unavailable");
            return;
        }
        QnnDevice_Infrastructure_t infrastructure = nullptr;
        const Qnn_ErrorHandle_t infrastructure_status =
            qnn_.deviceGetInfrastructure(&infrastructure);
        if (infrastructure_status != QNN_SUCCESS
            || infrastructure == nullptr) {
            log_info(
                "deviceGetInfrastructure returned "
                + std::to_string(infrastructure_status));
            return;
        }
        auto* htp = static_cast<QnnHtpDevice_Infrastructure_t*>(
            infrastructure);
        if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
            log_info("HTP performance infrastructure has an unknown type");
            return;
        }
        perf_infrastructure_ = htp->perfInfra;
        if (perf_infrastructure_.createPowerConfigId == nullptr
            || perf_infrastructure_.setPowerConfig == nullptr
            || perf_infrastructure_.destroyPowerConfigId == nullptr) {
            log_info("HTP performance infrastructure is incomplete");
            return;
        }
        const Qnn_ErrorHandle_t create_status =
            perf_infrastructure_.createPowerConfigId(
                0,
                0,
                &power_config_id_);
        if (create_status != QNN_SUCCESS) {
            log_info(
                "createPowerConfigId returned "
                + std::to_string(create_status));
            return;
        }
        power_config_active_ = true;

        QnnHtpPerfInfrastructure_PowerConfig_t dcvs =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        dcvs.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config.contextId = power_config_id_;
        dcvs.dcvsV3Config.setDcvsEnable = 1;
        dcvs.dcvsV3Config.dcvsEnable = 0;
        dcvs.dcvsV3Config.powerMode =
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        dcvs.dcvsV3Config.setSleepLatency = 1;
        dcvs.dcvsV3Config.sleepLatency = 40;
        dcvs.dcvsV3Config.setSleepDisable = 1;
        dcvs.dcvsV3Config.sleepDisable = 1;
        dcvs.dcvsV3Config.setBusParams = 1;
        dcvs.dcvsV3Config.busVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.setCoreParams = 1;
        dcvs.dcvsV3Config.coreVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        QnnHtpPerfInfrastructure_PowerConfig_t rpc_latency =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        rpc_latency.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
        rpc_latency.rpcControlLatencyConfig = 100;

        QnnHtpPerfInfrastructure_PowerConfig_t rpc_polling =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        rpc_polling.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        rpc_polling.rpcPollingTimeConfig =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME;

        const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {
            &dcvs,
            &rpc_latency,
            &rpc_polling,
            nullptr};
        const Qnn_ErrorHandle_t set_status =
            perf_infrastructure_.setPowerConfig(
                power_config_id_,
                configs);
        if (set_status != QNN_SUCCESS) {
            log_info(
                "HTP burst power vote returned "
                + std::to_string(set_status));
            return;
        }

        // SM8750 is HMX capable. Keep this optional so the normal maximum
        // HVX/bus vote remains active even if a firmware rejects HMX voting.
        QnnHtpPerfInfrastructure_PowerConfig_t hmx =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        hmx.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
        hmx.hmxV2Config.hmxPickDefault = 0;
        hmx.hmxV2Config.hmxVoltageCornerMin = DCVS_EXP_VCORNER_TUR;
        hmx.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_TUR;
        hmx.hmxV2Config.hmxVoltageCornerMax = DCVS_EXP_VCORNER_TUR;
        hmx.hmxV2Config.hmxPerfMode =
            QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;
        const QnnHtpPerfInfrastructure_PowerConfig_t* hmx_configs[] = {
            &hmx,
            nullptr};
        const Qnn_ErrorHandle_t hmx_status =
            perf_infrastructure_.setPowerConfig(
                power_config_id_,
                hmx_configs);
        log_info(
            "HTP burst vote enabled; HMX vote status "
            + std::to_string(hmx_status));
    }

    void parse_metadata()
    {
        check(
            qnn_system_.systemContextCreate(&system_context_),
            "systemContextCreate");
        const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
        check(
            qnn_system_.systemContextGetMetaData(
                system_context_,
                context_binary_.data(),
                context_binary_.size(),
                &binary_info),
            "systemContextGetMetadata");
        const GraphMetadata graph = first_graph(binary_info);
        if (graph.name == nullptr) {
            throw failure("QNN graph has no name");
        }
        graph_name_ = graph.name;
        input_tensors_.assign(graph.inputs, graph.inputs + graph.input_count);
        output_tensors_.assign(
            graph.outputs,
            graph.outputs + graph.output_count);
    }

    Buffer allocate_rpc_tensor(Qnn_Tensor_t& tensor)
    {
        Buffer result;
        result.bytes = tensor_bytes(tensor);
        result.allocation_bytes = (result.bytes + 4095U) & ~size_t{4095U};
        result.data = rpc_alloc_(
            kRpcMemHeapSystem,
            kRpcMemDefaultFlags,
            static_cast<int>(result.allocation_bytes));
        if (result.data == nullptr) {
            throw failure("rpcmem_alloc failed");
        }
        result.rpc_owned = true;
        std::memset(result.data, 0, result.allocation_bytes);
        result.fd = rpc_to_fd_(result.data);
        if (result.fd < 0) {
            rpc_free_(result.data);
            throw failure("rpcmem_to_fd failed");
        }
        Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
        descriptor.memShape.numDim = QNN_TENSOR_GET_RANK(tensor);
        descriptor.memShape.dimSize = QNN_TENSOR_GET_DIMENSIONS(tensor);
        descriptor.memShape.shapeConfig = nullptr;
        descriptor.dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);
        descriptor.memType = QNN_MEM_TYPE_ION;
        descriptor.ionInfo.fd = result.fd;
        check(
            qnn_.memRegister(context_, &descriptor, 1, &result.handle),
            "memRegister");
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, result.handle);
        return result;
    }

    bool try_allocate_hardware_output(
        Qnn_Tensor_t& tensor,
        Buffer& result)
    {
        result.bytes = tensor_bytes(tensor);
        result.allocation_bytes =
            (result.bytes + 4095U) & ~size_t{4095U};
        if (result.allocation_bytes > UINT32_MAX) {
            log_info("AHardwareBuffer output is too large");
            return false;
        }

        AHardwareBuffer_Desc description{};
        description.width =
            static_cast<uint32_t>(result.allocation_bytes);
        description.height = 1;
        description.layers = 1;
        description.format = AHARDWAREBUFFER_FORMAT_BLOB;
        description.usage =
            AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER
            | AHARDWAREBUFFER_USAGE_CPU_READ_RARELY
            | AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY;
        const int allocation_status =
            AHardwareBuffer_allocate(
                &description,
                &result.hardware_buffer);
        if (allocation_status != 0 || result.hardware_buffer == nullptr) {
            log_info(
                "AHardwareBuffer_allocate returned "
                + std::to_string(allocation_status));
            result.hardware_buffer = nullptr;
            return false;
        }

        auto get_native_handle =
            reinterpret_cast<AHardwareBufferGetNativeHandle>(
                dlsym(RTLD_DEFAULT, "AHardwareBuffer_getNativeHandle"));
        if (get_native_handle == nullptr) {
            log_info("AHardwareBuffer_getNativeHandle is unavailable");
            AHardwareBuffer_release(result.hardware_buffer);
            result.hardware_buffer = nullptr;
            return false;
        }
        const void* opaque_handle =
            get_native_handle(result.hardware_buffer);
        if (opaque_handle == nullptr) {
            log_info("AHardwareBuffer has no native handle");
            AHardwareBuffer_release(result.hardware_buffer);
            result.hardware_buffer = nullptr;
            return false;
        }
        const auto* header =
            static_cast<const NativeHandleHeader*>(opaque_handle);
        const int* handle_data = reinterpret_cast<const int*>(header + 1);
        if (header->num_fds <= 0 || header->num_fds > 16) {
            log_info(
                "AHardwareBuffer native handle has invalid fd count "
                + std::to_string(header->num_fds));
            AHardwareBuffer_release(result.hardware_buffer);
            result.hardware_buffer = nullptr;
            return false;
        }

        int selected_fd = -1;
        off_t selected_size = -1;
        for (int index = 0; index < header->num_fds; ++index) {
            struct stat attributes {};
            const int candidate = handle_data[index];
            const off_t candidate_size =
                fstat(candidate, &attributes) == 0
                ? attributes.st_size
                : 0;
            log_info(
                "AHardwareBuffer fd[" + std::to_string(index)
                + "]=" + std::to_string(candidate)
                + " size=" + std::to_string(candidate_size));
            if (candidate_size >= static_cast<off_t>(result.bytes)
                && candidate_size > selected_size) {
                selected_fd = candidate;
                selected_size = candidate_size;
            }
        }
        if (selected_fd < 0) {
            selected_fd = handle_data[0];
        }
        result.fd = dup(selected_fd);
        result.fd_owned = result.fd >= 0;
        if (result.fd < 0) {
            log_info(
                "dup(AHardwareBuffer fd) failed: "
                + std::string(std::strerror(errno)));
            AHardwareBuffer_release(result.hardware_buffer);
            result.hardware_buffer = nullptr;
            return false;
        }
        result.data = mmap(
            nullptr,
            result.allocation_bytes,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            result.fd,
            0);
        if (result.data == MAP_FAILED) {
            log_info(
                "mmap(AHardwareBuffer) failed: "
                + std::string(std::strerror(errno)));
            result.data = nullptr;
            close(result.fd);
            result.fd = -1;
            result.fd_owned = false;
            AHardwareBuffer_release(result.hardware_buffer);
            result.hardware_buffer = nullptr;
            return false;
        }
        result.mmap_owned = true;
        std::memset(result.data, 0, result.allocation_bytes);

        Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
        descriptor.memShape.numDim = QNN_TENSOR_GET_RANK(tensor);
        descriptor.memShape.dimSize = QNN_TENSOR_GET_DIMENSIONS(tensor);
        descriptor.memShape.shapeConfig = nullptr;
        descriptor.dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);
        descriptor.memType = QNN_MEM_TYPE_DMA_BUF;
        descriptor.dmaBufInfo.fd = result.fd;
        descriptor.dmaBufInfo.data = result.data;
        Qnn_ErrorHandle_t registration_status =
            qnn_.memRegister(
                context_,
                &descriptor,
                1,
                &result.handle);
        std::string registration_type = "DMA_BUF";
        if (registration_status != QNN_SUCCESS) {
            descriptor = QNN_MEM_DESCRIPTOR_INIT;
            descriptor.memShape.numDim = QNN_TENSOR_GET_RANK(tensor);
            descriptor.memShape.dimSize = QNN_TENSOR_GET_DIMENSIONS(tensor);
            descriptor.memShape.shapeConfig = nullptr;
            descriptor.dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);
            descriptor.memType = QNN_MEM_TYPE_ION;
            descriptor.ionInfo.fd = result.fd;
            registration_status =
                qnn_.memRegister(
                    context_,
                    &descriptor,
                    1,
                    &result.handle);
            registration_type = "ION";
        }
        if (registration_status != QNN_SUCCESS) {
            log_info(
                "QNN rejected AHardwareBuffer output: "
                + std::to_string(registration_status));
            munmap(result.data, result.allocation_bytes);
            result.data = nullptr;
            result.mmap_owned = false;
            close(result.fd);
            result.fd = -1;
            result.fd_owned = false;
            AHardwareBuffer_release(result.hardware_buffer);
            result.hardware_buffer = nullptr;
            return false;
        }
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, result.handle);
        log_info(
            "QNN output uses AHardwareBuffer/"
            + registration_type
            + " bytes=" + std::to_string(result.bytes));
        return true;
    }

    Buffer allocate_output_tensor(Qnn_Tensor_t& tensor)
    {
        Buffer result;
        if (prefer_hardware_output_
            && try_allocate_hardware_output(tensor, result)) {
            return result;
        }
        log_info(
            prefer_hardware_output_
                ? "falling back to rpcmem QNN output"
                : "using rpcmem QNN output");
        return allocate_rpc_tensor(tensor);
    }

    void allocate_tensors()
    {
        inputs_.reserve(input_tensors_.size());
        for (size_t index = 0; index < input_tensors_.size(); ++index) {
            const char* name = QNN_TENSOR_GET_NAME(input_tensors_[index]);
            if (name == nullptr) {
                throw failure("QNN input has no name");
            }
            input_by_name_[name] = index;
            inputs_.push_back(allocate_rpc_tensor(input_tensors_[index]));
        }
        outputs_.reserve(output_tensors_.size());
        for (size_t index = 0; index < output_tensors_.size(); ++index) {
            Qnn_Tensor_t& tensor = output_tensors_[index];
            const char* name = QNN_TENSOR_GET_NAME(tensor);
            if (name == nullptr) {
                throw failure("QNN output has no name");
            }
            output_by_name_[name] = index;
            outputs_.push_back(allocate_output_tensor(tensor));
        }
    }

    void release_tensors()
    {
        auto release = [&](std::vector<Buffer>& buffers) {
            for (Buffer& buffer : buffers) {
                if (buffer.handle != nullptr
                    && qnn_.memDeRegister != nullptr) {
                    Qnn_MemHandle_t handle = buffer.handle;
                    qnn_.memDeRegister(&handle, 1);
                }
                if (buffer.mmap_owned && buffer.data != nullptr) {
                    munmap(buffer.data, buffer.allocation_bytes);
                } else if (
                    buffer.rpc_owned
                    && buffer.data != nullptr
                    && rpc_free_ != nullptr) {
                    rpc_free_(buffer.data);
                }
                if (buffer.fd_owned && buffer.fd >= 0) {
                    close(buffer.fd);
                }
                if (buffer.hardware_buffer != nullptr) {
                    AHardwareBuffer_release(buffer.hardware_buffer);
                }
                buffer = {};
            }
        };
        release(outputs_);
        release(inputs_);
    }

    void* backend_library_{};
    void* system_library_{};
    void* rpc_library_{};
    RpcMemAlloc rpc_alloc_{};
    RpcMemFree rpc_free_{};
    RpcMemToFd rpc_to_fd_{};
    QNN_INTERFACE_VER_TYPE qnn_{};
    QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_{};
    Qnn_BackendHandle_t backend_{};
    Qnn_DeviceHandle_t device_{};
    QnnHtpDevice_PerfInfrastructure_t perf_infrastructure_{};
    uint32_t power_config_id_{};
    bool power_config_active_{};
    Qnn_ContextHandle_t context_{};
    Qnn_GraphHandle_t graph_{};
    QnnSystemContext_Handle_t system_context_{};
    std::vector<uint8_t> context_binary_;
    std::string graph_name_;
    std::vector<Qnn_Tensor_t> input_tensors_;
    std::vector<Qnn_Tensor_t> output_tensors_;
    std::vector<Buffer> inputs_;
    std::vector<Buffer> outputs_;
    std::unordered_map<std::string, size_t> input_by_name_;
    std::unordered_map<std::string, size_t> output_by_name_;
    bool prefer_hardware_output_{true};
};

QnnRunner::QnnRunner(
    const std::string& native_library_dir,
    const std::string& context_binary_path,
    const bool prefer_hardware_output)
    : impl_(std::make_unique<Impl>(
          native_library_dir,
          context_binary_path,
          prefer_hardware_output))
{
}

QnnRunner::~QnnRunner() = default;

void* QnnRunner::input_data(const std::string& name)
{
    return impl_->input_data(name);
}

size_t QnnRunner::input_bytes(const std::string& name) const
{
    return impl_->input_bytes(name);
}

const void* QnnRunner::output_data(const std::string& name) const
{
    return impl_->output_data(name);
}

size_t QnnRunner::output_bytes(const std::string& name) const
{
    return impl_->output_bytes(name);
}

const void* QnnRunner::output_data() const
{
    return impl_->output().data;
}

size_t QnnRunner::output_bytes() const
{
    return impl_->output().bytes;
}

size_t QnnRunner::output_allocation_bytes() const
{
    return impl_->output().allocation_bytes;
}

int QnnRunner::output_fd() const
{
    return impl_->output().fd;
}

AHardwareBuffer* QnnRunner::output_hardware_buffer() const
{
    return impl_->output().hardware_buffer;
}

const std::string& QnnRunner::graph_name() const
{
    return impl_->graph_name();
}

double QnnRunner::execute()
{
    return impl_->execute();
}

}  // namespace pulse_mobile
