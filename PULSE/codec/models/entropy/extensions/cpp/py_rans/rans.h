// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <condition_variable>
#include <exception>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

struct RansSymbol {
    uint16_t start;
    uint16_t range;
};

enum class WorkType {
    EncodeDecodeY,
    EncodeDecodeYSkipped,
    EncodeDecodeZ,
    EncodeDecodeZMetaPrior,
    Flush,
};

struct PendingTask {
    WorkType workType;
    const int16_t* symbols_y{ nullptr };
    const int16_t* symbols_z{ nullptr };
    const uint8_t* indexes{ nullptr };
    const uint8_t* meta_prior_indexes{ nullptr };
    int16_t* decoded_ptr{ nullptr };
    std::shared_ptr<void> owner;
    int total_size{ 0 };
    int cdf_offset{ 0 };
    int symbol_size{ 0 };
    int symbol_offset{ 0 };
    int skip_index_cutoff{ -1 };
    int ch{ 0 };

    PendingTask() = default;
    PendingTask(PendingTask&&) noexcept = default;
    PendingTask& operator=(PendingTask&&) noexcept = default;

    PendingTask(const PendingTask&) = delete;
    PendingTask& operator=(const PendingTask&) = delete;
};

using RansState = uint32_t;
constexpr int MIN_RANS_STREAM_SIZE = 4;

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class RansEncoderLib {
public:
    RansEncoderLib();
    virtual ~RansEncoderLib();

    RansEncoderLib(const RansEncoderLib&) = delete;
    RansEncoderLib(RansEncoderLib&&) = delete;
    RansEncoderLib& operator=(const RansEncoderLib&) = delete;
    RansEncoderLib& operator=(RansEncoderLib&&) = delete;

    void encode_y(const int16_t* symbols, const int symbol_size, const int symbol_offset,
                  const std::shared_ptr<void>& owner = nullptr);
    void encode_y_internal(const int16_t* symbols, const int symbol_size, const int symbol_offset);
    void encode_y_skipped(const int16_t* symbols, const int symbol_size, const int symbol_offset,
                          const int skip_index_cutoff,
                          const std::shared_ptr<void>& owner = nullptr);
    void encode_y_skipped_internal(const int16_t* symbols, const int symbol_size,
                                   const int symbol_offset, const int skip_index_cutoff);
    void encode_z(const int16_t* symbols, const int symbol_size, const int symbol_offset,
                  const int cdf_offset, const int ch, const std::shared_ptr<void>& owner = nullptr);
    void encode_z_internal(const int16_t* symbols, const int symbol_size, const int symbol_offset,
                           const int cdf_offset, const int ch);
    void encode_z_meta_prior(
        const int16_t* symbols, const uint8_t* meta_prior_indexes,
        const int symbol_size, const int symbol_offset, const int ch,
        const std::shared_ptr<void>& owner = nullptr);
    void encode_z_meta_prior_internal(
        const int16_t* symbols, const uint8_t* meta_prior_indexes,
        const int symbol_size, const int symbol_offset, const int ch);
    void flush();
    void flush_internal();
    std::shared_ptr<std::vector<uint8_t>> get_encoded_stream();
    void reset();
    void set_cdf(const std::shared_ptr<std::vector<std::vector<RansSymbol>>>& ransSymbols,
                 const std::shared_ptr<std::vector<int8_t>>& max_value, const int index);

    void worker();

private:
    size_t m_buffer_size{ 1024 * 1024 };
    void reserve_output(size_t bytes);
    uint8_t* _stream_buffer{ nullptr };
    RansState _rans;
    uint8_t* _ptr{ nullptr };
    std::shared_ptr<std::vector<uint8_t>> _stream;

    std::vector<std::shared_ptr<std::vector<std::vector<RansSymbol>>>> _ransSymbols;
    std::vector<std::shared_ptr<std::vector<int8_t>>> _max_value;

    bool m_finish{ false };
    bool m_result_ready{ false };
    std::exception_ptr m_error;
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    std::queue<PendingTask> m_pending;
};

class RansDecoderLib {
public:
    RansDecoderLib();
    virtual ~RansDecoderLib();

    RansDecoderLib(const RansDecoderLib&) = delete;
    RansDecoderLib(RansDecoderLib&&) = delete;
    RansDecoderLib& operator=(const RansDecoderLib&) = delete;
    RansDecoderLib& operator=(RansDecoderLib&&) = delete;

    void set_stream(const std::shared_ptr<std::vector<uint8_t>>& encoded, const int offset = 0,
                    const int size = -1);
    void decode_y_internal(int16_t* decoded_ptr, const uint8_t* indexes, const int symbol_size,
                           const int symbol_offset);
    void decode_y(int16_t* decoded_ptr, const uint8_t* indexes, const int symbol_size,
                  const int symbol_offset, const std::shared_ptr<void>& owner = nullptr);
    void decode_y_sync(int16_t* decoded_ptr, const uint8_t* indexes, const int symbol_size,
                       const int symbol_offset);
    void decode_y_skipped_internal(int16_t* decoded_ptr, const uint8_t* indexes,
                                   const int symbol_size, const int symbol_offset,
                                   const int skip_index_cutoff);
    void decode_y_skipped(int16_t* decoded_ptr, const uint8_t* indexes, const int symbol_size,
                          const int symbol_offset, const int skip_index_cutoff,
                          const std::shared_ptr<void>& owner = nullptr);
    void decode_y_skipped_sync(int16_t* decoded_ptr, const uint8_t* indexes,
                               const int symbol_size, const int symbol_offset,
                               const int skip_index_cutoff);
    void decode_z_internal(int16_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                           const int cdf_offset, const int ch);
    void decode_z(int16_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                  const int cdf_offset, const int ch);
    void decode_z_sync(int16_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                       const int cdf_offset, const int ch);
    void decode_z_meta_prior_internal(
        int16_t* decoded_ptr, const uint8_t* meta_prior_indexes,
        const int symbol_size, const int symbol_offset, const int ch);
    void decode_z_meta_prior(
        int16_t* decoded_ptr, const uint8_t* meta_prior_indexes,
        const int symbol_size, const int symbol_offset, const int ch,
        const std::shared_ptr<void>& owner = nullptr);
    void decode_z_meta_prior_sync(
        int16_t* decoded_ptr, const uint8_t* meta_prior_indexes,
        const int symbol_size, const int symbol_offset, const int ch);

    bool wait_for_decoding_finish();

    void set_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>>& cdfs,
                 const std::shared_ptr<std::vector<int8_t>>& max_value, const int index);
    void worker();

private:
    RansState _rans;
    const uint8_t* _ptr8{ nullptr };
    const uint8_t* _end8{ nullptr };
    std::shared_ptr<std::vector<uint8_t>> _stream;

    std::vector<std::shared_ptr<std::vector<std::vector<int32_t>>>> _cdfs;
    std::vector<std::shared_ptr<std::vector<int8_t>>> _max_value;

    bool m_finish{ false };
    bool m_result_ready{ false };
    std::exception_ptr m_error;
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    std::queue<PendingTask> m_pending;
};
