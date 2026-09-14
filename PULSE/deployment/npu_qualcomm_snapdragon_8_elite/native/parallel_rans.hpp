// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once
#include "rans.h"
#include "log_index_scalar.h"
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

inline uint32_t read_u32_le(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24;
}
inline void append_u32_le(std::vector<uint8_t>& out, uint32_t v) {
    for (int i=0;i<4;++i) out.push_back(uint8_t(v>>(8*i)));
}
struct RansTables {
    std::shared_ptr<std::vector<std::vector<RansSymbol>>> symbols;
    std::shared_ptr<std::vector<int8_t>> max_values;
};


class ParallelRansDecoder {
public:
    explicit ParallelRansDecoder(const int lanes)
        : m_lanes(lanes)
    {
        if (lanes < 1 || lanes > 8) {
            throw std::runtime_error("rANS lanes must be in [1, 8]");
        }
        m_decoders.reserve(static_cast<size_t>(lanes));
        m_lane_streams.resize(static_cast<size_t>(lanes));
        m_lane_offsets.resize(static_cast<size_t>(lanes));
        m_lane_sizes.resize(static_cast<size_t>(lanes));
        for (int lane = 0; lane < lanes; ++lane) {
            auto decoder = std::make_shared<RansDecoderLib>();
            m_decoders.push_back(std::move(decoder));
        }
    }

    void set_cdf(
        const std::shared_ptr<std::vector<std::vector<int32_t>>>& cdfs,
        const std::shared_ptr<std::vector<int8_t>>& max_value,
        const int index)
    {
        for (const auto& decoder : m_decoders) {
            decoder->set_cdf(cdfs, max_value, index);
        }
    }

    void set_stream(const std::shared_ptr<std::vector<uint8_t>>& stream)
    {
        std::fill(m_lane_streams.begin(), m_lane_streams.end(), nullptr);
        m_owned_streams.clear();
        if (m_lanes == 1) {
            bind_lane(
                0,
                stream,
                0,
                static_cast<int>(stream->size()));
            return;
        }
        if (m_lanes == 2) {
            auto reversed = std::make_shared<std::vector<uint8_t>>(
                stream->rbegin(),
                stream->rend());
            m_owned_streams.push_back(reversed);
            bind_lane(
                0,
                stream,
                0,
                static_cast<int>(stream->size()));
            bind_lane(
                1,
                reversed,
                0,
                static_cast<int>(reversed->size()));
            return;
        }

        const int pairs = m_lanes / 2;
        const bool has_tail = (m_lanes & 1) != 0;
        const int offset_count = pairs - 1 + (has_tail ? 1 : 0);
        const int header_size = offset_count * 4;
        if (stream->size() < static_cast<size_t>(header_size)) {
            throw std::runtime_error("parallel rANS header is truncated");
        }
        const int payload_size =
            static_cast<int>(stream->size()) - header_size;
        std::vector<int> offsets(static_cast<size_t>(offset_count));
        for (int index = 0; index < offset_count; ++index) {
            offsets[static_cast<size_t>(index)] = static_cast<int>(
                read_u32_le(stream->data() + index * 4));
            if (offsets[static_cast<size_t>(index)] < 0
                || offsets[static_cast<size_t>(index)] > payload_size
                || (index > 0
                    && offsets[static_cast<size_t>(index)]
                        <= offsets[static_cast<size_t>(index - 1)])) {
                throw std::runtime_error(
                    "parallel rANS offsets are invalid");
            }
        }
        auto reversed_payload = std::make_shared<std::vector<uint8_t>>(
            stream->begin() + header_size,
            stream->end());
        std::reverse(
            reversed_payload->begin(),
            reversed_payload->end());
        m_owned_streams.push_back(reversed_payload);

        std::vector<int> group_start(static_cast<size_t>(pairs));
        std::vector<int> group_size(static_cast<size_t>(pairs));
        group_start[0] = 0;
        group_size[0] = offsets[0];
        for (int pair = 1; pair < pairs; ++pair) {
            group_start[static_cast<size_t>(pair)] =
                offsets[static_cast<size_t>(pair - 1)];
            if (pair < offset_count) {
                group_size[static_cast<size_t>(pair)] =
                    offsets[static_cast<size_t>(pair)]
                    - offsets[static_cast<size_t>(pair - 1)];
            } else {
                const int groups_end =
                    has_tail ? offsets.back() : payload_size;
                group_size[static_cast<size_t>(pair)] =
                    groups_end
                    - offsets[static_cast<size_t>(pair - 1)];
            }
        }
        for (int pair = 0; pair < pairs; ++pair) {
            const int size = group_size[static_cast<size_t>(pair)];
            if (size < MIN_RANS_STREAM_SIZE) {
                throw std::runtime_error(
                    "parallel rANS group is too short");
            }
            const int forward_offset =
                header_size + group_start[static_cast<size_t>(pair)];
            const int reverse_offset =
                payload_size
                - (group_start[static_cast<size_t>(pair)] + size);
            bind_lane(
                pair * 2,
                stream,
                forward_offset,
                size);
            bind_lane(
                pair * 2 + 1,
                reversed_payload,
                reverse_offset,
                size);
        }
        if (has_tail) {
            const int tail_start = offsets.back();
            const int tail_size = payload_size - tail_start;
            if (tail_size < MIN_RANS_STREAM_SIZE) {
                throw std::runtime_error(
                    "parallel rANS tail is too short");
            }
            bind_lane(
                m_lanes - 1,
                stream,
                header_size + tail_start,
                tail_size);
        }
    }

    void reset_stream()
    {
        for (int lane = 0; lane < m_lanes; ++lane) {
            const size_t index = static_cast<size_t>(lane);
            m_decoders[index]->set_stream(
                m_lane_streams[index],
                m_lane_offsets[index],
                m_lane_sizes[index]);
        }
    }

    void decode_z(
        int16_t* output,
        const int total_size,
        const int cdf_offset,
        const int channels)
    {
        const int base = total_size / m_lanes;
        for (int lane = 0; lane < m_lanes; ++lane) {
            const int offset = base * lane;
            const int size = lane == m_lanes - 1
                ? total_size - offset
                : base;
            m_decoders[static_cast<size_t>(lane)]->decode_z(
                output,
                size,
                offset,
                cdf_offset,
                channels);
        }
        wait();
    }

    void decode_y(
        int16_t* output,
        const uint8_t* indexes,
        const int total_size)
    {
        start_decode_y(output, indexes, total_size);
        wait();
    }

    void decode_meta(int16_t* output, const uint8_t* indexes, int total, int channels)
    {
        const int base = total / m_lanes;
        for (int lane = 0; lane < m_lanes; ++lane) {
            const int offset = base * lane;
            const int size = lane == m_lanes-1 ? total-offset : base;
            m_decoders[lane]->decode_z_meta_prior(output, indexes, size, offset, channels);
        }
        wait();
    }

    void start_decode_y(
        int16_t* output,
        const uint8_t* indexes,
        const int total_size)
    {
        const int base = total_size / m_lanes;
        for (int lane = 0; lane < m_lanes; ++lane) {
            const int offset = base * lane;
            const int size = lane == m_lanes - 1
                ? total_size - offset
                : base;
            m_decoders[static_cast<size_t>(lane)]->decode_y(
                output,
                indexes,
                size,
                offset);
        }
    }

    void wait_decode()
    {
        wait();
    }

private:
    void bind_lane(
        const int lane,
        const std::shared_ptr<std::vector<uint8_t>>& stream,
        const int offset,
        const int size)
    {
        const size_t index = static_cast<size_t>(lane);
        m_lane_streams[index] = stream;
        m_lane_offsets[index] = offset;
        m_lane_sizes[index] = size;
        m_decoders[index]->set_stream(stream, offset, size);
    }

    void wait()
    {
        for (const auto& decoder : m_decoders) {
            decoder->wait_for_decoding_finish();
        }
    }

    int m_lanes;
    std::vector<std::shared_ptr<RansDecoderLib>> m_decoders;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> m_owned_streams;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> m_lane_streams;
    std::vector<int> m_lane_offsets;
    std::vector<int> m_lane_sizes;
};

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
using pulse::log_index_scalar::PreparedLinearDecoder;
class I8mmScaleExecutor {
public:
    I8mmScaleExecutor(
        const PreparedLinearDecoder& decoder,
        const int height,
        const int width,
        const int thread_count)
        : m_decoder(decoder),
          m_height(height),
          m_width(width),
          m_sites(height * width),
          m_site_pairs((m_sites + 1) / 2),
          m_one_part(
              pulse::log_index_scalar::packed_length(
                  decoder,
                  height,
                  width)),
          m_thread_count(
              std::clamp(thread_count, 1, m_site_pairs)),
          m_signed_nhwc(
              static_cast<size_t>(m_sites * decoder.z_channels))
    {
        m_workers.reserve(static_cast<size_t>(m_thread_count));
        for (int index = 0; index < m_thread_count; ++index) {
            m_workers.emplace_back(&I8mmScaleExecutor::worker, this, index);
        }
    }

    ~I8mmScaleExecutor()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_start.notify_all();
        for (auto& worker : m_workers) {
            worker.join();
        }
    }

    I8mmScaleExecutor(const I8mmScaleExecutor&) = delete;
    I8mmScaleExecutor& operator=(const I8mmScaleExecutor&) = delete;

    size_t run(
        const int16_t* z_hat_nchw,
        uint8_t* output,
        size_t* saturation_count)
    {
        size_t saturated = 0;
        pulse::log_index_scalar::quantize_signed_nhwc(
            m_decoder,
            z_hat_nchw,
            m_height,
            m_width,
            m_signed_nhwc,
            saturated);
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_output = output;
            m_remaining = m_thread_count;
            ++m_generation;
        }
        m_start.notify_all();
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_done.wait(lock, [this] { return m_remaining == 0; });
        }
        if (saturation_count != nullptr) {
            *saturation_count = saturated;
        }
        return m_one_part;
    }

private:
    void worker(const int worker_index)
    {
        uint64_t observed_generation = 0;
        while (true) {
            uint8_t* output = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_start.wait(lock, [&] {
                    return m_stop || m_generation != observed_generation;
                });
                if (m_stop) {
                    return;
                }
                observed_generation = m_generation;
                output = m_output;
            }
            const int begin =
                worker_index * m_site_pairs / m_thread_count;
            const int end =
                (worker_index + 1) * m_site_pairs / m_thread_count;
            pulse::log_index_scalar::infer_packed_i8mm_worker(
                m_decoder,
                m_signed_nhwc.data(),
                m_width,
                begin,
                end,
                m_sites,
                m_one_part,
                output);
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (--m_remaining == 0) {
                    m_done.notify_one();
                }
            }
        }
    }

    const PreparedLinearDecoder& m_decoder;
    int m_height;
    int m_width;
    int m_sites;
    int m_site_pairs;
    size_t m_one_part;
    int m_thread_count;
    std::vector<int8_t> m_signed_nhwc;
    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_start;
    std::condition_variable m_done;
    uint64_t m_generation{0};
    int m_remaining{0};
    bool m_stop{false};
    uint8_t* m_output{nullptr};
};

#endif

inline int trailing_identical_bytes(
    const std::vector<uint8_t>& first,
    const std::vector<uint8_t>& second)
{
    const int first_size = static_cast<int>(first.size());
    const int second_size = static_cast<int>(second.size());
    int identical = 0;
    const int checked = std::min({first_size, second_size, 8});
    for (int index = 0; index < checked; ++index) {
        if (first[static_cast<size_t>(first_size - 1 - index)] != 0
            || second[static_cast<size_t>(second_size - 1 - index)] != 0) {
            break;
        }
        ++identical;
    }
    if (identical == 0
        && !first.empty()
        && !second.empty()
        && first.back() == second.back()) {
        identical = 1;
    }
    return identical;
}

class ParallelRansEncoder {
public:
    explicit ParallelRansEncoder(const int lanes)
        : lanes_(lanes)
    {
        if (lanes_ < 1 || lanes_ > 8) {
            throw std::runtime_error("rANS lane count must be in [1, 8]");
        }
        encoders_.reserve(static_cast<size_t>(lanes_));
        for (int lane = 0; lane < lanes_; ++lane) {
            encoders_.push_back(std::make_shared<RansEncoderLib>());
        }
    }

    void set_cdf(const RansTables& tables, const int slot)
    {
        for (const auto& encoder : encoders_) {
            encoder->set_cdf(tables.symbols, tables.max_values, slot);
        }
    }

    void begin_y(const int16_t* symbols, const int total_size)
    {
        begin(
            symbols,
            total_size,
            [symbols](RansEncoderLib& encoder,
                      const int size,
                      const int offset) {
                encoder.encode_y(symbols, size, offset);
            });
    }

    void begin_z(
        const int16_t* symbols,
        const int total_size,
        const int cdf_offset,
        const int channels)
    {
        begin(
            symbols,
            total_size,
            [symbols, cdf_offset, channels](
                RansEncoderLib& encoder,
                const int size,
                const int offset) {
                encoder.encode_z(
                    symbols,
                    size,
                    offset,
                    cdf_offset,
                    channels);
            });
    }

    void begin_meta(const int16_t* symbols, const uint8_t* indexes, int total, int channels)
    {
        begin(symbols, total, [symbols,indexes,channels](RansEncoderLib& encoder, int size, int offset) {
            encoder.encode_z_meta_prior(symbols, indexes, size, offset, channels);
        });
    }

    std::vector<uint8_t> finish()
    {
        if (!active_) {
            throw std::runtime_error("rANS encoder has no active frame");
        }
        std::vector<std::shared_ptr<std::vector<uint8_t>>> streams;
        streams.reserve(static_cast<size_t>(lanes_));
        for (const auto& encoder : encoders_) {
            streams.push_back(encoder->get_encoded_stream());
        }
        active_ = false;
        if (lanes_ == 1) {
            return *streams[0];
        }

        const int pairs = lanes_ / 2;
        const bool has_tail = (lanes_ & 1) != 0;
        std::vector<int> overlaps(static_cast<size_t>(pairs));
        std::vector<int> group_sizes(static_cast<size_t>(pairs));
        size_t payload_bytes = 0;
        for (int pair = 0; pair < pairs; ++pair) {
            const auto& first = *streams[static_cast<size_t>(pair * 2)];
            const auto& second =
                *streams[static_cast<size_t>(pair * 2 + 1)];
            const int overlap = trailing_identical_bytes(first, second);
            overlaps[static_cast<size_t>(pair)] = overlap;
            const int size = static_cast<int>(
                first.size() + second.size() - overlap);
            group_sizes[static_cast<size_t>(pair)] = size;
            payload_bytes += static_cast<size_t>(size);
        }
        if (has_tail) {
            payload_bytes += streams.back()->size();
        }
        const int offset_count = pairs - 1 + (has_tail ? 1 : 0);
        const size_t header_bytes =
            static_cast<size_t>(offset_count) * sizeof(uint32_t);
        std::vector<uint8_t> output;
        output.reserve(header_bytes + payload_bytes);

        uint32_t cumulative =
            static_cast<uint32_t>(group_sizes.front());
        for (int index = 0; index < offset_count; ++index) {
            append_u32_le(output, cumulative);
            if (index + 1 < pairs) {
                cumulative += static_cast<uint32_t>(
                    group_sizes[static_cast<size_t>(index + 1)]);
            }
        }
        for (int pair = 0; pair < pairs; ++pair) {
            const auto& first = *streams[static_cast<size_t>(pair * 2)];
            const auto& second =
                *streams[static_cast<size_t>(pair * 2 + 1)];
            output.insert(output.end(), first.begin(), first.end());
            const int overlap = overlaps[static_cast<size_t>(pair)];
            output.insert(
                output.end(),
                second.rbegin() + overlap,
                second.rend());
        }
        if (has_tail) {
            output.insert(
                output.end(),
                streams.back()->begin(),
                streams.back()->end());
        }
        return output;
    }

private:
    template <typename Queue>
    void begin(
        const int16_t* symbols,
        const int total_size,
        Queue&& queue)
    {
        if (active_) {
            throw std::runtime_error("rANS encoder frame is already active");
        }
        if (symbols == nullptr || total_size < 0) {
            throw std::runtime_error("invalid rANS encoder input");
        }
        const int base = total_size / lanes_;
        for (int lane = 0; lane < lanes_; ++lane) {
            encoders_[static_cast<size_t>(lane)]->reset();
            const int offset = base * lane;
            const int size = lane == lanes_ - 1
                ? total_size - offset
                : base;
            queue(
                *encoders_[static_cast<size_t>(lane)],
                size,
                offset);
            encoders_[static_cast<size_t>(lane)]->flush();
        }
        active_ = true;
    }

    int lanes_;
    bool active_{false};
    std::vector<std::shared_ptr<RansEncoderLib>> encoders_;
};
