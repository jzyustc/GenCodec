// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_rans.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct MetaPriorEncodeBuffers {
    std::vector<int16_t> symbols;
    std::vector<uint8_t> indexes;
};

void validate_cdf_index(const int index)
{
    if (index < 0 || index >= 2) {
        throw std::runtime_error("rANS cdf index must be 0 or 1");
    }
}

void validate_entropy_coder_parallel(const int n)
{
    if (n < 1 || n > MAX_EC_PARALLEL) {
        throw std::runtime_error("rANS entropy_coder_parallel must be in [1, "
                                 + std::to_string(MAX_EC_PARALLEL) + "]");
    }
}

void validate_z_args(const int cdf_offset, const int ch)
{
    if (cdf_offset < 0) {
        throw std::runtime_error("rANS cdf_offset must be non-negative");
    }
    if (ch <= 0) {
        throw std::runtime_error("rANS ch must be positive");
    }
}

void validate_y_skip_cutoff(const int cutoff)
{
    if (cutoff < 0 || cutoff >= 64) {
        throw std::runtime_error("rANS y skip cutoff must be in [0, 63]");
    }
}

int validate_cdf_inputs(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                        const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index)
{
    validate_cdf_index(index);
    if (cdfs == nullptr || cdfs_sizes == nullptr) {
        throw std::runtime_error("rANS cdf inputs must not be null");
    }

    const int cdf_num = static_cast<int>(cdfs_sizes->size());
    if (cdf_num <= 0) {
        throw std::runtime_error("rANS cdf_sizes must not be empty");
    }
    if (cdfs->size() % static_cast<size_t>(cdf_num) != 0) {
        throw std::runtime_error("rANS cdfs size must be divisible by the number of CDFs");
    }

    const int per_vector_size = static_cast<int>(cdfs->size() / static_cast<size_t>(cdf_num));
    if (per_vector_size < 2) {
        throw std::runtime_error("rANS each CDF must contain at least 2 entries");
    }
    for (int i = 0; i < cdf_num; i++) {
        const int cdf_size = cdfs_sizes->at(i);
        if (cdf_size < 2 || cdf_size > per_vector_size) {
            throw std::runtime_error("rANS cdf size is out of range");
        }
    }

    return cdf_num;
}

}  // namespace

// Count trailing identical zero bytes shared between two encoded streams.
// This allows overlapping the reversed stream to save space.
static int compute_identical_bytes(const std::vector<uint8_t>& a, int na,
                                   const std::vector<uint8_t>& b, int nb)
{
    int identical_bytes = 0;
    int check_bytes = std::min({ na, nb, 8 });
    for (int i = 0; i < check_bytes; i++) {
        if (a[na - 1 - i] != 0) {
            break;
        }
        if (b[nb - 1 - i] != 0) {
            break;
        }
        identical_bytes++;
    }
    if (identical_bytes == 0 && a[na - 1] == b[nb - 1]) {
        identical_bytes = 1;
    }
    return identical_bytes;
}

static int32_t read_stream_offset(const uint8_t* ptr)
{
    int32_t offset = 0;
    std::memcpy(&offset, ptr, sizeof(offset));
    return offset;
}

static std::shared_ptr<std::vector<uint8_t>> make_stream_copy(const uint8_t* ptr, const int size)
{
    auto stream = std::make_shared<std::vector<uint8_t>>(size);
    std::copy(ptr, ptr + size, stream->begin());
    return stream;
}

static std::shared_ptr<std::vector<uint8_t>> make_reversed_stream_copy(const uint8_t* ptr, const int size)
{
    auto stream = std::make_shared<std::vector<uint8_t>>(size);
    std::reverse_copy(ptr, ptr + size, stream->begin());
    return stream;
}

RansEncoder::RansEncoder()
{
    // CPU_AMD is latency-oriented and defaults to one lane. Avoid creating
    // 31 unused workers and their 10 MB stream buffers.
    m_encoders.push_back(std::make_shared<RansEncoderLib>());
}

void RansEncoder::encode_y(const int16_t* symbols, const int symbolSize,
                           const std::shared_ptr<void>& owner)
{
    int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_encoders[0]->encode_y_internal(symbols, symbolSize, 0);
        return;
    }
    int size0 = symbolSize / n;
    for (int i = 0; i < n - 1; i++) {
        m_encoders[i]->encode_y(symbols, size0, size0 * i, owner);
    }
    m_encoders[n - 1]->encode_y(symbols, symbolSize - size0 * (n - 1), size0 * (n - 1), owner);
}

void RansEncoder::encode_y(const py::array_t<int16_t>& symbols)
{
    py::buffer_info symbols_buf = symbols.request();
    int16_t* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);

    int symbolSize = static_cast<int>(symbols.size());
    if (m_entropy_coder_parallel == 1) {
        encode_y(symbols_ptr, symbolSize);
        return;
    }
    auto vec_symbols = std::make_shared<std::vector<int16_t>>(symbolSize);
    std::copy(symbols_ptr, symbols_ptr + symbolSize, vec_symbols->data());
    encode_y(vec_symbols->data(), symbolSize, vec_symbols);
}

void RansEncoder::encode_y_borrowed(
    const py::array_t<int16_t>& symbols)
{
    py::buffer_info symbols_buf = symbols.request();
    auto* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);
    encode_y(
        symbols_ptr,
        static_cast<int>(symbols.size()));
}

void RansEncoder::encode_y_skip(const int16_t* symbols, const int symbolSize,
                                const int skipIndexCutoff,
                                const std::shared_ptr<void>& owner)
{
    validate_y_skip_cutoff(skipIndexCutoff);
    int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_encoders[0]->encode_y_skipped_internal(
            symbols, symbolSize, 0, skipIndexCutoff);
        return;
    }
    int size0 = symbolSize / n;
    for (int i = 0; i < n - 1; i++) {
        m_encoders[i]->encode_y_skipped(
            symbols, size0, size0 * i, skipIndexCutoff, owner);
    }
    m_encoders[n - 1]->encode_y_skipped(
        symbols, symbolSize - size0 * (n - 1), size0 * (n - 1),
        skipIndexCutoff, owner);
}

void RansEncoder::encode_y_skip(const py::array_t<int16_t>& symbols,
                                const int skipIndexCutoff)
{
    py::buffer_info symbols_buf = symbols.request();
    int16_t* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);
    int symbolSize = static_cast<int>(symbols.size());
    if (m_entropy_coder_parallel == 1) {
        encode_y_skip(symbols_ptr, symbolSize, skipIndexCutoff);
        return;
    }
    auto vec_symbols = std::make_shared<std::vector<int16_t>>(symbolSize);
    std::copy(symbols_ptr, symbols_ptr + symbolSize, vec_symbols->data());
    encode_y_skip(vec_symbols->data(), symbolSize, skipIndexCutoff, vec_symbols);
}

void RansEncoder::encode_y_skip_borrowed(
    const py::array_t<int16_t>& symbols,
    const int skipIndexCutoff)
{
    py::buffer_info symbols_buf = symbols.request();
    auto* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);
    encode_y_skip(
        symbols_ptr,
        static_cast<int>(symbols.size()),
        skipIndexCutoff);
}

void RansEncoder::encode_z(const int16_t* symbols, const int symbolSize, const int cdf_offset,
                           const int ch, const std::shared_ptr<void>& owner)
{
    validate_z_args(cdf_offset, ch);

    int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_encoders[0]->encode_z_internal(symbols, symbolSize, 0, cdf_offset, ch);
        return;
    }
    int size0 = symbolSize / n;
    for (int i = 0; i < n - 1; i++) {
        m_encoders[i]->encode_z(symbols, size0, size0 * i, cdf_offset, ch, owner);
    }
    m_encoders[n - 1]->encode_z(symbols, symbolSize - size0 * (n - 1), size0 * (n - 1), cdf_offset,
                                ch, owner);
}

void RansEncoder::encode_z(const py::array_t<int16_t>& symbols, const int cdf_offset, const int ch)
{
    py::buffer_info symbols_buf = symbols.request();
    int16_t* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);

    int symbolSize = static_cast<int>(symbols.size());
    if (m_entropy_coder_parallel == 1) {
        encode_z(symbols_ptr, symbolSize, cdf_offset, ch);
        return;
    }
    auto vec_symbols = std::make_shared<std::vector<int16_t>>(symbolSize);
    std::copy(symbols_ptr, symbols_ptr + symbolSize, vec_symbols->data());
    encode_z(vec_symbols->data(), symbolSize, cdf_offset, ch, vec_symbols);
}

void RansEncoder::encode_z_meta_prior(
    const int16_t* symbols, const uint8_t* meta_prior_indexes,
    const int symbolSize, const int positionCount, const int ch,
    const std::shared_ptr<void>& owner)
{
    validate_z_args(0, ch);
    if (symbolSize < 0 || positionCount < 0 || symbolSize != positionCount * ch) {
        throw std::runtime_error(
            "Meta Prior symbol count must equal position count times channels");
    }

    const int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_encoders[0]->encode_z_meta_prior_internal(
            symbols, meta_prior_indexes, symbolSize, 0, ch);
        return;
    }
    const int size0 = symbolSize / n;
    for (int i = 0; i < n - 1; ++i) {
        m_encoders[i]->encode_z_meta_prior(
            symbols,
            meta_prior_indexes,
            size0,
            size0 * i,
            ch,
            owner);
    }
    m_encoders[n - 1]->encode_z_meta_prior(
        symbols,
        meta_prior_indexes,
        symbolSize - size0 * (n - 1),
        size0 * (n - 1),
        ch,
        owner);
}

void RansEncoder::encode_z_meta_prior(
    const py::array_t<int16_t>& symbols,
    const py::array_t<uint8_t>& meta_prior_indexes,
    const int ch)
{
    py::buffer_info symbols_buf = symbols.request();
    py::buffer_info indexes_buf = meta_prior_indexes.request();
    auto* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);
    auto* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);
    const int symbolSize = static_cast<int>(symbols.size());
    const int positionCount = static_cast<int>(meta_prior_indexes.size());

    if (m_entropy_coder_parallel == 1) {
        encode_z_meta_prior(
            symbols_ptr,
            indexes_ptr,
            symbolSize,
            positionCount,
            ch);
        return;
    }
    auto buffers = std::make_shared<MetaPriorEncodeBuffers>();
    buffers->symbols.assign(symbols_ptr, symbols_ptr + symbolSize);
    buffers->indexes.assign(indexes_ptr, indexes_ptr + positionCount);
    encode_z_meta_prior(
        buffers->symbols.data(),
        buffers->indexes.data(),
        symbolSize,
        positionCount,
        ch,
        buffers);
}

void RansEncoder::encode_z_meta_prior_borrowed(
    const py::array_t<int16_t>& symbols,
    const py::array_t<uint8_t>& meta_prior_indexes,
    const int ch)
{
    py::buffer_info symbols_buf = symbols.request();
    py::buffer_info indexes_buf = meta_prior_indexes.request();
    auto* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);
    auto* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);
    encode_z_meta_prior(
        symbols_ptr,
        indexes_ptr,
        static_cast<int>(symbols.size()),
        static_cast<int>(meta_prior_indexes.size()),
        ch);
}

void RansEncoder::set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                          const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index)
{
    int cdf_num = validate_cdf_inputs(cdfs, cdfs_sizes, index);
    int per_vector_size = static_cast<int>(cdfs->size() / cdf_num);
    auto max_value = std::make_shared<std::vector<int8_t>>(cdf_num);
    auto ransSymbols = std::make_shared<std::vector<std::vector<RansSymbol>>>(cdf_num);
    for (int i = 0; i < cdf_num; i++) {
        max_value->at(i) = static_cast<int8_t>(cdfs_sizes->at(i) - 2);

        const int32_t* cdf = cdfs->data() + i * per_vector_size;
        std::vector<RansSymbol> ransSym(per_vector_size);
        const int ransSize = per_vector_size - 1;
        for (int j = 0; j < ransSize; j++) {
            ransSym[j] = RansSymbol(
                { static_cast<uint16_t>(cdf[j]), static_cast<uint16_t>(cdf[j + 1] - cdf[j]) });
        }
        ransSymbols->at(i) = std::move(ransSym);
    }

    for (const auto& encoder : m_encoders) {
        encoder->set_cdf(ransSymbols, max_value, index);
    }
    m_has_cdf = true;
}

void RansEncoder::set_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                          const int index)
{
    py::buffer_info cdfs_buf = cdfs.request();
    py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
    int32_t* cdfs_ptr = static_cast<int32_t*>(cdfs_buf.ptr);
    int32_t* cdfs_sizes_ptr = static_cast<int32_t*>(cdfs_sizes_buf.ptr);

    auto vec_cdfs = std::make_shared<std::vector<int32_t>>(cdfs.size());
    std::copy(cdfs_ptr, cdfs_ptr + cdfs.size(), vec_cdfs->data());
    auto vec_cdfs_sizes = std::make_shared<std::vector<int32_t>>(cdfs_sizes.size());
    std::copy(cdfs_sizes_ptr, cdfs_sizes_ptr + cdfs_sizes.size(), vec_cdfs_sizes->data());

    set_cdf(vec_cdfs, vec_cdfs_sizes, index);
}

void RansEncoder::flush()
{
    int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_encoders[0]->flush_internal();
        return;
    }
    for (int i = 0; i < n; i++) {
        m_encoders[i]->flush();
    }
}

py::array_t<uint8_t> RansEncoder::get_encoded_stream()
{
    int n = m_entropy_coder_parallel;

    // Collect all encoded streams
    std::vector<std::shared_ptr<std::vector<uint8_t>>> results(n);
    std::vector<int> nbytes(n);
    for (int i = 0; i < n; i++) {
        results[i] = m_encoders[i]->get_encoded_stream();
        nbytes[i] = static_cast<int>(results[i]->size());
    }

    if (n == 1) {
        /**
         * The input arguments to py::array_t<uint8_t> constructor are array size and stride.
         * The code may fail to perform correct encoding-decoding by omitting array stride.
         * It would be more robust by explicitly specifying the stride.
         *
         * By only passing the array size, i.e., "py::array_t<uint8_t> stream(nbytes0);",
         * our empirical results are as follows:
         *   pybind11==3.0.1,  numpy==2.3.3:  succeeded
         *   pybind11==2.10.4, numpy==1.26.0: succeeded
         *   pybind11==2.10.4, numpy==2.3.3:  failed. the array stride defaults to 0
         */
        py::array_t<uint8_t> stream({ nbytes[0] }, { sizeof(uint8_t) });
        py::buffer_info stream_buf = stream.request();
        uint8_t* stream_ptr = static_cast<uint8_t*>(stream_buf.ptr);
        std::copy(results[0]->begin(), results[0]->end(), stream_ptr);
        return stream;
    }

    // For n >= 2, streams are paired: (0,1), (2,3), ...
    // Each pair is merged: stream[2k] forward + stream[2k+1] reversed with zero-byte overlap.
    // If n is odd, the last stream stands alone (forward only).
    int num_pairs = n / 2;
    bool has_tail = (n % 2 != 0);

    // Compute group sizes (each pair merged) and the tail
    std::vector<int> group_sizes(num_pairs);
    std::vector<int> identical(num_pairs);
    for (int p = 0; p < num_pairs; p++) {
        int i0 = p * 2;
        int i1 = p * 2 + 1;
        identical[p] = compute_identical_bytes(*results[i0], nbytes[i0], *results[i1], nbytes[i1]);
        group_sizes[p] = nbytes[i0] + nbytes[i1] - identical[p];
    }
    int tail_size = has_tail ? nbytes[n - 1] : 0;

    // Header: (num_pairs - 1 + has_tail) int32_t offsets
    // For n==2 (1 pair, no tail): 0 offsets, no header
    int num_offsets = num_pairs - 1 + (has_tail ? 1 : 0);
    int header_size = num_offsets * 4;

    int total_size = header_size;
    for (int p = 0; p < num_pairs; p++) {
        total_size += group_sizes[p];
    }
    total_size += tail_size;

    py::array_t<uint8_t> stream({ total_size }, { sizeof(uint8_t) });
    py::buffer_info stream_buf = stream.request();
    uint8_t* stream_ptr = static_cast<uint8_t*>(stream_buf.ptr);

    // Write header: cumulative offsets for groups after the first
    // offset[k] = cumulative size of groups 0..k (so decoder knows where group k+1 starts)
    int cumulative = group_sizes[0];
    for (int k = 0; k < num_offsets; k++) {
        *reinterpret_cast<int32_t*>(stream_ptr + k * 4) = cumulative;
        if (k + 1 < num_pairs) {
            cumulative += group_sizes[k + 1];
        } else {
            // This offset points to the tail start (== total payload without header)
            // Already correct: cumulative == sum of all group_sizes
        }
    }

    // Write groups
    int pos = header_size;
    for (int p = 0; p < num_pairs; p++) {
        int i0 = p * 2;
        int i1 = p * 2 + 1;
        std::copy(results[i0]->begin(), results[i0]->end(), stream_ptr + pos);
        std::reverse_copy(results[i1]->begin(), results[i1]->end() - identical[p],
                          stream_ptr + pos + nbytes[i0]);
        pos += group_sizes[p];
    }

    // Write tail
    if (has_tail) {
        std::copy(results[n - 1]->begin(), results[n - 1]->end(), stream_ptr + pos);
    }

    return stream;
}

void RansEncoder::reset()
{
    for (const auto& encoder : m_encoders) {
        encoder->reset();
    }
}

void RansEncoder::set_entropy_coder_parallel(int n)
{
    validate_entropy_coder_parallel(n);
    if (n > static_cast<int>(m_encoders.size())) {
        if (m_has_cdf) {
            throw std::runtime_error(
                "set rANS lane count before registering CDF tables");
        }
        while (static_cast<int>(m_encoders.size()) < n) {
            m_encoders.push_back(std::make_shared<RansEncoderLib>());
        }
    }
    m_entropy_coder_parallel = n;
}

int RansEncoder::get_entropy_coder_parallel()
{
    return m_entropy_coder_parallel;
}

RansDecoder::RansDecoder()
{
    m_decoded_tensor = std::make_shared<PinnedHostBuffer<int16_t>>(3840 * 2160 / 16 / 16 * 128 * 2);
    m_decoders.push_back(std::make_shared<RansDecoderLib>());
}

void RansDecoder::set_stream(const uint8_t* ptr, const int size)
{
    int n = m_entropy_coder_parallel;
    if (size < MIN_RANS_STREAM_SIZE) {
        throw std::runtime_error("rANS bitstream is too short to initialize the decoder");
    }

    auto forward_stream = make_stream_copy(ptr, size);

    if (n == 1) {
        m_decoders[0]->set_stream(forward_stream);
        return;
    }

    if (n == 2) {
        // No header, entire buffer is one pair
        auto reverse_stream = make_reversed_stream_copy(forward_stream->data(), size);
        m_decoders[0]->set_stream(forward_stream, 0, size);
        m_decoders[1]->set_stream(reverse_stream, 0, size);
        return;
    }

    // n >= 3: header + groups + optional tail
    int num_pairs = n / 2;
    bool has_tail = (n % 2 != 0);
    int num_offsets = num_pairs - 1 + (has_tail ? 1 : 0);
    int header_size = num_offsets * 4;
    if (size < header_size) {
        throw std::runtime_error("rANS bitstream is too short for the header");
    }

    // Read cumulative offsets
    const int payload_size = size - header_size;
    std::vector<int> offsets(num_offsets);
    for (int k = 0; k < num_offsets; k++) {
        offsets[k] = read_stream_offset(ptr + k * 4);
        if (offsets[k] < 0 || offsets[k] > payload_size) {
            throw std::runtime_error("rANS bitstream offset is out of range");
        }
        if (k > 0 && offsets[k] <= offsets[k - 1]) {
            throw std::runtime_error("rANS bitstream offsets must be strictly increasing");
        }
    }

    auto reverse_payload_stream =
        make_reversed_stream_copy(forward_stream->data() + header_size, payload_size);

    // Compute group start/end positions (relative to after header)
    std::vector<int> group_start(num_pairs);
    std::vector<int> group_size(num_pairs);
    group_start[0] = 0;
    group_size[0] = offsets[0];  // first offset == size of first group
    if (group_size[0] < MIN_RANS_STREAM_SIZE) {
        throw std::runtime_error("rANS bitstream group is too short to initialize the decoder");
    }
    for (int p = 1; p < num_pairs; p++) {
        group_start[p] = offsets[p - 1];
        if (p < num_offsets) {
            group_size[p] = offsets[p] - offsets[p - 1];
        } else {
            // Last group: extends to end of payload (or to tail start)
            int groups_end = has_tail ? offsets[num_offsets - 1] : payload_size;
            group_size[p] = groups_end - offsets[p - 1];
        }
        if (group_size[p] < MIN_RANS_STREAM_SIZE) {
            throw std::runtime_error("rANS bitstream group is too short to initialize the decoder");
        }
    }

    // Set streams for each pair
    for (int p = 0; p < num_pairs; p++) {
        int i0 = p * 2;
        int i1 = p * 2 + 1;
        int gs = group_size[p];

        const int forward_offset = header_size + group_start[p];
        const int reverse_offset = payload_size - (group_start[p] + gs);
        m_decoders[i0]->set_stream(forward_stream, forward_offset, gs);
        m_decoders[i1]->set_stream(reverse_payload_stream, reverse_offset, gs);
    }

    // Tail
    if (has_tail) {
        int tail_start = offsets[num_offsets - 1];
        int tail_size = payload_size - tail_start;
        if (tail_size < MIN_RANS_STREAM_SIZE) {
            throw std::runtime_error("rANS bitstream tail is too short to initialize the decoder");
        }
        m_decoders[n - 1]->set_stream(forward_stream, header_size + tail_start, tail_size);
    }
}

void RansDecoder::set_stream(const py::array_t<uint8_t>& encoded)
{
    py::buffer_info encoded_buf = encoded.request();
    const uint8_t* encoded_ptr = static_cast<uint8_t*>(encoded_buf.ptr);
    const int encoded_size = static_cast<int>(encoded.size());
    set_stream(encoded_ptr, encoded_size);
}

void RansDecoder::decode_y(const uint8_t* indexes, const int indexSize, const std::shared_ptr<void>& owner)
{
    m_current_decoded_tensor_size = indexSize;
    if (m_decoded_tensor == nullptr || static_cast<int>(m_decoded_tensor->size()) < indexSize) {
        m_decoded_tensor = std::make_shared<PinnedHostBuffer<int16_t>>(indexSize * 2);
    }
    int16_t* decoded_ptr = m_decoded_tensor->data();

    int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_decoders[0]->decode_y_sync(decoded_ptr, indexes, indexSize, 0);
        return;
    }
    int size0 = indexSize / n;
    for (int i = 0; i < n - 1; i++) {
        m_decoders[i]->decode_y(decoded_ptr, indexes, size0, size0 * i, owner);
    }
    m_decoders[n - 1]->decode_y(decoded_ptr, indexes, indexSize - size0 * (n - 1), size0 * (n - 1),
                                owner);
}

void RansDecoder::decode_y(const py::array_t<uint8_t>& indexes)
{
    py::buffer_info indexes_buf = indexes.request();
    uint8_t* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);

    int indexSize = static_cast<int>(indexes.size());
    if (m_entropy_coder_parallel == 1) {
        decode_y(indexes_ptr, indexSize);
        return;
    }
    auto vec_indexes = std::make_shared<std::vector<uint8_t>>(indexSize);
    std::copy(indexes_ptr, indexes_ptr + indexSize, vec_indexes->data());

    decode_y(vec_indexes->data(), indexSize, vec_indexes);
}

void RansDecoder::decode_y_borrowed(
    const py::array_t<uint8_t>& indexes)
{
    py::buffer_info indexes_buf = indexes.request();
    auto* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);
    decode_y(
        indexes_ptr,
        static_cast<int>(indexes.size()));
}

void RansDecoder::decode_y_skip(const uint8_t* indexes, const int indexSize,
                                const int skipIndexCutoff,
                                const std::shared_ptr<void>& owner)
{
    validate_y_skip_cutoff(skipIndexCutoff);
    m_current_decoded_tensor_size = indexSize;
    if (m_decoded_tensor == nullptr || static_cast<int>(m_decoded_tensor->size()) < indexSize) {
        m_decoded_tensor = std::make_shared<PinnedHostBuffer<int16_t>>(indexSize * 2);
    }
    int16_t* decoded_ptr = m_decoded_tensor->data();

    int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_decoders[0]->decode_y_skipped_sync(
            decoded_ptr, indexes, indexSize, 0, skipIndexCutoff);
        return;
    }
    int size0 = indexSize / n;
    for (int i = 0; i < n - 1; i++) {
        m_decoders[i]->decode_y_skipped(
            decoded_ptr, indexes, size0, size0 * i, skipIndexCutoff, owner);
    }
    m_decoders[n - 1]->decode_y_skipped(
        decoded_ptr, indexes, indexSize - size0 * (n - 1), size0 * (n - 1),
        skipIndexCutoff, owner);
}

void RansDecoder::decode_y_skip(const py::array_t<uint8_t>& indexes,
                                const int skipIndexCutoff)
{
    py::buffer_info indexes_buf = indexes.request();
    uint8_t* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);
    int indexSize = static_cast<int>(indexes.size());
    if (m_entropy_coder_parallel == 1) {
        decode_y_skip(indexes_ptr, indexSize, skipIndexCutoff);
        return;
    }
    auto vec_indexes = std::make_shared<std::vector<uint8_t>>(indexSize);
    std::copy(indexes_ptr, indexes_ptr + indexSize, vec_indexes->data());
    decode_y_skip(
        vec_indexes->data(), indexSize, skipIndexCutoff, vec_indexes);
}

void RansDecoder::decode_y_skip_borrowed(
    const py::array_t<uint8_t>& indexes,
    const int skipIndexCutoff)
{
    py::buffer_info indexes_buf = indexes.request();
    auto* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);
    decode_y_skip(
        indexes_ptr,
        static_cast<int>(indexes.size()),
        skipIndexCutoff);
}

std::shared_ptr<PinnedHostBuffer<int16_t>> RansDecoder::decode_and_get_y(
    const uint8_t* indexes, const int indexSize, const std::shared_ptr<void>& owner)
{
    decode_y(indexes, indexSize, owner);
    return get_decoded_tensor_cpp();
}

py::array_t<int16_t> RansDecoder::decode_and_get_y(const py::array_t<uint8_t>& indexes)
{
    decode_y(indexes);
    return get_decoded_tensor();
}

py::array_t<int16_t> RansDecoder::decode_and_get_y_borrowed(
    const py::array_t<uint8_t>& indexes)
{
    py::buffer_info indexes_buf = indexes.request();
    auto* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);
    decode_y(
        indexes_ptr,
        static_cast<int>(indexes.size()));
    return get_decoded_tensor();
}

std::shared_ptr<PinnedHostBuffer<int16_t>> RansDecoder::decode_and_get_y_skip(
    const uint8_t* indexes, const int indexSize, const int skipIndexCutoff,
    const std::shared_ptr<void>& owner)
{
    decode_y_skip(indexes, indexSize, skipIndexCutoff, owner);
    return get_decoded_tensor_cpp();
}

py::array_t<int16_t> RansDecoder::decode_and_get_y_skip(
    const py::array_t<uint8_t>& indexes, const int skipIndexCutoff)
{
    decode_y_skip(indexes, skipIndexCutoff);
    return get_decoded_tensor();
}

py::array_t<int16_t> RansDecoder::decode_and_get_y_skip_borrowed(
    const py::array_t<uint8_t>& indexes,
    const int skipIndexCutoff)
{
    py::buffer_info indexes_buf = indexes.request();
    auto* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);
    decode_y_skip(
        indexes_ptr,
        static_cast<int>(indexes.size()),
        skipIndexCutoff);
    return get_decoded_tensor();
}

void RansDecoder::decode_z(const int total_size, const int cdf_offset, const int ch)
{
    if (total_size < 0) {
        throw std::runtime_error("rANS total_size must be non-negative");
    }
    validate_z_args(cdf_offset, ch);

    m_current_decoded_tensor_size = total_size;
    if (m_decoded_tensor == nullptr || static_cast<int>(m_decoded_tensor->size()) < total_size) {
        m_decoded_tensor = std::make_shared<PinnedHostBuffer<int16_t>>(total_size * 2);
    }
    int16_t* decoded_ptr = m_decoded_tensor->data();

    int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_decoders[0]->decode_z_sync(
            decoded_ptr, total_size, 0, cdf_offset, ch);
        return;
    }
    int size0 = total_size / n;
    for (int i = 0; i < n - 1; i++) {
        m_decoders[i]->decode_z(decoded_ptr, size0, size0 * i, cdf_offset, ch);
    }
    m_decoders[n - 1]->decode_z(decoded_ptr, total_size - size0 * (n - 1), size0 * (n - 1),
                                cdf_offset, ch);
}

void RansDecoder::decode_z_meta_prior(
    const int total_size,
    const py::array_t<uint8_t>& meta_prior_indexes,
    const int ch)
{
    if (total_size < 0) {
        throw std::runtime_error("rANS total_size must be non-negative");
    }
    validate_z_args(0, ch);
    py::buffer_info indexes_buf = meta_prior_indexes.request();
    auto* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);
    const int positionCount = static_cast<int>(meta_prior_indexes.size());
    if (total_size != positionCount * ch) {
        throw std::runtime_error(
            "Meta Prior symbol count must equal position count times channels");
    }

    m_current_decoded_tensor_size = total_size;
    if (m_decoded_tensor == nullptr
        || static_cast<int>(m_decoded_tensor->size()) < total_size) {
        m_decoded_tensor =
            std::make_shared<PinnedHostBuffer<int16_t>>(total_size * 2);
    }
    int16_t* decoded_ptr = m_decoded_tensor->data();
    const int n = m_entropy_coder_parallel;
    if (n == 1) {
        m_decoders[0]->decode_z_meta_prior_sync(
            decoded_ptr,
            indexes_ptr,
            total_size,
            0,
            ch);
        return;
    }
    auto indexes = std::make_shared<std::vector<uint8_t>>(
        indexes_ptr,
        indexes_ptr + positionCount);
    const int size0 = total_size / n;
    for (int i = 0; i < n - 1; ++i) {
        m_decoders[i]->decode_z_meta_prior(
            decoded_ptr,
            indexes->data(),
            size0,
            size0 * i,
            ch,
            indexes);
    }
    m_decoders[n - 1]->decode_z_meta_prior(
        decoded_ptr,
        indexes->data(),
        total_size - size0 * (n - 1),
        size0 * (n - 1),
        ch,
        indexes);
}

std::shared_ptr<PinnedHostBuffer<int16_t>> RansDecoder::get_decoded_tensor_cpp()
{
    int n = m_entropy_coder_parallel;
    for (int i = 0; i < n; i++) {
        m_decoders[i]->wait_for_decoding_finish();
    }
    return m_decoded_tensor;
}

py::array_t<int16_t> RansDecoder::get_decoded_tensor()
{
    auto decoded = get_decoded_tensor_cpp();
    // python will not delete the memory, C++ will do it.
    // the actual data is stored in m_decoded_tensor
    return py::array_t<int16_t>(m_current_decoded_tensor_size, decoded->data());
}

void RansDecoder::set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                          const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index)
{
    int cdf_num = validate_cdf_inputs(cdfs, cdfs_sizes, index);

    int per_vector_size = static_cast<int>(cdfs->size() / cdf_num);
    auto vec_cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(cdf_num);
    auto max_value = std::make_shared<std::vector<int8_t>>(cdf_num);
    for (int i = 0; i < cdf_num; i++) {
        max_value->at(i) = static_cast<int8_t>(cdfs_sizes->at(i) - 2);

        std::vector<int32_t> t(per_vector_size);
        std::copy(cdfs->data() + i * per_vector_size,
                  cdfs->data() + i * per_vector_size + per_vector_size, t.data());
        vec_cdfs->at(i) = std::move(t);
    }

    for (const auto& decoder : m_decoders) {
        decoder->set_cdf(vec_cdfs, max_value, index);
    }
    m_has_cdf = true;
}

void RansDecoder::set_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                          const int index)
{
    py::buffer_info cdfs_buf = cdfs.request();
    py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
    int32_t* cdfs_ptr = static_cast<int32_t*>(cdfs_buf.ptr);
    int32_t* cdfs_sizes_ptr = static_cast<int32_t*>(cdfs_sizes_buf.ptr);

    auto vec_cdfs = std::make_shared<std::vector<int32_t>>(cdfs.size());
    std::copy(cdfs_ptr, cdfs_ptr + cdfs.size(), vec_cdfs->data());
    auto vec_cdfs_sizes = std::make_shared<std::vector<int32_t>>(cdfs_sizes.size());
    std::copy(cdfs_sizes_ptr, cdfs_sizes_ptr + cdfs_sizes.size(), vec_cdfs_sizes->data());

    set_cdf(vec_cdfs, vec_cdfs_sizes, index);
}

void RansDecoder::set_entropy_coder_parallel(int n)
{
    validate_entropy_coder_parallel(n);
    if (n > static_cast<int>(m_decoders.size())) {
        if (m_has_cdf) {
            throw std::runtime_error(
                "set rANS lane count before registering CDF tables");
        }
        while (static_cast<int>(m_decoders.size()) < n) {
            m_decoders.push_back(std::make_shared<RansDecoderLib>());
        }
    }
    m_entropy_coder_parallel = n;
}

int RansDecoder::get_entropy_coder_parallel()
{
    return m_entropy_coder_parallel;
}

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float>& pmf)
{
    /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
     * although it's only run once per model after training. See TF/compression
     * implementation for an optimized version. */
    constexpr int precision = 16;
    constexpr uint32_t prob_max = (1u << precision);
    constexpr int min_freq = 1;

    std::vector<uint32_t> cdf(pmf.size() + 1);
    cdf[0] = 0; /* freq 0 */

    std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                   [=](float p) { return static_cast<uint32_t>(p * prob_max + 0.5); });

    const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);

    std::transform(cdf.begin(), cdf.end(), cdf.begin(), [=](uint32_t p) {
        return static_cast<uint32_t>(((static_cast<uint64_t>(prob_max) * p) / total));
    });

    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
    cdf.back() = prob_max;

    for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
        if (cdf[i] + min_freq > cdf[i + 1]) {
            /* Try to steal frequency from low-frequency symbols */
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
                uint32_t freq = cdf[j + 1] - cdf[j];
                if (freq >= min_freq * 2 && freq < best_freq) {
                    best_freq = freq;
                    best_steal = j;
                }
            }

            assert(best_steal != -1);

            if (best_steal < i) {
                for (int j = best_steal + 1; j <= i; ++j) {
                    cdf[j] -= min_freq;
                }
            } else {
                assert(best_steal > i);
                for (int j = i + 1; j <= best_steal; ++j) {
                    cdf[j] += min_freq;
                }
            }
        }
    }

    assert(cdf[0] == 0);
    assert(cdf.back() == prob_max);
    for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
        assert(cdf[i + 1] > cdf[i]);
    }

    return cdf;
}
