// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rans.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

constexpr int SCALE_BITS = 16;
constexpr int RANS_SHIFT_BITS = 23;
constexpr uint32_t RANS_BYTE_L = 1u << RANS_SHIFT_BITS;
constexpr int ENC_RENORM_SHIFT_BITS = RANS_SHIFT_BITS - SCALE_BITS + 8;
constexpr uint32_t DEC_MASK = (1u << SCALE_BITS) - 1;
constexpr uint16_t BYPASS_PRECISION = 2;  // number of bits in bypass mode
constexpr uint16_t MAX_BYPASS_VAL = (1 << BYPASS_PRECISION) - 1;

#if defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline
#endif

FORCE_INLINE void RansEncInit(RansState& r)
{
    r = RANS_BYTE_L;
}

FORCE_INLINE void RansEncPut(RansState& r, uint8_t*& ptr, uint32_t start, uint32_t freq)
{
    // renormalize
    const uint32_t r_max = freq << ENC_RENORM_SHIFT_BITS;
    while (r >= r_max) {
        // converting to uint8_t will only keep the lowest 8 bits, equal to r & 0xff
        *(--ptr) = static_cast<uint8_t>(r);
        r >>= 8;
    }

    r = ((r / freq) << SCALE_BITS) + (r % freq) + start;
}

FORCE_INLINE void RansEncFlush(const RansState& r, uint8_t*& ptr)
{
    ptr -= 4;
    ptr[0] = static_cast<uint8_t>(r >> 0);
    ptr[1] = static_cast<uint8_t>(r >> 8);
    ptr[2] = static_cast<uint8_t>(r >> 16);
    ptr[3] = static_cast<uint8_t>(r >> 24);
}

FORCE_INLINE void RansDecInit(RansState& r, const uint8_t*& ptr)
{
    r = (*ptr++) << 0;
    r |= (*ptr++) << 8;
    r |= (*ptr++) << 16;
    r |= (*ptr++) << 24;
}

FORCE_INLINE int32_t RansDecGet(RansState& r)
{
    return r & DEC_MASK;
}

FORCE_INLINE void RansDecAdvance(RansState& r, const uint8_t*& ptr, uint32_t start, uint32_t freq, const uint8_t* end)
{
    r = freq * (r >> SCALE_BITS) + (r & DEC_MASK) - start;

    // renormalize
    while (r < RANS_BYTE_L) {
        if (ptr >= end) throw std::runtime_error("truncated rANS payload");
        r = (r << 8) | *ptr++;
    }
}

FORCE_INLINE void RansEncPutBits(RansState& r, uint8_t*& ptr, uint32_t val)
{
    static_assert(BYPASS_PRECISION <= 8);
    assert(val < (1u << BYPASS_PRECISION));

    constexpr uint32_t freq = 1 << (SCALE_BITS - BYPASS_PRECISION);
    constexpr uint32_t x_max = freq << ENC_RENORM_SHIFT_BITS;
    while (r >= x_max) {
        *(--ptr) = static_cast<uint8_t>(r);
        r >>= 8;
    }

    r = (r << BYPASS_PRECISION) | val;
}

FORCE_INLINE uint32_t RansDecGetBits(RansState& r, const uint8_t*& ptr, const uint8_t* end)
{
    uint32_t val = r & ((1u << BYPASS_PRECISION) - 1);

    // renormalize
    r = r >> BYPASS_PRECISION;
    if (r < RANS_BYTE_L) {
        if (ptr >= end) throw std::runtime_error("truncated rANS payload");
        r = (r << 8) | *ptr++;
        if (r < RANS_BYTE_L) throw std::runtime_error("invalid rANS state");
    }

    return val;
}

RansEncoderLib::RansEncoderLib()
{
    _ransSymbols.resize(2);
    _max_value.resize(2);
    _stream_buffer = new uint8_t[m_buffer_size];
    _stream = std::make_shared<std::vector<uint8_t>>();
    m_thread = std::thread(&RansEncoderLib::worker, this);
}

RansEncoderLib::~RansEncoderLib()
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_pending);
        m_finish = true;
    }
    m_cv_pending.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }

    if (_stream_buffer != nullptr) {
        delete[] _stream_buffer;
        _stream_buffer = nullptr;
    }
}

void RansEncoderLib::reserve_output(size_t bytes)
{
    if (static_cast<size_t>(_ptr - _stream_buffer) >= bytes) return;
    const size_t used = _stream_buffer + m_buffer_size - _ptr;
    const size_t capacity = std::max(m_buffer_size * 2, used + bytes);
    if (capacity > 512u * 1024u * 1024u)
        throw std::runtime_error("rANS output exceeds the size limit");
    auto replacement = std::make_unique<uint8_t[]>(capacity);
    std::copy(_ptr, _stream_buffer + m_buffer_size, replacement.get() + capacity - used);
    delete[] _stream_buffer;
    _stream_buffer = replacement.release();
    m_buffer_size = capacity;
    _ptr = _stream_buffer + capacity - used;
}

void RansEncoderLib::set_cdf(const std::shared_ptr<std::vector<std::vector<RansSymbol>>>& ransSymbols,
                             const std::shared_ptr<std::vector<int8_t>>& max_value, const int index)
{
    if (index < 0 || index >= 2) {
        throw std::runtime_error("rANS cdf index must be 0 or 1");
    }
    _ransSymbols[index] = ransSymbols;
    _max_value[index] = max_value;
}

FORCE_INLINE void encode_one_symbol(uint8_t*& ptr, RansState& rans, const int32_t symbol,
                                    const int8_t max_value, const std::vector<RansSymbol>& ransSymbols,
                                    std::vector<uint16_t>& bypassBins)
{
    int32_t value = abs(symbol) * 2 - (symbol > 0);

    if (value >= max_value) {
        const uint32_t raw_val = value - max_value;
        value = max_value;

        bypassBins.clear();
        // Determine the number of bypasses (in BYPASS_PRECISION size)
        // needed to encode the raw value.
        int32_t n_bypass = 0;
        while ((raw_val >> (n_bypass * BYPASS_PRECISION)) != 0) {
            ++n_bypass;
        }

        // Encode number of bypasses
        int32_t val = n_bypass;
        while (val >= MAX_BYPASS_VAL) {
            bypassBins.push_back(MAX_BYPASS_VAL);
            val -= MAX_BYPASS_VAL;
        }
        bypassBins.push_back(static_cast<uint16_t>(val));

        // Encode raw value
        for (int32_t j = 0; j < n_bypass; ++j) {
            const int32_t val1 = (raw_val >> (j * BYPASS_PRECISION)) & MAX_BYPASS_VAL;
            bypassBins.push_back(static_cast<uint16_t>(val1));
        }

        for (auto it = bypassBins.rbegin(); it != bypassBins.rend(); ++it) {
            RansEncPutBits(rans, ptr, *it);
        }
    }
    RansEncPut(rans, ptr, ransSymbols[value].start, ransSymbols[value].range);
}

void RansEncoderLib::encode_y_internal(const int16_t* symbols_ptr, const int symbol_size,
                                       const int symbol_offset)
{
    // backward loop on symbols from the end;
    const int8_t* max_value_ptr = _max_value[1]->data();
    const std::vector<RansSymbol>* ransSymbols_ptr = _ransSymbols[1]->data();
    std::vector<uint16_t> bypassBins;
    bypassBins.reserve(20);
    const int symbol_start = symbol_offset;
    const int symbol_end = symbol_offset + symbol_size - 1;

    for (int i = symbol_end; i >= symbol_start; i--) {
        const int16_t combined_symbol = symbols_ptr[i];
        const int32_t cdf_idx = combined_symbol & 0xff;
        const int32_t s = static_cast<int8_t>(combined_symbol >> 8);
        reserve_output(16);  // upper bound for one int16 symbol, including bypass bits
        encode_one_symbol(_ptr, _rans, s, max_value_ptr[cdf_idx], ransSymbols_ptr[cdf_idx], bypassBins);
    }
}

void RansEncoderLib::encode_y_skipped_internal(const int16_t* symbols_ptr, const int symbol_size,
                                               const int symbol_offset,
                                               const int skip_index_cutoff)
{
    // Preserve the original full-position partitioning and coder order, but
    // omit entries whose decoded-z CDF row is in the deterministic skip set.
    const int8_t* max_value_ptr = _max_value[1]->data();
    const std::vector<RansSymbol>* ransSymbols_ptr = _ransSymbols[1]->data();
    std::vector<uint16_t> bypassBins;
    bypassBins.reserve(20);
    const int symbol_start = symbol_offset;
    const int symbol_end = symbol_offset + symbol_size - 1;

    for (int i = symbol_end; i >= symbol_start; i--) {
        const int16_t combined_symbol = symbols_ptr[i];
        const int32_t cdf_idx = combined_symbol & 0xff;
        if (cdf_idx <= skip_index_cutoff) {
            continue;
        }
        const int32_t s = static_cast<int8_t>(combined_symbol >> 8);
        reserve_output(16);  // upper bound for one int16 symbol, including bypass bits
        encode_one_symbol(_ptr, _rans, s, max_value_ptr[cdf_idx], ransSymbols_ptr[cdf_idx],
                          bypassBins);
    }
}

void RansEncoderLib::encode_z_internal(const int16_t* symbols_ptr, const int symbol_size,
                                       const int symbol_offset, const int cdf_offset, const int ch)
{
    // backward loop on symbols from the end;
    const int8_t* max_value_ptr = _max_value[0]->data();
    const std::vector<RansSymbol>* ransSymbols_ptr = _ransSymbols[0]->data();
    std::vector<uint16_t> bypassBins;
    bypassBins.reserve(20);
    const int symbol_start = symbol_offset;
    const int symbol_end = symbol_offset + symbol_size - 1;

    for (int i = symbol_end; i >= symbol_start; i--) {
        const int32_t cdf_idx = (i % ch) + cdf_offset;
        reserve_output(16);  // upper bound for one int16 symbol, including bypass bits
        encode_one_symbol(_ptr, _rans, symbols_ptr[i], max_value_ptr[cdf_idx],
                          ransSymbols_ptr[cdf_idx], bypassBins);
    }
}

void RansEncoderLib::encode_z_meta_prior_internal(
    const int16_t* symbols_ptr, const uint8_t* meta_prior_indexes,
    const int symbol_size, const int symbol_offset, const int ch)
{
    const int8_t* max_value_ptr = _max_value[0]->data();
    const std::vector<RansSymbol>* ransSymbols_ptr = _ransSymbols[0]->data();
    const int cdf_count = static_cast<int>(_ransSymbols[0]->size());
    std::vector<uint16_t> bypassBins;
    bypassBins.reserve(20);
    const int symbol_start = symbol_offset;
    const int symbol_end = symbol_offset + symbol_size - 1;

    for (int i = symbol_end; i >= symbol_start; --i) {
        const int position = i / ch;
        const int channel = i % ch;
        const int32_t cdf_idx =
            static_cast<int32_t>(meta_prior_indexes[position]) * ch + channel;
        if (cdf_idx < 0 || cdf_idx >= cdf_count) {
            throw std::runtime_error("Meta Prior CDF index is out of range");
        }
        reserve_output(16);  // upper bound for one int16 symbol, including bypass bits
        encode_one_symbol(
            _ptr,
            _rans,
            symbols_ptr[i],
            max_value_ptr[cdf_idx],
            ransSymbols_ptr[cdf_idx],
            bypassBins);
    }
}

void RansEncoderLib::flush_internal()
{
    reserve_output(4);
    RansEncFlush(_rans, _ptr);

    uint8_t* ptrEnd = _stream_buffer + m_buffer_size;
    const int nbytes = static_cast<int>(std::distance(_ptr, ptrEnd));

    if (_ptr < _stream_buffer || static_cast<size_t>(nbytes) > m_buffer_size) {
        throw std::runtime_error("rANS stream buffer overflow: encoded size ("
                                 + std::to_string(nbytes) + ") exceeds buffer capacity ("
                                 + std::to_string(m_buffer_size) + ")");
    }

    _stream->resize(nbytes);
    std::copy(_ptr, _ptr + nbytes, _stream->data());
    {
        std::lock_guard<std::mutex> lk_result(m_mutex_result);
        m_result_ready = true;
    }
    m_cv_result.notify_one();
}

void RansEncoderLib::flush()
{
    PendingTask p;
    p.workType = WorkType::Flush;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansEncoderLib::encode_y_skipped(const int16_t* symbols, const int symbol_size,
                                      const int symbol_offset, const int skip_index_cutoff,
                                      const std::shared_ptr<void>& owner)
{
    PendingTask p;
    p.workType = WorkType::EncodeDecodeYSkipped;
    p.symbols_y = symbols;
    p.owner = owner;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    p.skip_index_cutoff = skip_index_cutoff;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansEncoderLib::encode_y(const int16_t* symbols, const int symbol_size,
                              const int symbol_offset, const std::shared_ptr<void>& owner)
{
    PendingTask p;
    p.workType = WorkType::EncodeDecodeY;
    p.symbols_y = symbols;
    p.owner = owner;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansEncoderLib::encode_z(const int16_t* symbols, const int symbol_size, const int symbol_offset,
                              const int cdf_offset, const int ch, const std::shared_ptr<void>& owner)
{
    PendingTask p;
    p.workType = WorkType::EncodeDecodeZ;
    p.symbols_z = symbols;
    p.owner = owner;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    p.cdf_offset = cdf_offset;
    p.ch = ch;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansEncoderLib::encode_z_meta_prior(
    const int16_t* symbols, const uint8_t* meta_prior_indexes,
    const int symbol_size, const int symbol_offset, const int ch,
    const std::shared_ptr<void>& owner)
{
    PendingTask p;
    p.workType = WorkType::EncodeDecodeZMetaPrior;
    p.symbols_z = symbols;
    p.meta_prior_indexes = meta_prior_indexes;
    p.owner = owner;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    p.ch = ch;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

std::shared_ptr<std::vector<uint8_t>> RansEncoderLib::get_encoded_stream()
{
    std::unique_lock<std::mutex> lk(m_mutex_result);
    m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
    if (m_error) std::rethrow_exception(m_error);
    return _stream;
}

void RansEncoderLib::reset()
{
    _stream->clear();

    RansEncInit(_rans);

    uint8_t* ptrEnd = _stream_buffer + m_buffer_size;
    _ptr = ptrEnd;

    std::lock_guard<std::mutex> lk(m_mutex_result);
    m_result_ready = false;
    m_error = nullptr;
}

void RansEncoderLib::worker()
{
    while (!m_finish) {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_cv_pending.wait(lk, [this] { return !m_pending.empty() || m_finish; });
        if (m_finish) {
            break;
        }
        while (!m_pending.empty()) {
            auto p = std::move(m_pending.front());
            m_pending.pop();
            lk.unlock();
            try {
                { std::lock_guard<std::mutex> lock(m_mutex_result);
                  if (m_error) std::rethrow_exception(m_error); }
                if (p.workType == WorkType::EncodeDecodeY) {
                    RansEncoderLib::encode_y_internal(p.symbols_y, p.symbol_size, p.symbol_offset);
                } else if (p.workType == WorkType::EncodeDecodeYSkipped) {
                    RansEncoderLib::encode_y_skipped_internal(
                        p.symbols_y, p.symbol_size, p.symbol_offset, p.skip_index_cutoff);
                } else if (p.workType == WorkType::EncodeDecodeZ) {
                    RansEncoderLib::encode_z_internal(p.symbols_z, p.symbol_size, p.symbol_offset,
                                                      p.cdf_offset, p.ch);
                } else if (p.workType == WorkType::EncodeDecodeZMetaPrior) {
                    RansEncoderLib::encode_z_meta_prior_internal(
                        p.symbols_z,
                        p.meta_prior_indexes,
                        p.symbol_size,
                        p.symbol_offset,
                        p.ch);
                } else if (p.workType == WorkType::Flush) {
                    RansEncoderLib::flush_internal();
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(m_mutex_result);
                if (!m_error) m_error = std::current_exception();
            }
            if (p.workType == WorkType::Flush) {
                // Successful flush_internal already signalled completion.
                // A second notification can race the next stream's reset.
                std::lock_guard<std::mutex> lock(m_mutex_result);
                if (m_error) {
                    m_result_ready = true;
                    m_cv_result.notify_one();
                }
            }
            lk.lock();
        }
        lk.unlock();
    }
}

RansDecoderLib::RansDecoderLib()
{
    _cdfs.resize(2);
    _max_value.resize(2);
    m_thread = std::thread(&RansDecoderLib::worker, this);
}

RansDecoderLib::~RansDecoderLib()
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_pending);
        m_finish = true;
    }
    m_cv_pending.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void RansDecoderLib::set_stream(const std::shared_ptr<std::vector<uint8_t>>& encoded,
                                const int offset, const int size)
{
    const int stream_size = (size >= 0) ? size : static_cast<int>(encoded->size()) - offset;
    if (offset < 0 || stream_size < MIN_RANS_STREAM_SIZE
        || offset + stream_size > static_cast<int>(encoded->size())) {
        throw std::runtime_error("rANS bitstream is too short to initialize the decoder");
    }

    _stream = encoded;
    { std::lock_guard<std::mutex> lock(m_mutex_result); m_error = nullptr; }
    _ptr8 = _stream->data() + offset;
    _end8 = _ptr8 + stream_size;
    RansDecInit(_rans, _ptr8);
}

void RansDecoderLib::set_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>>& cdfs,
                             const std::shared_ptr<std::vector<int8_t>>& max_value, const int index)
{
    if (index < 0 || index >= 2) {
        throw std::runtime_error("rANS cdf index must be 0 or 1");
    }
    _cdfs[index] = cdfs;
    _max_value[index] = max_value;
}

FORCE_INLINE int16_t decode_one_symbol(const uint8_t*& ptr8, RansState& rans, const int32_t* cdf,
                                       const int8_t max_value, const uint8_t* end)
{
    const int32_t cum_freq = RansDecGet(rans);

    int s = 1;
    while (cdf[s] <= cum_freq) {
        s++;
    }
    s--;

    RansDecAdvance(rans, ptr8, cdf[s], cdf[s + 1] - cdf[s], end);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
        // Bypass decoding mode
        int32_t val = RansDecGetBits(rans, ptr8, end);
        int32_t n_bypass = val;

        while (val == MAX_BYPASS_VAL) {
            val = RansDecGetBits(rans, ptr8, end);
            n_bypass += val;
            if (n_bypass > 9) throw std::runtime_error("invalid rANS bypass length");
        }

        if (n_bypass > 9) throw std::runtime_error("invalid rANS bypass length");
        int32_t raw_val = 0;
        for (int j = 0; j < n_bypass; ++j) {
            val = RansDecGetBits(rans, ptr8, end);
            raw_val |= val << (j * BYPASS_PRECISION);
        }
        value = raw_val + max_value;
        if (value > 65536) throw std::runtime_error("rANS symbol exceeds int16 range");
    }

    return static_cast<int16_t>((value % 2 == 1) ? (value + 1) / 2 : -(value + 1) / 2);
}

void RansDecoderLib::decode_y_internal(int16_t* decoded_ptr, const uint8_t* indexes_ptr,
                                       const int symbol_size, const int symbol_offset)
{
    const int8_t* max_value_ptr = _max_value[1]->data();
    const std::vector<int32_t>* cdfs = _cdfs[1]->data();
    for (int i = 0; i < symbol_size; ++i) {
        const int32_t cdf_idx = indexes_ptr[i + symbol_offset];
        decoded_ptr[i + symbol_offset] =
            decode_one_symbol(_ptr8, _rans, cdfs[cdf_idx].data(), max_value_ptr[cdf_idx], _end8);
    }
}

void RansDecoderLib::decode_y_skipped_internal(int16_t* decoded_ptr,
                                               const uint8_t* indexes_ptr,
                                               const int symbol_size,
                                               const int symbol_offset,
                                               const int skip_index_cutoff)
{
    const int8_t* max_value_ptr = _max_value[1]->data();
    const std::vector<int32_t>* cdfs = _cdfs[1]->data();
    for (int i = 0; i < symbol_size; ++i) {
        const int position = i + symbol_offset;
        const int32_t cdf_idx = indexes_ptr[position];
        if (cdf_idx <= skip_index_cutoff) {
            decoded_ptr[position] = 0;
            continue;
        }
        decoded_ptr[position] =
            decode_one_symbol(_ptr8, _rans, cdfs[cdf_idx].data(), max_value_ptr[cdf_idx], _end8);
    }
}

void RansDecoderLib::decode_z_internal(int16_t* decoded_ptr, const int symbol_size,
                                       const int symbol_offset, const int cdf_offset, const int ch)
{
    const int8_t* max_value_ptr = _max_value[0]->data();
    const std::vector<int32_t>* cdfs = _cdfs[0]->data();

    for (int i = 0; i < symbol_size; ++i) {
        const int32_t cdf_idx = ((i + symbol_offset) % ch) + cdf_offset;
        decoded_ptr[i + symbol_offset] =
            decode_one_symbol(_ptr8, _rans, cdfs[cdf_idx].data(), max_value_ptr[cdf_idx], _end8);
    }
}

void RansDecoderLib::decode_z_meta_prior_internal(
    int16_t* decoded_ptr, const uint8_t* meta_prior_indexes,
    const int symbol_size, const int symbol_offset, const int ch)
{
    const int8_t* max_value_ptr = _max_value[0]->data();
    const std::vector<int32_t>* cdfs = _cdfs[0]->data();
    const int cdf_count = static_cast<int>(_cdfs[0]->size());

    for (int i = 0; i < symbol_size; ++i) {
        const int global_index = i + symbol_offset;
        const int position = global_index / ch;
        const int channel = global_index % ch;
        const int32_t cdf_idx =
            static_cast<int32_t>(meta_prior_indexes[position]) * ch + channel;
        if (cdf_idx < 0 || cdf_idx >= cdf_count) {
            throw std::runtime_error("Meta Prior CDF index is out of range");
        }
        decoded_ptr[global_index] = decode_one_symbol(
            _ptr8,
            _rans,
            cdfs[cdf_idx].data(),
            max_value_ptr[cdf_idx], _end8);
    }
}

void RansDecoderLib::decode_y(int16_t* decoded_ptr, const uint8_t* indexes, const int symbol_size,
                              const int symbol_offset, const std::shared_ptr<void>& owner)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    PendingTask p;
    p.workType = WorkType::EncodeDecodeY;
    p.indexes = indexes;
    p.decoded_ptr = decoded_ptr;
    p.owner = owner;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansDecoderLib::decode_y_sync(int16_t* decoded_ptr, const uint8_t* indexes,
                                   const int symbol_size, const int symbol_offset)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    decode_y_internal(decoded_ptr, indexes, symbol_size, symbol_offset);
    {
        std::lock_guard<std::mutex> lk_result(m_mutex_result);
        m_result_ready = true;
    }
    m_cv_result.notify_one();
}

void RansDecoderLib::decode_y_skipped(int16_t* decoded_ptr, const uint8_t* indexes,
                                      const int symbol_size, const int symbol_offset,
                                      const int skip_index_cutoff,
                                      const std::shared_ptr<void>& owner)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    PendingTask p;
    p.workType = WorkType::EncodeDecodeYSkipped;
    p.indexes = indexes;
    p.decoded_ptr = decoded_ptr;
    p.owner = owner;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    p.skip_index_cutoff = skip_index_cutoff;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansDecoderLib::decode_y_skipped_sync(int16_t* decoded_ptr, const uint8_t* indexes,
                                           const int symbol_size, const int symbol_offset,
                                           const int skip_index_cutoff)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    decode_y_skipped_internal(
        decoded_ptr, indexes, symbol_size, symbol_offset, skip_index_cutoff);
    {
        std::lock_guard<std::mutex> lk_result(m_mutex_result);
        m_result_ready = true;
    }
    m_cv_result.notify_one();
}

void RansDecoderLib::decode_z(int16_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                              const int cdf_offset, const int ch)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    PendingTask p;
    p.workType = WorkType::EncodeDecodeZ;
    p.decoded_ptr = decoded_ptr;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    p.cdf_offset = cdf_offset;
    p.ch = ch;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansDecoderLib::decode_z_sync(int16_t* decoded_ptr, const int symbol_size,
                                   const int symbol_offset, const int cdf_offset, const int ch)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    decode_z_internal(decoded_ptr, symbol_size, symbol_offset, cdf_offset, ch);
    {
        std::lock_guard<std::mutex> lk_result(m_mutex_result);
        m_result_ready = true;
    }
    m_cv_result.notify_one();
}

void RansDecoderLib::decode_z_meta_prior(
    int16_t* decoded_ptr, const uint8_t* meta_prior_indexes,
    const int symbol_size, const int symbol_offset, const int ch,
    const std::shared_ptr<void>& owner)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    PendingTask p;
    p.workType = WorkType::EncodeDecodeZMetaPrior;
    p.decoded_ptr = decoded_ptr;
    p.meta_prior_indexes = meta_prior_indexes;
    p.owner = owner;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    p.ch = ch;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansDecoderLib::decode_z_meta_prior_sync(
    int16_t* decoded_ptr, const uint8_t* meta_prior_indexes,
    const int symbol_size, const int symbol_offset, const int ch)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    decode_z_meta_prior_internal(
        decoded_ptr,
        meta_prior_indexes,
        symbol_size,
        symbol_offset,
        ch);
    {
        std::lock_guard<std::mutex> lk_result(m_mutex_result);
        m_result_ready = true;
    }
    m_cv_result.notify_one();
}

bool RansDecoderLib::wait_for_decoding_finish()
{
    std::unique_lock<std::mutex> lk(m_mutex_result);
    m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
    if (m_error) std::rethrow_exception(m_error);
    return true;
}

void RansDecoderLib::worker()
{
    while (!m_finish) {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_cv_pending.wait(lk, [this] { return !m_pending.empty() || m_finish; });
        if (m_finish) {
            break;
        }
        while (!m_pending.empty()) {
            auto p = std::move(m_pending.front());
            m_pending.pop();
            lk.unlock();
            try {
                { std::lock_guard<std::mutex> lock(m_mutex_result);
                  if (m_error) std::rethrow_exception(m_error); }
                if (p.workType == WorkType::EncodeDecodeY) {
                    decode_y_internal(p.decoded_ptr, p.indexes, p.symbol_size, p.symbol_offset);
                } else if (p.workType == WorkType::EncodeDecodeYSkipped) {
                    decode_y_skipped_internal(p.decoded_ptr, p.indexes, p.symbol_size,
                                              p.symbol_offset, p.skip_index_cutoff);
                } else if (p.workType == WorkType::EncodeDecodeZ) {
                    decode_z_internal(p.decoded_ptr, p.symbol_size, p.symbol_offset, p.cdf_offset, p.ch);
                } else if (p.workType == WorkType::EncodeDecodeZMetaPrior) {
                    decode_z_meta_prior_internal(
                        p.decoded_ptr,
                        p.meta_prior_indexes,
                        p.symbol_size,
                        p.symbol_offset,
                        p.ch);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(m_mutex_result);
                if (!m_error) m_error = std::current_exception();
            }
            { std::lock_guard<std::mutex> lock(m_mutex_result); m_result_ready = true; }
            m_cv_result.notify_one();
            lk.lock();
        }
        lk.unlock();
    }
}
