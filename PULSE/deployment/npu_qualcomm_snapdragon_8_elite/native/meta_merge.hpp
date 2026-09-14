// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint16_t kProbabilityTotal = 2048;
constexpr uint16_t kProbabilityHalf = 1024;

struct Layout {
    int index_bits{};
    size_t tree_nodes{};
    size_t merge_left_offset{};
    size_t merge_up_offset{};
    size_t escape_offset{};
    size_t total_probabilities{};
};

Layout make_layout(const int index_bits)
{
    if (index_bits < 1 || index_bits > 8) {
        throw std::runtime_error("Meta Prior index_bits must be in [1, 8]");
    }
    const size_t tree_nodes =
        (static_cast<size_t>(1) << index_bits) - 1;
    return Layout{
        index_bits,
        tree_nodes,
        0,
        2,
        4,
        4 + tree_nodes * 4,
    };
}

void validate_shape(
    const int batch,
    const int height,
    const int width,
    const size_t count)
{
    if (batch <= 0 || height <= 0 || width <= 0) {
        throw std::runtime_error(
            "Meta Prior index map dimensions must be positive");
    }
    const uint64_t expected =
        static_cast<uint64_t>(batch)
        * static_cast<uint64_t>(height)
        * static_cast<uint64_t>(width);
    if (expected > std::numeric_limits<size_t>::max()
        || static_cast<size_t>(expected) != count) {
        throw std::runtime_error(
            "Meta Prior index map size does not match BxHxW");
    }
}

std::vector<uint16_t> load_probabilities(const std::vector<uint16_t>& source, const Layout& layout) {
    if (source.size() != layout.total_probabilities) throw std::runtime_error("invalid probability table size");
    for (auto p : source) if (p == 0 || p >= kProbabilityTotal) throw std::runtime_error("invalid probability");
    return source;
}

size_t position_of(
    const int batch_index,
    const int h,
    const int w,
    const int height,
    const int width)
{
    return (
        (static_cast<size_t>(batch_index) * height + h) * width + w
    );
}

int neighbor_bits(
    const uint8_t* indexes,
    const int batch_index,
    const int h,
    const int w,
    const int height,
    const int width,
    const int bit_position)
{
    int context = 0;
    if (w > 0) {
        const uint8_t left = indexes[
            position_of(batch_index, h, w - 1, height, width)];
        context |= ((left >> bit_position) & 1U) != 0 ? 1 : 0;
    }
    if (h > 0) {
        const uint8_t up = indexes[
            position_of(batch_index, h - 1, w, height, width)];
        context |= ((up >> bit_position) & 1U) != 0 ? 2 : 0;
    }
    return context;
}

size_t tree_node(const int level, const int prefix)
{
    return (
        (static_cast<size_t>(1) << level) - 1
        + static_cast<size_t>(prefix)
    );
}

void update_probability(
    uint16_t& probability,
    const bool bit,
    const int adaptation_shift)
{
    if (adaptation_shift <= 0) {
        return;
    }
    if (!bit) {
        probability = static_cast<uint16_t>(
            probability
            + ((kProbabilityTotal - probability) >> adaptation_shift));
    } else {
        probability = static_cast<uint16_t>(
            probability - (probability >> adaptation_shift));
    }
    probability = std::clamp<uint16_t>(
        probability,
        1,
        kProbabilityTotal - 1);
}

class BitWriter {
public:
    void write(const bool bit)
    {
        current_ = static_cast<uint8_t>(
            (current_ << 1) | (bit ? 1 : 0));
        ++used_;
        if (used_ == 8) {
            bytes_.push_back(current_);
            current_ = 0;
            used_ = 0;
        }
    }

    std::vector<uint8_t> finish()
    {
        if (used_ != 0) {
            bytes_.push_back(
                static_cast<uint8_t>(current_ << (8 - used_)));
        }
        return bytes_;
    }

private:
    uint8_t current_{};
    int used_{};
    std::vector<uint8_t> bytes_;
};

class BitReader {
public:
    explicit BitReader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}

    bool read()
    {
        if (byte_ >= bytes_.size()) {
            return false;
        }
        const bool value =
            ((bytes_[byte_] >> (7 - bit_)) & 1U) != 0;
        ++bit_;
        if (bit_ == 8) {
            bit_ = 0;
            ++byte_;
        }
        return value;
    }

private:
    const std::vector<uint8_t>& bytes_;
    size_t byte_{};
    int bit_{};
};

class ArithmeticEncoder {
public:
    void encode(
        uint16_t& probability,
        const bool bit,
        const int adaptation_shift)
    {
        const uint64_t range = high_ - low_ + 1;
        const uint64_t zero =
            range * static_cast<uint64_t>(probability)
            / kProbabilityTotal;
        if (zero == 0 || zero >= range) {
            throw std::runtime_error("arithmetic interval collapsed");
        }
        const uint64_t split = low_ + zero;
        if (!bit) {
            high_ = split - 1;
        } else {
            low_ = split;
        }
        update_probability(probability, bit, adaptation_shift);
        normalize();
    }

    std::vector<uint8_t> finish()
    {
        ++pending_;
        emit(low_ < kQuarter ? false : true);
        return writer_.finish();
    }

private:
    static constexpr uint64_t kMask =
        (static_cast<uint64_t>(1) << 32) - 1;
    static constexpr uint64_t kHalf =
        static_cast<uint64_t>(1) << 31;
    static constexpr uint64_t kQuarter =
        static_cast<uint64_t>(1) << 30;
    static constexpr uint64_t kThreeQuarter =
        static_cast<uint64_t>(3) << 30;

    void emit(const bool bit)
    {
        writer_.write(bit);
        while (pending_ != 0) {
            writer_.write(!bit);
            --pending_;
        }
    }

    void normalize()
    {
        while (true) {
            if (high_ < kHalf) {
                emit(false);
            } else if (low_ >= kHalf) {
                emit(true);
                low_ -= kHalf;
                high_ -= kHalf;
            } else if (
                low_ >= kQuarter && high_ < kThreeQuarter
            ) {
                ++pending_;
                low_ -= kQuarter;
                high_ -= kQuarter;
            } else {
                break;
            }
            low_ = (low_ << 1) & kMask;
            high_ = ((high_ << 1) | 1U) & kMask;
        }
    }

    uint64_t low_{};
    uint64_t high_{kMask};
    uint64_t pending_{};
    BitWriter writer_;
};

class ArithmeticDecoder {
public:
    explicit ArithmeticDecoder(const std::vector<uint8_t>& bytes)
        : reader_(bytes)
    {
        for (int i = 0; i < 32; ++i) {
            code_ = (code_ << 1) | (reader_.read() ? 1U : 0U);
        }
    }

    bool decode(
        uint16_t& probability,
        const int adaptation_shift)
    {
        const uint64_t range = high_ - low_ + 1;
        const uint64_t zero =
            range * static_cast<uint64_t>(probability)
            / kProbabilityTotal;
        if (zero == 0 || zero >= range) {
            throw std::runtime_error("arithmetic interval collapsed");
        }
        const uint64_t split = low_ + zero;
        const bool bit = code_ >= split;
        if (!bit) {
            high_ = split - 1;
        } else {
            low_ = split;
        }
        update_probability(probability, bit, adaptation_shift);
        normalize();
        return bit;
    }

private:
    static constexpr uint64_t kMask =
        (static_cast<uint64_t>(1) << 32) - 1;
    static constexpr uint64_t kHalf =
        static_cast<uint64_t>(1) << 31;
    static constexpr uint64_t kQuarter =
        static_cast<uint64_t>(1) << 30;
    static constexpr uint64_t kThreeQuarter =
        static_cast<uint64_t>(3) << 30;

    void normalize()
    {
        while (true) {
            if (high_ < kHalf) {
                // No offset change.
            } else if (low_ >= kHalf) {
                low_ -= kHalf;
                high_ -= kHalf;
                code_ -= kHalf;
            } else if (
                low_ >= kQuarter && high_ < kThreeQuarter
            ) {
                low_ -= kQuarter;
                high_ -= kQuarter;
                code_ -= kQuarter;
            } else {
                break;
            }
            low_ = (low_ << 1) & kMask;
            high_ = ((high_ << 1) | 1U) & kMask;
            code_ =
                ((code_ << 1) | (reader_.read() ? 1U : 0U)) & kMask;
        }
    }

    BitReader reader_;
    uint64_t low_{};
    uint64_t high_{kMask};
    uint64_t code_{};
};

void encode_index(
    ArithmeticEncoder& coder,
    std::vector<uint16_t>& probabilities,
    const Layout& layout,
    const uint8_t value,
    const uint8_t* indexes,
    const int batch_index,
    const int h,
    const int w,
    const int height,
    const int width,
    const int adaptation_shift)
{
    int prefix = 0;
    for (int level = 0; level < layout.index_bits; ++level) {
        const int bit_position = layout.index_bits - 1 - level;
        const size_t node = tree_node(level, prefix);
        const int context = neighbor_bits(
            indexes,
            batch_index,
            h,
            w,
            height,
            width,
            bit_position);
        const size_t probability_index =
            layout.escape_offset + node * 4
            + static_cast<size_t>(context);
        const bool bit = ((value >> bit_position) & 1U) != 0;
        coder.encode(
            probabilities[probability_index],
            bit,
            adaptation_shift);
        prefix = (prefix << 1) | (bit ? 1 : 0);
    }
}

uint8_t decode_index(
    ArithmeticDecoder& coder,
    std::vector<uint16_t>& probabilities,
    const Layout& layout,
    const uint8_t* indexes,
    const int batch_index,
    const int h,
    const int w,
    const int height,
    const int width,
    const int adaptation_shift)
{
    int prefix = 0;
    uint8_t value = 0;
    for (int level = 0; level < layout.index_bits; ++level) {
        const int bit_position = layout.index_bits - 1 - level;
        const size_t node = tree_node(level, prefix);
        const int context = neighbor_bits(
            indexes,
            batch_index,
            h,
            w,
            height,
            width,
            bit_position);
        const size_t probability_index =
            layout.escape_offset + node * 4
            + static_cast<size_t>(context);
        const bool bit = coder.decode(
            probabilities[probability_index],
            adaptation_shift);
        value = static_cast<uint8_t>(
            value
            | (static_cast<uint8_t>(bit ? 1 : 0) << bit_position));
        prefix = (prefix << 1) | (bit ? 1 : 0);
    }
    return value;
}

void validate_adaptation_shift(const int adaptation_shift)
{
    if (adaptation_shift < 0 || adaptation_shift > 15) {
        throw std::runtime_error(
            "Meta Prior adaptation_shift must be in [0, 15]");
    }
}

}  // namespace

inline std::vector<uint8_t> encode_meta_prior_index_merge(
    const std::vector<uint8_t>& indexes, int batch, int height, int width, int index_bits,
    const std::vector<uint16_t>& initial_probabilities, int adaptation_shift)
{
    validate_adaptation_shift(adaptation_shift);
    const Layout layout = make_layout(index_bits);
    const size_t count = indexes.size();
    validate_shape(batch, height, width, count);
    const auto* values = indexes.data();
    const uint16_t limit = static_cast<uint16_t>(
        static_cast<uint16_t>(1U) << index_bits);
    for (size_t i = 0; i < count; ++i) {
        if (values[i] >= limit) {
            throw std::runtime_error(
                "Meta Prior index exceeds configured bit width");
        }
    }

    std::vector<uint16_t> probabilities =
        load_probabilities(initial_probabilities, layout);
    ArithmeticEncoder coder;
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const size_t position = position_of(
                    batch_index,
                    h,
                    w,
                    height,
                    width);
                const uint8_t value = values[position];
                bool merged = false;
                if (w > 0) {
                    const uint8_t left = values[position - 1];
                    const bool up_matches_left =
                        h > 0
                        && values[position - static_cast<size_t>(width)]
                            == left;
                    const bool same_left = value == left;
                    coder.encode(
                        probabilities[
                            layout.merge_left_offset
                            + (up_matches_left ? 1 : 0)],
                        same_left,
                        adaptation_shift);
                    merged = same_left;
                }
                if (!merged && h > 0) {
                    const uint8_t up =
                        values[position - static_cast<size_t>(width)];
                    const bool distinct_left =
                        w > 0 && values[position - 1] != up;
                    if (w == 0 || distinct_left) {
                        const bool same_up = value == up;
                        coder.encode(
                            probabilities[
                                layout.merge_up_offset
                                + (distinct_left ? 1 : 0)],
                            same_up,
                            adaptation_shift);
                        merged = same_up;
                    }
                }
                if (!merged) {
                    encode_index(
                        coder,
                        probabilities,
                        layout,
                        value,
                        values,
                        batch_index,
                        h,
                        w,
                        height,
                        width,
                        adaptation_shift);
                }
            }
        }
    }
    const std::vector<uint8_t> payload = coder.finish();
    return payload;
}

inline std::vector<uint8_t> decode_meta_prior_index_merge(
    const std::vector<uint8_t>& payload, int batch, int height, int width, int index_bits,
    int bank_count, const std::vector<uint16_t>& initial_probabilities, int adaptation_shift)
{
    validate_adaptation_shift(adaptation_shift);
    const Layout layout = make_layout(index_bits);
    if (
        bank_count <= 1
        || bank_count > (1 << index_bits)
        || bank_count <= (1 << (index_bits - 1))
    ) {
        throw std::runtime_error(
            "bank_count does not match Meta Prior index bit width");
    }
    const uint64_t count64 =
        static_cast<uint64_t>(batch)
        * static_cast<uint64_t>(height)
        * static_cast<uint64_t>(width);
    if (
        batch <= 0
        || height <= 0
        || width <= 0
        || count64 > std::numeric_limits<int32_t>::max()
    ) {
        throw std::runtime_error("invalid Meta Prior index map dimensions");
    }
    const size_t count = static_cast<size_t>(count64);

    const auto& encoded = payload;
    if (encoded.empty()) {
        throw std::runtime_error("empty Meta Prior index-merge payload");
    }
    const std::vector<uint8_t> bytes(encoded.begin(), encoded.end());
    std::vector<uint16_t> probabilities =
        load_probabilities(initial_probabilities, layout);

    std::vector<uint8_t> output(count);
    auto* values = output.data();
    std::fill(values, values + count, 0);
    ArithmeticDecoder coder(bytes);
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const size_t position = position_of(
                    batch_index,
                    h,
                    w,
                    height,
                    width);
                uint8_t value = 0;
                bool merged = false;
                if (w > 0) {
                    const uint8_t left = values[position - 1];
                    const bool up_matches_left =
                        h > 0
                        && values[position - static_cast<size_t>(width)]
                            == left;
                    const bool same_left = coder.decode(
                        probabilities[
                            layout.merge_left_offset
                            + (up_matches_left ? 1 : 0)],
                        adaptation_shift);
                    if (same_left) {
                        value = left;
                        merged = true;
                    }
                }
                if (!merged && h > 0) {
                    const uint8_t up =
                        values[position - static_cast<size_t>(width)];
                    const bool distinct_left =
                        w > 0 && values[position - 1] != up;
                    if (w == 0 || distinct_left) {
                        const bool same_up = coder.decode(
                            probabilities[
                                layout.merge_up_offset
                                + (distinct_left ? 1 : 0)],
                            adaptation_shift);
                        if (same_up) {
                            value = up;
                            merged = true;
                        }
                    }
                }
                if (!merged) {
                    value = decode_index(
                        coder,
                        probabilities,
                        layout,
                        values,
                        batch_index,
                        h,
                        w,
                        height,
                        width,
                        adaptation_shift);
                }
                if (value >= bank_count) {
                    throw std::runtime_error(
                        "decoded Meta Prior index exceeds bank_count");
                }
                values[position] = value;
            }
        }
    }
    return output;
}
