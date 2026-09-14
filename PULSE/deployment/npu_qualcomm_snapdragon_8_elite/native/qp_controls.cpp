#include "qp_controls.hpp"

#include "qnn_runner.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace pulse_mobile {
namespace {

constexpr size_t kHeaderBytes = 32;
constexpr size_t kScalarCount = 6;
constexpr char kMagic[] = "PQPCTRL1";

uint16_t read_u16_le(const uint8_t* data)
{
    return static_cast<uint16_t>(data[0])
        | static_cast<uint16_t>(data[1]) << 8;
}

uint32_t read_u32_le(const uint8_t* data)
{
    return static_cast<uint32_t>(data[0])
        | static_cast<uint32_t>(data[1]) << 8
        | static_cast<uint32_t>(data[2]) << 16
        | static_cast<uint32_t>(data[3]) << 24;
}

std::vector<uint8_t> read_file(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open QP controls " + path);
    }
    return {
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()};
}

}  // namespace

QpControls::QpControls(const std::string& path)
{
    const std::vector<uint8_t> data = read_file(path);
    if (data.size() < kHeaderBytes) throw std::runtime_error("truncated QP controls");
    dico_channels_ = read_u32_le(data.data()+16);
    render_channels_ = read_u32_le(data.data()+20);
    latent_height_ = read_u32_le(data.data()+24);
    latent_width_ = read_u32_le(data.data()+28);
    if (!latent_height_ || !latent_width_ || latent_height_%4 || latent_width_%4 ||
        !((dico_channels_==96 && render_channels_==32) ||
          (dico_channels_==144 && render_channels_==48) ||
          (dico_channels_==288 && render_channels_==96)))
        throw std::runtime_error("unsupported QP control dimensions");
    const size_t expected = kHeaderBytes + kQpCount * (kScalarCount + dico_channels_ + render_channels_) * 2;
    if (data.size() != expected
        || !std::equal(
            std::begin(kMagic),
            std::end(kMagic) - 1,
            data.begin())
        || read_u32_le(data.data() + 8) != 1
        || read_u32_le(data.data() + 12) != kQpCount
        || read_u32_le(data.data() + 16) != dico_channels_
        || read_u32_le(data.data() + 20) != render_channels_
        || read_u32_le(data.data() + 24) != latent_height_
        || read_u32_le(data.data() + 28) != latent_width_) {
        throw std::runtime_error("QP controls do not match the mobile graph");
    }

    const uint8_t* cursor = data.data() + kHeaderBytes;
    const auto next = [&cursor]() {
        const uint16_t value = read_u16_le(cursor);
        cursor += sizeof(uint16_t);
        return value;
    };
    for (Record& item : records_) {
        item.q_dico.resize(dico_channels_);
        item.q_render.resize(render_channels_);
        item.q_image = next();
        item.q_hyper_z = next();
        item.q_hyper_m = next();
        item.q_prior = next();
        item.q_basic = next();
        item.q_codec_dec = next();
        for (uint16_t& value : item.q_dico) {
            value = next();
        }
        for (uint16_t& value : item.q_render) {
            value = next();
        }
    }
}

const QpControls::Record& QpControls::record(const int qp) const
{
    if (qp < 0 || qp >= kQpCount) {
        throw std::runtime_error("QP must be in [0, 7]");
    }
    return records_[static_cast<size_t>(qp)];
}

void QpControls::write_scalar(
    QnnRunner& runner,
    const char* name,
    const uint16_t value)
{
    write_array(runner, name, &value, 1);
}

void QpControls::write_repeated(
    QnnRunner& runner,
    const char* name,
    const uint16_t value,
    const size_t count)
{
    if (runner.input_bytes(name) != count * sizeof(uint16_t)) {
        throw std::runtime_error(
            std::string("unexpected QNN QP input size for ") + name);
    }
    auto* output = static_cast<uint16_t*>(runner.input_data(name));
    std::fill(output, output + count, value);
}

void QpControls::write_array(
    QnnRunner& runner,
    const char* name,
    const uint16_t* values,
    const size_t count)
{
    const size_t bytes = count * sizeof(uint16_t);
    if (runner.input_bytes(name) != bytes) {
        throw std::runtime_error(
            std::string("unexpected QNN QP input size for ") + name);
    }
    std::memcpy(runner.input_data(name), values, bytes);
}

void QpControls::apply_encoder(QnnRunner& runner, const int qp) const
{
    const Record& item = record(qp);
    write_scalar(runner, "q_image", item.q_image);
    write_scalar(runner, "q_hyper_z", item.q_hyper_z);
    write_scalar(runner, "q_hyper_m", item.q_hyper_m);
    write_scalar(runner, "q_prior", item.q_prior);
    write_repeated(
        runner,
        "q_basic",
        item.q_basic,
        static_cast<size_t>(latent_height_) * latent_width_);
}

void QpControls::apply_decoder(QnnRunner& runner, const int qp) const
{
    const Record& item = record(qp);
    write_scalar(runner, "q_hyper_z", item.q_hyper_z);
    write_scalar(runner, "q_hyper_m", item.q_hyper_m);
    write_scalar(runner, "q_prior", item.q_prior);
    write_repeated(
        runner,
        "q_basic",
        item.q_basic,
        static_cast<size_t>(latent_height_) * latent_width_);
    write_scalar(runner, "q_codec_dec", item.q_codec_dec);
    write_array(
        runner,
        "q_dico",
        item.q_dico.data(),
        item.q_dico.size());
    write_array(
        runner,
        "q_render",
        item.q_render.data(),
        item.q_render.size());
}

}  // namespace pulse_mobile
