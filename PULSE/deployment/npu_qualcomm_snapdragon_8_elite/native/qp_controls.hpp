#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pulse_mobile {

class QnnRunner;

class QpControls {
public:
    static constexpr int kQpCount = 8;

    explicit QpControls(const std::string& path);

    void apply_encoder(QnnRunner& runner, int qp) const;
    void apply_decoder(QnnRunner& runner, int qp) const;

private:
    uint32_t dico_channels_{}, render_channels_{}, latent_height_{}, latent_width_{};
    struct Record {
        uint16_t q_image{};
        uint16_t q_hyper_z{};
        uint16_t q_hyper_m{};
        uint16_t q_prior{};
        uint16_t q_basic{};
        uint16_t q_codec_dec{};
        std::vector<uint16_t> q_dico{};
        std::vector<uint16_t> q_render{};
    };

    const Record& record(int qp) const;
    static void write_scalar(
        QnnRunner& runner,
        const char* name,
        uint16_t value);
    static void write_repeated(
        QnnRunner& runner,
        const char* name,
        uint16_t value,
        size_t count);
    static void write_array(
        QnnRunner& runner,
        const char* name,
        const uint16_t* values,
        size_t count);

    std::array<Record, kQpCount> records_{};
};

}  // namespace pulse_mobile
