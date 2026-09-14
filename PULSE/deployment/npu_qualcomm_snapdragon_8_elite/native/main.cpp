// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "entropy.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <set>

#ifndef PULSE_HOST_ONLY
#include "qnn_runner.hpp"
#include "qp_controls.hpp"
#endif

using namespace pulse_mobile;
using Clock = std::chrono::steady_clock;
using Args = std::map<std::string,std::string>;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
template<class T> void read_tensor(const std::string& path, std::vector<T>& out) {
    auto bytes=read_file_bytes(path);
    if(bytes.size()!=out.size()*sizeof(T)) throw std::runtime_error("tensor size mismatch: "+path);
    std::memcpy(out.data(),bytes.data(),bytes.size());
}
std::string stats(std::vector<double> samples) {
    double mean=std::accumulate(samples.begin(),samples.end(),0.0)/samples.size();
    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    size_t n=sorted.size();double median=(sorted[(n-1)/2]+sorted[n/2])/2;
    std::ostringstream o;o<<"{\"mean_ms\":"<<mean<<",\"median_ms\":"<<median<<",\"samples_ms\":[";
    for(size_t i=0;i<n;++i){if(i)o<<',';o<<samples[i];}o<<"]}";return o.str();
}

#ifndef PULSE_HOST_ONLY
static_assert(sizeof(__fp16)==2, "FP16 ARM build required");
void to_half(const std::vector<int16_t>& input, QnnRunner& runner, const char* name) {
    if(runner.input_bytes(name)!=input.size()*2)throw std::runtime_error("QNN FP16 input shape mismatch");
    auto* out=static_cast<__fp16*>(runner.input_data(name));
    for(size_t i=0;i<input.size();++i)out[i]=__fp16(input[i]);
}
void from_half(QnnRunner& runner, const char* name, std::vector<int16_t>& output) {
    if(runner.output_bytes(name)!=output.size()*2)throw std::runtime_error("QNN FP16 output shape mismatch");
    const auto* in=static_cast<const __fp16*>(runner.output_data(name));
    for(size_t i=0;i<output.size();++i) {
        float v=float(in[i]);
        if(!std::isfinite(v) || v < -32768 || v > 32767)throw std::runtime_error("invalid QNN entropy symbol");
        output[i]=int16_t(std::lrint(v));
    }
}

class MobileCodec {
public:
    EntropyRuntime entropy;
    QpControls controls;
    std::unique_ptr<QnnRunner> encoder,decoder;
    MobileCodec(const std::string& assets, const std::string& libs, int qp, bool enc, bool dec)
        :entropy(assets+"/entropy_qp"+std::to_string(qp)+".bin"),controls(assets+"/qp_controls.bin") {
        if(enc)encoder=std::make_unique<QnnRunner>(libs,assets+"/encoder.bin",false);
        if(dec)decoder=std::make_unique<QnnRunner>(libs,assets+"/decoder.bin",false);
    }
    Bytes compress(const Bytes& rgb) {
        auto& b=entropy.bundle;size_t pixels=size_t(b.width)*b.height;
        if(rgb.size()!=3*pixels || encoder->input_bytes("image")!=6*pixels)
            throw std::runtime_error("expected packed uint8 RGB and an FP16 whole-image graph");
        controls.apply_encoder(*encoder,b.qp);
        auto* image=static_cast<__fp16*>(encoder->input_data("image"));
        for(int c=0;c<3;++c)
            for(size_t i=0;i<pixels;++i)image[c*pixels+i]=__fp16(float(rgb[3*i+c])/255.0f);
        encoder->execute();
        from_half(*encoder,"z_hat",entropy.z);
        from_half(*encoder,"symbols0_packed",entropy.s0);
        from_half(*encoder,"symbols1_packed",entropy.s1);
        return entropy.encode();
    }
    void decompress(const Bytes& stream) {
        entropy.decode(stream);
        controls.apply_decoder(*decoder,entropy.bundle.qp);
        to_half(entropy.z,*decoder,"z_hat");
        to_half(entropy.s0,*decoder,"symbols0_packed");
        to_half(entropy.s1,*decoder,"symbols1_packed");
        decoder->execute();
        if(decoder->output_bytes()!=size_t(entropy.bundle.width)*entropy.bundle.height*6)
            throw std::runtime_error("expected FP16 NCHW RGB output");
    }
};
#endif

int main(int argc,char** argv) {
    try {
        if(argc<2)throw std::runtime_error("usage: pulse_mobile compress|decompress|benchmark [--assets DIR --input FILE --output FILE --qp 3 --warmup 3 --repeats 20]");
        std::string action=argv[1];Args args;
        std::set<std::string> allowed={"--assets","--lib","--input","--output","--qp","--warmup",
                                      "--repeats","--bundle","--z","--y0","--y1"};
        for(int i=2;i<argc;i+=2) {
            if(i+1>=argc || !allowed.count(argv[i]) || args.count(argv[i]))
                throw std::runtime_error("unknown, duplicate or incomplete argument");
            args[argv[i]]=argv[i+1];
        }
        if(action=="encode-symbols" || action=="decode-symbols") {
            EntropyRuntime codec(args.at("--bundle"));
            if(action=="encode-symbols") {
                read_tensor(args.at("--z"),codec.z);read_tensor(args.at("--y0"),codec.s0);read_tensor(args.at("--y1"),codec.s1);
                auto result=codec.encode();write_file_bytes(args.at("--output"),result.data(),result.size());
            } else {
                codec.decode(read_file_bytes(args.at("--input")));
                auto path=args.at("--output");std::filesystem::create_directories(path);
                write_file_bytes(path+"/z.i16",codec.z.data(),codec.z.size()*2);
                write_file_bytes(path+"/y0.i16",codec.s0.data(),codec.s0.size()*2);
                write_file_bytes(path+"/y1.i16",codec.s1.data(),codec.s1.size()*2);
            }
            auto path=args.at("--output")+".meta";
            write_file_bytes(path,codec.meta.data(),codec.meta.size());
            write_file_bytes(args.at("--output")+".indexes",codec.indexes.data(),codec.indexes.size());
            return 0;
        }
#ifdef PULSE_HOST_ONLY
        throw std::runtime_error("host build supports encode-symbols and decode-symbols only");
#else
        if(action!="compress" && action!="decompress" && action!="benchmark")throw std::runtime_error("unsupported action");
        const auto assets=args.at("--assets");
        const auto libs=args.count("--lib")?args.at("--lib"):assets+"/lib";
        auto input=read_file_bytes(args.at("--input"));
        int qp=args.count("--qp")?std::stoi(args.at("--qp")):3;
        if(action=="decompress")unpack_stream(input,nullptr,&qp);
        if(qp<0 || qp>7)throw std::runtime_error("QP must be in [0,7]");
        MobileCodec codec(assets,libs,qp,action!="decompress",action!="compress");
        if(action=="compress") {
            auto stream=codec.compress(input);write_file_bytes(args.at("--output"),stream.data(),stream.size());return 0;
        }
        if(action=="decompress") {
            codec.decompress(input);
            write_file_bytes(args.at("--output"),codec.decoder->output_data(),codec.decoder->output_bytes());return 0;
        }
        int warmup=args.count("--warmup")?std::stoi(args.at("--warmup")):3;
        int repeats=args.count("--repeats")?std::stoi(args.at("--repeats")):20;
        if(warmup<1 || repeats<1)throw std::runtime_error("warmup/repeats must be positive");
        std::vector<double> encode,decode;Bytes stream,first_stream;
        for(int i=0;i<warmup+repeats;++i) {
            auto t=Clock::now();stream=codec.compress(input);double enc=milliseconds(t);
            t=Clock::now();codec.decompress(stream);double dec=milliseconds(t);
            if(i>=warmup){
                encode.push_back(enc);decode.push_back(dec);
                if(i==warmup)first_stream=stream;
                else if(stream!=first_stream)throw std::runtime_error("non-deterministic encoded stream");
            }
        }
        auto rgb=static_cast<const uint16_t*>(codec.decoder->output_data());
        for(size_t i=0;i<codec.decoder->output_bytes()/2;++i)
            if((rgb[i]&0x7c00U)==0x7c00U)throw std::runtime_error("non-finite reconstructed RGB");
        auto& b=codec.entropy.bundle;std::ostringstream report;
        report<<"{\"width\":"<<b.width<<",\"height\":"<<b.height<<",\"qp\":"<<qp
              <<",\"bytes\":"<<stream.size()<<",\"bpp\":"<<stream.size()*8.0/(b.width*b.height)
              <<",\"warmup\":"<<warmup<<",\"repeats\":"<<repeats
              <<",\"deterministic_stream\":true,\"finite_rgb\":true"
              <<",\"encoding\":"<<stats(encode)<<",\"decoding\":"<<stats(decode)
              <<",\"scope\":\"full image; RGB8 in memory to .pulse in memory; .pulse to FP16 RGB; includes entropy and buffer conversion; excludes file IO, initialization, warmup and display\"}\n";
        auto text=report.str();std::cout<<text;
        write_file_bytes(args.at("--output"),text.data(),text.size());
        return 0;
#endif
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';return 1;
    }
}
