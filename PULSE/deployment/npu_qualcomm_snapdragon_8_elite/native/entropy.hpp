// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once
#include "parallel_rans.hpp"
#include "meta_merge.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace pulse_mobile {
using Bytes = std::vector<uint8_t>;

inline uint32_t crc32(uint32_t initial, const uint8_t* data, size_t count) {
    uint32_t crc=~initial;
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    while(count>=8){uint64_t v;std::memcpy(&v,data,8);crc=__crc32d(crc,v);data+=8;count-=8;}
    while(count--)crc=__crc32b(crc,*data++);
#else
    static const auto table=[]{
        std::array<uint32_t,256> t{};
        for(uint32_t i=0;i<256;++i){uint32_t c=i;for(int k=0;k<8;++k)c=(c>>1)^((c&1)?0xedb88320U:0);t[i]=c;}
        return t;
    }();
    while(count--)crc=table[(crc^*data++)&255]^(crc>>8);
#endif
    return ~crc;
}
inline Bytes read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return Bytes(std::istreambuf_iterator<char>(f), {});
}
inline void write_file_bytes(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f || !f.write(static_cast<const char*>(data), size))
        throw std::runtime_error("cannot write " + path);
}
struct Reader {
    const Bytes& data; size_t pos{};
    const uint8_t* take(size_t count) {
        if (count > data.size()-pos) throw std::runtime_error("truncated PULSE data");
        const auto* p=data.data()+pos; pos+=count; return p;
    }
    uint32_t u32() { return read_u32_le(take(4)); }
    uint16_t u16() { auto p=take(2); return uint16_t(p[0]) | uint16_t(p[1])<<8; }
    uint8_t u8() { return *take(1); }
    template<class T> std::vector<T> array() {
        size_t count=u32();
        if (count > (64U<<20)/sizeof(T)) throw std::runtime_error("array too large");
        auto p=take(count*sizeof(T)); std::vector<T> v(count);
        std::memcpy(v.data(), p, count*sizeof(T)); return v;
    }
};
inline void put16(Bytes& b, uint16_t v) { b.push_back(uint8_t(v)); b.push_back(uint8_t(v>>8)); }
inline void put64(Bytes& b, uint64_t v) { for(int i=0;i<8;++i)b.push_back(uint8_t(v>>(8*i))); }
inline void append(Bytes& b, const Bytes& v) { b.insert(b.end(),v.begin(),v.end()); }

struct Tables {
    std::shared_ptr<std::vector<std::vector<int32_t>>> cdf;
    std::shared_ptr<std::vector<int8_t>> maximum;
    RansTables encode;
    Tables(const std::vector<int32_t>& flat, const std::vector<int32_t>& lengths) {
        if(lengths.empty() || flat.size()%lengths.size()) throw std::runtime_error("invalid CDF");
        const size_t cols=flat.size()/lengths.size();
        cdf=std::make_shared<std::vector<std::vector<int32_t>>>(lengths.size());
        maximum=std::make_shared<std::vector<int8_t>>(lengths.size());
        encode.symbols=std::make_shared<std::vector<std::vector<RansSymbol>>>(lengths.size());
        encode.max_values=maximum;
        for(size_t r=0;r<lengths.size();++r) {
            int len=lengths[r];
            if(len<3 || size_t(len)>cols || len-2>127) throw std::runtime_error("invalid CDF length");
            auto first=flat.begin()+r*cols;
            (*cdf)[r].assign(first,first+cols);
            if(first[0]!=0 || first[len-1]!=65536) throw std::runtime_error("invalid CDF total");
            (*maximum)[r]=int8_t(len-2);
            auto& symbols=(*encode.symbols)[r]; symbols.resize(cols-1);
            for(int j=0;j<len-1;++j) {
                const int freq=first[j+1]-first[j];
                if(freq<1 || freq>65535) throw std::runtime_error("invalid CDF frequency");
                symbols[j]={uint16_t(first[j]),uint16_t(freq)};
            }
        }
    }
};

struct Bundle {
    uint32_t width{},height{},m{},z{},qp{},clip{},mode{},shift{};
    int cutoff{};
    std::array<uint8_t,16> id{};
    pulse::log_index_scalar::PreparedLinearDecoder linear;
    std::vector<int32_t> z_cdf,z_lengths,y_cdf,y_lengths;
    std::vector<float> cost;
    std::vector<uint16_t> probabilities;
    explicit Bundle(const std::string& path) {
        auto data=read_file_bytes(path); Reader r{data};
        if(std::memcmp(r.take(8),"PULSENT2",8)) throw std::runtime_error("bad entropy bundle");
        std::memcpy(id.data(),r.take(16),16);
        if(r.u32()!=2) throw std::runtime_error("unsupported entropy bundle");
        width=r.u32();height=r.u32();m=r.u32();z=r.u32();qp=r.u32();clip=r.u32();
        cutoff=int(r.u32())-1;mode=r.u32();shift=r.u32();
        if(!width || !height || width%64 || height%64 || uint64_t(width)*height>100000000 ||
           qp>7 || (m!=256 && m!=320) || (z!=80 && z!=96 && z!=144) ||
           cutoff < -1 || cutoff > 63 || mode>1) throw std::runtime_error("invalid bundle dimensions");
        auto divisor=r.array<int32_t>(); auto weight=r.array<int8_t>();
        auto multiplier=r.array<int32_t>(); auto bias=r.array<int32_t>();
        z_cdf=r.array<int32_t>();z_lengths=r.array<int32_t>();
        y_cdf=r.array<int32_t>();y_lengths=r.array<int32_t>();
        cost=r.array<float>();probabilities=r.array<uint16_t>();
        if(r.pos!=data.size() || z_lengths.size()!=64*z || y_lengths.size()!=(mode ? 128U:64U) ||
           cost.size()!=size_t(64)*z*128) throw std::runtime_error("incomplete entropy bundle");
        linear=pulse::log_index_scalar::prepare(m,z,4,clip,divisor,weight,multiplier,bias);
    }
    size_t sites() const { return size_t(height/64)*(width/64); }
    size_t part_size() const { return size_t(height/16)*(width/16)*(m/2); }
};

inline Bytes meta_pack(const Bundle& b, const std::vector<uint8_t>& indexes, const Bytes& raw) {
    if(!b.mode || b.cutoff!=2 || b.probabilities.empty())
        throw std::runtime_error("re-export assets with calibrated CDF, skip=2 and Index Merge");
    Bytes merged=encode_meta_prior_index_merge(indexes,1,b.height/64,b.width/64,6,b.probabilities,b.shift);
    Bytes out={'M','P','R','I',2,6,64,0};
    append_u32_le(out,merged.size());append(out,merged);
    append(out,raw);return out;
}
inline std::vector<uint8_t> meta_read(const Bundle& b, Reader& r) {
    if(std::memcmp(r.take(4),"MPRI",4)) throw std::runtime_error("missing Meta Prior");
    auto version=r.u8();
    if(r.u8()!=6 || r.u16()!=64)throw std::runtime_error("invalid Meta Prior bank count");
    if(version==2) {
        if(b.probabilities.empty()) throw std::runtime_error("missing index-merge probabilities");
        auto size=r.u32();auto p=r.take(size);
        return decode_meta_prior_index_merge(Bytes(p,p+size),1,b.height/64,b.width/64,6,64,b.probabilities,b.shift);
    }
    if(version!=1)throw std::runtime_error("unsupported Meta Prior syntax");
    std::vector<uint8_t> out(b.sites());auto p=r.take((out.size()*6+7)/8);
    uint32_t acc=0;int bits=0;size_t i=0;
    for(auto& v:out){while(bits<6){acc|=uint32_t(p[i++])<<bits;bits+=8;}v=acc&63;acc>>=6;bits-=6;}
    return out;
}

inline Bytes pack_stream(const Bundle& b, const Bytes& payload) {
    Bytes native={4,3,3};
    uint8_t magic=b.mode ? (b.cutoff==2?0xe1:(b.cutoff<0?0xe2:0xe3)) : (b.cutoff<0?0xde:0xdf);
    native.push_back(0xc0);native.push_back(magic);
    put16(native,b.width);put16(native,b.height);native.push_back(b.qp);
    if(magic==0xe3 || magic==0xdf)native.push_back(uint8_t(b.cutoff));
    put16(native,b.m);put16(native,b.height/16);put16(native,b.width/16);
    put16(native,b.z);put16(native,b.height/64);put16(native,b.width/64);append(native,payload);
    Bytes out={'P','L','S','3'};out.insert(out.end(),b.id.begin(),b.id.end());put64(out,native.size());
    uint32_t crc=crc32(0,out.data(),out.size());crc=crc32(crc,native.data(),native.size());
    append_u32_le(out,crc);append(out,native);return out;
}
inline Bytes unpack_stream(const Bytes& data, const Bundle* bundle=nullptr, int* qp=nullptr) {
    Reader r{data};
    if(std::memcmp(r.take(4),"PLS3",4))throw std::runtime_error("expected mobile .pulse (PLS3)");
    auto id=r.take(16);uint64_t len=r.u32();len|=uint64_t(r.u32())<<32;
    auto crc=r.u32();
    if(len!=data.size()-32 || len>(512U<<20))throw std::runtime_error("invalid .pulse length");
    uint32_t check=crc32(0,data.data(),28);check=crc32(check,data.data()+32,len);
    if(crc!=check)throw std::runtime_error(".pulse checksum mismatch");
    if(bundle && std::memcmp(id,bundle->id.data(),16))throw std::runtime_error("different model bundle");
    if(r.u8()!=4 || r.u8()!=3 || r.u8()!=3)throw std::runtime_error("expected z4/y3/y3 lanes");
    if(r.u8()!=0xc0)throw std::runtime_error("invalid native header");
    auto magic=r.u8();int mode=magic>=0xe1 && magic<=0xe3;
    if(magic!=0xde && magic!=0xdf && !mode)throw std::runtime_error("invalid CDF mode");
    auto w=r.u16(),h=r.u16();auto q=r.u8();
    int cutoff=(magic==0xdf || magic==0xe3)?int(r.u8()):(magic==0xe1?2:-1);
    auto m=r.u16(),yh=r.u16(),yw=r.u16(),z=r.u16(),zh=r.u16(),zw=r.u16();
    if(q>7 || !w || !h || w%64 || h%64 || yh!=h/16 || yw!=w/16 || zh!=h/64 || zw!=w/64)
        throw std::runtime_error("invalid image header");
    if(bundle && (w!=bundle->width || h!=bundle->height || m!=bundle->m || z!=bundle->z ||
                  q!=bundle->qp || mode!=int(bundle->mode) || cutoff!=bundle->cutoff))
        throw std::runtime_error("stream and runtime settings differ");
    if(qp)*qp=q;return Bytes(data.begin()+r.pos,data.end());
}

class EntropyRuntime {
public:
    Bundle bundle;
    std::vector<int16_t> z,s0,s1;
    std::vector<uint8_t> indexes,meta;
    explicit EntropyRuntime(const std::string& path)
        :bundle(path),z(bundle.sites()*bundle.z),s0(bundle.part_size()),s1(s0.size()),
         indexes(2*s0.size()),meta(bundle.sites()),zt(bundle.z_cdf,bundle.z_lengths),
         yt(bundle.y_cdf,bundle.y_lengths),ze(4),e0(3),e1(3),zd(4),d0(3),d1(3),
         nhwc(z.size()),p0(s0.size()),p1(s0.size()) {
        ze.set_cdf(zt.encode,0);e0.set_cdf(yt.encode,1);e1.set_cdf(yt.encode,1);
        zd.set_cdf(zt.cdf,zt.maximum,0);d0.set_cdf(yt.cdf,yt.maximum,1);d1.set_cdf(yt.cdf,yt.maximum,1);
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
        scale=std::make_unique<I8mmScaleExecutor>(bundle.linear,bundle.height/64,bundle.width/64,4);
#endif
    }
    void predict_indexes() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
        scale->run(z.data(),indexes.data(),nullptr);
#else
        // Explicit host-reference build; Android is built with I8MM enabled.
        pulse::log_index_scalar::infer_packed(bundle.linear,z.data(),bundle.height/64,bundle.width/64,indexes.data());
#endif
    }
    void select_meta() {
        for(size_t site=0;site<meta.size();++site) {
            float best=std::numeric_limits<float>::infinity();int selected=0;
            for(int bank=0;bank<64;++bank) {
                float sum=0;
                for(uint32_t c=0;c<bundle.z;++c) {
                    int value=z[size_t(c)*meta.size()+site], mapped=std::abs(value)*2-(value>0);
                    size_t row=size_t(bank)*bundle.z+c;
                    int maximum=bundle.z_lengths[row]-2, encoded=std::min(mapped,maximum);
                    sum+=bundle.cost[row*128+encoded];
                    if(mapped>=maximum) {
                        uint32_t tail=mapped-maximum;int groups=0;
                        while(tail) { ++groups;tail>>=2; }
                        sum+=float(2*(groups+1+groups/3));
                    }
                }
                if(sum<best){best=sum;selected=bank;}
            }
            meta[site]=uint8_t(selected);
        }
    }
    Bytes encode() {
        select_meta();predict_indexes();
        for(size_t site=0;site<meta.size();++site)
            for(uint32_t c=0;c<bundle.z;++c)nhwc[site*bundle.z+c]=z[c*meta.size()+site];
        ze.begin_meta(nhwc.data(),meta.data(),nhwc.size(),bundle.z);
        size_t n0=0,n1=0;
        for(size_t i=0;i<s0.size();++i) {
            if(s0[i]<-63 || s0[i]>63 || s1[i]<-63 || s1[i]>63)
                throw std::runtime_error("main-latent symbols must be in [-63,63]");
            int a=indexes[i],b=indexes[i+s0.size()];
            if(a>bundle.cutoff)p0[n0++]=int16_t(int(s0[i])*256+a);
            if(b>bundle.cutoff)p1[n1++]=int16_t(int(s1[i])*256+b+(bundle.mode?64:0));
        }
        e0.begin_y(p0.data(),n0);e1.begin_y(p1.data(),n1);
        auto a=ze.finish(),b=e0.finish(),c=e1.finish();Bytes raw;
        append_u32_le(raw,a.size());append_u32_le(raw,b.size());append_u32_le(raw,c.size());
        append(raw,a);append(raw,b);append(raw,c);
        return pack_stream(bundle,meta_pack(bundle,meta,raw));
    }
    void decode(const Bytes& stream) {
        auto payload=unpack_stream(stream,&bundle);Reader r{payload};meta=meta_read(bundle,r);
        std::array<uint32_t,3> sizes={r.u32(),r.u32(),r.u32()};
        if(uint64_t(sizes[0])+sizes[1]+sizes[2]!=payload.size()-r.pos ||
           *std::min_element(sizes.begin(),sizes.end())<4)throw std::runtime_error("invalid entropy sections");
        std::array<ParallelRansDecoder*,3> decoders={&zd,&d0,&d1};
        for(int i=0;i<3;++i){auto p=r.take(sizes[i]);decoders[i]->set_stream(std::make_shared<Bytes>(p,p+sizes[i]));}
        zd.decode_meta(nhwc.data(),meta.data(),nhwc.size(),bundle.z);
        for(size_t site=0;site<meta.size();++site)
            for(uint32_t c=0;c<bundle.z;++c)z[c*meta.size()+site]=nhwc[site*bundle.z+c];
        predict_indexes();std::vector<uint8_t> i0,i1;std::vector<size_t> o0,o1;
        i0.reserve(s0.size());i1.reserve(s1.size());o0.reserve(s0.size());o1.reserve(s1.size());
        for(size_t i=0;i<s0.size();++i) {
            if(indexes[i]>bundle.cutoff){i0.push_back(indexes[i]);o0.push_back(i);}
            if(indexes[i+s0.size()]>bundle.cutoff){i1.push_back(indexes[i+s0.size()]+(bundle.mode?64:0));o1.push_back(i);}
        }
        d0.start_decode_y(p0.data(),i0.data(),i0.size());d1.start_decode_y(p1.data(),i1.data(),i1.size());
        d0.wait_decode();d1.wait_decode();
        std::fill(s0.begin(),s0.end(),0);std::fill(s1.begin(),s1.end(),0);
        for(size_t i=0;i<o0.size();++i)s0[o0[i]]=p0[i];
        for(size_t i=0;i<o1.size();++i)s1[o1[i]]=p1[i];
    }
private:
    Tables zt,yt;ParallelRansEncoder ze,e0,e1;ParallelRansDecoder zd,d0,d1;
    std::vector<int16_t> nhwc,p0,p1;
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    std::unique_ptr<I8mmScaleExecutor> scale;
#endif
};
} // namespace pulse_mobile
