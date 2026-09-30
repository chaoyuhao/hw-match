// Bounded synchronous instruction doubles. These do not model CANN scheduling.
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <map>
#define __aicore__
#define __gm__
#define GM_ADDR uint8_t*
constexpr int PIPE_V = 0;
using half = _Float16;
struct bfloat16_t {
    uint16_t bits;
    bfloat16_t() = default;
    explicit bfloat16_t(float x) { uint32_t u; std::memcpy(&u,&x,4); u+=0x7fff+((u>>16)&1); bits=u>>16; }
    operator float() const { uint32_t u=uint32_t(bits)<<16; float x; std::memcpy(&x,&u,4); return x; }
};
namespace AscendC {
enum class TPosition { VECIN, VECOUT, VECCALC };
enum class HardEvent { V_S, S_V, V_MTE2, S_MTE3, MTE3_S };
enum class RoundMode { CAST_NONE };
enum class ReduceOrder { ORDER_ONLY_VALUE };
static uint32_t blockIdx=0, blockNum=1;
static uint64_t vectorToScalar=0, repeatedMulCalls=0;
static std::map<uintptr_t,uint32_t> owners;
inline uint32_t GetBlockIdx(){return blockIdx;}
inline uint32_t GetBlockNum(){return blockNum;}
template<int> void PipeBarrier(){}
template<HardEvent> void SetFlag(int){}
template<HardEvent event> void WaitFlag(int){if(event==HardEvent::V_S)++vectorToScalar;}
struct Storage { std::vector<uint8_t> bytes, valid; explicit Storage(size_t n):bytes(n),valid(n){} };
template<typename T> struct LocalTensor {
    std::shared_ptr<Storage> s; size_t offset=0;
    LocalTensor operator[](size_t i) const { assert(offset+i*sizeof(T)<s->bytes.size()); return {s,offset+i*sizeof(T)}; }
    T GetValue(size_t i) const { size_t a=offset+i*sizeof(T); assert(a+sizeof(T)<=s->bytes.size());
        for(size_t j=0;j<sizeof(T);++j) assert(s->valid[a+j]); T v; std::memcpy(&v,s->bytes.data()+a,sizeof(T)); return v; }
    void SetValue(size_t i,T v) const { size_t a=offset+i*sizeof(T); assert(a+sizeof(T)<=s->bytes.size());
        std::memcpy(s->bytes.data()+a,&v,sizeof(T)); std::fill_n(s->valid.data()+a,sizeof(T),1); }
};
template<typename T> struct GlobalTensor {
    T* p=nullptr; size_t count=0;
    void SetGlobalBuffer(T* ptr,size_t n){p=ptr;count=n;}
    GlobalTensor operator[](size_t n)const{assert(n<count);return {p+n,count-n};}
};
template<TPosition> struct TBuf {
    std::shared_ptr<Storage> s;
    template<typename T> LocalTensor<T> Get(){return {s,0};}
};
template<TPosition, int> struct TQue {
    std::shared_ptr<Storage> s; int state=0;
    template<typename T> LocalTensor<T> AllocTensor(){assert(state==0);state=1;std::fill(s->valid.begin(),s->valid.end(),0);return {s,0};}
    template<typename T> void EnQue(LocalTensor<T>){assert(state==1);state=2;}
    template<typename T> LocalTensor<T> DeQue(){assert(state==2);state=3;return {s,0};}
    template<typename T> void FreeTensor(LocalTensor<T>){assert(state==3);state=0;}
};
struct TPipe {
    uint32_t bytes=0;
    template<typename Q> void InitBuffer(Q& q,uint32_t n){assert(n%32==0);bytes+=n;assert(bytes<=65536);q.s=std::make_shared<Storage>(n);}
    template<typename Q> void InitBuffer(Q& q,int depth,uint32_t n){assert(depth==1);InitBuffer(q,n);}
    int FetchEventID(HardEvent){return 0;}
};
struct DataCopyExtParams {uint16_t blockCount;uint32_t blockLen,srcStride,dstStride,rsv;};
template<typename T> struct DataCopyPadExtParams {bool isPad;uint8_t leftPadding,rightPadding;T paddingValue;};
template<typename T> void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,DataCopyExtParams c,DataCopyPadExtParams<T> pad){
    assert(c.srcStride==0 && c.dstStride==0 && pad.isPad && pad.leftPadding==0);
    size_t width=c.blockLen/sizeof(T),pitch=(width*sizeof(T)+31)/32*32/sizeof(T);
    assert(pad.rightPadding==pitch-width);
    for(size_t r=0;r<c.blockCount;++r) for(size_t j=0;j<pitch;++j){
        if(j<width){assert(r*width+j<src.count);dst.SetValue(r*pitch+j,src.p[r*width+j]);}
        else dst.SetValue(r*pitch+j,pad.paddingValue);
    }
}
template<typename T> void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyExtParams c){
    assert(c.blockCount==1 && c.srcStride==0 && c.dstStride==0);
    for(size_t j=0;j<c.blockLen/sizeof(T);++j){assert(j<dst.count);auto key=reinterpret_cast<uintptr_t>(dst.p+j)/32;
        auto inserted=owners.emplace(key,blockIdx);assert(inserted.second || inserted.first->second==blockIdx);dst.p[j]=src.GetValue(j);}
}
template<typename T> void Cast(LocalTensor<float> d,LocalTensor<T>s,RoundMode,uint32_t n){for(uint32_t i=0;i<n;++i)d.SetValue(i,float(s.GetValue(i)));}
inline void Duplicate(LocalTensor<float>d,float v,uint32_t n){for(uint32_t i=0;i<n;++i)d.SetValue(i,v);}
inline void Mul(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<float>b,uint32_t n){assert(a.offset%32==0&&b.offset%32==0);for(uint32_t i=0;i<n;++i)d.SetValue(i,a.GetValue(i)*b.GetValue(i));}
struct BinaryRepeatParams {
    uint8_t dstBlkStride,src0BlkStride,src1BlkStride,dstRepStride,src0RepStride,src1RepStride;
};
inline void Mul(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<float>b,uint64_t mask,uint8_t repeats,const BinaryRepeatParams& p){
    assert(mask>0&&mask<=64&&repeats>0&&repeats<=64);
    assert(d.offset%32==0&&a.offset%32==0&&b.offset%32==0);
    ++repeatedMulCalls;
    for(uint32_t r=0;r<repeats;++r)for(uint32_t i=0;i<mask;++i){
        auto at=[r,i](uint8_t block,uint8_t repeat){return r*repeat*8+(i/8)*block*8+i%8;};
        d.SetValue(at(p.dstBlkStride,p.dstRepStride),a.GetValue(at(p.src0BlkStride,p.src0RepStride))*b.GetValue(at(p.src1BlkStride,p.src1RepStride)));
    }
}
inline void Muls(LocalTensor<float>d,LocalTensor<float>a,float b,uint32_t n){assert(a.offset%32==0);for(uint32_t i=0;i<n;++i)d.SetValue(i,a.GetValue(i)*b);}
inline void Add(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<float>b,uint32_t n){assert(d.offset%32==0&&a.offset%32==0&&b.offset%32==0);for(uint32_t i=0;i<n;++i)d.SetValue(i,a.GetValue(i)+b.GetValue(i));}
inline void WholeReduceSum(LocalTensor<float>d,LocalTensor<float>s,int mask,int repeats,int ds,int bs,int rs){
    assert(mask>0&&mask<=64&&repeats>0&&repeats<=255&&ds>0&&s.offset%32==0&&d.offset%4==0);
    for(int r=0;r<repeats;++r){
        float a[64]={};for(int i=0;i<mask;++i)a[i]=s.GetValue(r*rs*8+(i/8)*bs*8+i%8);
        for(int width=64;width>1;width/=2)for(int i=0;i<width/2;++i)a[i]=a[2*i]+a[2*i+1];
        d.SetValue(r*ds,a[0]);
    }
}
inline void WholeReduceMax(LocalTensor<float>d,LocalTensor<float>s,int mask,int repeats,int ds,int bs,int rs,ReduceOrder){
    assert(mask>0&&mask<=64&&repeats==1&&ds==1&&bs==1&&rs==8&&s.offset%32==0);
    float v=s.GetValue(0);for(int i=1;i<mask;++i)v=std::max(v,s.GetValue(i));d.SetValue(0,v);
}
}
