// CPU implementation of documented ND/NZ/zZ/nZ storage only. All operations
// complete synchronously; these doubles cannot prove real pipeline timing.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <deque>
#include <vector>
#include <stdexcept>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
enum {PIPE_ALL, PIPE_MTE2, PIPE_MTE1, PIPE_M, PIPE_FIX, EVENT_ID0};
using half=_Float16;
struct bf16 {
 uint16_t bits;
 bf16()=default;
 explicit bf16(float value){uint32_t word;std::memcpy(&word,&value,4);word+=0x7fff+((word>>16)&1);bits=word>>16;}
 operator float()const{uint32_t word=uint32_t(bits)<<16;float f;std::memcpy(&f,&word,4);return f;}
};
namespace AscendC {
enum class TPosition {A1,B1,A2,B2,CO1};
enum class HardEvent {MTE2_MTE1,MTE1_MTE2,MTE1_M,M_MTE1,M_FIX,FIX_M};
using TEventID=int;
static bool eventFlags[6][8], allocated[6][8];
// Execute recorded event instructions in independent engine FIFO order.
// This checks progress/token reuse, not hardware latency or memory ordering.
struct EventOp{bool set;int event,id;};
static std::deque<EventOp> engines[4];
inline int Source(HardEvent e){const int v[]={0,1,1,2,2,3};return v[int(e)];}
inline int Target(HardEvent e){const int v[]={1,0,2,1,3,2};return v[int(e)];}
inline void PumpEvents(){
 bool progress=true;
 while(progress){progress=false;for(int i=3;i>=0;--i)if(!engines[i].empty()){
  auto op=engines[i].front();bool& flag=eventFlags[op.event][op.id];
  if(op.set){assert(!flag);flag=true;}else{if(!flag)continue;flag=false;}
  engines[i].pop_front();progress=true;
 }}
 for(auto& queue:engines)assert(queue.empty());
}
template<HardEvent E>void SetFlag(int id){assert(id>=0&&id<8);engines[Source(E)].push_back({true,int(E),id});}
template<HardEvent E>void WaitFlag(int id){assert(id>=0&&id<8);engines[Target(E)].push_back({false,int(E),id});}
template<int P>void PipeBarrier(){}
template<typename T>struct LocalTensor {
 std::shared_ptr<std::vector<uint8_t>> data;size_t offset=0;
 LocalTensor operator[](size_t n)const{return {data,offset+n*sizeof(T)};}
 T get(size_t n)const{T x;assert(offset+(n+1)*sizeof(T)<=data->size());std::memcpy(&x,data->data()+offset+n*sizeof(T),sizeof(T));return x;}
 void put(size_t n,T v)const{assert(offset+(n+1)*sizeof(T)<=data->size());std::memcpy(data->data()+offset+n*sizeof(T),&v,sizeof(T));}
};
template<typename T>struct GlobalTensor {
 T* data=nullptr;
 void SetGlobalBuffer(T* p){data=p;}
 GlobalTensor operator[](size_t n)const{return {data+n};}
};
template<TPosition P>struct TBuf{
 std::shared_ptr<std::vector<uint8_t>> data;
 template<typename T> LocalTensor<T> Get(){return {data,0};}
};
struct TPipe {
 TPipe(){for(auto& row:eventFlags)for(bool& f:row)f=false;for(auto& row:allocated)for(bool& f:row)f=false;for(int i=0;i<3;++i)allocated[int(HardEvent::M_MTE1)][i]=true;}
 ~TPipe(){PumpEvents();for(auto& row:eventFlags)for(bool f:row)assert(!f);for(int e=0;e<6;++e)for(int i=0;i<8;++i)assert(allocated[e][i]==(e==int(HardEvent::M_MTE1)&&i<3));}
 int FetchEventID(HardEvent e){for(int i=0;i<8;++i)if(!allocated[int(e)][i])return i;throw std::runtime_error("event exhaustion");}
 template<HardEvent E>TEventID AllocEventID(){int i=FetchEventID(E);allocated[int(E)][i]=true;return i;}
 template<HardEvent E>void ReleaseEventID(int i){PumpEvents();assert(allocated[int(E)][i]&&!eventFlags[int(E)][i]);allocated[int(E)][i]=false;}
 template<typename B>void InitBuffer(B& b,size_t n){b.data=std::make_shared<std::vector<uint8_t>>(n,0xcd);}
};
struct Nd2NzParams {uint16_t ndNum=0,nValue=0,dValue=0,srcNdMatrixStride=0,srcDValue=0,dstNzC0Stride=0,dstNzNStride=0,dstNzMatrixStride=0;};
struct LoadData2DParams{uint16_t startIndex=0;uint8_t repeatTimes=0;uint16_t srcStride=0;uint8_t sid=0;uint16_t dstGap=0;bool ifTranspose=false;uint8_t addrMode=0;};
struct MmadParams{uint16_t m=0,n=0,k=0;bool cmatrixInitVal=false,cmatrixSource=true;uint8_t unitFlag=0;};
enum class QuantMode_t {NoQuant};
constexpr int CFG_ROW_MAJOR=0;
struct FixpipeParamsV220 {uint16_t nSize=0,mSize=0,srcStride=0;uint32_t dstStride=0;uint64_t deqScalar=0;QuantMode_t quantPre=QuantMode_t::NoQuant;bool reluEn=false;uint8_t unitFlag=0;bool isChannelSplit=false;uint16_t ndNum=1;uint32_t srcNdStride=0,dstNdStride=0;};
template<typename T>struct InitConstValueParams{uint16_t repeatTimes=0,blockNum=0,dstGap=0;T initValue{};};
template<typename T>void InitConstValue(LocalTensor<T> dst,InitConstValueParams<T> p){
 assert(p.repeatTimes==1&&!p.dstGap&&p.blockNum);
 for(unsigned i=0;i<unsigned(p.blockNum)*32/sizeof(T);++i)dst.put(i,p.initValue);
}
template<typename T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,Nd2NzParams p){
 assert(p.ndNum==1 && p.dstNzNStride==1 && p.dstNzC0Stride%16==0);
 assert(p.srcNdMatrixStride==0 && p.dstNzMatrixStride==0 && p.srcDValue>=p.dValue);
 for(unsigned r=0;r<p.nValue;++r)for(unsigned c=0;c<(p.dValue+15u)/16*16;++c)
   dst.put((c/16)*p.dstNzC0Stride*16+r*16+c%16,c<p.dValue?src.data[r*p.srcDValue+c]:T(0));
}
template<typename T>void LoadData(LocalTensor<T> dst,LocalTensor<T> src,LoadData2DParams p){
 assert(p.startIndex==0 && p.repeatTimes && !p.dstGap && !p.sid && !p.addrMode);
 for(unsigned q=0;q<p.repeatTimes;++q)for(unsigned r=0;r<16;++r)for(unsigned c=0;c<16;++c)
   dst.put(q*256+r*16+c,src.get(q*p.srcStride*256+(p.ifTranspose?c*16+r:r*16+c)));
}
template<typename T>void Mmad(LocalTensor<float> c,LocalTensor<T> a,LocalTensor<T> b,MmadParams p){
 assert(!p.cmatrixSource && !p.unitFlag && p.m%16==0 && p.n%16==0 && p.k%16==0);
 for(unsigned i=0;i<p.m;++i)for(unsigned j=0;j<p.n;++j){
   size_t ci=(j/16)*p.m*16+i*16+j%16;
   float value=p.cmatrixInitVal?0:c.get(ci);
   for(unsigned k=0;k<p.k;++k)
     value+=float(a.get((i/16)*p.k*16+(k/16)*256+(i%16)*16+k%16))*
            float(b.get((k/16)*p.n*16+(j/16)*256+(j%16)*16+k%16));
   c.put(ci,value);
 }
}
template<typename D,typename S,int F>void Fixpipe(GlobalTensor<D> dst,LocalTensor<S> src,FixpipeParamsV220 p){
 assert(F==CFG_ROW_MAJOR && !p.reluEn && !p.unitFlag && !p.isChannelSplit && p.ndNum==1 && p.quantPre==QuantMode_t::NoQuant);
 assert(!p.srcNdStride && !p.dstNdStride && p.srcStride>=p.mSize && p.dstStride>=p.nSize);
 for(unsigned i=0;i<p.mSize;++i)for(unsigned j=0;j<p.nSize;++j)
   dst.data[i*p.dstStride+j]=src.get((j/16)*p.srcStride*16+i*16+j%16);
}
}
