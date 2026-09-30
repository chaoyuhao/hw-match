from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
class SmallVectorTests(unittest.TestCase):
    def test_actual_device_helper_with_bounded_instruction_doubles(self):
        source = r'''
#include "small_vector.h"
#include <iostream>
using namespace local_baseline;
template<typename T> std::vector<float> run(ProblemDesc p,int pattern,bool focused=false,uint64_t ub=192*1024,bool legacy=false) {
 auto s=MakeSmallPlan(p,{2,ub}); if(s.variant==SmallVariant::None)return {};
 if(legacy){s.dotColumns=0;s.productElements=256;s.partialElements=8;s.ubBytes=6*(s.aElements+s.bElements)+1344;}
 // Keep the exhaustive legacy matrix bounded; new auto selections plus focused
 // probes below cover expansion without turning CPU checks into a stress sweep.
 if(!focused && s.variant==SmallVariant::Rows && p.m*p.k>64 && !UseSmallAutomatically(p,s))return {};
 std::vector<T>a(p.batches*p.m*p.k),b(p.batches*p.k*p.n);
 for(size_t i=0;i<a.size();++i)a[i]=T(pattern==1?0.5f:pattern==2?0.0f:float(int((i*13+7)%31)-15)/16);
 for(size_t i=0;i<b.size();++i)b[i]=T(pattern==1?-0.5f:pattern==3?((i%2)?-1.f:1.f):float(int((i*17+11)%29)-14)/16);
 if(pattern==6){
   uint32_t state=12345;
   for(auto& v:a){state=1664525*state+1013904223;v=T(float(int(state>>8)-8388608)/16777216.f);}
   for(auto& v:b){state=1664525*state+1013904223;v=T(float(int(state>>8)-8388608)/16777216.f);}
 }
 // Near ties in N, with residual K cancellation and exactly representable inputs.
 if(pattern==4)for(uint64_t bi=0;bi<p.batches;++bi)for(uint32_t k=0;k<p.k;++k)for(uint32_t n=0;n<p.n;++n)
   b[bi*p.k*p.n+(p.tb?n*p.k+k:k*p.n+n)]=T((k%2?-1.f:1.f)+(n==0?1.f/128:0));
 // Exposes an accidental half multiply before promotion (256*256 overflows half).
 if(pattern==5){std::fill(a.begin(),a.end(),T(256.f));std::fill(b.begin(),b.end(),T(256.f));}
 if(pattern==7){
   std::fill(a.begin(),a.end(),T(1.f));
   for(uint64_t bi=0;bi<p.batches;++bi)for(uint32_t k=0;k<p.k;++k)for(uint32_t n=0;n<p.n;++n)
     b[bi*p.k*p.n+(p.tb?n*p.k+k:k*p.n+n)]=T(n==0?(k<64?1.f:-1.f):(k<64?-0.25f:0.5f));
 }
 auto beforeA=a,beforeB=b;
 std::vector<float> yStorage(p.batches+23,123456.f);
 auto* y=reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(yStorage.data())+31)&~uintptr_t(31));
 AscendC::owners.clear();AscendC::blockNum=s.blocks;AscendC::vectorToScalar=AscendC::repeatedMulCalls=0;
 for(uint32_t core=0;core<s.blocks;++core){AscendC::blockIdx=core;AscendC::TPipe pipe;
   SmallVectorOp<T> op;op.Init(pipe,(uint8_t*)a.data(),(uint8_t*)b.data(),(uint8_t*)(y+8),p.batches,p.m,p.n,p.k,p.ta,s);op.Process();assert(pipe.bytes==s.ubBytes);}
 if(s.dotColumns){
   assert(AscendC::vectorToScalar==p.batches*(1+p.m));
   assert(AscendC::repeatedMulCalls==p.batches*p.m*CeilDiv(p.n,s.dotColumns)*CeilDiv(p.k,64));
 }
 for(uint64_t bi=0;bi<p.batches;++bi){double expected=0;
   for(uint32_t m=0;m<p.m;++m){double maximum=-INFINITY;
     for(uint32_t n=0;n<p.n;++n){double dot=0;for(uint32_t k=0;k<p.k;++k)
       dot+=double(float(a[bi*p.m*p.k+(p.ta?k*p.m+m:m*p.k+k)]))*double(float(b[bi*p.k*p.n+(p.tb?n*p.k+k:k*p.n+n)]));
       maximum=std::max(maximum,dot);}expected+=maximum;}
   assert(std::isfinite(y[8+bi])&&std::abs(y[8+bi]-expected)<=1e-4+std::abs(expected)*1e-4);
 }
 for(size_t i=0;i<8;++i)assert(y[i]==123456.f);for(size_t i=8+p.batches;i<p.batches+16;++i)assert(y[i]==123456.f);
 assert(std::memcmp(a.data(),beforeA.data(),a.size()*sizeof(T))==0&&std::memcmp(b.data(),beforeB.data(),b.size()*sizeof(T))==0);
 return {y+8,y+8+p.batches};
}
int main(){for(uint64_t b:{1,7,8,9,17,32})for(uint32_t m:{1,2,3,16})for(uint32_t n:{1,5,16,63,64})for(uint32_t k:{8,24,64,72,256})
 for(bool ta:{false,true})for(bool tb:{false,true})for(int pat=0;pat<7;++pat){run<half>({b,m,n,k,1,ta,tb},pat);run<bfloat16_t>({b,m,n,k,2,ta,tb},pat);}
 // Generated probes cross removed B/M/Rows-MK limits, then pair dtype/layout.
 std::vector<ProblemDesc> expanded;
 for(uint64_t b:{33,191,192,193,257})expanded.push_back({b,1,5,8,1,false,false});
 for(uint32_t m:{17,31,32,33,127,128,129,200})expanded.push_back({1,m,1,8,1,false,false});
 for(uint32_t k:{72,128,256})expanded.push_back({1,1,5,k,1,false,false});
 expanded.push_back({1,1,64,128,1,false,false});
 expanded.push_back({1,32,5,8,1,false,false});
 // Column-tile tails and partial 64-lane K chunks.
 for(uint32_t n:{2,7,8,9,15,17,31,33,63})for(uint32_t k:{8,56,64,120,128,136,248,256})
   expanded.push_back({1,2,n,k,1,false,true});
 for(auto p:expanded)for(bool ta:{false,true})for(bool tb:{false,true})for(int pat=0;pat<7;++pat){
   p.ta=ta;p.tb=tb;p.dtype=1;run<half>(p,pat,true);p.dtype=2;run<bfloat16_t>(p,pat,true);
 }
 // Scratch pressure forces 32/16/8-column tiles, then the original Dot path.
 for(uint64_t ub:{192*1024,32768+53600,32768+51520,32768+51519})for(int pat=0;pat<7;++pat){
   run<half>({9,1,63,128,1,false,true},pat,true,ub);
   run<bfloat16_t>({9,1,63,128,2,false,true},pat,true,ub);
 }
 for(int pat=0;pat<8;++pat)for(uint32_t n:{17,64}){
   ProblemDesc p{1,1,n,128,1,false,true};
   auto a=run<half>(p,pat,true),b=run<half>(p,pat,true,192*1024,true);
   assert(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
   p.dtype=2;
   a=run<bfloat16_t>(p,pat,true);b=run<bfloat16_t>(p,pat,true,192*1024,true);
   assert(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
 }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            (tmp/'kernel_operator.h').write_text((ROOT/'tests/small_vector_cpu_stubs.h').read_text())
            (tmp/'main.cpp').write_text(source)
            done=subprocess.run(['/usr/bin/g++','-std=c++14','-O2','-ffp-contract=off','-I',str(tmp),'-I',str(ROOT),str(tmp/'main.cpp'),'-o',str(tmp/'test')],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(tmp/'test')],capture_output=True,text=True,timeout=60)
            self.assertEqual(done.returncode,0,done.stderr)
