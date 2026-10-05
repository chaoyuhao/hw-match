using namespace local_baseline;
struct AlignedFloats {
 std::vector<float> owner;size_t count;float*ptr;
 explicit AlignedFloats(size_t n,float value):owner(n+8,value),count(n),ptr(reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(owner.data())+31)&~uintptr_t(31))){}
 float*data(){return ptr;}size_t size()const{return count;}float&operator[](size_t i){return ptr[i];}
};
template<unsigned BM,unsigned BN>void run(unsigned splits,unsigned cores,ReductionPolicy policy){
 using namespace AscendC;
 ProblemDesc p{3,BM+3,BN*4+7,137,1,0,0};HardwareCaps caps{cores,192*1024};
 auto physical=PipelineCandidate(p,caps,{true,524288,65536,65536,131072},BM,BN,splits,policy);
 require(physical.enabled,"pipeline plan rejected");auto s=physical.stream;
 AlignedFloats scratch(s.scratchBytes/4,std::numeric_limits<float>::quiet_NaN()),out(p.batches,0);
 regions.clear();reads.clear();regions.push_back({scratch.data(),scratch.size(),false,std::vector<unsigned>(scratch.size()),std::vector<bool>(scratch.size(),false)});
 regions.push_back({out.data(),out.size(),false,std::vector<unsigned>(out.size()),std::vector<bool>(out.size(),false)});
 readHook=observeRead;blockNum=s.blocks;
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){
  TPipe cube,vec;
  std::thread producer([&]{PipelineStreamCube<float,false,false,BM,BN>(cube,nullptr,nullptr,(GM_ADDR)scratch.data(),p.m,p.n,p.k,physical);});
  std::thread consumer([&]{PipelineStreamVector(vec,(GM_ADDR)scratch.data(),p.m,p.n,s);});
  producer.join();consumer.join();
  for(bool f:flags)require(!f,"final slot notification was not drained");
 }
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){TPipe pipe;if(s.splits>1)MergeStreamMaxima(pipe,(GM_ADDR)scratch.data(),p.batches,p.m,s);}
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){TPipe pipe;
  if(s.reduction.mode)FinalizePartialSums(pipe,(GM_ADDR)scratch.data()+s.maximaOffset,(GM_ADDR)out.data(),p.batches,s.reduction);
  else SumRowMaxima(pipe,(GM_ADDR)scratch.data()+s.maximaOffset,(GM_ADDR)out.data(),p.batches,p.m,s.rowPitch);
 }
 for(unsigned batch=0;batch<p.batches;++batch){double want=0;for(unsigned row=0;row<p.m;++row)want+=batch+(row%7)*.125;
  require(out[batch]==want,"cross-owner max/sum is wrong");}
}
int main(){for(unsigned splits:{1u,2u,4u})for(unsigned cores:{1u,7u,24u})for(auto policy:{ReductionPolicy::Rows,ReductionPolicy::Partials}){
 run<16,32>(splits,cores,policy);run<32,64>(splits,cores,policy);run<64,128>(splits,cores,policy);run<128,128>(splits,cores,policy);
}}
