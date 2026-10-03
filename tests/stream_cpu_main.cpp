#include <cstdlib>
#include <iostream>
struct Aligned {
 float* data=nullptr;
 explicit Aligned(size_t count) { require(posix_memalign(reinterpret_cast<void**>(&data),64,count*4)==0,"allocation");std::fill(data,data+count,123456.0f); }
 ~Aligned(){std::free(data);}
};
static float* scratchStart; static size_t cFloats; static uint32_t bufferCount;
static std::vector<bool> consumed;
static std::vector<unsigned> lastReadFence;
static float* pendingStart=nullptr;static size_t pendingSize=0;
static unsigned asyncIssues=0,asyncWaits=0,overlappedReads=0;
static unsigned coldWaits=0,pendingReads=0,overlappedFinalWrites=0;
static size_t scratchFloats=0;
void trackRead(float* p) {
 auto a=reinterpret_cast<uintptr_t>(p),b=reinterpret_cast<uintptr_t>(scratchStart);
 if(a>=b && a<b+cFloats*4) {
  auto pending=reinterpret_cast<uintptr_t>(pendingStart);
  require(!pendingStart || a<pending || a>=pending+pendingSize*4,"read of pending Cube output");
  if(pendingStart){++overlappedReads;++pendingReads;}
  const size_t i=(a-b)/4;consumed[i]=true;lastReadFence[i]=AscendC::mte2Fences;
 }
}
void trackWrite(float* p) {
 const auto a=reinterpret_cast<uintptr_t>(p),b=reinterpret_cast<uintptr_t>(scratchStart);
 if(pendingStart && a>=b+cFloats*4 && a<b+scratchFloats*4)++overlappedFinalWrites;
}
struct CpuMatmul {
 uint32_t m=0,n=0,k=0,pitch=0,rows=0,cols=0; bool ta=false,tb=false,pending=false,open=false;
 AscendC::GlobalTensor<float> a,b,destination;
 void ready(){require(!pending&&!open,"Matmul reconfigured before wait/End");}
 void SetOrgShape(uint32_t M,uint32_t N,uint32_t K,uint32_t Kb,uint32_t P){ready();require(K==Kb,"K mismatch");m=M;n=N;k=K;pitch=P;}
 void SetSingleShape(uint32_t R,uint32_t C,uint32_t K){ready();require(K==k,"partial K");rows=R;cols=C;}
 void SetTensorA(AscendC::GlobalTensor<float> A,bool T){ready();a=A;ta=T;}
 void SetTensorB(AscendC::GlobalTensor<float> B,bool T){ready();b=B;tb=T;}
 void materialize() {
  using namespace AscendC;
  for(uint32_t r=0;r<rows;++r) for(uint32_t col=0;col<cols;++col) {
   float sum=0;
   for(uint32_t q=0;q<k;++q) sum+=readGm(a.data+(ta?q*m+r:r*k+q))*readGm(b.data+(tb?col*k+q:q*n+col));
   writeGm(destination.data+r*pitch+col,sum);
  }
 }
 template<bool Sync=true>void IterateAll(AscendC::GlobalTensor<float> c,uint8_t atomic=0,bool sequential=false,bool wait=false) {
  using namespace AscendC;ready();
  require(!atomic&&!sequential,"unsupported test call");
  require(Sync || wait,"async call must enable WaitIterateAll");
  size_t slot=0;auto& region=locate(c.data,slot);
  const size_t coreSize=cFloats/blockNum,size=coreSize/bufferCount;
  require(slot>=GetBlockIdx()*coreSize && slot+size<=(GetBlockIdx()+1)*coreSize && slot%size==0,"C slot not owned by core");
  for(size_t i=slot;i<slot+size;++i){
   require(!region.readable[i]||consumed[i],"overwriting unread C");
   require(!region.readable[i]||mte2Fences>lastReadFence[i],"C overwritten before MTE2->Scalar fence");
   region.writes[i]=0;region.readable[i]=false;consumed[i]=false;
  }
  destination=c;open=true;
  if(Sync)materialize();
  else {pending=true;pendingStart=c.data;pendingSize=size;pendingReads=0;++asyncIssues;}
 }
 void WaitIterateAll(){require(pending,"wait without pending Matmul");if(!pendingReads)++coldWaits;materialize();pending=false;pendingStart=nullptr;pendingSize=0;++asyncWaits;}
 void End(){require(!pending&&open,"End without completed call");open=false;}
};
template<bool TA,bool TB>
void check(uint32_t B,uint32_t M,uint32_t N,uint32_t K,uint32_t tileM,uint32_t tileN,uint32_t cores,uint32_t splits,int pattern,uint32_t buffers,bool records,bool chainTasks) {
 using namespace AscendC; using namespace local_baseline;
 ProblemDesc p{B,M,N,K,1,TA,TB};HardwareCaps h{cores,192*1024};
 auto mm=MakePlan(p,h,{tileM,tileN});auto plan=SplitStreamPlan(p,h,mm,splits,buffers,records?ReductionPolicy::Partials:ReductionPolicy::Rows);
 size_t ac=size_t(B)*M*K,bc=size_t(B)*K*N,sc=plan.scratchBytes/4;
 Aligned aa(ac+32),bb(bc+32),ss(sc+32),yy(B+32);
 float* a=aa.data+16;float* b=bb.data+16;float* s=ss.data+16;float* y=yy.data+16;
 regions={{a,ac,true,std::vector<unsigned>(ac),std::vector<bool>(ac,true)},
          {b,bc,true,std::vector<unsigned>(bc),std::vector<bool>(bc,true)},
          {s,sc,false,std::vector<unsigned>(sc),std::vector<bool>(sc)},
          {y,B,false,std::vector<unsigned>(B),std::vector<bool>(B)}};
 auto ai=[&](uint32_t batch,uint32_t r,uint32_t q){return size_t(batch)*M*K+(TA?q*M+r:r*K+q);};
 auto bi=[&](uint32_t batch,uint32_t q,uint32_t c){return size_t(batch)*K*N+(TB?c*K+q:q*N+c);};
 for(uint32_t batch=0;batch<B;++batch) {
  for(uint32_t r=0;r<M;++r)for(uint32_t q=0;q<K;++q){
   float v=(int((batch*11+r*7+q*3)%17)-8)*0.125f;
   if(pattern==1)v=-std::abs(v)-0.125f;
   if(pattern==2)v=0;
   if(pattern==3)v=q==r?1:0;
   if(pattern==4){const float vals[]={4096,0.03125f,-4096,-0.015625f};v=q==0?vals[r%4]:0;}
   a[ai(batch,r,q)]=v;
  }
  for(uint32_t q=0;q<K;++q)for(uint32_t c=0;c<N;++c){
   float v=(int((batch*3+q*5+c*11)%19)-9)*0.125f;
   if(pattern==1)v=std::abs(v)+0.125f;
   if(pattern==3)v=((q==0&&c==0)||(q==1&&c==16))?100:0;
   if(pattern==4)v=q==0?1:0;
   b[bi(batch,q,c)]=v;
  }
 }
 std::vector<float> beforeA(a,a+ac),beforeB(b,b+bc),golden(B);
 for(uint32_t batch=0;batch<B;++batch){double sum=0;
  for(uint32_t r=0;r<M;++r){double best=-std::numeric_limits<double>::infinity();
   for(uint32_t c=0;c<N;++c){double dot=0;for(uint32_t q=0;q<K;++q)dot+=double(a[ai(batch,r,q)])*b[bi(batch,q,c)];best=std::max(best,dot);}
   sum+=best;
  }golden[batch]=sum;
 }
 if(pattern==3)require(golden[0]==200,"bad Max-before-Sum counterexample");
 blockNum=plan.blocks;scratchStart=s;cFloats=plan.maximaOffset/4;consumed.assign(cFloats,false);lastReadFence.assign(cFloats,0);readHook=trackRead;
 bufferCount=buffers;asyncIssues=asyncWaits=overlappedReads=0;scalarReads=0;wholeMaxCalls=0;
 coldWaits=pendingReads=overlappedFinalWrites=0;scratchFloats=sc;writeHook=trackWrite;
 std::vector<TPipe> pipes(blockNum);
 for(blockIdx=0;blockIdx<blockNum;++blockIdx){CpuMatmul cpu;StreamProduce<float,TA,TB>(pipes[blockIdx],cpu,(GM_ADDR)a,(GM_ADDR)b,(GM_ADDR)s,B,M,N,K,plan,chainTasks);require(!cpu.pending&&!cpu.open,"producer finished with pending/unclosed Matmul");}
 uint64_t rowChunks=0;for(uint32_t row=0;row<M;row+=tileM)rowChunks+=(std::min(tileM,M-row)+31)/32;
 require(wholeMaxCalls==B*rowChunks*((N+tileN-1)/tileN),"Stream still repeats horizontal Max within C tile");
 if(splits>1)for(blockIdx=0;blockIdx<blockNum;++blockIdx)MergeStreamMaxima(pipes[blockIdx],(GM_ADDR)s,B,M,plan);
 require(scalarReads==0,"producer/merge unexpectedly reads Scalar");
 for(blockIdx=0;blockIdx<blockNum;++blockIdx)if(records)FinalizePartialSums(pipes[blockIdx],(GM_ADDR)s+plan.maximaOffset,(GM_ADDR)y,B,plan.reduction);else SumRowMaxima(pipes[blockIdx],(GM_ADDR)s+plan.maximaOffset,(GM_ADDR)y,B,M,plan.rowPitch);
 readHook=nullptr;writeHook=nullptr;
 require(asyncIssues==asyncWaits,"async work not drained");
 if(buffers==2){require(asyncIssues>0,"double buffer path never issues async Matmul");if((N+tileN-1)/tileN>splits)require(overlappedReads>0,"no producer/consumer overlap");}
 else require(!asyncIssues&&!overlappedReads,"legacy path changed to async");
 if(buffers==2){
  require(coldWaits==(chainTasks?plan.blocks:plan.tasks),"unexpected pipeline cold-wait count");
  if(chainTasks && plan.tasks>plan.blocks)require(overlappedFinalWrites>0,"next Matmul does not overlap previous task finalization");
  else require(!overlappedFinalWrites,"unexpected cross-task overlap in legacy/single-task schedule");
  require(asyncIssues==uint64_t(B)*((M+tileM-1)/tileM)*((N+tileN-1)/tileN),"missing/duplicate Matmul tiles");
 }
 for(uint32_t batch=0;batch<B;++batch)require(y[batch]==golden[batch]&&regions[3].writes[batch]==1,"wrong/missing output");
 require(std::equal(a,a+ac,beforeA.begin())&&std::equal(b,b+bc,beforeB.begin()),"input mutation");
 for(size_t i=0;i<16;++i)require(ss.data[i]==123456 &&ss.data[16+sc+i]==123456&&yy.data[i]==123456&&yy.data[16+B+i]==123456,"guard mutation");
 for(auto& pipe:pipes)require(pipe.bytes==plan.ubBytes,"UB accounting differs from real helper allocation");
}
template<bool TA,bool TB>void layouts(){
 for(uint32_t buffers:{1u,2u})for(bool records:{false,true})for(bool chainTasks:{false,true}){
 for(uint32_t tileM:{16u,64u,256u})for(uint32_t tileN:{16u,64u,256u})for(uint32_t edge:{0u,1u}){
  auto m=tileM+edge,n=tileN*2+edge;
  for(uint32_t splits:{1u,2u}) check<TA,TB>(3,m,n,7,tileM,tileN,3,splits,edge,buffers,records,chainTasks);
 }
 check<TA,TB>(9,17,33,257,16,16,24,2,0,buffers,records,chainTasks);
 check<TA,TB>(1,2,32,2,16,16,24,2,3,buffers,records,chainTasks);
 check<TA,TB>(2,1027,1,1,256,16,3,1,4,buffers,records,chainTasks);
 check<TA,TB>(1,1,1,1,16,16,24,1,2,buffers,records,chainTasks);
 check<TA,TB>(5,65,97,8,32,16,2,3,1,buffers,records,chainTasks);
 check<TA,TB>(2,17,65,8,16,16,4,4,0,buffers,records,chainTasks);
 check<TA,TB>(9,1,1,8,16,16,3,1,0,buffers,records,chainTasks);
}}
int main(){layouts<false,false>();layouts<false,true>();layouts<true,false>();layouts<true,true>();}
