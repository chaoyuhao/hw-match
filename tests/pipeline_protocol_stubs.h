#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <map>
#define __gm__
namespace AscendC {
static bool flags[4];static std::mutex mutex;static std::condition_variable cv;
static std::map<float*,unsigned> reads;
template<int Mode,int Pipe>void CrossCoreSetFlag(unsigned flag){
 static_assert(Mode==2,"wrong participant group");static_assert(Pipe==PIPE_FIX || Pipe==PIPE_MTE2,"wrong publishing pipe");
 std::lock_guard<std::mutex> lock(mutex);require(flag<4 && !flags[flag],"notification overwritten before wait");flags[flag]=true;cv.notify_all();
}
template<int Mode>void CrossCoreWaitFlag(unsigned flag){
 static_assert(Mode==2,"wrong participant group");std::unique_lock<std::mutex> lock(mutex);
 require(flag<4,"reserved flag used");require(cv.wait_for(lock,std::chrono::seconds(3),[&]{return flags[flag];}),"unmatched READY/FREE");flags[flag]=false;
}
inline void observeRead(float*p){std::lock_guard<std::mutex>lock(mutex);++reads[p];}
}
namespace local_baseline {
template<typename T,bool TA,bool TB,unsigned BM,unsigned BN>struct PipelineCubeTile {
 struct Pending{uint64_t batch;unsigned row,rows,col,cols;} pending[2];
 void Init(AscendC::TPipe&,GM_ADDR,GM_ADDR,unsigned,unsigned,unsigned,PipelinePlan){}
 void Compute(uint64_t batch,unsigned row,unsigned rows,unsigned col,unsigned cols,unsigned slot){pending[slot]={batch,row,rows,col,cols};}
 void Finish(){}
 void Store(unsigned slot,AscendC::GlobalTensor<float> dst){
  using namespace AscendC;const auto p=pending[slot];
  std::lock_guard<std::mutex> lock(mutex);
  for(unsigned i=0;i<p.rows;++i)for(unsigned j=0;j<p.cols;++j){
   float*ptr=dst.data+i*BN+j;size_t index=0;auto&r=locate(ptr,index);
   require(!r.input && reads[ptr]==r.writes[index],"C slot overwritten before its final GM read");
   *ptr=float(p.batch)+float((p.row+i)%7)*.125f-float(std::abs(int(p.col+j)-17))*.25f;
   ++r.writes[index];r.readable[index]=true;
  }
 }
};
}
