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
template<typename T,bool TA,bool TB>struct DirectCubeTile {
 void Init(AscendC::TPipe&,GM_ADDR,GM_ADDR,unsigned,unsigned,unsigned,DirectPlan){}
 void Compute(uint64_t batch,unsigned row,unsigned rows,unsigned col,unsigned cols,AscendC::GlobalTensor<float> dst,unsigned pitch){
  using namespace AscendC;
  std::lock_guard<std::mutex> lock(mutex);
  for(unsigned i=0;i<rows;++i)for(unsigned j=0;j<cols;++j){
   float*ptr=dst.data+i*pitch+j;size_t index=0;auto&r=locate(ptr,index);
   require(!r.input && reads[ptr]==r.writes[index],"C slot overwritten before its final GM read");
   *ptr=float(batch)+float((row+i)%7)*.125f-float(std::abs(int(col+j)-17))*.25f;
   ++r.writes[index];r.readable[index]=true;
  }
 }
};
}
