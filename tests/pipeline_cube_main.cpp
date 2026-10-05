
using namespace local_baseline;
template<typename T,bool TA,bool TB,unsigned BM,unsigned BN>
void run(unsigned m,unsigned n,unsigned k,bool resident){
 DirectCaps caps{true,resident?524288u:24576u,65536,65536,131072};
 ProblemDesc p{2,m,n,k,1,TA,TB};auto q=PipelineCandidate(p,{7,196608},caps,BM,BN,1,ReductionPolicy::Partials);
 assert(q.enabled);assert(q.residentA==resident);
 std::vector<T>a(2*m*k),b(2*k*n);std::vector<float>tile(2*BM*BN+16,-98765);
 for(unsigned ba=0;ba<2;++ba){
  for(unsigned i=0;i<m;++i)for(unsigned h=0;h<k;++h)
   a[ba*m*k+(TA?h*m+i:i*k+h)]=T(float(int((i*13+h*7+ba*5)%41)-20)/32);
  for(unsigned j=0;j<n;++j)for(unsigned h=0;h<k;++h)
   b[ba*k*n+(TB?j*k+h:h*n+j)]=T(float(int((j*11+h*3+ba*7)%37)-18)/32);
 }
 AscendC::TPipe pipe;PipelineCubeTile<T,TA,TB,BM,BN> op;
 op.Init(pipe,(GM_ADDR)a.data(),(GM_ADDR)b.data(),m,n,k,q);
 AscendC::GlobalTensor<float> out;out.SetGlobalBuffer(tile.data());unsigned seq=0;
 for(unsigned ba=0;ba<2;++ba)for(unsigned i=0;i<m;i+=BM)for(unsigned j=0;j<n;j+=BN,++seq){
  const unsigned slot=seq%2,rows=std::min(BM,m-i),cols=std::min(BN,n-j);
  op.Compute(ba,i,rows,j,cols,slot);op.Store(slot,out[slot*BM*BN]);
  for(unsigned r=0;r<BM;++r)for(unsigned c=0;c<BN;++c){double want=0;
   if(r<rows && c<cols)for(unsigned h=0;h<k;++h)
    want+=float(a[ba*m*k+(TA?h*m+i+r:(i+r)*k+h)])*float(b[ba*k*n+(TB?(j+c)*k+h:h*n+j+c)]);
   assert(std::abs(tile[slot*BM*BN+r*BN+c]-want)<1e-5);
  }
 }
 op.Finish();for(unsigned i=2*BM*BN;i<tile.size();++i)assert(tile[i]==-98765);
}
template<typename T,bool A,bool B>void suite(){
 run<T,A,B,16,32>(1,1,1,true);
 run<T,A,B,16,32>(17,35,8,true);
 run<T,A,B,32,64>(35,67,137,true);
 run<T,A,B,32,64>(35,67,520,false);
 run<T,A,B,64,128>(65,131,24,true);
 run<T,A,B,128,32>(129,33,16,true);
}
int main(){
 suite<half,0,0>();suite<half,0,1>();suite<half,1,0>();suite<half,1,1>();
 suite<bf16,0,0>();suite<bf16,0,1>();suite<bf16,1,0>();suite<bf16,1,1>();
}
