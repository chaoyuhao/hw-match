using namespace local_baseline;
template<typename T,bool TA,bool TB>void run(unsigned m,unsigned n,unsigned k,unsigned bm,unsigned bn,bool resident){
 DirectCaps caps{true,resident?512u*1024:24u*1024,65536,65536,131072};
 ProblemDesc p{2,m,n,k,1,TA,TB};auto plan=MakeDirectPlan(p,{bm,bn},caps);
 assert(plan.enabled);if(resident)assert(plan.residentA);
 std::vector<T>a(2*m*k),b(2*k*n);std::vector<float>c(2*m*(n+16),-98765);
 for(unsigned ba=0;ba<2;++ba){
  for(unsigned i=0;i<m;++i)for(unsigned h=0;h<k;++h)
    a[ba*m*k+(TA?h*m+i:i*k+h)]=T(float(int((i*13+h*7+ba*5)%41)-20)/32);
  for(unsigned j=0;j<n;++j)for(unsigned h=0;h<k;++h)
    b[ba*k*n+(TB?j*k+h:h*n+j)]=T(float(int((j*11+h*3+ba*7)%37)-18)/32);
 }
 AscendC::TPipe pipe;DirectCubeTile<T,TA,TB> op;
 op.Init(pipe,(GM_ADDR)a.data(),(GM_ADDR)b.data(),m,n,k,plan);
 AscendC::GlobalTensor<float> out;out.SetGlobalBuffer(c.data());
 for(unsigned ba=0;ba<2;++ba)for(unsigned i=0;i<m;i+=bm)for(unsigned j=0;j<n;j+=bn)
   op.Compute(ba,i,std::min(bm,m-i),j,std::min(bn,n-j),out[ba*m*(n+16)+i*(n+16)+j],n+16);
 for(unsigned ba=0;ba<2;++ba)for(unsigned i=0;i<m;++i){
  for(unsigned j=0;j<n;++j){double want=0;
   for(unsigned h=0;h<k;++h)want+=float(a[ba*m*k+(TA?h*m+i:i*k+h)])*float(b[ba*k*n+(TB?j*k+h:h*n+j)]);
   assert(std::abs(c[ba*m*(n+16)+i*(n+16)+j]-want)<1e-5);
  }
  for(unsigned j=n;j<n+16;++j)assert(c[ba*m*(n+16)+i*(n+16)+j]==-98765);
 }
}
template<typename T,bool TA,bool TB>void suite(){
 run<T,TA,TB>(48,80,144,32,64,true);
 run<T,TA,TB>(32,64,512,32,64,false);
 run<T,TA,TB>(16,32,16,256,256,true);
}
int main(){
 suite<half,0,0>();suite<half,0,1>();suite<half,1,0>();suite<half,1,1>();
 suite<bf16,0,0>();suite<bf16,0,1>();suite<bf16,1,0>();suite<bf16,1,1>();
}
