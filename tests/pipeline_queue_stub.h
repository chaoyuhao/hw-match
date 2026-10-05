template<TPosition P,int N>struct TQue {
 std::shared_ptr<std::vector<float>> memory[N];int state[N]{};std::deque<int> fifo;
 void Init(int depth,size_t bytes){require(depth==N,"wrong queue depth");for(auto& m:memory)m=std::make_shared<std::vector<float>>(bytes/4,std::numeric_limits<float>::quiet_NaN());}
 template<typename T>LocalTensor<T> AllocTensor(){for(int i=0;i<N;++i)if(!state[i]){state[i]=1;return{memory[i],0};}throw std::runtime_error("queue over-allocated");}
 template<typename T>int Index(LocalTensor<T> t){for(int i=0;i<N;++i)if(t.data==memory[i])return i;throw std::runtime_error("foreign tensor");}
 template<typename T>void EnQue(LocalTensor<T> t){int i=Index(t);require(state[i]==1,"bad enqueue");state[i]=2;fifo.push_back(i);}
 template<typename T>LocalTensor<T> DeQue(){require(!fifo.empty(),"empty queue");int i=fifo.front();fifo.pop_front();require(state[i]==2,"bad dequeue");state[i]=3;return{memory[i],0};}
 template<typename T>void FreeTensor(LocalTensor<T> t){int i=Index(t);require(state[i]==3,"bad free");state[i]=0;}
};
