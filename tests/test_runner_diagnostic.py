"""Execute the actual runner allocation/diagnostic branches with fake ACL I/O."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class RunnerDiagnosticTests(unittest.TestCase):
    def test_full_similarity_host_storage_is_only_allocated_for_actual_gm_dump(self):
        runner=(ROOT/'local/runner.asc').read_text()
        start=runner.index('        PinnedBuffer aHost(')
        allocations=runner[start:runner.index('        local_baseline::DeviceBuffer aDevice',start)]
        start=runner.index('                if (dump && iteration == 0)')
        block=runner[start:runner.index('                local_baseline::Check(aclrtSynchronize',start)]
        block=re.sub(r'local_baseline::RunKernel\([^;]+\)', 'local_baseline::RunKernel(&actual)', block)
        code=r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
static int hugeAllocations,copies,writes,launches,family;
struct PinnedBuffer { void* data=nullptr; explicit PinnedBuffer(size_t size){if(size>1024*1024)++hugeAllocations;} };
constexpr int ACL_MEMCPY_DEVICE_TO_HOST=1;
int aclrtMemcpy(void*,size_t,void*,size_t,int){++copies;return 0;}
bool WriteFile(const std::string&,void*,size_t){++writes;return true;}
void Require(bool ok,const char*){assert(ok);}
namespace local_baseline {
struct ExecutionInfo { bool isSmall=false,isStream=false; };
struct Buffer{void*data=nullptr;};
Buffer RunKernel(ExecutionInfo* actual){++launches;actual->isSmall=family==1;actual->isStream=family==2;return {};}
void Check(int error,const char*){assert(!error);}
}
void check(bool dump,int iteration) {
 const size_t aBytes=64,bBytes=64,yBytes=4,sBytes=size_t(1)<<40;
 const std::string directory="case";
 hugeAllocations=copies=writes=launches=0;
 local_baseline::ExecutionInfo actual;
'''+allocations+block+r'''
 const bool needDump=dump&&iteration==0&&family==0;
 assert(hugeAllocations==(needDump?1:0));
 assert(copies==(needDump?1:0)&&writes==copies&&launches==1);
}
int main(){for(family=0;family<3;++family)for(bool dump:{false,true})for(int iteration:{0,1})check(dump,iteration);}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp=Path(tmp)/'runner.cpp';binary=Path(tmp)/'runner';cpp.write_text(code)
            done=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
