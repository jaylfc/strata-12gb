#include "strata/core/kv_persistence.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <io.h>
#include <process.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#endif

namespace strata::core {
namespace {
constexpr size_t chunk = 65536;
constexpr uint64_t max_metadata = 32 * 1024 * 1024;
constexpr uint64_t max_sections = 16384;
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error("KV persistence: " + s); }
uint64_t plus(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) fail("size overflow");
    return a + b;
}
void seek(FILE* f,uint64_t n) {
    if(n>uint64_t(INT64_MAX))fail("file offset overflow");
#if defined(_WIN32)
    if(_fseeki64(f,static_cast<int64_t>(n),SEEK_SET)!=0)fail("seek failed");
#else
    if(fseeko(f,static_cast<off_t>(n),SEEK_SET)!=0)fail("seek failed");
#endif
}
void get(FILE* f,void* dst,size_t n){if(n&&std::fread(dst,1,n,f)!=n)fail("truncated or unreadable file");}
void put(FILE* f,const void* src,size_t n){if(n&&std::fwrite(src,1,n,f)!=n)fail("disk write failed");}
uint64_t file_size(FILE* f) {
#if defined(_WIN32)
    if(_fseeki64(f,0,SEEK_END)!=0) fail("cannot size file");
    auto n=_ftelli64(f);
#else
    if(fseeko(f,0,SEEK_END)!=0) fail("cannot size file");
    auto n=ftello(f);
#endif
    if(n<0) fail("invalid file size");
    seek(f,0); return uint64_t(n);
}
FILE* open_read(const std::filesystem::path& path) {
#if defined(_WIN32)
    HANDLE h=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE)fail("cannot open committed cache with write exclusion");
    int fd=_open_osfhandle(reinterpret_cast<intptr_t>(h),_O_RDONLY|_O_BINARY);
    if(fd<0){CloseHandle(h);fail("cannot create read descriptor");}
    FILE* f=_fdopen(fd,"rb");if(!f){_close(fd);fail("cannot create read stream");}return f;
#else
    FILE* f=std::fopen(path.c_str(),"rb");if(!f)fail("cannot open committed cache");return f;
#endif
}
FILE* open_new(const std::filesystem::path& path) {
#if defined(_WIN32)
    HANDLE h=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE)fail("temporary cache path exists or is not writable");
    int fd=_open_osfhandle(reinterpret_cast<intptr_t>(h),_O_WRONLY|_O_BINARY);
    if(fd<0){CloseHandle(h);fail("cannot create write descriptor");}
    FILE* f=_fdopen(fd,"wb");if(!f){_close(fd);fail("cannot create write stream");}return f;
#else
    int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)fail("temporary cache path exists or is not writable");
    FILE* f=fdopen(fd,"wb");if(!f){::close(fd);fail("cannot create write stream");}return f;
#endif
}
void flush(FILE* f) {
    if(std::fflush(f)!=0)fail("flush failed");
#if defined(_WIN32)
    if(!FlushFileBuffers(reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(f)))))fail("durable flush failed");
#else
    if(::fsync(fileno(f))!=0)fail("durable flush failed");
#endif
}
void replace(const std::filesystem::path& from,const std::filesystem::path& to) {
#if defined(_WIN32)
    if(!MoveFileExW(from.c_str(),to.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))fail("atomic durable replacement failed");
#else
    if(::rename(from.c_str(),to.c_str())!=0)fail("atomic replacement failed");
    int fd=::open(to.parent_path().c_str(),O_RDONLY|O_DIRECTORY);if(fd<0)fail("cannot open parent directory");
    int rc=::fsync(fd);::close(fd);if(rc!=0)fail("directory flush failed");
#endif
}

}
void PersistentEncoder::number(uint64_t v){for(size_t i=0;i<8;++i)data.push_back(uint8_t(v>>(i*8)));}
void PersistentEncoder::text(const std::string& v){number(v.size());data.insert(data.end(),v.begin(),v.end());}
uint64_t PersistentDecoder::number(){if(at>data.size()||data.size()-at<8)fail("short metadata integer");uint64_t v=0;for(size_t i=0;i<8;++i)v|=uint64_t(data[at++])<<(i*8);return v;}
std::string PersistentDecoder::text(size_t maximum){auto n=number();if(n>maximum||at>data.size()||n>data.size()-at)fail("invalid metadata string extent");std::string v(data.begin()+at,data.begin()+at+size_t(n));at+=size_t(n);return v;}
void PersistentDecoder::finish()const{if(at!=data.size())fail("trailing metadata");}
struct KvDisk::Impl {
    std::filesystem::path path;uint64_t cap;std::string identity;
    FILE* input=nullptr;uint64_t bytes=0,payload=0;std::vector<uint8_t> metadata;std::vector<PersistentSection> sections;
#if defined(_WIN32)
    HANDLE lock = INVALID_HANDLE_VALUE;
#else
    int lock = -1;
#endif
    ~Impl(){
        if(input)std::fclose(input);
#if defined(_WIN32)
        if(lock!=INVALID_HANDLE_VALUE)CloseHandle(lock);
#else
        if(lock>=0)::close(lock);
#endif
    }
};
KvDisk::KvDisk(std::string path,uint64_t cap,std::string identity):impl_(std::make_unique<Impl>()){
    std::u8string utf8(reinterpret_cast<const char8_t*>(path.data()),path.size());
    impl_->path=std::filesystem::path(utf8);impl_->cap=cap;impl_->identity=std::move(identity);
    if(impl_->identity.empty()||impl_->identity.size()>4096)fail("model identity must contain 1..4096 bytes");
    if(!impl_->path.is_absolute()||cap<1024*1024)fail("require absolute path and at least 1 MiB bounded SSD budget");
    if(!std::filesystem::is_directory(impl_->path.parent_path()))fail("cache parent directory does not exist");
    auto lock_path=impl_->path;lock_path+=".lock";
#if defined(_WIN32)
    impl_->lock=CreateFileW(lock_path.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(impl_->lock==INVALID_HANDLE_VALUE)fail("cache namespace is already owned or cannot be locked");
#else
    impl_->lock=::open(lock_path.c_str(),O_RDWR|O_CREAT,0600);
    if(impl_->lock<0||flock(impl_->lock,LOCK_EX|LOCK_NB)!=0)fail("cache namespace is already owned or cannot be locked");
#endif
    auto temp=impl_->path;temp+=".tmp";
    if(std::filesystem::exists(temp)){
        if(!std::filesystem::is_regular_file(temp)||std::filesystem::is_symlink(temp))fail("uncommitted temporary cache is not an ordinary file");
        std::filesystem::remove(temp);
        std::fprintf(stderr,"KV persistence: removed uncommitted temporary file\n");
    }
}
KvDisk::~KvDisk()=default;
bool KvDisk::inspect(){
    auto& x=*impl_;discard_read();
    if(!std::filesystem::exists(x.path))return false;
    x.input=open_read(x.path);x.bytes=file_size(x.input);
    if(x.bytes>x.cap/2||x.bytes<32)fail("cache outside size budget");
    std::array<uint8_t,16> head{};get(x.input,head.data(),head.size());
    if(std::memcmp(head.data(),"STKV0002",8)!=0)fail("unsupported disk format");
    std::vector<uint8_t> v(head.begin()+8,head.end());PersistentDecoder id_len{v};auto len=id_len.number();
    if(len>4096||len>x.bytes-32)fail("identity extent exceeds file");
    std::string identity(size_t(len),'\0');get(x.input,identity.data(),identity.size());
    if(identity!=x.identity)fail("model/layout identity mismatch");
    std::array<uint8_t,16> nums{};get(x.input,nums.data(),nums.size());v.assign(nums.begin(),nums.end());
    PersistentDecoder d{v};uint64_t meta=d.number(),count=d.number();
    if(meta>max_metadata||count>max_sections||meta>x.bytes-32-len)fail("invalid metadata/directory bounds");
    x.metadata.resize(size_t(meta));get(x.input,x.metadata.data(),x.metadata.size());x.payload=32+len+meta;
    for(uint64_t i=0;i<count;++i){
        std::array<uint8_t,8> n{};get(x.input,n.data(),8);v.assign(n.begin(),n.end());PersistentDecoder nd{v};auto size=nd.number();
        if(!size||size>256)fail("invalid section name length");
        std::string name(size_t(size),'\0');get(x.input,name.data(),name.size());get(x.input,n.data(),8);
        v.assign(n.begin(),n.end());PersistentDecoder bd{v};auto bytes=bd.number();
        x.sections.push_back({name,bytes});x.payload=plus(x.payload,plus(16,size));
    }
    uint64_t expected=x.payload;for(const auto& section:x.sections)expected=plus(expected,section.bytes);
    if(expected!=x.bytes)fail("section extents do not equal file length");
    return true;
}
const std::vector<uint8_t>& KvDisk::metadata()const{return impl_->metadata;}
const std::vector<PersistentSection>& KvDisk::sections()const{return impl_->sections;}
uint64_t KvDisk::file_bytes()const{return impl_->bytes;}
void KvDisk::discard_read(){auto& x=*impl_;if(x.input){std::fclose(x.input);x.input=nullptr;}x.metadata.clear();x.sections.clear();x.bytes=0;x.payload=0;}
void KvDisk::restore(const std::vector<PersistentSection>& expected,const PersistentWrite& write){
    auto& x=*impl_;if(!x.input||expected.size()!=x.sections.size())fail("restore directory mismatch");
    for(size_t i=0;i<expected.size();++i)if(expected[i].name!=x.sections[i].name||expected[i].bytes!=x.sections[i].bytes)fail("restore section layout mismatch");
    seek(x.input,x.payload);std::array<uint8_t,chunk> b{};
    for(size_t i=0;i<expected.size();++i)for(uint64_t at=0;at<expected[i].bytes;){auto n=size_t(std::min<uint64_t>(b.size(),expected[i].bytes-at));get(x.input,b.data(),n);write(i,at,b.data(),n);at+=n;}
}
uint64_t KvDisk::encoded_bytes(size_t metadata,const std::vector<PersistentSection>& sections)const{
    uint64_t n=plus(32+impl_->identity.size(),metadata);
    for(const auto& section:sections)n=plus(n,plus(16+section.name.size(),section.bytes));
    return n;
}
void KvDisk::save(const std::vector<uint8_t>& metadata,const std::vector<PersistentSection>& sections,const PersistentRead& read){
    auto& x=*impl_;discard_read();if(metadata.size()>max_metadata||sections.size()>max_sections)fail("metadata/directory exceeds bound");
    PersistentEncoder head;head.data.insert(head.data.end(),{'S','T','K','V','0','0','0','2'});head.text(x.identity);head.number(metadata.size());head.number(sections.size());head.data.insert(head.data.end(),metadata.begin(),metadata.end());
    for(size_t i=0;i<sections.size();++i){const auto& s=sections[i];if(s.name.empty()||s.name.size()>256)fail("invalid section name");for(size_t j=0;j<i;++j)if(sections[j].name==s.name)fail("duplicate section name");head.text(s.name);head.number(s.bytes);}
    uint64_t total=head.data.size();for(const auto& s:sections)total=plus(total,s.bytes);if(total>x.cap/2)fail("snapshot exceeds half of committed+temporary SSD budget");
    uint64_t prior=std::filesystem::exists(x.path)?std::filesystem::file_size(x.path):0;if(plus(prior,total)>x.cap)fail("replacement exceeds total SSD budget");
    auto temp=x.path;temp+=".tmp";FILE* out=open_new(temp);bool committed=false;
    try{
        put(out,head.data.data(),head.data.size());std::array<uint8_t,chunk> b{};
        for(size_t i=0;i<sections.size();++i)for(uint64_t at=0;at<sections[i].bytes;){auto n=size_t(std::min<uint64_t>(b.size(),sections[i].bytes-at));read(i,at,b.data(),n);put(out,b.data(),n);at+=n;}
        flush(out);if(std::fclose(out)!=0){out=nullptr;fail("closing cache failed");}out=nullptr;replace(temp,x.path);committed=true;x.bytes=total;
    }catch(...){if(out)std::fclose(out);if(!committed){std::error_code e;std::filesystem::remove(temp,e);}throw;}
}
} // namespace strata::core
