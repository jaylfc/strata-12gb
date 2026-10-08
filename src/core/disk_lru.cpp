#include "strata/core/disk_lru.hpp"
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace strata::core {
namespace {
constexpr uint64_t index_budget=2*1024*1024;
std::filesystem::path native(const std::string& s){return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));}
std::string utf8(const std::filesystem::path& p){auto s=p.u8string();return {reinterpret_cast<const char*>(s.data()),s.size()};}
[[noreturn]] void fail(const std::string& s){throw std::runtime_error("KV disk LRU: "+s);}
}
DiskLru::DiskLru(std::string directory,uint64_t budget):directory_(utf8(std::filesystem::absolute(native(directory)))),budget_(budget){
    if(budget_<=index_budget)fail("require a directory budget above 2 MiB");
    std::filesystem::create_directories(native(directory_));
    index_=std::make_unique<KvDisk>(utf8(native(directory_)/"index.kv"),index_budget,"strata-disk-lru-1");
    if(index_->inspect()){
        PersistentDecoder d{index_->metadata()};clock_=d.number();next_id_=d.number();auto n=d.number();
        if(n>(d.data.size()-d.at)/32)fail("index entry count exceeds file");
        for(uint64_t i=0;i<n;++i){DiskLruEntry e{d.number(),d.number(),d.number(),d.text(4096)};entries_.push_back(std::move(e));}
        d.finish();index_->discard_read();
        for(const auto& e:entries_)if(std::filesystem::file_size(native(path(e.id)))!=e.bytes)fail("indexed entry is missing or has a different extent");
    }else commit_index();
    for(const auto& f:std::filesystem::directory_iterator(native(directory_))){
        const auto name=f.path().filename().string();
        if(!name.starts_with("entry-")||(!name.ends_with(".kv")&&!name.ends_with(".kv.tmp")))continue;
        const auto end=name.size()-(name.ends_with(".tmp")?7:3);
        const std::string digits=name.substr(6,end-6);uint64_t id=0;
        auto parsed=std::from_chars(digits.data(),digits.data()+digits.size(),id);
        if(parsed.ec!=std::errc()||parsed.ptr!=digits.data()+digits.size()||!id||id>=next_id_)continue;
        if(std::any_of(entries_.begin(),entries_.end(),[&](const auto& e){return e.id==id;}))continue;
        if(!name.ends_with(".tmp")){
            std::ifstream file(f.path(),std::ios::binary);char magic[8]{};file.read(magic,8);
            if(std::string(magic,8)!="STKV0002")fail("reserved orphan entry has an unknown format");
        }
        std::filesystem::remove(f.path());
        std::fprintf(stderr,"strata serve: KV_PERSIST removed_uncommitted entry=%llu\n",(unsigned long long)id);
    }
    evict_for(0);
}
std::string DiskLru::path(uint64_t id)const{return utf8(native(directory_)/("entry-"+std::to_string(id)+".kv"));}
uint64_t DiskLru::bytes()const{uint64_t n=0;for(const auto& e:entries_)n+=e.bytes;return n;}
void DiskLru::commit_index(){
    PersistentEncoder e;e.number(clock_);e.number(next_id_);e.number(entries_.size());
    for(const auto& entry:entries_){e.number(entry.id);e.number(entry.used);e.number(entry.bytes);e.text(entry.identity);}
    index_->save(e.data,{},[](size_t,uint64_t,uint8_t*,size_t){});
}
void DiskLru::evict_for(uint64_t incoming){
    if(incoming>budget_-index_budget)fail("snapshot exceeds configured directory budget");
    while(bytes()>budget_-index_budget-incoming){
        const auto it=std::min_element(entries_.begin(),entries_.end(),[](const auto& a,const auto& b){return a.used<b.used;});
        if(it==entries_.end())fail("cannot reserve snapshot space");
        const auto id=it->id;entries_.erase(it);commit_index();std::filesystem::remove(native(path(id)));
        std::fprintf(stderr,"strata serve: KV_PERSIST evicted entry=%llu reason=lru\n",(unsigned long long)id);
    }
}
void DiskLru::touch(uint64_t id){
    auto it=std::find_if(entries_.begin(),entries_.end(),[&](const auto& e){return e.id==id;});
    if(it==entries_.end())fail("selected entry is absent from index");
    it->used=++clock_;commit_index();
}
uint64_t DiskLru::save(const std::string& identity,const std::vector<uint8_t>& metadata,const std::vector<PersistentSection>& sections,const PersistentRead& read){
    const uint64_t id=next_id_++;
    KvDisk disk(path(id),budget_*2,identity);
    const auto total=disk.encoded_bytes(metadata.size(),sections);
    evict_for(total);commit_index();
    disk.save(metadata,sections,read);
    entries_.push_back({id,++clock_,total,identity});commit_index();
    return id;
}
}
