#include "strata/core/disk_lru.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace strata::core;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::string utf8(const std::filesystem::path& p){auto s=p.u8string();return {reinterpret_cast<const char*>(s.data()),s.size()};}
std::vector<uint8_t> contents(const std::filesystem::path& p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
}
int main(){
    const auto working_directory=std::filesystem::current_path();
    auto directory=std::filesystem::temp_directory_path()/("strata-kv-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    try{
        std::vector<PersistentSection> sections{{"stage/0/kv",131137},{"stage/1/gdn",37},{"draft/kv",65536}};
        std::vector<std::vector<uint8_t>> payload;
        for(size_t i=0;i<sections.size();++i){payload.emplace_back(size_t(sections[i].bytes));for(size_t j=0;j<payload.back().size();++j)payload.back()[j]=uint8_t(i*31+j*17);}
        PersistentEncoder metadata;metadata.number(262144);metadata.text("iq3_s:int8:split25");
        const auto cache=directory/"codec.kv";
        auto reader=[&](size_t i,uint64_t at,uint8_t* dst,size_t n){require(n<=65536,"stream buffer exceeds 64 KiB");std::copy_n(payload.at(i).data()+size_t(at),n,dst);};
        {
            KvDisk disk(utf8(cache),2*1024*1024,"model-revision:iq3_s:int8");
            require(!disk.inspect(),"absent cache should be a miss");disk.save(metadata.data,sections,reader);
        }
        const auto committed=contents(cache);
        {
            KvDisk disk(utf8(cache),2*1024*1024,"model-revision:iq3_s:int8");
            require(disk.inspect()&&disk.metadata()==metadata.data,"metadata roundtrip");
            std::vector<std::vector<uint8_t>> restored;for(const auto& s:sections)restored.emplace_back(size_t(s.bytes));
            disk.restore(sections,[&](size_t i,uint64_t at,const uint8_t* src,size_t n){std::copy_n(src,n,restored.at(i).data()+size_t(at));});
            require(restored==payload,"all streamed state sections must roundtrip");disk.discard_read();
            bool failed=false;
            try{disk.save(metadata.data,sections,[](size_t,uint64_t,uint8_t*,size_t){throw std::runtime_error("write source unavailable");});}catch(const std::runtime_error&){failed=true;}
            require(failed&&contents(cache)==committed,"failed save must preserve the committed file");
        }
        const auto lru_dir=directory/"lru";
        constexpr uint64_t budget=8*1024*1024;
        const std::vector<PersistentSection> big{{"main-kv",2800000}};
        auto fill=[](size_t,uint64_t at,uint8_t* dst,size_t n){for(size_t i=0;i<n;++i)dst[i]=uint8_t(at+i);};
        uint64_t a,b,c;
        {
            std::filesystem::current_path(directory);
            DiskLru lru("lru",budget);
            std::filesystem::current_path(working_directory);
            a=lru.save("model-a",{},big,fill);b=lru.save("model-b",{},big,fill);lru.touch(a);
            require(lru.entries().size()==2,"two snapshots should fit");
        }
        {
            DiskLru lru(utf8(lru_dir),budget);c=lru.save("model-c",{},big,fill);
            require(lru.entries().size()==2,"directory budget must evict one snapshot");
            require(std::filesystem::exists(lru.path(a))&&!std::filesystem::exists(lru.path(b))&&std::filesystem::exists(lru.path(c)),"durable touch order must evict least recently used across model identities");
            require(lru.bytes()+2*1024*1024<=budget,"payload plus index/temp reservation fits global budget");
        }
        std::filesystem::remove_all(directory);
        std::cout<<"kv_persistence_test: PASS (streamed state roundtrip, atomic write failure, durable global LRU, cross-identity eviction, directory budget)\n";
        return 0;
    }catch(const std::exception& e){std::filesystem::current_path(working_directory);std::cerr<<"kv_persistence_test: FAIL: "<<e.what()<<'\n';std::filesystem::remove_all(directory);return 1;}
}
