#include "strata/core/persistent_conversation.hpp"
#include "strata/core/on_device.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace strata::core {
namespace {
constexpr const char* slot_signature = "/batch-slot-v1";
[[noreturn]] void bad(const std::string& s){throw std::runtime_error("KV persistence: "+s);}
void tokens(PersistentEncoder& e,const std::vector<int32_t>& ids){e.number(ids.size());for(auto id:ids)e.number(uint64_t(id));}
std::vector<int32_t> tokens(PersistentDecoder& d,int64_t context){
    uint64_t n=d.number();if(n>uint64_t(context)||n>(d.data.size()-d.at)/8)bad("token extent exceeds context/metadata");
    std::vector<int32_t> ids;ids.reserve(size_t(n));for(uint64_t i=0;i<n;++i){auto id=d.number();if(id>INT32_MAX)bad("invalid token ID");ids.push_back(int32_t(id));}return ids;
}
void pictures(PersistentEncoder& e,const std::vector<ConversationImageKey>& imgs){e.number(imgs.size());for(auto im:imgs){e.number(uint64_t(im.start));e.number(im.hash);}}
std::vector<ConversationImageKey> pictures(PersistentDecoder& d,size_t n){
    uint64_t count=d.number();if(count>n||count>(d.data.size()-d.at)/16)bad("image extent exceeds tokens/metadata");
    std::vector<ConversationImageKey> imgs;int64_t previous=-1;
    for(uint64_t i=0;i<count;++i){auto at=d.number(),hash=d.number();if(at>=n||at>INT64_MAX||int64_t(at)<=previous)bad("invalid image order");previous=int64_t(at);imgs.push_back({previous,hash});}return imgs;
}
void checkpoint_metadata(PersistentEncoder& e,const ConversationCheckpoint& c){
    tokens(e,c.ids);pictures(e,c.imgs);e.number(c.used);e.number(c.stage_parts.size());
    for(const auto& part:c.stage_parts){if(!part.stage_parts.empty())bad("nested stage parts");checkpoint_metadata(e,part);}
}
ConversationCheckpoint checkpoint_metadata(PersistentDecoder& d,int64_t context,size_t stages,bool part=false){
    ConversationCheckpoint c;c.ids=tokens(d,context);c.imgs=pictures(d,c.ids.size());c.used=d.number();auto count=d.number();
    if(count!=(part?0:stages))bad("checkpoint stage count mismatch");
    for(uint64_t i=0;i<count;++i)c.stage_parts.push_back(checkpoint_metadata(d,context,0,true));
    return c;
}
std::vector<PersistentSection> directory(const std::vector<PersistentGpuSpan>& spans){
    std::vector<PersistentSection> out;out.reserve(spans.size());for(const auto& s:spans)out.push_back(s.section);return out;
}
void transfer(const PersistentGpuSpan& s,uint64_t at,uint8_t* buffer,size_t n,bool restoring){
    if(at>s.section.bytes||n>s.section.bytes-at||(!s.address&&n))bad("transfer extent outside authoritative storage");
    if(s.host){if(restoring)std::memcpy(s.address+size_t(at),buffer,n);else std::memcpy(buffer,s.address+size_t(at),n);return;}
    OnDevice device(s.device);auto rc=restoring?cudaMemcpy(s.address+size_t(at),buffer,n,cudaMemcpyDefault):cudaMemcpy(buffer,s.address+size_t(at),n,cudaMemcpyDefault);
    if(rc!=cudaSuccess)bad(std::string("state transfer failed: ")+cudaGetErrorString(rc));
}
}

PersistentConversation::PersistentConversation(const std::string& directory,uint64_t cap,const std::string& identity,
        std::string signature,const ModelGeometry& g,std::vector<PersistentSession> sessions,MtpDrafter& draft,size_t limit)
    :lru_(directory,cap),identity_(identity),signature_(std::move(signature)),g_(g),sessions_(std::move(sessions)),draft_(draft),checkpoint_limit_(limit){
    if(sessions_.empty()||!limit||draft_.device()<0)bad("require sessions, MTP and prompt checkpoints");
    PersistentEncoder shape;shape.text(signature_);
    for(auto n:{g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,g.ssm_d_conv,g.ssm_conv_channels,
                g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff})shape.number(uint64_t(n));
    shape.number(sessions_.size());
    int64_t next=0;
    for(auto s:sessions_){if(!s.state||s.state->layer_lo!=next||s.state->layer_hi<=next)bad("noncontiguous session carve");next=s.state->layer_hi;
        shape.number(uint64_t(s.device));shape.number(uint64_t(s.state->layer_lo));shape.number(uint64_t(s.state->layer_hi));shape.number(uint64_t(s.state->max_cells));}
    if(next!=g.n_layers)bad("session carves do not cover the model");
    shape.number(uint64_t(draft_.device()));
    // Geometry and device carves describe the state independently of the model identity.
    signature_.assign(reinterpret_cast<const char*>(shape.data.data()),shape.data.size());
    loaded_=uint64_t(std::count_if(lru_.entries().begin(),lru_.entries().end(),[&](const auto& e){return e.identity==identity_;}));
    std::fprintf(stderr,"strata serve: KV_PERSIST disk_lru entries=%llu managed_bytes=%llu budget_bytes=%llu\n",(unsigned long long)loaded_,(unsigned long long)lru_.bytes(),(unsigned long long)lru_.budget());
}

void PersistentConversation::read_metadata(KvDisk& disk){
    saved_live_.clear();saved_images_.clear();saved_checks_.clear();
    PersistentDecoder d{disk.metadata()};const auto signature=d.text(65536);
    saved_draft_=signature==signature_;
    if(!saved_draft_&&signature!=signature_+slot_signature)bad("effective engine geometry/configuration differs from committed cache");
    auto cv=d.number();if(cv>1)bad("invalid steering state");saved_cvec_=cv!=0;
    saved_live_=tokens(d,sessions_[0].state->max_cells);if(saved_live_.empty())bad("empty persisted conversation");saved_images_=pictures(d,saved_live_.size());
    auto count=d.number();if(count>checkpoint_limit_)bad("checkpoint count exceeds configuration");
    for(uint64_t i=0;i<count;++i)saved_checks_.push_back(checkpoint_metadata(d,sessions_[0].state->max_cells,sessions_.size()-1));
    d.finish();
    auto expected=directory(spans(saved_checks_,int64_t(saved_live_.size()),false,false,sessions_,saved_draft_));const auto& actual=disk.sections();
    if(expected.size()!=actual.size())bad("persistent section count differs from actual session");
    for(size_t i=0;i<expected.size();++i)if(expected[i].name!=actual[i].name||expected[i].bytes!=actual[i].bytes)bad("persistent section shape differs from actual session");
    for(const auto& c:saved_checks_){
        if(c.ids.empty()||c.ids.size()>saved_live_.size()||!std::equal(c.ids.begin(),c.ids.end(),saved_live_.begin()))bad("checkpoint is not live prefix");
        if(!conversation_image_prefix(c.imgs,c.ids.size(),saved_images_))bad("checkpoint image prefix differs from live conversation");
    }
 }

void PersistentConversation::sync(const std::vector<PersistentSession>& sessions,bool draft){
    for(auto s:sessions){OnDevice on(s.device);auto rc=cudaDeviceSynchronize();if(rc!=cudaSuccess)bad(std::string("session synchronization failed: ")+cudaGetErrorString(rc));}
    if(draft){OnDevice on(draft_.device());std::string error;if(!draft_.idle(error))bad(error);}
}

std::vector<PersistentGpuSpan> PersistentConversation::spans(std::vector<ConversationCheckpoint>& checks,int64_t n,bool allocate,bool saving,
        const std::vector<PersistentSession>& sessions,bool draft){
    std::vector<PersistentGpuSpan> out;
    auto add=[&](std::string name,void* ptr,size_t bytes,bool host,int device){if(bytes&&!ptr&&(saving||allocate))bad("missing state buffer");out.push_back({{std::move(name),uint64_t(bytes)},static_cast<uint8_t*>(ptr),host,device});};
    auto cp=[&](ConversationCheckpoint& c,const PersistentSession& s,const std::string& prefix){
        ConversationStateSizes z;std::string error;OnDevice on(s.device);
        if(!conversation_session_sizes(g_,*s.state,z,error))bad(error);
        const size_t layers=size_t(s.state->qsa_alloc);
        const std::array<size_t,5> sizes={z.gdn,s.state->ple_hist?z.ple:0,layers*z.tail,layers*z.dead,layers*z.block_pos};
        std::array<std::vector<uint8_t>*,5> buffers={&c.gdn,&c.ple,&c.tails,&c.dead,&c.block_pos};
        if(allocate)for(size_t i=0;i<5;++i)buffers[i]->resize(sizes[i]);
        if((saving||allocate)&&!conversation_checkpoint_validate(c,*s.state,g_,error))bad(error);
        for(size_t i=0;i<5;++i)add(prefix+"/"+std::to_string(i),buffers[i]->empty()?nullptr:buffers[i]->data(),sizes[i],true,s.device);
    };
    for(size_t i=0;i<sessions.size();++i){
        auto s=sessions[i];auto& ss=*s.state;OnDevice on(s.device);ConversationStateSizes z;std::string error;
        if(!conversation_session_sizes(g_,ss,z,error))bad(error);
        const auto prefix="stage/"+std::to_string(i);
        add(prefix+"/gdn",ss.gdn_state,z.gdn,false,s.device);add(prefix+"/ple",ss.ple_hist,ss.ple_hist?z.ple:0,false,s.device);
        add(prefix+"/ple_prev",ss.ple_prev,sizeof ss.ple_prev,true,s.device);add(prefix+"/ple_token",&ss.ple_token,sizeof ss.ple_token,true,s.device);
        for(int64_t j=0;j<ss.qsa_alloc;++j){auto& st=ss.qsa_states[ss.qsa_ord0+j];auto qp=prefix+"/qsa/"+std::to_string(ss.qsa_ord0+j);
            add(qp+"/tail",st.idx_tail,z.tail,false,s.device);add(qp+"/dead",st.idx_dead,z.dead,false,s.device);add(qp+"/block_pos",st.idx_block_pos,z.block_pos,false,s.device);
            if(!conversation_kv_spans(st,g_,n,true,qp,s.device,out,error))bad(error);
        }
    }
    if(draft){OnDevice on(draft_.device());std::string error;if(!conversation_kv_spans(draft_.kv_state(),g_,n,false,"draft",draft_.device(),out,error))bad(error);}
    for(size_t i=0;i<checks.size();++i){auto& c=checks[i];if(c.stage_parts.size()!=sessions.size()-1)bad("checkpoint session carve count differs");
        cp(c,sessions[0],"check/"+std::to_string(i)+"/stage/0");
        for(size_t j=1;j<sessions.size();++j){auto& part=c.stage_parts[j-1];if(!part.stage_parts.empty()||part.ids!=c.ids)bad("checkpoint stage token prefix differs");cp(part,sessions[j],"check/"+std::to_string(i)+"/stage/"+std::to_string(j));}
    }
    return out;
}

bool PersistentConversation::restore_if_matching(const std::vector<int64_t>& prompt,const std::vector<ConversationImageKey>& images,
        bool cvec,int64_t resident_prefix,std::vector<int32_t>& live,std::vector<ConversationImageKey>& live_imgs,std::vector<ConversationCheckpoint>& checks,uint64_t& clock){
    uint64_t selected=0,selected_used=0;int64_t best=resident_prefix;
    for(const auto& entry:lru_.entries()){
        if(entry.identity!=identity_)continue;
        KvDisk candidate(lru_.path(entry.id),lru_.budget()*2,identity_);
        if(!candidate.inspect())bad("indexed entry disappeared");
        read_metadata(candidate);
        ConversationCheckpoint current;current.ids=saved_live_;current.imgs=saved_images_;
        int64_t prefix=saved_cvec_==cvec?conversation_prefix(current,prompt,images):0;
        if(saved_cvec_==cvec)for(const auto& checkpoint:saved_checks_)prefix=std::max(prefix,conversation_prefix(checkpoint,prompt,images));
        if(prefix>best||(selected&&prefix==best&&entry.used>selected_used)){best=prefix;selected=entry.id;selected_used=entry.used;}
    }
    if(!selected){saved_live_.clear();saved_images_.clear();saved_checks_.clear();return false;}
    KvDisk disk(lru_.path(selected),lru_.budget()*2,identity_);
    if(!disk.inspect())bad("selected entry disappeared");
    read_metadata(disk);
    sync(sessions_,true);checks.clear();auto buffers=spans(saved_checks_,int64_t(saved_live_.size()),true,false,sessions_,saved_draft_);
    auto t0=std::chrono::steady_clock::now();bytes_=disk.file_bytes();
    disk.restore(directory(buffers),[&](size_t i,uint64_t at,const uint8_t* b,size_t n){transfer(buffers.at(i),at,const_cast<uint8_t*>(b),n,true);});
    for(auto s:sessions_){OnDevice on(s.device);std::string error;for(int64_t j=0;j<s.state->qsa_alloc;++j)if(!conversation_kv_residency_restore(s.state->qsa_states[s.state->qsa_ord0+j],g_,int64_t(saved_live_.size()),error))bad(error);}
    {OnDevice on(draft_.device());std::string error;
        if(saved_draft_){if(!conversation_kv_residency_restore(draft_.kv_state(),g_,int64_t(saved_live_.size()),error))bad(error);}
        else {
            qsa_state_zero(draft_.kv_state_rw(),g_,nullptr);
            std::vector<PersistentGpuSpan> empty;
            if(!conversation_kv_spans(draft_.kv_state(),g_,int64_t(saved_live_.size()),false,"draft",draft_.device(),empty,error))bad(error);
            for(const auto& span:empty){
                if(span.host)std::memset(span.address,0,size_t(span.section.bytes));
                else if(span.section.bytes){auto status=cudaMemset(span.address,0,size_t(span.section.bytes));if(status!=cudaSuccess)bad(cudaGetErrorString(status));}
            }
            if(!conversation_kv_residency_restore(draft_.kv_state(),g_,int64_t(saved_live_.size()),error))bad(error);
        }
    }
    sync(sessions_,true);for(const auto& span:buffers)read_bytes_+=span.section.bytes;
    const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();restore_ms_+=elapsed;
    live=std::move(saved_live_);live_imgs=std::move(saved_images_);checks=std::move(saved_checks_);for(const auto& c:checks)clock=std::max(clock,c.used);
    restored_+=uint64_t(best);lru_.touch(selected);
    std::fprintf(stderr,"strata serve: KV_PERSIST restored entry=%llu tokens=%lld stored_tokens=%zu bytes=%llu stages=%zu draft=%d ms=%.1f\n",(unsigned long long)selected,(long long)best,live.size(),(unsigned long long)bytes_,sessions_.size(),saved_draft_,elapsed);return true;
}

void PersistentConversation::save(const std::vector<int32_t>& live,const std::vector<ConversationImageKey>& imgs,const std::vector<ConversationCheckpoint>& checks,bool cvec){
    save_state(live,imgs,checks,cvec,sessions_,true);
}
void PersistentConversation::save_slot(const std::vector<int32_t>& live,const std::vector<ConversationImageKey>& imgs,
        const std::vector<ConversationCheckpoint>& checks,bool cvec,const std::vector<PersistentSession>& sessions){
    save_state(live,imgs,checks,cvec,sessions,false);
}
void PersistentConversation::save_state(const std::vector<int32_t>& live,const std::vector<ConversationImageKey>& imgs,
        const std::vector<ConversationCheckpoint>& checks,bool cvec,const std::vector<PersistentSession>& sessions,bool draft){
    if(live.empty())bad("cannot commit empty conversation");
    sync(sessions,draft);PersistentEncoder e;e.text(draft?signature_:signature_+slot_signature);e.number(cvec?1:0);tokens(e,live);pictures(e,imgs);e.number(checks.size());
    if(checks.size()>checkpoint_limit_)bad("checkpoint limit exceeded");
    for(const auto& c:checks)checkpoint_metadata(e,c);
    // Existing checkpoints are read-only while the serialized request owns the engine.
    auto buffers=spans(const_cast<std::vector<ConversationCheckpoint>&>(checks),int64_t(live.size()),false,true,sessions,draft);
    auto t0=std::chrono::steady_clock::now();auto entry=lru_.save(identity_,e.data,directory(buffers),[&](size_t i,uint64_t at,uint8_t* b,size_t n){transfer(buffers.at(i),at,b,n,false);});++saved_;
    const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    commit_ms_+=elapsed;bytes_=lru_.entries().back().bytes;write_bytes_+=bytes_;
    std::fprintf(stderr,"strata serve: KV_PERSIST saved entry=%llu tokens=%zu bytes=%llu stages=%zu draft=%d commits=%llu ms=%.1f\n",(unsigned long long)entry,live.size(),(unsigned long long)bytes_,sessions.size(),draft,(unsigned long long)saved_,elapsed);
}
} // namespace strata::core
