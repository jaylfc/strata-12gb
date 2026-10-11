#pragma once
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/disk_lru.hpp"

namespace strata::core {
struct PersistentSession { int device; SessionState* state; };
class PersistentConversation {
public:
    PersistentConversation(const std::string& directory, uint64_t disk_bytes, const std::string& identity,
                           std::string signature, const ModelGeometry& geometry,
                           std::vector<PersistentSession> sessions, MtpDrafter& draft,
                           size_t checkpoint_limit);
    // Exact prefix search uses metadata. The selected file is fully validated before restoring any state.
    bool restore_if_matching(const std::vector<int64_t>& prompt, const std::vector<ConversationImageKey>& images,
                             bool cvec, int64_t resident_prefix, std::vector<int32_t>& live, std::vector<ConversationImageKey>& live_images,
                             std::vector<ConversationCheckpoint>& checkpoints, uint64_t& clock);
    void save(const std::vector<int32_t>& live, const std::vector<ConversationImageKey>& images,
              const std::vector<ConversationCheckpoint>& checkpoints, bool cvec);
    void save_slot(const std::vector<int32_t>& live, const std::vector<ConversationImageKey>& images,
                   const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                   const std::vector<PersistentSession>& sessions);
    uint64_t loaded() const { return loaded_; }
    uint64_t restored() const { return restored_; }
    uint64_t saved() const { return saved_; }
    uint64_t bytes() const { return bytes_; }
    uint64_t entries() const { return lru_.entries().size(); }
    uint64_t managed_bytes() const { return lru_.bytes(); }
    uint64_t read_bytes() const { return read_bytes_; }
    uint64_t write_bytes() const { return write_bytes_; }
    double restore_ms() const { return restore_ms_; }
    double commit_ms() const { return commit_ms_; }
private:
    DiskLru lru_;
    std::string identity_;
    uint64_t bytes_=0;
    std::string signature_;
    const ModelGeometry& g_;
    std::vector<PersistentSession> sessions_;
    MtpDrafter& draft_;
    size_t checkpoint_limit_;
    bool saved_cvec_ = true;
    bool saved_draft_ = true;
    uint64_t loaded_ = 0, restored_ = 0, saved_ = 0;
    uint64_t read_bytes_ = 0, write_bytes_ = 0;
    double restore_ms_ = 0, commit_ms_ = 0;
    std::vector<int32_t> saved_live_;
    std::vector<ConversationImageKey> saved_images_;
    std::vector<ConversationCheckpoint> saved_checks_;
    std::vector<PersistentGpuSpan> spans(std::vector<ConversationCheckpoint>& checks, int64_t tokens,
                                         bool allocate, bool saving,
                                         const std::vector<PersistentSession>& sessions, bool draft);
    void sync(const std::vector<PersistentSession>& sessions, bool draft);
    void save_state(const std::vector<int32_t>& live, const std::vector<ConversationImageKey>& images,
                    const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                    const std::vector<PersistentSession>& sessions, bool draft);
    void read_metadata(KvDisk& disk);
};
} // namespace strata::core
