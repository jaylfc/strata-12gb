#pragma once
#include "strata/core/kv_persistence.hpp"

namespace strata::core {
struct DiskLruEntry { uint64_t id, used, bytes; std::string identity; };
class DiskLru {
public:
    DiskLru(std::string directory, uint64_t budget);
    const std::vector<DiskLruEntry>& entries() const { return entries_; }
    std::string path(uint64_t id) const;
    uint64_t bytes() const;
    uint64_t budget() const { return budget_; }
    void touch(uint64_t id);
    uint64_t save(const std::string& identity, const std::vector<uint8_t>& metadata,
                  const std::vector<PersistentSection>& sections, const PersistentRead& read);
private:
    std::string directory_;
    uint64_t budget_, clock_=0, next_id_=1;
    std::vector<DiskLruEntry> entries_;
    std::unique_ptr<KvDisk> index_;
    void commit_index();
    void evict_for(uint64_t incoming);
};
}
