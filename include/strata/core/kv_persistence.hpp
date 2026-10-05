#pragma once
// Pointer-free, bounded disk snapshot transport. No CUDA dependency.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
struct PersistentSection { std::string name; uint64_t bytes = 0; };
using PersistentRead = std::function<void(size_t, uint64_t, uint8_t*, size_t)>;
using PersistentWrite = std::function<void(size_t, uint64_t, const uint8_t*, size_t)>;
class KvDisk {
public:
    // total_disk_bytes bounds committed + temporary file together; one file may use at most half.
    KvDisk(std::string path, uint64_t total_disk_bytes, std::string identity);
    ~KvDisk();
    KvDisk(const KvDisk&) = delete;
    KvDisk& operator=(const KvDisk&) = delete;
    bool inspect(); // reads metadata and section extents; state payload is read only on restore
    uint64_t encoded_bytes(size_t metadata_bytes, const std::vector<PersistentSection>& sections) const;
    const std::vector<uint8_t>& metadata() const;
    const std::vector<PersistentSection>& sections() const;
    uint64_t file_bytes() const;
    void restore(const std::vector<PersistentSection>& expected, const PersistentWrite& write);
    void discard_read();
    void save(const std::vector<uint8_t>& metadata, const std::vector<PersistentSection>& sections,
              const PersistentRead& read);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Metadata integers are unsigned little-endian; readers enforce bounds before allocating.
struct PersistentEncoder {
    std::vector<uint8_t> data;
    void number(uint64_t value);
    void text(const std::string& value);
};
struct PersistentDecoder {
    const std::vector<uint8_t>& data;
    size_t at = 0;
    uint64_t number();
    std::string text(size_t maximum = 256);
    void finish() const;
};
} // namespace strata::core
