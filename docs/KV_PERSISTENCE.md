# Disk KV cache

The engine can preserve completed text and image conversations across graceful restarts. Entries contain the actual INT8 K/V, QSA index state, GDN recurrence and convolution state, PLE history, prompt checkpoints for every GPU stage, image identity keys, and the MTP draft K/V. Transfers stream through 64 KiB buffers into the existing state arenas.

Configure the engine flags in the server's existing `args` list:

```json
{
  "args": [
    "--kv-persist",
    "--kv-persist-dir", "data/kv-cache",
    "--kv-persist-max-mib", "30720",
    "--kv-persist-identity", "model-v1:int8"
  ],
  "engine_close_s": 300
}
```

Add these flags to the model's other engine arguments. `--kv-persist-dir` defaults to `data/kv-cache`; relative paths resolve against the engine working directory (`cwd` in the server config). Absolute paths are also supported. `--kv-persist-max-mib` defaults to 30720 MiB, or 30 GiB. These options also work when invoking the engine directly.

`--kv-persist-identity` identifies the model revision, quantization and state representation; change it when those facts change, including MTP weights, RoPE settings or other state-affecting configuration. Entries with different identities share the directory budget and LRU order. Engine geometry, GPU layer carves and KV layout must match before restoration; this is not an automatic fingerprint of all model weights or configuration. The cache supports INT8 serving, including `--vision`, with MTP and prompt checkpoints enabled. RAM conversation parking must be disabled. Batch slots (`--batch`, `--slots`, server `parallel`) share the same disk LRU.

Image keys include their token position, embedding bytes and grid shape. Image-bearing prefixes must match these keys; a text-only prefix before an image can still be shared. Every image request supplies its embeddings and rebuilds/uploads the current M-RoPE position table on every GPU stage before cache selection, including the MTP device. The table is not restored from a previous conversation, whose suffix may differ. Text requests reset it to identity positions. Checkpoint image keys are validated against their live conversation before any restored GPU state is written. These metadata fields already exist in disk format 2; the format is unchanged.

A completed outgoing conversation is saved before a switch or branch rewind. The current completed state is saved on QUIT or stdin EOF. Live continuation and repeated prompts use in-memory state without snapshot writes. Cancelled or incomplete work is not committed. After a process kill, the last committed entries remain available; work since the last save needs another prompt pass.

A completed batch slot is saved before its sessions are overwritten and on QUIT/EOF. Pipelined groups save completed slots before another window's padding can overwrite them; in-flight groups finish before disk transfers start. A slot snapshot contains every GPU stage, its image keys and turn checkpoint. Slots do not run MTP, so their snapshots omit draft K/V and use the explicit `batch-slot-v1` signature. Restoring one into the main session clears draft K/V; new tokens populate the drafter while the main verifier checks every generated token. Solo snapshots retain their MTP state and existing signature. A yielded prompt or cancelled slot is not committed. A request's disk metrics accumulate across its solo, admission and resumed segments.

The cache chooses the longest exact token/checkpoint prefix that exceeds the resident prefix. It restores the entry's state and resumes from its live state or deepest matching checkpoint. Candidate metadata is read one entry at a time. Startup reads the index, without scanning K/V payloads.

One engine owns a cache directory at a time. The durable index records a monotonic usage sequence. The directory budget covers snapshots, temporary writes and 2 MiB reserved for the atomic index. Saves evict the least recently used entries when space is needed. `index.kv`, `entry-N.kv` and their lock/temporary files belong to the cache; files outside that namespace are left intact.

Writes use a temporary file, durable flush and atomic replacement. I/O failures and incompatible selected state end the operation with an error. `engine_close_s` controls the server's orderly shutdown deadline: 300 seconds by default with persistence enabled, 20 otherwise. Unloading succeeds after a successful engine exit.

Engine logs report `KV_PERSIST saved`, `restored` and `evicted`. Request metrics expose `kv_persist_restored_tokens`, `kv_persist_read_bytes`, `kv_persist_restore_ms`, `kv_persist_write_bytes` and `kv_persist_commit_ms` for snapshot transfers. Prompt/decode timing excludes disk switching; request wall time includes it.

Build `kv_persistence_test` with `STRATA_BUILD_CONVERSATION_TESTS=ON` for streamed state roundtrip, atomic failed-write, relative paths and durable global LRU eviction. Server lifecycle and metrics tests are in `serve/test_kv_persistence.py`.

`tools/kv_persistence_batch.py` runs private GPU tests for slot overwrite, cancellation, graceful restart and continuation on both the batch and solo MTP paths. Use `--slots 2 --groups 1` for ordinary slots and `--slots 4 --groups 2` for pipelined groups. Supply `--config`, `--engine` and a new `--output` directory; stop other model engines before running it.

`tools/kv_persistence_vision.py` is a private-engine vision gate. It runs a no-persistence baseline and a disk-cache candidate sequentially, with image/embedding/grid changes, text switching, repeated images and a graceful engine restart. It requires real disk-read/write metrics, matching greedy output and valid primary-GPU state fingerprints; output parity also exercises all configured GPU stages, but is not a bytewise fingerprint of secondary stages. Synthetic embeddings test cache identity and M-RoPE, not the image encoder's quality. The two documented spare-indexer hashes are retained in results but excluded from parity; the valid indexer tail is compared. Run only in an available GPU window, with a new private output directory:

```text
python tools/kv_persistence_vision.py --config CONFIG.json --engine build/strata.exe --tokenizer TOKENIZER_DIR --output NEW_OUTPUT_DIR --run
```

Without `--run` it creates no files and loads no model. CPU verifier tests are in `tools/test_kv_persistence_vision.py`. Production HTTP services and their cache directory are not contacted by this gate.

The restart baseline primes the same image prefix in live memory before the continuation; the candidate restores that prefix from disk. This keeps prefill boundaries equal. A cold full-prompt pass can round differently from a prefix/continuation pass and is not a bitwise baseline for disk restoration.
