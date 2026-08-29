#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Qwen3.8-Flash-Next constants from the public model card. The engine is sized
// for a single CUDA/Ampere RTX 3060 12 GB host with 32 GB RAM and PCIe 3.0.
#define GGML_QWEN38_FLASH_NEXT_N_LAYER          48
#define GGML_QWEN38_FLASH_NEXT_N_EXPERT         512
#define GGML_QWEN38_FLASH_NEXT_N_EXPERT_USED    10
#define GGML_QWEN38_FLASH_NEXT_N_SHARED_EXPERT  1
#define GGML_QWEN38_FLASH_NEXT_HIDDEN_SIZE      2560
#define GGML_QWEN38_FLASH_NEXT_EXPERT_FF        640
#define GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS       320001536ull
#define GGML_QWEN38_FLASH_NEXT_NGRAM_LOOKUPS    16
#define GGML_QWEN38_FLASH_NEXT_NGRAM_ROW_BYTES  160
#define GGML_QWEN38_FLASH_NEXT_NGRAM_TABLE_BYTES \
    (GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS * GGML_QWEN38_FLASH_NEXT_NGRAM_ROW_BYTES)
#define GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS_PER_PAGE 256

typedef enum ggml_qwen38_ssd_level {
    GGML_QWEN38_SSD_LEVEL_NONE = 0,
    GGML_QWEN38_SSD_LEVEL_VRAM = 1,
    GGML_QWEN38_SSD_LEVEL_RAM  = 2,
    GGML_QWEN38_SSD_LEVEL_SSD  = 3,
} ggml_qwen38_ssd_level;

typedef enum ggml_qwen38_ssd_state {
    GGML_QWEN38_SSD_STATE_SSD = 0,
    GGML_QWEN38_SSD_STATE_SSD_READ_PENDING = 1,
    GGML_QWEN38_SSD_STATE_RAM_READY = 2,
    GGML_QWEN38_SSD_STATE_H2D_PENDING = 3,
    GGML_QWEN38_SSD_STATE_VRAM_READY = 4,
    GGML_QWEN38_SSD_STATE_EVICT_PENDING = 5,
} ggml_qwen38_ssd_state;

typedef enum ggml_qwen38_ssd_quant {
    GGML_QWEN38_SSD_QUANT_Q4 = 0,
} ggml_qwen38_ssd_quant;

typedef struct ggml_qwen38_ssd_expert_key {
    int32_t layer;
    int32_t expert;
} ggml_qwen38_ssd_expert_key;

typedef struct ggml_qwen38_ssd_expert_meta {
    ggml_qwen38_ssd_expert_key key;
    uint64_t file_offset;
    size_t   size_bytes;
    ggml_qwen38_ssd_quant quant;
    ggml_qwen38_ssd_level resident;
    ggml_qwen38_ssd_state state;
    uint64_t last_use_tick;
    uint32_t frequency;
    int32_t  pin_priority;
} ggml_qwen38_ssd_expert_meta;

typedef struct ggml_qwen38_ssd_config {
    const char * expert_weights_path;
    const char * ngram_lut_path;
    size_t max_vram_cache_bytes;
    size_t max_ram_cache_bytes;
    size_t max_staging_bytes;
    size_t ngram_cache_bytes;
    size_t pinned_staging_bytes;
    int    max_inflight_reads;
    bool   use_direct_io;
    bool   prefer_io_uring;
} ggml_qwen38_ssd_config;

typedef void * ggml_qwen38_cuda_stream;
typedef void * ggml_qwen38_cuda_event;

typedef void * (*ggml_qwen38_ssd_host_alloc_fn)(void * user, size_t size, bool pinned);
typedef void (*ggml_qwen38_ssd_host_free_fn)(void * user, void * ptr, size_t size, bool pinned);
typedef bool (*ggml_qwen38_ssd_upload_fn)(void * user, const ggml_qwen38_ssd_expert_meta * meta,
        const void * host, size_t size, ggml_qwen38_cuda_stream stream,
        ggml_qwen38_cuda_event * done, void ** device_handle);
typedef bool (*ggml_qwen38_ssd_event_ready_fn)(void * user, ggml_qwen38_cuda_event done);
typedef void (*ggml_qwen38_ssd_release_event_fn)(void * user, ggml_qwen38_cuda_event done);
typedef void (*ggml_qwen38_ssd_release_device_fn)(void * user, void * device_handle);

typedef struct ggml_qwen38_ssd_stats {
    uint64_t expert_vram_hits;
    uint64_t expert_ram_hits;
    uint64_t expert_ssd_misses;
    uint64_t expert_duplicate_requests;
    uint64_t expert_h2d_submitted;
    uint64_t expert_h2d_completed;
    uint64_t ngram_hits;
    uint64_t ngram_misses;
    uint64_t ngram_duplicate_requests;
    uint64_t bytes_read;
    uint64_t bytes_uploaded;
    size_t ram_cache_bytes;
    size_t vram_cache_bytes;
    size_t ngram_cache_bytes;
}
ggml_qwen38_ssd_stats;

typedef struct ggml_qwen38_ssd_cuda_hooks {
    ggml_qwen38_ssd_host_alloc_fn host_alloc;
    ggml_qwen38_ssd_host_free_fn host_free;
    ggml_qwen38_ssd_upload_fn upload_async;
    ggml_qwen38_ssd_event_ready_fn event_ready;
    ggml_qwen38_ssd_release_event_fn release_event;
    ggml_qwen38_ssd_release_device_fn release_device;
    ggml_qwen38_cuda_stream h2d_stream;
    void * user;
} ggml_qwen38_ssd_cuda_hooks;

typedef struct ggml_qwen38_ssd_context ggml_qwen38_ssd_context;
typedef struct ggml_qwen38_ssd_ngram_handle ggml_qwen38_ssd_ngram_handle;

ggml_qwen38_ssd_config ggml_qwen38_ssd_default_config(void);
ggml_qwen38_ssd_context * ggml_qwen38_ssd_init(const ggml_qwen38_ssd_config * cfg,
        const ggml_qwen38_ssd_cuda_hooks * hooks);
void ggml_qwen38_ssd_free(ggml_qwen38_ssd_context * ctx);

bool ggml_qwen38_ssd_register_expert(ggml_qwen38_ssd_context * ctx,
        const ggml_qwen38_ssd_expert_meta * meta);

bool ggml_qwen38_ssd_schedule_experts(ggml_qwen38_ssd_context * ctx,
        int32_t layer, const int32_t * selected_ids, int32_t n_selected);
bool ggml_qwen38_ssd_prefetch_experts(ggml_qwen38_ssd_context * ctx,
        int32_t layer, const int32_t * predicted_ids, int32_t n_predicted);
bool ggml_qwen38_ssd_poll(ggml_qwen38_ssd_context * ctx);

ggml_qwen38_ssd_state ggml_qwen38_ssd_get_expert_state(ggml_qwen38_ssd_context * ctx,
        int32_t layer, int32_t expert);
void * ggml_qwen38_ssd_get_device_expert(ggml_qwen38_ssd_context * ctx,
        int32_t layer, int32_t expert);
void * ggml_qwen38_ssd_acquire_device_expert(ggml_qwen38_ssd_context * ctx,
        int32_t layer, int32_t expert);
void ggml_qwen38_ssd_release_device_expert(ggml_qwen38_ssd_context * ctx,
        int32_t layer, int32_t expert);
bool ggml_qwen38_ssd_get_stats(ggml_qwen38_ssd_context * ctx, ggml_qwen38_ssd_stats * stats);

bool ggml_qwen38_ssd_prefetch_ngram_rows(ggml_qwen38_ssd_context * ctx,
        const uint64_t * row_ids, int32_t n_rows);
ggml_qwen38_ssd_ngram_handle * ggml_qwen38_ssd_acquire_ngram_row(ggml_qwen38_ssd_context * ctx, uint64_t row_id);
const void * ggml_qwen38_ssd_ngram_handle_data(const ggml_qwen38_ssd_ngram_handle * handle);
void ggml_qwen38_ssd_release_ngram_row(ggml_qwen38_ssd_context * ctx, ggml_qwen38_ssd_ngram_handle * handle);

// Compatibility helper for old callers. Prefer acquire/release above; this
// returns null unless the row is already cached and pins only until the next
// call on the same thread.
const void * ggml_qwen38_ssd_get_ngram_row(ggml_qwen38_ssd_context * ctx, uint64_t row_id);

#ifdef __cplusplus
}
#endif
