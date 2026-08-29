#include "ggml-qwen38-ssd-stream.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#if defined(__has_include)
#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#define GGML_QWEN38_HAS_IO_URING 1
#endif
#endif
#endif

namespace {

static uint64_t pack_key(int32_t layer, int32_t expert) {
    return (uint64_t(uint32_t(layer)) << 32) | uint32_t(expert);
}

static constexpr size_t NGRAM_PAGE_BYTES =
    size_t(GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS_PER_PAGE) * GGML_QWEN38_FLASH_NEXT_NGRAM_ROW_BYTES;

struct staging_buffer {
    uint8_t * ptr = nullptr;
    size_t capacity = 0;
    bool in_use = false;
    bool pinned = false;
};

class staging_pool {
public:
    void init(size_t total, size_t pinned, int inflight, const ggml_qwen38_ssd_cuda_hooks * hooks) {
        this->hooks = hooks;
        const int n = std::max(1, inflight);
        const size_t per = std::max(NGRAM_PAGE_BYTES, total / size_t(n));
        const size_t count = std::max<size_t>(1, total / per);
        slots.resize(count);
        size_t pinned_left = pinned;
        for (auto & s : slots) {
            s.capacity = per;
            s.pinned = pinned_left >= per;
            if (s.pinned) pinned_left -= per;
            s.ptr = hooks && hooks->host_alloc ? static_cast<uint8_t *>(hooks->host_alloc(hooks->user, per, s.pinned)) : nullptr;
            if (!s.ptr) {
#if defined(_WIN32)
                s.ptr = static_cast<uint8_t *>(malloc(per));
#else
                void * mem = nullptr;
                if (posix_memalign(&mem, 4096, per) == 0) s.ptr = static_cast<uint8_t *>(mem);
#endif
            }
        }
    }

    ~staging_pool() {
        for (auto & s : slots) {
            if (!s.ptr) continue;
            if (hooks && hooks->host_free) {
                hooks->host_free(hooks->user, s.ptr, s.capacity, s.pinned);
            } else {
                free(s.ptr);
            }
        }
    }

    int acquire(size_t need) {
        for (size_t i = 0; i < slots.size(); ++i) {
            if (!slots[i].in_use && slots[i].capacity >= need && slots[i].ptr) {
                slots[i].in_use = true;
                return int(i);
            }
        }
        return -1;
    }

    void release(int idx) {
        if (idx >= 0 && size_t(idx) < slots.size()) slots[idx].in_use = false;
    }

    uint8_t * data(int idx) { return slots[idx].ptr; }
    size_t size(int idx) const { return slots[idx].capacity; }
    bool is_pinned(int idx) const { return slots[idx].pinned; }

private:
    const ggml_qwen38_ssd_cuda_hooks * hooks = nullptr;
    std::vector<staging_buffer> slots;
};

struct resident_blob {
    std::vector<uint8_t> host;
    void * device = nullptr;
    ggml_qwen38_cuda_event event = nullptr;
    size_t size = 0;
    uint32_t pins = 0;
};

struct ngram_page {
    uint64_t page = 0;
    uint64_t last_use = 0;
    uint32_t frequency = 0;
    uint32_t pins = 0;
    bool io_pending = false;
    std::shared_ptr<std::vector<uint8_t>> data;
};

struct io_request {
    enum kind_t { EXPERT, NGRAM_PAGE } kind;
    uint64_t key;
    int fd;
    uint64_t offset;
    size_t size;
    int staging = -1;
    bool promote_to_vram = false;
};

struct io_completion {
    io_request req;
    int64_t result = 0;
};

class storage_backend {
public:
    void init(int max_inflight, bool prefer_uring) {
        this->max_inflight = std::max(1, max_inflight);
        this->prefer_uring = prefer_uring;
#if defined(GGML_QWEN38_HAS_IO_URING)
        use_uring = prefer_uring && setup_uring();
#else
        use_uring = false;
#endif
        if (!use_uring) {
            fallback = std::thread([this] { fallback_loop(); });
        }
    }

    ~storage_backend() { stop(); }

    void shutdown_backend() { stop(); }

    static uint64_t storage_key(const io_request & req) {
        return req.key ^ (req.kind == io_request::NGRAM_PAGE ? (1ull << 63) : 0ull);
    }

    bool submit(const io_request & req, uint8_t * dst) {
        std::lock_guard<std::mutex> lock(mtx);
        buffers[req.staging] = dst;
        pending.insert(storage_key(req));
#if defined(GGML_QWEN38_HAS_IO_URING)
        if (use_uring) {
            if (submit_uring_locked(req, dst)) return true;
            pending.erase(storage_key(req));
            buffers.erase(req.staging);
            return false;
        }
#endif
        queue.push_back(req);
        cv.notify_one();
        return true;
    }

    bool is_pending(const io_request & req) const {
        std::lock_guard<std::mutex> lock(mtx);
        return pending.count(storage_key(req)) != 0;
    }

    bool poll(std::vector<io_completion> & out) {
#if defined(GGML_QWEN38_HAS_IO_URING)
        if (use_uring) poll_uring(out);
#endif
        std::lock_guard<std::mutex> lock(done_mtx);
        while (!done.empty()) {
            out.push_back(done.front());
            done.pop_front();
        }
        for (const auto & c : out) {
            std::lock_guard<std::mutex> lock2(mtx);
            pending.erase(storage_key(c.req));
            buffers.erase(c.req.staging);
        }
        return !out.empty();
    }

    bool using_io_uring() const { return use_uring; }

private:
    int max_inflight = 1;
    bool prefer_uring = true;
    bool use_uring = false;
    bool shutdown = false;
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::deque<io_request> queue;
    std::unordered_set<uint64_t> pending;
    std::unordered_map<int, uint8_t *> buffers;
    std::mutex done_mtx;
    std::deque<io_completion> done;
    std::thread fallback;

#if defined(GGML_QWEN38_HAS_IO_URING)
    int ring_fd = -1;
    void * sq_ptr = MAP_FAILED;
    void * cq_ptr = MAP_FAILED;
    void * sqes_ptr = MAP_FAILED;
    size_t sq_sz = 0;
    size_t cq_sz = 0;
    size_t sqes_sz = 0;
    io_uring_params params{};
    uint32_t * sq_head = nullptr;
    uint32_t * sq_tail = nullptr;
    uint32_t * sq_ring_mask = nullptr;
    uint32_t * sq_ring_entries = nullptr;
    uint32_t * sq_array = nullptr;
    uint32_t * cq_head = nullptr;
    uint32_t * cq_tail = nullptr;
    uint32_t * cq_ring_mask = nullptr;
    io_uring_cqe * cqes = nullptr;
    io_uring_sqe * sqes = nullptr;
    std::unordered_map<int, io_request> uring_reqs;

    void cleanup_uring() {
        if (sqes_ptr != MAP_FAILED) { munmap(sqes_ptr, sqes_sz); sqes_ptr = MAP_FAILED; }
        if (cq_ptr != MAP_FAILED && cq_ptr != sq_ptr) { munmap(cq_ptr, cq_sz); cq_ptr = MAP_FAILED; }
        if (sq_ptr != MAP_FAILED) { munmap(sq_ptr, sq_sz); sq_ptr = MAP_FAILED; }
        if (ring_fd >= 0) { close(ring_fd); ring_fd = -1; }
    }

    bool setup_uring() {
        memset(&params, 0, sizeof(params));
        ring_fd = int(syscall(__NR_io_uring_setup, uint32_t(max_inflight), &params));
        if (ring_fd < 0) return false;

        sq_sz = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
        cq_sz = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
        if (params.features & IORING_FEAT_SINGLE_MMAP) {
            if (cq_sz > sq_sz) sq_sz = cq_sz;
            cq_sz = sq_sz;
        }

        sq_ptr = mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED) { cleanup_uring(); return false; }
        if (params.features & IORING_FEAT_SINGLE_MMAP) {
            cq_ptr = sq_ptr;
        } else {
            cq_ptr = mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
            if (cq_ptr == MAP_FAILED) { cleanup_uring(); return false; }
        }
        sqes_sz = params.sq_entries * sizeof(io_uring_sqe);
        sqes_ptr = mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQES);
        if (sqes_ptr == MAP_FAILED) { cleanup_uring(); return false; }

        auto * sq = static_cast<char *>(sq_ptr);
        auto * cq = static_cast<char *>(cq_ptr);
        sq_head = reinterpret_cast<uint32_t *>(sq + params.sq_off.head);
        sq_tail = reinterpret_cast<uint32_t *>(sq + params.sq_off.tail);
        sq_ring_mask = reinterpret_cast<uint32_t *>(sq + params.sq_off.ring_mask);
        sq_ring_entries = reinterpret_cast<uint32_t *>(sq + params.sq_off.ring_entries);
        sq_array = reinterpret_cast<uint32_t *>(sq + params.sq_off.array);
        cq_head = reinterpret_cast<uint32_t *>(cq + params.cq_off.head);
        cq_tail = reinterpret_cast<uint32_t *>(cq + params.cq_off.tail);
        cq_ring_mask = reinterpret_cast<uint32_t *>(cq + params.cq_off.ring_mask);
        cqes = reinterpret_cast<io_uring_cqe *>(cq + params.cq_off.cqes);
        sqes = static_cast<io_uring_sqe *>(sqes_ptr);
        return true;
    }

    bool submit_uring_locked(const io_request & req, uint8_t * dst) {
        const uint32_t head = __atomic_load_n(sq_head, __ATOMIC_ACQUIRE);
        const uint32_t tail = __atomic_load_n(sq_tail, __ATOMIC_RELAXED);
        if (tail - head >= *sq_ring_entries) return false;

        const uint32_t index = tail & *sq_ring_mask;
        io_uring_sqe * sqe = &sqes[index];
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = req.fd;
        sqe->off = req.offset;
        sqe->addr = reinterpret_cast<uint64_t>(dst);
        sqe->len = uint32_t(req.size);
        sqe->user_data = uint64_t(uint32_t(req.staging));
        sq_array[index] = index;
        uring_reqs[req.staging] = req;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        const int entered = int(syscall(__NR_io_uring_enter, ring_fd, 1u, 0u, 0u, nullptr, 0u));
        if (entered < 0) {
            uring_reqs.erase(req.staging);
            return false;
        }
        return true;
    }

    void poll_uring(std::vector<io_completion> & out) {
        uint32_t head = __atomic_load_n(cq_head, __ATOMIC_ACQUIRE);
        const uint32_t tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
        while (head != tail) {
            const io_uring_cqe & cqe = cqes[head & *cq_ring_mask];
            const int staging = int(uint32_t(cqe.user_data));
            auto it = uring_reqs.find(staging);
            if (it != uring_reqs.end()) {
                out.push_back({it->second, cqe.res});
                uring_reqs.erase(it);
            }
            ++head;
        }
        __atomic_store_n(cq_head, head, __ATOMIC_RELEASE);
    }
#endif

    void finish(const io_completion & c) {
        {
            std::lock_guard<std::mutex> lock(done_mtx);
            done.push_back(c);
        }
    }

    void fallback_loop() {
        for (;;) {
            io_request req{};
            uint8_t * dst = nullptr;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return shutdown || !queue.empty(); });
                if (shutdown) return;
                req = queue.front();
                queue.pop_front();
                dst = buffers[req.staging];
            }
            ssize_t got = 0;
#if defined(__linux__)
            while (got >= 0 && size_t(got) < req.size) {
                const ssize_t n = pread(req.fd, dst + got, req.size - size_t(got), off_t(req.offset + uint64_t(got)));
                if (n <= 0) break;
                got += n;
            }
#endif
            finish({req, got});
        }
    }

    void stop() {
        if (shutdown) return;
        {
            std::lock_guard<std::mutex> lock(mtx);
            shutdown = true;
            cv.notify_all();
        }
        if (fallback.joinable()) fallback.join();
#if defined(GGML_QWEN38_HAS_IO_URING)
        cleanup_uring();
#endif
    }
};

class residency_cache {
public:
    uint64_t tick = 1;
    size_t vram_bytes = 0;
    size_t ram_bytes = 0;
    size_t ngram_bytes = 0;

    std::unordered_map<uint64_t, ggml_qwen38_ssd_expert_meta> meta;
    std::unordered_map<uint64_t, resident_blob> ram;
    std::unordered_map<uint64_t, resident_blob> vram;
    std::unordered_map<uint64_t, ngram_page> ngram;

    static uint64_t eviction_score(const ggml_qwen38_ssd_expert_meta & m, uint64_t now) {
        const uint64_t age = now > m.last_use_tick ? now - m.last_use_tick : 0;
        const uint64_t keep = uint64_t(m.frequency) * 1024u + uint64_t(std::max(0, m.pin_priority)) * 4096u;
        return age > keep ? age - keep : 0;
    }
};

struct ggml_qwen38_ssd_context {
    ggml_qwen38_ssd_config cfg;
    ggml_qwen38_ssd_cuda_hooks hooks;
#if defined(__linux__)
    int fd_expert = -1;
    int fd_ngram = -1;
#endif
    mutable std::mutex mtx;
    staging_pool staging;
    storage_backend storage;
    residency_cache cache;
    ggml_qwen38_ssd_stats stats{};

    ~ggml_qwen38_ssd_context() {
        storage.shutdown_backend();
#if defined(__linux__)
        if (fd_expert >= 0) close(fd_expert);
        if (fd_ngram >= 0) close(fd_ngram);
#endif
    }

    bool can_evict_expert_locked(uint64_t key) const {
        auto m = cache.meta.find(key);
        if (m == cache.meta.end()) return true;
        const auto st = m->second.state;
        const auto r = cache.ram.find(key);
        const auto v = cache.vram.find(key);
        return st != GGML_QWEN38_SSD_STATE_SSD_READ_PENDING &&
               st != GGML_QWEN38_SSD_STATE_H2D_PENDING &&
               st != GGML_QWEN38_SSD_STATE_EVICT_PENDING &&
               (r == cache.ram.end() || r->second.pins == 0) &&
               (v == cache.vram.end() || v->second.pins == 0);
    }

    void evict_ram_locked(size_t need) {
        while (cache.ram_bytes + need > cfg.max_ram_cache_bytes && !cache.ram.empty()) {
            auto victim = cache.ram.end();
            uint64_t best = 0;
            for (auto it = cache.ram.begin(); it != cache.ram.end(); ++it) {
                if (!can_evict_expert_locked(it->first)) continue;
                uint64_t s = residency_cache::eviction_score(cache.meta[it->first], cache.tick);
                if (victim == cache.ram.end() || s > best) { best = s; victim = it; }
            }
            if (victim == cache.ram.end()) break;
            cache.meta[victim->first].state = GGML_QWEN38_SSD_STATE_EVICT_PENDING;
            cache.meta[victim->first].resident = GGML_QWEN38_SSD_LEVEL_SSD;
            cache.ram_bytes -= victim->second.size;
            cache.ram.erase(victim);
        }
    }

    void evict_vram_locked(size_t need) {
        while (cache.vram_bytes + need > cfg.max_vram_cache_bytes && !cache.vram.empty()) {
            auto victim = cache.vram.end();
            uint64_t best = 0;
            for (auto it = cache.vram.begin(); it != cache.vram.end(); ++it) {
                if (!can_evict_expert_locked(it->first)) continue;
                uint64_t s = residency_cache::eviction_score(cache.meta[it->first], cache.tick);
                if (victim == cache.vram.end() || s > best) { best = s; victim = it; }
            }
            if (victim == cache.vram.end()) break;
            cache.meta[victim->first].state = GGML_QWEN38_SSD_STATE_EVICT_PENDING;
            if (hooks.release_event && victim->second.event) hooks.release_event(hooks.user, victim->second.event);
            if (hooks.release_device && victim->second.device) hooks.release_device(hooks.user, victim->second.device);
            cache.meta[victim->first].resident = cache.ram.count(victim->first) ? GGML_QWEN38_SSD_LEVEL_RAM : GGML_QWEN38_SSD_LEVEL_SSD;
            cache.meta[victim->first].state = cache.ram.count(victim->first) ? GGML_QWEN38_SSD_STATE_RAM_READY : GGML_QWEN38_SSD_STATE_SSD;
            cache.vram_bytes -= victim->second.size;
            cache.vram.erase(victim);
        }
    }


    void evict_ngram_locked(size_t need = 0) {
        while (cache.ngram_bytes + need > cfg.ngram_cache_bytes && !cache.ngram.empty()) {
            auto victim = cache.ngram.end();
            for (auto it = cache.ngram.begin(); it != cache.ngram.end(); ++it) {
                if (it->second.pins || it->second.io_pending || !it->second.data) continue;
                if (victim == cache.ngram.end() ||
                        std::make_pair(it->second.frequency, it->second.last_use) <
                        std::make_pair(victim->second.frequency, victim->second.last_use)) {
                    victim = it;
                }
            }
            if (victim == cache.ngram.end()) break;
            cache.ngram_bytes -= victim->second.data->size();
            cache.ngram.erase(victim);
        }
    }

    bool enqueue_read_locked(io_request req) {
        if (req.fd < 0 || storage.is_pending(req)) return false;
        int idx = staging.acquire(req.size);
        if (idx < 0) return false;
        req.staging = idx;
        if (!storage.submit(req, staging.data(idx))) {
            staging.release(idx);
            return false;
        }
        return true;
    }

    void maybe_start_h2d_locked(uint64_t key) {
        auto m = cache.meta.find(key);
        auto r = cache.ram.find(key);
        if (m == cache.meta.end() || r == cache.ram.end() || cache.vram.count(key)) return;
        if (m->second.state == GGML_QWEN38_SSD_STATE_H2D_PENDING || m->second.state == GGML_QWEN38_SSD_STATE_VRAM_READY) return;
        evict_vram_locked(r->second.size);
        resident_blob dev;
        dev.size = r->second.size;
        r->second.pins++;
        m->second.state = GGML_QWEN38_SSD_STATE_H2D_PENDING;
        if (hooks.upload_async) {
            const bool ok = hooks.upload_async(hooks.user, &m->second, r->second.host.data(), r->second.size,
                    hooks.h2d_stream, &dev.event, &dev.device);
            if (!ok) {
                r->second.pins--;
                m->second.state = GGML_QWEN38_SSD_STATE_RAM_READY;
                return;
            }
        } else {
            // CPU-only/static path: no CUDA backend installed, so a null handle is
            // immediately considered ready for integration tests/future callers.
            m->second.state = GGML_QWEN38_SSD_STATE_VRAM_READY;
            stats.expert_h2d_completed++;
        }
        cache.vram.emplace(key, dev);
        cache.vram_bytes += dev.size;
        stats.expert_h2d_submitted++;
        stats.bytes_uploaded += dev.size;
        m->second.resident = GGML_QWEN38_SSD_LEVEL_VRAM;
    }

    void complete_h2d_locked(uint64_t key) {
        auto m = cache.meta.find(key);
        auto r = cache.ram.find(key);
        auto v = cache.vram.find(key);
        if (m == cache.meta.end() || v == cache.vram.end()) return;
        if (m->second.state != GGML_QWEN38_SSD_STATE_H2D_PENDING) return;
        if (hooks.event_ready && v->second.event && !hooks.event_ready(hooks.user, v->second.event)) return;
        if (hooks.release_event && v->second.event) hooks.release_event(hooks.user, v->second.event);
        v->second.event = nullptr;
        if (r != cache.ram.end() && r->second.pins > 0) r->second.pins--;
        m->second.state = GGML_QWEN38_SSD_STATE_VRAM_READY;
        stats.expert_h2d_completed++;
    }
};

static bool valid_expert(int32_t layer, int32_t expert) {
    return layer >= 0 && layer < GGML_QWEN38_FLASH_NEXT_N_LAYER && expert >= 0 && expert < GGML_QWEN38_FLASH_NEXT_N_EXPERT;
}

}

ggml_qwen38_ssd_config ggml_qwen38_ssd_default_config(void) {
    ggml_qwen38_ssd_config c{};
    c.max_vram_cache_bytes = 6ull * 1024ull * 1024ull * 1024ull;
    c.max_ram_cache_bytes = 18ull * 1024ull * 1024ull * 1024ull;
    c.max_staging_bytes = 512ull * 1024ull * 1024ull;
    c.ngram_cache_bytes = 512ull * 1024ull * 1024ull;
    c.pinned_staging_bytes = 32ull * 1024ull * 1024ull;
    c.max_inflight_reads = 8;
    c.use_direct_io = false;
    c.prefer_io_uring = true;
    return c;
}

ggml_qwen38_ssd_context * ggml_qwen38_ssd_init(const ggml_qwen38_ssd_config * cfg, const ggml_qwen38_ssd_cuda_hooks * hooks) {
    auto * ctx = new ggml_qwen38_ssd_context();
    ctx->cfg = cfg ? *cfg : ggml_qwen38_ssd_default_config();
    if (hooks) ctx->hooks = *hooks;
#if defined(__linux__)
    const int flags = O_RDONLY | (ctx->cfg.use_direct_io ? O_DIRECT : 0);
    if (ctx->cfg.expert_weights_path) ctx->fd_expert = open(ctx->cfg.expert_weights_path, flags);
    if (ctx->cfg.ngram_lut_path) ctx->fd_ngram = open(ctx->cfg.ngram_lut_path, flags);
#endif
    ctx->staging.init(ctx->cfg.max_staging_bytes, ctx->cfg.pinned_staging_bytes, ctx->cfg.max_inflight_reads, &ctx->hooks);
    ctx->storage.init(ctx->cfg.max_inflight_reads, ctx->cfg.prefer_io_uring);
    return ctx;
}

void ggml_qwen38_ssd_free(ggml_qwen38_ssd_context * ctx) { delete ctx; }

bool ggml_qwen38_ssd_register_expert(ggml_qwen38_ssd_context * ctx, const ggml_qwen38_ssd_expert_meta * meta) {
    if (!ctx || !meta || !valid_expert(meta->key.layer, meta->key.expert)) return false;
    std::lock_guard<std::mutex> lock(ctx->mtx);
    auto m = *meta;
    m.quant = GGML_QWEN38_SSD_QUANT_Q4;
    m.resident = GGML_QWEN38_SSD_LEVEL_SSD;
    m.state = GGML_QWEN38_SSD_STATE_SSD;
    ctx->cache.meta[pack_key(m.key.layer, m.key.expert)] = m;
    return true;
}

bool ggml_qwen38_ssd_poll(ggml_qwen38_ssd_context * ctx) {
    if (!ctx) return false;
    std::vector<io_completion> completions;
    ctx->storage.poll(completions);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    bool progress = false;
    for (const auto & c : completions) {
        progress = true;
        if (c.req.kind == io_request::EXPERT) {
            auto m = ctx->cache.meta.find(c.req.key);
            if (m != ctx->cache.meta.end() && c.result == int64_t(c.req.size)) {
                ctx->evict_ram_locked(c.req.size);
                resident_blob rb;
                rb.size = c.req.size;
                rb.host.resize(c.req.size);
                memcpy(rb.host.data(), ctx->staging.data(c.req.staging), c.req.size);
                ctx->cache.ram_bytes += rb.size;
                ctx->stats.bytes_read += c.req.size;
                ctx->cache.ram[c.req.key] = std::move(rb);
                m->second.resident = GGML_QWEN38_SSD_LEVEL_RAM;
                m->second.state = GGML_QWEN38_SSD_STATE_RAM_READY;
                if (c.req.promote_to_vram) ctx->maybe_start_h2d_locked(c.req.key);
            } else if (m != ctx->cache.meta.end()) {
                m->second.state = GGML_QWEN38_SSD_STATE_SSD;
                m->second.resident = GGML_QWEN38_SSD_LEVEL_SSD;
            }
        } else {
            auto & p = ctx->cache.ngram[c.req.key];
            if (c.result == int64_t(c.req.size)) {
                p.page = c.req.key;
                p.last_use = ctx->cache.tick++;
                ++p.frequency;
                p.io_pending = false;
                ctx->evict_ngram_locked(c.req.size);
                p.data = std::make_shared<std::vector<uint8_t>>(c.req.size);
                memcpy(p.data->data(), ctx->staging.data(c.req.staging), c.req.size);
                ctx->cache.ngram_bytes += c.req.size;
                ctx->stats.bytes_read += c.req.size;
            } else {
                p.io_pending = false;
            }
        }
        ctx->staging.release(c.req.staging);
    }
    for (const auto & v : ctx->cache.vram) ctx->complete_h2d_locked(v.first);
    return progress;
}

bool ggml_qwen38_ssd_schedule_experts(ggml_qwen38_ssd_context * ctx, int32_t layer, const int32_t * ids, int32_t n) {
    if (!ctx || !ids || n < 0) return false;
    ggml_qwen38_ssd_poll(ctx);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    for (int32_t i = 0; i < n; ++i) {
        if (!valid_expert(layer, ids[i])) continue;
        const uint64_t key = pack_key(layer, ids[i]);
        auto m = ctx->cache.meta.find(key);
        if (m == ctx->cache.meta.end()) continue;
        m->second.last_use_tick = ctx->cache.tick++;
        ++m->second.frequency;
        if (m->second.state == GGML_QWEN38_SSD_STATE_VRAM_READY) {
            ctx->stats.expert_vram_hits++;
            continue;
        }
        if (m->second.state == GGML_QWEN38_SSD_STATE_H2D_PENDING || m->second.state == GGML_QWEN38_SSD_STATE_SSD_READ_PENDING) {
            ctx->stats.expert_duplicate_requests++;
            continue;
        }
        if (m->second.state == GGML_QWEN38_SSD_STATE_RAM_READY || ctx->cache.ram.count(key)) {
            ctx->stats.expert_ram_hits++;
            ctx->maybe_start_h2d_locked(key);
            continue;
        }
#if defined(__linux__)
        m->second.state = GGML_QWEN38_SSD_STATE_SSD_READ_PENDING;
        ctx->stats.expert_ssd_misses++;
        if (!ctx->enqueue_read_locked({io_request::EXPERT, key, ctx->fd_expert, m->second.file_offset, m->second.size_bytes, -1, true})) {
            m->second.state = GGML_QWEN38_SSD_STATE_SSD;
        }
#endif
    }
    return true;
}

bool ggml_qwen38_ssd_prefetch_experts(ggml_qwen38_ssd_context * ctx, int32_t layer, const int32_t * ids, int32_t n) {
    if (!ctx || !ids || n < 0) return false;
    ggml_qwen38_ssd_poll(ctx);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    for (int32_t i = 0; i < n; ++i) {
        if (!valid_expert(layer, ids[i])) continue;
        const uint64_t key = pack_key(layer, ids[i]);
        auto m = ctx->cache.meta.find(key);
        if (m == ctx->cache.meta.end()) continue;
        if (m->second.state == GGML_QWEN38_SSD_STATE_VRAM_READY || m->second.state == GGML_QWEN38_SSD_STATE_H2D_PENDING ||
                m->second.state == GGML_QWEN38_SSD_STATE_SSD_READ_PENDING) {
            ctx->stats.expert_duplicate_requests++;
            continue;
        }
        if (m->second.state == GGML_QWEN38_SSD_STATE_RAM_READY || ctx->cache.ram.count(key)) {
            ctx->maybe_start_h2d_locked(key);
            continue;
        }
#if defined(__linux__)
        m->second.state = GGML_QWEN38_SSD_STATE_SSD_READ_PENDING;
        ctx->stats.expert_ssd_misses++;
        if (!ctx->enqueue_read_locked({io_request::EXPERT, key, ctx->fd_expert, m->second.file_offset, m->second.size_bytes, -1, true})) {
            m->second.state = GGML_QWEN38_SSD_STATE_SSD;
        }
#endif
    }
    return true;
}

ggml_qwen38_ssd_state ggml_qwen38_ssd_get_expert_state(ggml_qwen38_ssd_context * ctx, int32_t layer, int32_t expert) {
    if (!ctx || !valid_expert(layer, expert)) return GGML_QWEN38_SSD_STATE_SSD;
    ggml_qwen38_ssd_poll(ctx);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    auto it = ctx->cache.meta.find(pack_key(layer, expert));
    return it == ctx->cache.meta.end() ? GGML_QWEN38_SSD_STATE_SSD : it->second.state;
}

void * ggml_qwen38_ssd_get_device_expert(ggml_qwen38_ssd_context * ctx, int32_t layer, int32_t expert) {
    if (!ctx || !valid_expert(layer, expert)) return nullptr;
    ggml_qwen38_ssd_poll(ctx);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    const uint64_t key = pack_key(layer, expert);
    auto m = ctx->cache.meta.find(key);
    auto it = ctx->cache.vram.find(key);
    if (m == ctx->cache.meta.end() || it == ctx->cache.vram.end() || m->second.state != GGML_QWEN38_SSD_STATE_VRAM_READY) return nullptr;
    return it->second.device;
}

void * ggml_qwen38_ssd_acquire_device_expert(ggml_qwen38_ssd_context * ctx, int32_t layer, int32_t expert) {
    if (!ctx || !valid_expert(layer, expert)) return nullptr;
    ggml_qwen38_ssd_poll(ctx);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    const uint64_t key = pack_key(layer, expert);
    auto m = ctx->cache.meta.find(key);
    auto it = ctx->cache.vram.find(key);
    if (m == ctx->cache.meta.end() || it == ctx->cache.vram.end() || m->second.state != GGML_QWEN38_SSD_STATE_VRAM_READY) return nullptr;
    ++it->second.pins;
    return it->second.device;
}

void ggml_qwen38_ssd_release_device_expert(ggml_qwen38_ssd_context * ctx, int32_t layer, int32_t expert) {
    if (!ctx || !valid_expert(layer, expert)) return;
    std::lock_guard<std::mutex> lock(ctx->mtx);
    auto it = ctx->cache.vram.find(pack_key(layer, expert));
    if (it != ctx->cache.vram.end() && it->second.pins > 0) --it->second.pins;
}

bool ggml_qwen38_ssd_get_stats(ggml_qwen38_ssd_context * ctx, ggml_qwen38_ssd_stats * stats) {
    if (!ctx || !stats) return false;
    ggml_qwen38_ssd_poll(ctx);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    *stats = ctx->stats;
    stats->ram_cache_bytes = ctx->cache.ram_bytes;
    stats->vram_cache_bytes = ctx->cache.vram_bytes;
    stats->ngram_cache_bytes = ctx->cache.ngram_bytes;
    return true;
}

bool ggml_qwen38_ssd_prefetch_ngram_rows(ggml_qwen38_ssd_context * ctx, const uint64_t * rows, int32_t n) {
    if (!ctx || !rows || n < 0) return false;
    ggml_qwen38_ssd_poll(ctx);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    for (int32_t i = 0; i < n; ++i) {
        if (rows[i] >= GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS) continue;
        const uint64_t page = rows[i] / GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS_PER_PAGE;
        auto it = ctx->cache.ngram.find(page);
        if (it != ctx->cache.ngram.end() && it->second.data) { ctx->stats.ngram_hits++; continue; }
        if (it != ctx->cache.ngram.end() && it->second.io_pending) { ctx->stats.ngram_duplicate_requests++; continue; }
        ctx->stats.ngram_misses++;
        auto & p = ctx->cache.ngram[page];
        p.page = page;
        p.io_pending = true;
#if defined(__linux__)
        if (!ctx->enqueue_read_locked({io_request::NGRAM_PAGE, page, ctx->fd_ngram, page * NGRAM_PAGE_BYTES, NGRAM_PAGE_BYTES, -1, false})) {
            p.io_pending = false;
        }
#endif
    }
    return true;
}

struct ggml_qwen38_ssd_ngram_handle {
    uint64_t page;
    uint64_t row;
    std::shared_ptr<std::vector<uint8_t>> data;
};

ggml_qwen38_ssd_ngram_handle * ggml_qwen38_ssd_acquire_ngram_row(ggml_qwen38_ssd_context * ctx, uint64_t row_id) {
    if (!ctx || row_id >= GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS) return nullptr;
    ggml_qwen38_ssd_poll(ctx);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    const uint64_t page = row_id / GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS_PER_PAGE;
    auto it = ctx->cache.ngram.find(page);
    if (it == ctx->cache.ngram.end() || !it->second.data) return nullptr;
    it->second.last_use = ctx->cache.tick++;
    ++it->second.frequency;
    ++it->second.pins;
    auto * h = new ggml_qwen38_ssd_ngram_handle{page, row_id % GGML_QWEN38_FLASH_NEXT_NGRAM_ROWS_PER_PAGE, it->second.data};
    return h;
}

const void * ggml_qwen38_ssd_ngram_handle_data(const ggml_qwen38_ssd_ngram_handle * h) {
    return h && h->data ? h->data->data() + h->row * GGML_QWEN38_FLASH_NEXT_NGRAM_ROW_BYTES : nullptr;
}

void ggml_qwen38_ssd_release_ngram_row(ggml_qwen38_ssd_context * ctx, ggml_qwen38_ssd_ngram_handle * h) {
    if (!h) return;
    if (ctx) {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        auto it = ctx->cache.ngram.find(h->page);
        if (it != ctx->cache.ngram.end() && it->second.pins > 0) --it->second.pins;
        ctx->evict_ngram_locked();
    }
    delete h;
}

const void * ggml_qwen38_ssd_get_ngram_row(ggml_qwen38_ssd_context * ctx, uint64_t row_id) {
    thread_local ggml_qwen38_ssd_ngram_handle * handle = nullptr;
    if (handle) {
        ggml_qwen38_ssd_release_ngram_row(ctx, handle);
        handle = nullptr;
    }
    handle = ggml_qwen38_ssd_acquire_ngram_row(ctx, row_id);
    return ggml_qwen38_ssd_ngram_handle_data(handle);
}
