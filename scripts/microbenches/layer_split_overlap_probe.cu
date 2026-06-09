// Minimal two-GPU layer-split overlap probe.
//
// This intentionally bypasses Rust/Cargo and model loading. It simulates the
// scheduling shape of Ferrum's current 2-stage layer-split decode path:
//
//   stage0 compute on GPU0 -> D2H hidden
//   stage1 H2D hidden -> stage1 compute on GPU1 -> D2H hidden
//   logits H2D hidden -> logits compute on GPU1
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 scripts/microbenches/layer_split_overlap_probe.cu \
//     -o /tmp/layer_split_overlap_probe
//
// Example:
//   /tmp/layer_split_overlap_probe --out /tmp/layer_split_overlap_probe.json \
//     --batch-sizes 4,8,16 --microbatch-sizes 4,8 --repeats 30

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(expr)                                                                  \
    do {                                                                                  \
        cudaError_t _err = (expr);                                                        \
        if (_err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << cudaGetErrorString(_err) << " (" << static_cast<int>(_err)      \
                      << ")" << std::endl;                                               \
            std::exit(2);                                                                 \
        }                                                                                 \
    } while (0)

__global__ void busy_wait_kernel(unsigned long long cycles)
{
    const unsigned long long start = clock64();
    while (clock64() - start < cycles) {
        asm volatile("");
    }
}

struct Config {
    int device0 = 0;
    int device1 = 1;
    int hidden_size = 8192;
    std::vector<int> batch_sizes = {4, 8, 16};
    std::vector<int> microbatch_sizes = {4, 8};
    int repeats = 30;
    int warmup = 5;
    double stage0_ms = 26.0;
    double stage1_ms = 26.0;
    double logits_ms = 4.7;
    double stage_fixed_frac = 0.60;
    int d2h_elem_bytes = 4;
    int h2d_elem_bytes = 2;
    int threads = 128;
    int blocks_per_sm = 1;
    int queue_capacity = 1;
    std::string out_path;
};

struct DeviceCtx {
    int ordinal = 0;
    cudaDeviceProp prop {};
    cudaStream_t stream = nullptr;
    char* buffer = nullptr;
    size_t buffer_bytes = 0;
};

struct CaseResult {
    int batch = 0;
    int microbatch = 0;
    int chunk_count = 0;
    double sequential_batch_ms = 0.0;
    double sequential_micro_ms = 0.0;
    double overlapped_micro_ms = 0.0;
    double speedup_vs_batch = 0.0;
    double speedup_vs_micro = 0.0;
    std::string verdict;
};

static std::vector<int> parse_csv_ints(const std::string& text)
{
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stoi(item));
        }
    }
    return out;
}

static std::string json_escape(const std::string& text)
{
    std::ostringstream out;
    for (char c : text) {
        switch (c) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << c;
            break;
        }
    }
    return out.str();
}

static Config parse_args(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << std::endl;
                std::exit(2);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--device0") {
            cfg.device0 = std::stoi(need_value("--device0"));
        } else if (arg == "--device1") {
            cfg.device1 = std::stoi(need_value("--device1"));
        } else if (arg == "--hidden-size") {
            cfg.hidden_size = std::stoi(need_value("--hidden-size"));
        } else if (arg == "--batch-sizes") {
            cfg.batch_sizes = parse_csv_ints(need_value("--batch-sizes"));
        } else if (arg == "--microbatch-sizes") {
            cfg.microbatch_sizes = parse_csv_ints(need_value("--microbatch-sizes"));
        } else if (arg == "--repeats") {
            cfg.repeats = std::stoi(need_value("--repeats"));
        } else if (arg == "--warmup") {
            cfg.warmup = std::stoi(need_value("--warmup"));
        } else if (arg == "--stage0-ms") {
            cfg.stage0_ms = std::stod(need_value("--stage0-ms"));
        } else if (arg == "--stage1-ms") {
            cfg.stage1_ms = std::stod(need_value("--stage1-ms"));
        } else if (arg == "--logits-ms") {
            cfg.logits_ms = std::stod(need_value("--logits-ms"));
        } else if (arg == "--stage-fixed-frac") {
            cfg.stage_fixed_frac = std::stod(need_value("--stage-fixed-frac"));
        } else if (arg == "--d2h-elem-bytes") {
            cfg.d2h_elem_bytes = std::stoi(need_value("--d2h-elem-bytes"));
        } else if (arg == "--h2d-elem-bytes") {
            cfg.h2d_elem_bytes = std::stoi(need_value("--h2d-elem-bytes"));
        } else if (arg == "--threads") {
            cfg.threads = std::stoi(need_value("--threads"));
        } else if (arg == "--blocks-per-sm") {
            cfg.blocks_per_sm = std::stoi(need_value("--blocks-per-sm"));
        } else if (arg == "--queue-capacity") {
            cfg.queue_capacity = std::stoi(need_value("--queue-capacity"));
        } else if (arg == "--out") {
            cfg.out_path = need_value("--out");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: layer_split_overlap_probe [options]\n"
                << "  --out PATH\n"
                << "  --batch-sizes 4,8,16\n"
                << "  --microbatch-sizes 4,8\n"
                << "  --repeats 30 --warmup 5\n"
                << "  --stage0-ms 26 --stage1-ms 26 --logits-ms 4.7\n"
                << "  --stage-fixed-frac 0.60\n";
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            std::exit(2);
        }
    }

    if (cfg.batch_sizes.empty() || cfg.microbatch_sizes.empty()) {
        std::cerr << "batch and microbatch size lists must be non-empty" << std::endl;
        std::exit(2);
    }
    if (cfg.repeats <= 0 || cfg.warmup < 0) {
        std::cerr << "repeats must be > 0 and warmup must be >= 0" << std::endl;
        std::exit(2);
    }
    if (cfg.stage_fixed_frac < 0.0 || cfg.stage_fixed_frac > 1.0) {
        std::cerr << "stage-fixed-frac must be in [0, 1]" << std::endl;
        std::exit(2);
    }
    return cfg;
}

static size_t max_batch_size(const Config& cfg)
{
    return static_cast<size_t>(*std::max_element(cfg.batch_sizes.begin(), cfg.batch_sizes.end()));
}

static double scaled_stage_ms(double full_batch_ms, int rows, int full_batch, double fixed_frac)
{
    const double row_frac = static_cast<double>(rows) / static_cast<double>(full_batch);
    return full_batch_ms * (fixed_frac + (1.0 - fixed_frac) * row_frac);
}

static void busy(DeviceCtx& dev, double ms, int threads, int blocks_per_sm)
{
    if (ms <= 0.0) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(dev.ordinal));
    const int blocks = std::max(1, dev.prop.multiProcessorCount * blocks_per_sm);
    const auto cycles = static_cast<unsigned long long>(ms * static_cast<double>(dev.prop.clockRate));
    busy_wait_kernel<<<blocks, threads, 0, dev.stream>>>(cycles);
    CUDA_CHECK(cudaGetLastError());
}

static void run_stage0(const Config& cfg, DeviceCtx& dev0, int rows, int full_batch, char* host)
{
    CUDA_CHECK(cudaSetDevice(dev0.ordinal));
    busy(dev0, scaled_stage_ms(cfg.stage0_ms, rows, full_batch, cfg.stage_fixed_frac), cfg.threads,
         cfg.blocks_per_sm);
    const size_t bytes = static_cast<size_t>(rows) * cfg.hidden_size * cfg.d2h_elem_bytes;
    CUDA_CHECK(cudaMemcpyAsync(host, dev0.buffer, bytes, cudaMemcpyDeviceToHost, dev0.stream));
    CUDA_CHECK(cudaStreamSynchronize(dev0.stream));
}

static void run_stage1_and_logits(const Config& cfg,
                                  DeviceCtx& dev1,
                                  int rows,
                                  int full_batch,
                                  char* host)
{
    CUDA_CHECK(cudaSetDevice(dev1.ordinal));
    const size_t d2h_bytes = static_cast<size_t>(rows) * cfg.hidden_size * cfg.d2h_elem_bytes;
    const size_t h2d_bytes = static_cast<size_t>(rows) * cfg.hidden_size * cfg.h2d_elem_bytes;

    CUDA_CHECK(cudaMemcpyAsync(dev1.buffer, host, h2d_bytes, cudaMemcpyHostToDevice, dev1.stream));
    busy(dev1, scaled_stage_ms(cfg.stage1_ms, rows, full_batch, cfg.stage_fixed_frac), cfg.threads,
         cfg.blocks_per_sm);

    CUDA_CHECK(cudaMemcpyAsync(host, dev1.buffer, d2h_bytes, cudaMemcpyDeviceToHost, dev1.stream));
    CUDA_CHECK(cudaMemcpyAsync(dev1.buffer, host, h2d_bytes, cudaMemcpyHostToDevice, dev1.stream));
    busy(dev1, scaled_stage_ms(cfg.logits_ms, rows, full_batch, cfg.stage_fixed_frac), cfg.threads,
         cfg.blocks_per_sm);
    CUDA_CHECK(cudaStreamSynchronize(dev1.stream));
}

static std::vector<int> chunk_rows(int batch, int microbatch)
{
    std::vector<int> rows;
    for (int remaining = batch; remaining > 0;) {
        const int n = std::min(remaining, microbatch);
        rows.push_back(n);
        remaining -= n;
    }
    return rows;
}

static double elapsed_ms(const std::chrono::steady_clock::time_point& start)
{
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static double measure_sequential_batch(const Config& cfg,
                                       DeviceCtx& dev0,
                                       DeviceCtx& dev1,
                                       int batch,
                                       char* host)
{
    const auto start = std::chrono::steady_clock::now();
    run_stage0(cfg, dev0, batch, batch, host);
    run_stage1_and_logits(cfg, dev1, batch, batch, host);
    return elapsed_ms(start);
}

static double measure_sequential_micro(const Config& cfg,
                                       DeviceCtx& dev0,
                                       DeviceCtx& dev1,
                                       int batch,
                                       int microbatch,
                                       char* host)
{
    const auto rows = chunk_rows(batch, microbatch);
    const auto start = std::chrono::steady_clock::now();
    for (int n : rows) {
        run_stage0(cfg, dev0, n, batch, host);
        run_stage1_and_logits(cfg, dev1, n, batch, host);
    }
    return elapsed_ms(start);
}

class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    void push(int value)
    {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [&] { return q_.size() < capacity_; });
        q_.push_back(value);
        not_empty_.notify_one();
    }

    int pop()
    {
        std::unique_lock<std::mutex> lock(mu_);
        not_empty_.wait(lock, [&] { return !q_.empty(); });
        int value = q_.front();
        q_.pop_front();
        not_full_.notify_one();
        return value;
    }

private:
    size_t capacity_;
    std::mutex mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<int> q_;
};

static double measure_overlapped_micro(const Config& cfg,
                                       DeviceCtx& dev0,
                                       DeviceCtx& dev1,
                                       int batch,
                                       int microbatch,
                                       const std::vector<char*>& hosts)
{
    const auto rows = chunk_rows(batch, microbatch);
    if (rows.size() <= 1) {
        return measure_sequential_micro(cfg, dev0, dev1, batch, microbatch, hosts[0]);
    }

    BoundedQueue q(static_cast<size_t>(std::max(1, cfg.queue_capacity)));
    const auto start = std::chrono::steady_clock::now();
    std::thread producer([&] {
        for (size_t i = 0; i < rows.size(); ++i) {
            run_stage0(cfg, dev0, rows[i], batch, hosts[i]);
            q.push(static_cast<int>(i));
        }
    });

    for (size_t consumed = 0; consumed < rows.size(); ++consumed) {
        const int idx = q.pop();
        run_stage1_and_logits(cfg, dev1, rows[idx], batch, hosts[idx]);
    }
    producer.join();
    return elapsed_ms(start);
}

static double measure_many(int warmup, int repeats, const std::function<double()>& f)
{
    for (int i = 0; i < warmup; ++i) {
        (void)f();
    }
    std::vector<double> values;
    values.reserve(static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        values.push_back(f());
    }
    return median(values);
}

static std::string device_json(const DeviceCtx& dev)
{
    std::ostringstream out;
    out << "{"
        << "\"ordinal\":" << dev.ordinal << ","
        << "\"name\":\"" << json_escape(dev.prop.name) << "\","
        << "\"sm_count\":" << dev.prop.multiProcessorCount << ","
        << "\"clock_rate_khz\":" << dev.prop.clockRate << ","
        << "\"memory_bytes\":" << static_cast<unsigned long long>(dev.prop.totalGlobalMem) << "}";
    return out.str();
}

static std::string results_json(const Config& cfg,
                                const DeviceCtx& dev0,
                                const DeviceCtx& dev1,
                                const std::vector<CaseResult>& cases)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"probe\": \"layer_split_overlap_probe\",\n";
    out << "  \"devices\": [" << device_json(dev0) << ", " << device_json(dev1) << "],\n";
    out << "  \"config\": {";
    out << "\"hidden_size\":" << cfg.hidden_size << ",";
    out << "\"repeats\":" << cfg.repeats << ",";
    out << "\"warmup\":" << cfg.warmup << ",";
    out << "\"stage0_ms\":" << cfg.stage0_ms << ",";
    out << "\"stage1_ms\":" << cfg.stage1_ms << ",";
    out << "\"logits_ms\":" << cfg.logits_ms << ",";
    out << "\"stage_fixed_frac\":" << cfg.stage_fixed_frac << ",";
    out << "\"d2h_elem_bytes\":" << cfg.d2h_elem_bytes << ",";
    out << "\"h2d_elem_bytes\":" << cfg.h2d_elem_bytes << ",";
    out << "\"threads\":" << cfg.threads << ",";
    out << "\"blocks_per_sm\":" << cfg.blocks_per_sm << ",";
    out << "\"queue_capacity\":" << cfg.queue_capacity << "},\n";
    out << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        out << "    {";
        out << "\"batch\":" << c.batch << ",";
        out << "\"microbatch\":" << c.microbatch << ",";
        out << "\"chunk_count\":" << c.chunk_count << ",";
        out << "\"sequential_batch_ms\":" << c.sequential_batch_ms << ",";
        out << "\"sequential_micro_ms\":" << c.sequential_micro_ms << ",";
        out << "\"overlapped_micro_ms\":" << c.overlapped_micro_ms << ",";
        out << "\"speedup_vs_batch\":" << c.speedup_vs_batch << ",";
        out << "\"speedup_vs_micro\":" << c.speedup_vs_micro << ",";
        out << "\"verdict\":\"" << c.verdict << "\"}";
        if (i + 1 != cases.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

int main(int argc, char** argv)
{
    Config cfg = parse_args(argc, argv);

    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count <= std::max(cfg.device0, cfg.device1)) {
        std::cerr << "need devices " << cfg.device0 << " and " << cfg.device1 << ", but CUDA sees "
                  << count << " device(s)" << std::endl;
        return 2;
    }
    if (cfg.device0 == cfg.device1) {
        std::cerr << "device0 and device1 must be different" << std::endl;
        return 2;
    }

    DeviceCtx dev0;
    dev0.ordinal = cfg.device0;
    CUDA_CHECK(cudaSetDevice(dev0.ordinal));
    CUDA_CHECK(cudaGetDeviceProperties(&dev0.prop, dev0.ordinal));
    CUDA_CHECK(cudaStreamCreateWithFlags(&dev0.stream, cudaStreamNonBlocking));

    DeviceCtx dev1;
    dev1.ordinal = cfg.device1;
    CUDA_CHECK(cudaSetDevice(dev1.ordinal));
    CUDA_CHECK(cudaGetDeviceProperties(&dev1.prop, dev1.ordinal));
    CUDA_CHECK(cudaStreamCreateWithFlags(&dev1.stream, cudaStreamNonBlocking));

    const size_t max_batch = max_batch_size(cfg);
    const size_t max_d2h_bytes = max_batch * static_cast<size_t>(cfg.hidden_size) *
                                 static_cast<size_t>(cfg.d2h_elem_bytes);
    const size_t max_h2d_bytes = max_batch * static_cast<size_t>(cfg.hidden_size) *
                                 static_cast<size_t>(cfg.h2d_elem_bytes);
    dev0.buffer_bytes = max_d2h_bytes;
    dev1.buffer_bytes = std::max(max_d2h_bytes, max_h2d_bytes);

    CUDA_CHECK(cudaSetDevice(dev0.ordinal));
    CUDA_CHECK(cudaMalloc(&dev0.buffer, dev0.buffer_bytes));
    CUDA_CHECK(cudaMemset(dev0.buffer, 7, dev0.buffer_bytes));
    CUDA_CHECK(cudaSetDevice(dev1.ordinal));
    CUDA_CHECK(cudaMalloc(&dev1.buffer, dev1.buffer_bytes));
    CUDA_CHECK(cudaMemset(dev1.buffer, 3, dev1.buffer_bytes));

    int min_microbatch = *std::min_element(cfg.microbatch_sizes.begin(), cfg.microbatch_sizes.end());
    min_microbatch = std::max(1, min_microbatch);
    const size_t max_chunks = (max_batch + static_cast<size_t>(min_microbatch) - 1) /
                              static_cast<size_t>(min_microbatch);
    std::vector<char*> hosts(max_chunks, nullptr);
    for (size_t i = 0; i < hosts.size(); ++i) {
        CUDA_CHECK(cudaHostAlloc(&hosts[i], max_d2h_bytes, cudaHostAllocPortable));
        std::memset(hosts[i], 0, max_d2h_bytes);
    }

    std::vector<CaseResult> cases;
    for (int batch : cfg.batch_sizes) {
        if (batch <= 0) {
            continue;
        }
        for (int microbatch : cfg.microbatch_sizes) {
            if (microbatch <= 0 || microbatch > batch) {
                continue;
            }
            CaseResult result;
            result.batch = batch;
            result.microbatch = microbatch;
            result.chunk_count = static_cast<int>(chunk_rows(batch, microbatch).size());

            std::cerr << "[probe] batch=" << batch << " microbatch=" << microbatch
                      << " chunks=" << result.chunk_count << std::endl;

            result.sequential_batch_ms = measure_many(cfg.warmup, cfg.repeats, [&] {
                return measure_sequential_batch(cfg, dev0, dev1, batch, hosts[0]);
            });
            result.sequential_micro_ms = measure_many(cfg.warmup, cfg.repeats, [&] {
                return measure_sequential_micro(cfg, dev0, dev1, batch, microbatch, hosts[0]);
            });
            result.overlapped_micro_ms = measure_many(cfg.warmup, cfg.repeats, [&] {
                return measure_overlapped_micro(cfg, dev0, dev1, batch, microbatch, hosts);
            });
            result.speedup_vs_batch = result.sequential_batch_ms / result.overlapped_micro_ms;
            result.speedup_vs_micro = result.sequential_micro_ms / result.overlapped_micro_ms;
            if (result.speedup_vs_batch >= 1.02) {
                result.verdict = "overlap_beats_full_batch";
            } else if (result.speedup_vs_batch <= 0.98) {
                result.verdict = "overlap_loses_to_full_batch";
            } else {
                result.verdict = "overlap_ties_full_batch";
            }
            cases.push_back(result);
        }
    }

    const std::string json = results_json(cfg, dev0, dev1, cases);
    std::cout << json;
    if (!cfg.out_path.empty()) {
        std::ofstream file(cfg.out_path);
        if (!file) {
            std::cerr << "failed to open output path: " << cfg.out_path << std::endl;
            return 2;
        }
        file << json;
    }

    for (char* host : hosts) {
        CUDA_CHECK(cudaFreeHost(host));
    }
    CUDA_CHECK(cudaSetDevice(dev0.ordinal));
    CUDA_CHECK(cudaFree(dev0.buffer));
    CUDA_CHECK(cudaStreamDestroy(dev0.stream));
    CUDA_CHECK(cudaSetDevice(dev1.ordinal));
    CUDA_CHECK(cudaFree(dev1.buffer));
    CUDA_CHECK(cudaStreamDestroy(dev1.stream));
    return 0;
}
