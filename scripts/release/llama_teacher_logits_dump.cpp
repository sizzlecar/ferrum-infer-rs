#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

constexpr int32_t kWorkerThreads = 4;
constexpr uint32_t kMinimumContextTokens = 512;
constexpr uint32_t kPhysicalBatchTokens = 512;
constexpr size_t kMaximumInputTokens = 128;

struct BackendGuard {
    BackendGuard() {
        ggml_backend_load_all();
        llama_backend_init();
    }

    ~BackendGuard() { llama_backend_free(); }

    BackendGuard(const BackendGuard &) = delete;
    BackendGuard & operator=(const BackendGuard &) = delete;
};

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;

class Batch {
  public:
    explicit Batch(int32_t capacity) : value_(llama_batch_init(capacity, 0, 1)) {
        if (capacity <= 0 || value_.token == nullptr || value_.pos == nullptr ||
            value_.n_seq_id == nullptr || value_.seq_id == nullptr || value_.logits == nullptr) {
            llama_batch_free(value_);
            throw std::runtime_error("llama.cpp failed to allocate a token batch");
        }
    }

    ~Batch() { llama_batch_free(value_); }

    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;

    llama_batch & get() { return value_; }

  private:
    llama_batch value_{};
};

struct DecisionRecord {
    size_t index;
    std::string segment;
    size_t context_token_count;
    std::string file;
    size_t bytes;
};

std::string errno_message(std::string_view operation, const std::filesystem::path & path) {
    return std::string(operation) + " " + path.string() + ": " + std::strerror(errno);
}

class PendingFile {
  public:
    explicit PendingFile(std::filesystem::path path) : path_(std::move(path)) {}

    ~PendingFile() {
        if (!released_) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }

    void release() { released_ = true; }

  private:
    std::filesystem::path path_;
    bool released_ = false;
};

void write_all(int fd, const uint8_t * data, size_t size, const std::filesystem::path & path) {
    size_t written = 0;
    while (written < size) {
        const size_t remaining = size - written;
        const size_t request =
            std::min(remaining, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t result = ::write(fd, data + written, request);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(errno_message("cannot write", path));
        }
        if (result == 0) {
            throw std::runtime_error("short write while writing " + path.string());
        }
        written += static_cast<size_t>(result);
    }
}

void close_checked(int fd, const std::filesystem::path & path) {
    if (::close(fd) != 0) {
        throw std::runtime_error(errno_message("cannot close", path));
    }
}

void write_atomic(
    const std::filesystem::path & directory,
    const std::string & file_name,
    const uint8_t * data,
    size_t size) {
    const auto final_path = directory / file_name;
    const auto temporary_path = directory / ("." + file_name + ".tmp");
    if (std::filesystem::exists(final_path) || std::filesystem::is_symlink(final_path) ||
        std::filesystem::exists(temporary_path) || std::filesystem::is_symlink(temporary_path)) {
        throw std::runtime_error("refusing to replace an output file: " + final_path.string());
    }

    int fd = ::open(
        temporary_path.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
        0600);
    if (fd < 0) {
        throw std::runtime_error(errno_message("cannot create", temporary_path));
    }
    PendingFile cleanup(temporary_path);

    try {
        write_all(fd, data, size, temporary_path);
        if (::fsync(fd) != 0) {
            throw std::runtime_error(errno_message("cannot fsync", temporary_path));
        }
        const int descriptor_to_close = fd;
        fd = -1;
        close_checked(descriptor_to_close, temporary_path);
    } catch (...) {
        const int saved_errno = errno;
        if (fd >= 0) {
            ::close(fd);
        }
        errno = saved_errno;
        throw;
    }

    std::filesystem::rename(temporary_path, final_path);
    cleanup.release();
    const auto status = std::filesystem::symlink_status(final_path);
    if (!std::filesystem::is_regular_file(status) || std::filesystem::is_symlink(status) ||
        std::filesystem::file_size(final_path) != size) {
        throw std::runtime_error("output file failed its post-write size check: " +
                                 final_path.string());
    }
}

void fsync_directory(const std::filesystem::path & directory) {
    const int fd = ::open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
    if (fd < 0) {
        throw std::runtime_error(errno_message("cannot open output directory", directory));
    }
    if (::fsync(fd) != 0) {
        const int saved_errno = errno;
        ::close(fd);
        errno = saved_errno;
        throw std::runtime_error(errno_message("cannot fsync output directory", directory));
    }
    close_checked(fd, directory);
}

void prepare_output_directory(const std::filesystem::path & directory) {
    std::error_code error;
    auto initial_status = std::filesystem::symlink_status(directory, error);
    if (error == std::errc::no_such_file_or_directory) {
        error.clear();
        initial_status = std::filesystem::file_status(std::filesystem::file_type::not_found);
    } else if (error) {
        throw std::runtime_error("cannot inspect output directory: " + error.message());
    }
    if (std::filesystem::is_symlink(initial_status)) {
        throw std::runtime_error("output directory must not be a symlink");
    }
    if (std::filesystem::exists(initial_status) &&
        !std::filesystem::is_directory(initial_status)) {
        throw std::runtime_error("output path exists but is not a directory");
    }
    if (!std::filesystem::exists(initial_status)) {
        std::filesystem::create_directories(directory, error);
        if (error) {
            throw std::runtime_error("cannot create output directory: " + error.message());
        }
    }

    const auto final_status = std::filesystem::symlink_status(directory);
    if (std::filesystem::is_symlink(final_status) ||
        !std::filesystem::is_directory(final_status)) {
        throw std::runtime_error("output directory must be a real directory");
    }
    if (std::filesystem::directory_iterator(directory) !=
        std::filesystem::directory_iterator()) {
        throw std::runtime_error("output directory must be empty");
    }
}

std::vector<llama_token> load_tokens(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open token file: " + path.string());
    }
    std::vector<llama_token> tokens;
    int64_t value = 0;
    while (input >> value) {
        if (value < 0 || value > std::numeric_limits<llama_token>::max()) {
            throw std::runtime_error("token id is outside the llama_token range");
        }
        tokens.push_back(static_cast<llama_token>(value));
    }
    if (!input.eof()) {
        throw std::runtime_error("token file contains a non-decimal token id");
    }
    if (tokens.empty()) {
        throw std::runtime_error("token file is empty");
    }
    if (tokens.size() > kMaximumInputTokens) {
        throw std::runtime_error(
            "token file exceeds the reviewed 128-token teacher fixture limit");
    }
    if (tokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("token file exceeds llama.cpp batch limits");
    }
    return tokens;
}

size_t parse_prefill_tokens(std::string_view text) {
    size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value == 0) {
        throw std::runtime_error("PREFILL_TOKENS must be a positive decimal integer");
    }
    return value;
}

std::string lower_ascii(std::string value) {
    for (char & character : value) {
        if (character >= 'A' && character <= 'Z') {
            character = static_cast<char>(character - 'A' + 'a');
        }
    }
    return value;
}

ggml_backend_dev_t find_metal_device() {
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr) {
            continue;
        }
        const auto registry = ggml_backend_dev_backend_reg(device);
        const std::string registry_name =
            registry == nullptr ? std::string() : ggml_backend_reg_name(registry);
        const std::string device_name = ggml_backend_dev_name(device);
        const std::string normalized_registry = lower_ascii(registry_name);
        const std::string normalized_device = lower_ascii(device_name);
        const auto device_type = ggml_backend_dev_type(device);
        const bool is_gpu = device_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                            device_type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        const bool is_metal = normalized_registry == "metal" ||
                              normalized_registry == "mtl" ||
                              normalized_device.rfind("mtl", 0) == 0;
        if (is_gpu && is_metal) {
            return device;
        }
    }
    throw std::runtime_error("llama.cpp has no registered Metal device");
}

std::string require_qwen35_architecture(const llama_model * model) {
    std::array<char, 64> buffer{};
    const int32_t length =
        llama_model_meta_val_str(model, "general.architecture", buffer.data(), buffer.size());
    if (length < 0 || static_cast<size_t>(length) >= buffer.size()) {
        throw std::runtime_error("GGUF has no complete general.architecture metadata");
    }
    const std::string architecture(buffer.data(), static_cast<size_t>(length));
    if (architecture != "qwen35") {
        throw std::runtime_error(
            "GGUF general.architecture must be qwen35, got " + architecture);
    }
    return architecture;
}

std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '\"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    output << "\\u00" << hex[character >> 4] << hex[character & 0x0f];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

void fill_batch(
    llama_batch & batch,
    const llama_token * tokens,
    int32_t token_count,
    llama_pos first_position) {
    batch.n_tokens = token_count;
    for (int32_t index = 0; index < token_count; ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = first_position + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = index == token_count - 1 ? 1 : 0;
    }
}

void decode_or_throw(
    llama_context * context,
    const llama_token * tokens,
    int32_t token_count,
    llama_pos first_position,
    Batch & batch) {
    fill_batch(batch.get(), tokens, token_count, first_position);
    const int32_t status = llama_decode(context, batch.get());
    if (status != 0) {
        throw std::runtime_error(
            "llama_decode failed at position " + std::to_string(first_position) +
            " with status " + std::to_string(status));
    }
}

std::string decision_file_name(size_t index) {
    std::array<char, 64> buffer{};
    const int length = std::snprintf(buffer.data(), buffer.size(), "decision-%04zu.f32", index);
    if (length < 0 || static_cast<size_t>(length) >= buffer.size()) {
        throw std::runtime_error("decision file index is too large");
    }
    return std::string(buffer.data(), static_cast<size_t>(length));
}

DecisionRecord save_decision(
    llama_context * context,
    const std::filesystem::path & output_directory,
    size_t decision_index,
    std::string segment,
    size_t context_token_count,
    int32_t vocabulary_size) {
    const float * logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) {
        throw std::runtime_error("llama.cpp did not expose final-token logits");
    }
    const size_t bytes = static_cast<size_t>(vocabulary_size) * sizeof(float);
    const std::string file_name = decision_file_name(decision_index);
    write_atomic(
        output_directory,
        file_name,
        reinterpret_cast<const uint8_t *>(logits),
        bytes);
    return DecisionRecord{
        decision_index,
        std::move(segment),
        context_token_count,
        file_name,
        bytes,
    };
}

std::string build_manifest(
    const std::filesystem::path & model_path,
    const std::filesystem::path & token_path,
    std::string_view metal_device,
    std::string_view architecture,
    const std::vector<llama_token> & tokens,
    size_t prefill_tokens,
    int32_t vocabulary_size,
    const std::vector<DecisionRecord> & decisions) {
    const size_t total_tokens = tokens.size();
    const size_t decode_tokens = total_tokens - prefill_tokens;
    std::ostringstream output;
    output << "{\n"
           << "  \"schema\": \"ferrum.llama-teacher-logits-dump.v1\",\n"
           << "  \"schema_version\": 1,\n"
           << "  \"status\": \"pass\",\n"
           << "  \"backend\": {\"name\": \"metal\", \"device\": \""
           << json_escape(metal_device) << "\", \"n_gpu_layers\": -1},\n"
           << "  \"input\": {\"model\": \"" << json_escape(model_path.string())
           << "\", \"token_ids_file\": \"" << json_escape(token_path.string())
           << "\", \"architecture\": \"" << json_escape(architecture)
           << "\", \"token_count\": " << total_tokens << ", \"token_ids\": [";
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << tokens[index];
    }
    output << "]},\n"
           << "  \"prefill\": {\"token_count\": " << prefill_tokens
           << ", \"decode_calls\": 1},\n"
           << "  \"decode\": {\"teacher_token_count\": " << decode_tokens
           << ", \"decode_calls\": " << decode_tokens << "},\n"
           << "  \"decision\": {\"count\": " << decisions.size()
           << ", \"records\": [\n";
    for (size_t index = 0; index < decisions.size(); ++index) {
        const auto & decision = decisions[index];
        output << "    {\"index\": " << decision.index << ", \"segment\": \""
               << decision.segment << "\", \"context_token_count\": "
               << decision.context_token_count << ", \"file\": \""
               << decision.file << "\", \"bytes\": " << decision.bytes << "}"
               << (index + 1 == decisions.size() ? "\n" : ",\n");
    }
    output << "  ]},\n"
           << "  \"vocab\": {\"size\": " << vocabulary_size << "},\n"
           << "  \"dtype\": \"f32\",\n"
           << "  \"execution\": {\"parallel_sequences\": 1, \"n_seq_max\": 1, "
              "\"worker_threads\": "
           << kWorkerThreads << "}\n"
           << "}\n";
    return output.str();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::cerr << "usage: llama_teacher_logits_dump MODEL.gguf TOKEN_IDS.txt "
                     "PREFILL_TOKENS OUTPUT_DIR\n";
        return 2;
    }

    try {
        const std::filesystem::path model_path = argv[1];
        const std::filesystem::path token_path = argv[2];
        const size_t prefill_tokens = parse_prefill_tokens(argv[3]);
        const std::filesystem::path output_directory = argv[4];
        const auto tokens = load_tokens(token_path);
        if (prefill_tokens > tokens.size()) {
            throw std::runtime_error("PREFILL_TOKENS exceeds the token file length");
        }
        prepare_output_directory(output_directory);

        BackendGuard backend;
        ggml_backend_dev_t metal_device = find_metal_device();
        std::array<ggml_backend_dev_t, 2> devices{metal_device, nullptr};

        auto model_params = llama_model_default_params();
        model_params.devices = devices.data();
        model_params.n_gpu_layers = -1;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.main_gpu = 0;
        model_params.use_mmap = true;
        model_params.use_mlock = false;
        model_params.check_tensors = false;
        Model model(llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
        if (!model) {
            throw std::runtime_error("llama.cpp failed to load the model on Metal");
        }
        const std::string architecture = require_qwen35_architecture(model.get());

        const llama_vocab * vocabulary = llama_model_get_vocab(model.get());
        const int32_t vocabulary_size = llama_vocab_n_tokens(vocabulary);
        if (vocabulary_size <= 0) {
            throw std::runtime_error("llama.cpp reported an invalid vocabulary size");
        }
        for (const llama_token token : tokens) {
            if (token < 0 || token >= vocabulary_size) {
                throw std::runtime_error(
                    "token id " + std::to_string(token) + " is outside the model vocabulary");
            }
        }

        const uint32_t context_tokens = static_cast<uint32_t>(std::max<size_t>(
            kMinimumContextTokens,
            tokens.size()));
        const uint32_t logical_batch_tokens = static_cast<uint32_t>(std::max<size_t>(
            kPhysicalBatchTokens,
            prefill_tokens));
        auto context_params = llama_context_default_params();
        context_params.n_ctx = context_tokens;
        context_params.n_batch = logical_batch_tokens;
        context_params.n_ubatch = kPhysicalBatchTokens;
        context_params.n_seq_max = 1;
        context_params.n_threads = kWorkerThreads;
        context_params.n_threads_batch = kWorkerThreads;
        context_params.embeddings = false;
        context_params.offload_kqv = true;
        context_params.op_offload = true;
        context_params.no_perf = true;
        Context context(llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) {
            throw std::runtime_error("llama.cpp failed to create a Metal context");
        }

        std::vector<DecisionRecord> decisions;
        decisions.reserve(tokens.size() - prefill_tokens + 1);

        Batch prefill_batch(static_cast<int32_t>(prefill_tokens));
        decode_or_throw(
            context.get(),
            tokens.data(),
            static_cast<int32_t>(prefill_tokens),
            0,
            prefill_batch);
        decisions.push_back(save_decision(
            context.get(),
            output_directory,
            0,
            "prefill",
            prefill_tokens,
            vocabulary_size));

        Batch decode_batch(1);
        for (size_t token_index = prefill_tokens; token_index < tokens.size(); ++token_index) {
            decode_or_throw(
                context.get(),
                &tokens[token_index],
                1,
                static_cast<llama_pos>(token_index),
                decode_batch);
            decisions.push_back(save_decision(
                context.get(),
                output_directory,
                token_index - prefill_tokens + 1,
                "decode",
                token_index + 1,
                vocabulary_size));
        }

        const size_t expected_decisions = tokens.size() - prefill_tokens + 1;
        if (decisions.size() != expected_decisions) {
            throw std::runtime_error("internal decision count mismatch");
        }
        // A PASS manifest is published only after every referenced decision is durable.
        fsync_directory(output_directory);

        const std::string manifest = build_manifest(
            model_path,
            token_path,
            ggml_backend_dev_name(metal_device),
            architecture,
            tokens,
            prefill_tokens,
            vocabulary_size,
            decisions);
        write_atomic(
            output_directory,
            "manifest.json",
            reinterpret_cast<const uint8_t *>(manifest.data()),
            manifest.size());
        fsync_directory(output_directory);

        std::cout << "LLAMA TEACHER LOGITS DUMP PASS: " << output_directory.string() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama_teacher_logits_dump: " << error.what() << '\n';
        return 1;
    }
}
