#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct BackendGuard {
    BackendGuard() {
        ggml_backend_load_all();
        llama_backend_init();
    }
    ~BackendGuard() { llama_backend_free(); }
};

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct CheckpointRecord {
    std::string tensor_name;
    std::string raw_file;
    std::string source_dtype;
    std::array<int64_t, 4> source_dimensions;
    std::vector<int64_t> logical_shape;
    size_t element_count;
};

class CheckpointDumper {
  public:
    CheckpointDumper(std::filesystem::path output_dir, std::vector<std::string> checkpoint_names)
        : output_dir_(std::move(output_dir)), checkpoint_names_(std::move(checkpoint_names)) {
        std::filesystem::create_directories(output_dir_);
    }

    bool wants(const char * tensor_name) const {
        return std::find(checkpoint_names_.begin(), checkpoint_names_.end(), tensor_name) !=
               checkpoint_names_.end();
    }

    bool capture(ggml_tensor * tensor) noexcept {
        try {
            if (!wants(tensor->name)) {
                return true;
            }
            if (std::any_of(records_.begin(), records_.end(), [tensor](const CheckpointRecord & record) {
                    return record.tensor_name == tensor->name;
                })) {
                throw std::runtime_error(std::string("duplicate checkpoint tensor: ") + tensor->name);
            }
            if (!ggml_is_contiguous(tensor)) {
                throw std::runtime_error(std::string("checkpoint tensor is not contiguous: ") +
                                         tensor->name);
            }

            const auto element_count = static_cast<size_t>(ggml_nelements(tensor));
            std::vector<uint8_t> source(ggml_nbytes(tensor));
            std::vector<float> normalized(element_count);
            ggml_backend_tensor_get(tensor, source.data(), 0, source.size());
            for (size_t index = 0; index < element_count; ++index) {
                if (tensor->type == GGML_TYPE_F32) {
                    std::memcpy(&normalized[index], source.data() + index * sizeof(float), sizeof(float));
                } else if (tensor->type == GGML_TYPE_F16) {
                    ggml_fp16_t value;
                    std::memcpy(&value, source.data() + index * sizeof(value), sizeof(value));
                    normalized[index] = ggml_fp16_to_fp32(value);
                } else if (tensor->type == GGML_TYPE_BF16) {
                    ggml_bf16_t value;
                    std::memcpy(&value, source.data() + index * sizeof(value), sizeof(value));
                    normalized[index] = ggml_bf16_to_fp32(value);
                } else {
                    throw std::runtime_error(std::string("unsupported checkpoint tensor dtype: ") +
                                             ggml_type_name(tensor->type));
                }
            }

            const std::string raw_file = std::string(tensor->name) + ".f32";
            write_f32(output_dir_ / raw_file, normalized);
            std::vector<int64_t> logical_shape;
            int highest_axis = 3;
            while (highest_axis > 0 && tensor->ne[highest_axis] == 1) {
                --highest_axis;
            }
            for (int axis = highest_axis; axis >= 0; --axis) {
                logical_shape.push_back(tensor->ne[axis]);
            }
            records_.push_back(CheckpointRecord{
                tensor->name,
                raw_file,
                ggml_type_name(tensor->type),
                {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]},
                std::move(logical_shape),
                element_count,
            });
            return true;
        } catch (const std::exception & error) {
            error_ = error.what();
            return false;
        }
    }

    const std::string & error() const { return error_; }

    void write_manifest() {
        if (!error_.empty()) {
            throw std::runtime_error(error_);
        }
        for (const std::string & expected : checkpoint_names_) {
            const auto count = std::count_if(
                records_.begin(), records_.end(), [&expected](const CheckpointRecord & record) {
                    return record.tensor_name == expected;
                });
            if (count != 1) {
                throw std::runtime_error(std::string("missing checkpoint tensor: ") + expected);
            }
        }
        std::sort(records_.begin(), records_.end(), [](const auto & left, const auto & right) {
            return left.tensor_name < right.tensor_name;
        });

        std::ofstream output(output_dir_ / "checkpoint-manifest.json", std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot create checkpoint manifest");
        }
        output << "{\n  \"schema_version\": 1,\n  \"status\": \"pass\",\n"
               << "  \"output_dtype\": \"f32\",\n  \"records\": [\n";
        for (size_t index = 0; index < records_.size(); ++index) {
            const auto & record = records_[index];
            output << "    {\"tensor_name\": \"" << record.tensor_name
                   << "\", \"raw_file\": \"" << record.raw_file
                   << "\", \"source_dtype\": \"" << record.source_dtype
                   << "\", \"source_dimensions\": [" << record.source_dimensions[0] << ", "
                   << record.source_dimensions[1] << ", " << record.source_dimensions[2] << ", "
                   << record.source_dimensions[3] << "], \"logical_shape\": [";
            for (size_t axis = 0; axis < record.logical_shape.size(); ++axis) {
                if (axis != 0) {
                    output << ", ";
                }
                output << record.logical_shape[axis];
            }
            output << "], \"element_count\": " << record.element_count << "}";
            output << (index + 1 == records_.size() ? "\n" : ",\n");
        }
        output << "  ]\n}\n";
        output.flush();
        if (!output) {
            throw std::runtime_error("cannot write complete checkpoint manifest");
        }
    }

  private:
    static void write_f32(const std::filesystem::path & path, const std::vector<float> & values) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot create checkpoint tensor file");
        }
        output.write(
            reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        output.flush();
        if (!output) {
            throw std::runtime_error("cannot write complete checkpoint tensor file");
        }
    }

    std::filesystem::path output_dir_;
    std::vector<std::string> checkpoint_names_;
    std::vector<CheckpointRecord> records_;
    std::string error_;
};

bool checkpoint_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * dumper = static_cast<CheckpointDumper *>(user_data);
    if (ask) {
        return dumper->wants(tensor->name);
    }
    return dumper->capture(tensor);
}

std::vector<llama_token> load_tokens(const char * path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open token file");
    }
    std::vector<llama_token> tokens;
    int64_t value = 0;
    while (input >> value) {
        if (value < 0 || value > std::numeric_limits<llama_token>::max()) {
            throw std::runtime_error("token id is outside llama_token range");
        }
        tokens.push_back(static_cast<llama_token>(value));
    }
    if (!input.eof()) {
        throw std::runtime_error("token file contains a non-integer value");
    }
    if (tokens.empty()) {
        throw std::runtime_error("token file is empty");
    }
    return tokens;
}

std::vector<std::string> load_checkpoint_names(const char * path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open checkpoint names file");
    }
    std::vector<std::string> names;
    std::string name;
    while (std::getline(input, name)) {
        if (!name.empty() && name.back() == '\r') {
            name.pop_back();
        }
        if (name.empty()) {
            continue;
        }
        const bool portable = std::all_of(name.begin(), name.end(), [](unsigned char character) {
            return std::isalnum(character) != 0 || character == '.' || character == '_' ||
                   character == '-';
        });
        if (!portable || name == "." || name == "..") {
            throw std::runtime_error("checkpoint name is not a portable file name: " + name);
        }
        if (std::find(names.begin(), names.end(), name) != names.end()) {
            throw std::runtime_error("duplicate checkpoint name: " + name);
        }
        names.push_back(std::move(name));
    }
    if (!input.eof()) {
        throw std::runtime_error("cannot read checkpoint names file");
    }
    if (names.empty()) {
        throw std::runtime_error("checkpoint names file is empty");
    }
    return names;
}

void write_logits(const char * path, const float * logits, int32_t vocabulary_size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create logits output file");
    }
    output.write(
        reinterpret_cast<const char *>(logits),
        static_cast<std::streamsize>(vocabulary_size) * sizeof(float));
    output.flush();
    if (!output) {
        throw std::runtime_error("cannot write complete logits output file");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "usage: llama_logits_dump MODEL.gguf TOKEN_IDS.txt OUTPUT.f32 "
                     "[CHECKPOINT_DIR CHECKPOINT_NAMES.txt]\n";
        return 2;
    }

    try {
        const auto tokens = load_tokens(argv[2]);
        BackendGuard backend;

        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.use_mmap = true;
        model_params.use_mlock = false;
        model_params.check_tensors = false;
        Model model(llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) {
            throw std::runtime_error("llama.cpp failed to load the model");
        }

        const auto worker_threads = static_cast<int32_t>(std::clamp(
            std::thread::hardware_concurrency(),
            1U,
            8U));
        auto context_params = llama_context_default_params();
        context_params.n_ctx = std::max<uint32_t>(128, tokens.size());
        context_params.n_batch = static_cast<uint32_t>(tokens.size());
        context_params.n_ubatch = static_cast<uint32_t>(tokens.size());
        context_params.n_seq_max = 1;
        context_params.n_threads = worker_threads;
        context_params.n_threads_batch = worker_threads;
        context_params.embeddings = false;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        context_params.no_perf = false;
        std::unique_ptr<CheckpointDumper> checkpoint_dumper;
        if (argc == 6) {
            checkpoint_dumper = std::make_unique<CheckpointDumper>(
                argv[4], load_checkpoint_names(argv[5]));
            context_params.cb_eval = checkpoint_callback;
            context_params.cb_eval_user_data = checkpoint_dumper.get();
        }
        Context context(llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) {
            throw std::runtime_error("llama.cpp failed to create a CPU context");
        }

        auto batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (int32_t index = 0; index < batch.n_tokens; ++index) {
            batch.token[index] = tokens[static_cast<size_t>(index)];
            batch.pos[index] = index;
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = index == batch.n_tokens - 1;
        }
        const int32_t decode_status = llama_decode(context.get(), batch);
        llama_batch_free(batch);
        if (decode_status != 0) {
            if (checkpoint_dumper && !checkpoint_dumper->error().empty()) {
                throw std::runtime_error(checkpoint_dumper->error());
            }
            throw std::runtime_error(
                "llama_decode failed with status " + std::to_string(decode_status));
        }
        if (checkpoint_dumper) {
            checkpoint_dumper->write_manifest();
        }

        const auto * vocabulary = llama_model_get_vocab(model.get());
        const int32_t vocabulary_size = llama_vocab_n_tokens(vocabulary);
        if (vocabulary_size <= 0) {
            throw std::runtime_error("llama.cpp reported an invalid vocabulary size");
        }
        const float * logits = llama_get_logits_ith(context.get(), -1);
        if (logits == nullptr) {
            throw std::runtime_error("llama.cpp did not expose final-token logits");
        }
        write_logits(argv[3], logits, vocabulary_size);
        std::cout << "{\"backend\":\"cpu\",\"n_gpu_layers\":0,\"offload_kqv\":false,"
                  << "\"op_offload\":false,\"token_count\":" << tokens.size()
                  << ",\"vocabulary_size\":" << vocabulary_size
                  << ",\"output_dtype\":\"f32\"}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama_logits_dump: " << error.what() << '\n';
        return 1;
    }
}
