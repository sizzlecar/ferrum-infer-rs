#include <llama.h>

#include <algorithm>
#include <cstdint>
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
    BackendGuard() { llama_backend_init(); }
    ~BackendGuard() { llama_backend_free(); }
};

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;

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
    if (argc != 4) {
        std::cerr << "usage: llama_logits_dump MODEL.gguf TOKEN_IDS.txt OUTPUT.f32\n";
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
            throw std::runtime_error(
                "llama_decode failed with status " + std::to_string(decode_status));
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
