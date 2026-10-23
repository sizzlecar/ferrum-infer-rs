// Test-only installed ATen CUDA BF16 linear oracle. No Ferrum CUDA kernel,
// Python wrapper, hand-written dot, or reduced output-channel matrix is used.
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Version.h>
#include <ATen/ops/linear.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>
#include <bit>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

static uint64_t number(const char* text) {
    uint64_t value = 0;
    const std::string_view s(text);
    const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != s.data() + s.size())
        throw std::runtime_error("expected unsigned decimal argument");
    return value;
}
static void u32(uint32_t value) {
    char bytes[4];
    for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<char>(value >> (8 * i));
    std::cout.write(bytes, 4);
}
static void u64(uint64_t value) {
    char bytes[8];
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>(value >> (8 * i));
    std::cout.write(bytes, 8);
}
static std::string quote(std::string_view text) {
    std::ostringstream out;
    out << '"';
    const char hex[] = "0123456789abcdef";
    for (unsigned char ch : text) {
        if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
        else if (ch < 32) out << "\\u00" << hex[ch >> 4] << hex[ch & 15];
        else out << static_cast<char>(ch);
    }
    out << '"';
    return out.str();
}
static std::set<std::string> mapped_libraries() {
    std::ifstream maps("/proc/self/maps");
    if (!maps) throw std::runtime_error("cannot inspect actual mapped libraries");
    std::set<std::string> paths;
    std::string line;
    while (std::getline(maps, line)) {
        const auto begin = line.find('/');
        if (begin != std::string::npos && line.find(".so", begin) != std::string::npos)
            paths.insert(line.substr(begin));
    }
    return paths;
}
static at::Tensor read_bf16(const char* path, uint64_t offset, uint64_t rows, uint64_t columns) {
    const uint64_t bytes = rows * columns * 2; // bounded geometry checked before call
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open oracle input segment");
    const auto length = file.tellg();
    if (length < 0 || offset > static_cast<uint64_t>(length) ||
        bytes > static_cast<uint64_t>(length) - offset ||
        offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("oracle input segment exceeds file");
    auto tensor = at::empty({static_cast<int64_t>(rows), static_cast<int64_t>(columns)},
        at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(tensor.data_ptr<c10::BFloat16>()),
        static_cast<std::streamsize>(bytes));
    if (!file || static_cast<uint64_t>(file.gcount()) != bytes)
        throw std::runtime_error("oracle input segment was truncated");
    const auto* values = tensor.const_data_ptr<c10::BFloat16>();
    for (uint64_t i = 0; i < rows * columns; ++i) {
        if ((std::bit_cast<uint16_t>(values[i]) & 0x7f80) == 0x7f80)
            throw std::runtime_error("linear diagnostic input must be finite BF16");
    }
    return tensor;
}
int main(int argc, char** argv) {
    try {
        if (argc != 10) throw std::runtime_error(
            "usage: oracle DEVICE POLICY ROWS K N INPUT_FILE INPUT_OFFSET WEIGHT_FILE WEIGHT_OFFSET");
        static_assert(std::endian::native == std::endian::little,
                      "raw BF16 protocol requires little-endian host");
        const uint64_t ordinal = number(argv[1]);
        const std::string policy(argv[2]);
        const uint64_t rows = number(argv[3]), k = number(argv[4]), n = number(argv[5]);
        const uint64_t input_offset = number(argv[7]), weight_offset = number(argv[9]);
        if (ordinal > static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max()) ||
            !rows || rows > 16 || !k || k > 65536 || !n || n > 65536 ||
            n * k * 2 > 64 * 1024 * 1024 ||
            (policy != "library_defaults" && policy != "no_reduced_precision"))
            throw std::runtime_error("unsupported device, linear extent, weight bound, or explicit policy");
        c10::InferenceMode inference;
        auto input = read_bf16(argv[6], input_offset, rows, k);
        auto weight = read_bf16(argv[8], weight_offset, n, k);
        auto& context = at::globalContext();
        if (policy == "no_reduced_precision") context.setAllowBF16ReductionCuBLAS(false);
        const auto device = c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(ordinal));
        c10::DeviceGuard guard(device);
        // Preserve complete OI weight geometry. Selecting output channels here
        // would change GEMM dispatch and invalidate comparison with the artifact.
        auto result = at::linear(input.to(device), weight.to(device), std::nullopt);
        auto cpu = result.cpu().contiguous(); // blocking; preserves BF16 bytes
        const uint64_t bytes = rows * n * 2;
        if (cpu.scalar_type() != at::kBFloat16 || cpu.dim() != 2 ||
            cpu.size(0) != static_cast<int64_t>(rows) || cpu.size(1) != static_cast<int64_t>(n) ||
            static_cast<uint64_t>(cpu.nbytes()) != bytes)
            throw std::runtime_error("unexpected ATen linear output");
        std::ostringstream metadata;
        metadata << "{\"operation\":\"aten.linear.no_bias\",\"torch_version\":" << quote(TORCH_VERSION)
            << ",\"torch_config\":" << quote(at::show_config())
            << ",\"cxx11_abi\":" << _GLIBCXX_USE_CXX11_ABI
            << ",\"device\":" << ordinal << ",\"policy\":" << quote(policy)
            << ",\"rows\":" << rows << ",\"in_features\":" << k << ",\"out_features\":" << n
            << ",\"dtype\":\"BF16\",\"input_path\":" << quote(argv[6])
            << ",\"input_offset\":" << input_offset << ",\"weight_path\":" << quote(argv[8])
            << ",\"weight_offset\":" << weight_offset
            << ",\"bf16_reduction_option\":" << static_cast<int>(context.allowBF16ReductionCuBLAS())
            << ",\"allow_tf32\":" << (context.allowTF32CuBLAS() ? "true" : "false")
            << ",\"float32_matmul_precision\":" << static_cast<int>(context.float32MatmulPrecision())
            << ",\"blas_preferred_backend\":" << static_cast<int>(context.blasPreferredBackend())
            << ",\"deterministic_algorithms\":" << (context.deterministicAlgorithms() ? "true" : "false")
            << ",\"cuda_runtime_version\":" << at::Context::versionCUDART()
            << ",\"loaded_libraries\":[";
        bool first = true;
        for (const auto& library : mapped_libraries()) {
            if (!first) metadata << ',';
            first = false;
            metadata << quote(library);
        }
        metadata << "]}";
        const auto encoded = metadata.str();
        if (encoded.size() > 65536) throw std::runtime_error("linear metadata exceeds 64KiB");
        std::cout.write("FLINOR01", 8);
        u32(1); u32(static_cast<uint32_t>(encoded.size()));
        std::cout.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        u64(bytes);
        std::cout.write(reinterpret_cast<const char*>(cpu.const_data_ptr<c10::BFloat16>()),
            static_cast<std::streamsize>(bytes));
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Torch linear oracle: " << error.what() << '\n';
        return 1;
    }
}
