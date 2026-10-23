// Independent test-only ATen block propagation. Inputs are six original files
// and nine complete checkpoint segments; no Ferrum intermediate enters here.
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Version.h>
#include <ATen/ops/linear.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
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
#include <vector>

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
    for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<char>(value >> (8*i));
    std::cout.write(bytes, 4);
}
static void u64(uint64_t value) {
    char bytes[8];
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>(value >> (8*i));
    std::cout.write(bytes, 8);
}
static std::string quote(std::string_view text) {
    std::ostringstream out; out << '"';
    const char hex[] = "0123456789abcdef";
    for (unsigned char ch : text) {
        if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
        else if (ch < 32) out << "\\u00" << hex[ch >> 4] << hex[ch & 15];
        else out << static_cast<char>(ch);
    }
    out << '"'; return out.str();
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
struct Segment { std::string path; uint64_t offset; uint64_t bytes; };
static at::Tensor read(const Segment& source, const std::vector<int64_t>& shape, bool f32) {
    std::ifstream file(source.path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open original block source");
    const auto length = file.tellg();
    if (length < 0 || !source.bytes || source.bytes > 128*1024*1024 ||
        source.offset > static_cast<uint64_t>(length) ||
        source.bytes > static_cast<uint64_t>(length) - source.offset ||
        source.offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("bounded block source exceeds file");
    auto tensor = at::empty(shape, at::TensorOptions().dtype(f32 ? at::kFloat : at::kBFloat16).device(at::kCPU));
    if (static_cast<uint64_t>(tensor.nbytes()) != source.bytes)
        throw std::runtime_error("source geometry differs from bytes");
    file.seekg(static_cast<std::streamoff>(source.offset));
    file.read(reinterpret_cast<char*>(tensor.data_ptr()), static_cast<std::streamsize>(source.bytes));
    if (!file || static_cast<uint64_t>(file.gcount()) != source.bytes)
        throw std::runtime_error("truncated original block source");
    if (f32) {
        const auto* values = tensor.const_data_ptr<float>();
        for (int64_t i = 0; i < tensor.numel(); i += 2) {
            if (std::bit_cast<uint32_t>(values[i]) != 0x3f800000 ||
                std::bit_cast<uint32_t>(values[i+1]) != 0)
                throw std::runtime_error("oracle scope requires identity adjacent RoPE probe");
        }
    } else {
        const auto* values = tensor.const_data_ptr<c10::BFloat16>();
        for (int64_t i = 0; i < tensor.numel(); ++i)
            if ((std::bit_cast<uint16_t>(values[i]) & 0x7f80) == 0x7f80)
                throw std::runtime_error("nonfinite original block source");
    }
    return tensor;
}
static at::Tensor b(const at::Tensor& value) { return value.to(at::kBFloat16); }
static at::Tensor f(const at::Tensor& value) { return value.to(at::kFloat); }

// standard_operations/layer_norm_bf16: FP32 population variance and one final
// BF16 boundary. ATen reductions deliberately have their own parallel order.
static at::Tensor layer_norm(const at::Tensor& x, float epsilon) {
    const auto xf = f(x);
    const auto centered = xf - xf.mean(-1, true);
    const auto variance = (centered * centered).mean(-1, true);
    return b(centered * at::reciprocal(at::sqrt(variance + epsilon)));
}
// standard_operations/head_rms_norm_bf16 requires the inner BF16 round trip.
static at::Tensor head_norm(const at::Tensor& x, const at::Tensor& gamma,
                            int64_t rows, int64_t heads, int64_t dim, float epsilon) {
    const auto xf = f(x).reshape({rows, heads, dim});
    const auto mean = (xf * xf).mean(-1, true);
    const auto normalized = b(xf * at::reciprocal(at::sqrt(mean + epsilon)));
    return b(f(normalized) * f(gamma)).reshape({rows, heads * dim});
}
static at::Tensor rope(const at::Tensor& x, const at::Tensor& frequencies,
                       int64_t rows, int64_t heads, int64_t dim) {
    const auto pairs = f(x).reshape({rows, heads, dim/2, 2});
    const auto a = pairs.select(-1, 0), z = pairs.select(-1, 1);
    const auto c = frequencies.select(-1, 0).unsqueeze(1);
    const auto s = frequencies.select(-1, 1).unsqueeze(1);
    // This artifact has c=1,s=+0. Do not claim nonidentity FMA equivalence for
    // ATen's independent expression, or replace the source with a saved output.
    return b(at::stack({a*c-z*s, z*c+a*s}, -1)).reshape({rows, heads*dim});
}
static at::Tensor modulation(const at::Tensor& x, const at::Tensor& scale) {
    return b(f(x) * f(b(1.0f + f(scale))));
}
static at::Tensor residual(const at::Tensor& x, const at::Tensor& gate, const at::Tensor& update) {
    return b(f(x) + f(b(f(gate) * f(update))));
}
static at::Tensor silu_mul(const at::Tensor& gate, const at::Tensor& up) {
    // Unary SiLU's exp/div FP32 contract, then BF16 before multiplying up.
    // No stable sigmoid replacement or fusion across that BF16 boundary.
    const auto gf = f(gate);
    const auto activation = b(gf / (1.0f + at::exp(-gf)));
    return b(f(activation) * f(up));
}
static const std::array<const char*, 21> OUTPUT_IDS = {
    "value.qwen_image21.block0.normalized1", "value.qwen_image21.block0.modulated1",
    "value.qwen_image21.block0.tanh_gate1", "value.qwen_image21.block0_qk_norm.query",
    "value.qwen_image21.block0_qk_norm.key", "value.qwen_image21.block0_qk_norm.query_normalized",
    "value.qwen_image21.block0_qk_norm.key_normalized", "value.qwen_image21.block0_qk_rotary.query_rotated",
    "value.qwen_image21.block0_qk_rotary.key_rotated", "value.qwen_image21.block0_attention.value",
    "value.qwen_image21.block0_attention.attention", "value.qwen_image21.block0_attention.projected",
    "value.qwen_image21.block0.residual1", "value.qwen_image21.block0.normalized2",
    "value.qwen_image21.block0.modulated2", "value.qwen_image21.block0.tanh_gate2",
    "value.qwen_image21.block0_mlp.gate", "value.qwen_image21.block0_mlp.up",
    "value.qwen_image21.block0_mlp.activation", "value.qwen_image21.block0_mlp.output",
    "value.qwen_image21.block0.output"
};
int main(int argc, char** argv) {
    try {
        if (argc != 39) throw std::runtime_error("usage: oracle DEVICE POLICY ROWS HEADS DIM MLP EPS_BITS SCALE_BITS then 15 original PATH OFFSET pairs");
        static_assert(std::endian::native == std::endian::little, "raw tensor protocol requires little endian");
        const uint64_t ordinal = number(argv[1]);
        const std::string policy(argv[2]);
        const uint64_t rows = number(argv[3]), heads = number(argv[4]), dim = number(argv[5]), mlp = number(argv[6]);
        const uint64_t epsilon_bits = number(argv[7]), scale_bits = number(argv[8]);
        if (ordinal > static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max()) ||
            !rows || rows > 16 || !heads || heads > 512 || dim < 2 || dim > 256 ||
            (dim & (dim-1)) || !mlp || mlp > 65536 || heads*dim > 65536 ||
            epsilon_bits > UINT32_MAX || scale_bits > UINT32_MAX ||
            (policy != "library_defaults" && policy != "no_reduced_precision"))
            throw std::runtime_error("unsupported block geometry/policy");
        const float epsilon = std::bit_cast<float>(static_cast<uint32_t>(epsilon_bits));
        const float scale = std::bit_cast<float>(static_cast<uint32_t>(scale_bits));
        if (!std::isfinite(epsilon) || !std::isfinite(scale) || epsilon <= 0 || scale <= 0)
            throw std::runtime_error("nonpositive or nonfinite epsilon/scale");
        const uint64_t h = heads*dim;
        std::array<std::vector<int64_t>, 15> shapes;
        for (size_t i=0;i<5;++i) shapes[i] = {static_cast<int64_t>(rows),static_cast<int64_t>(h)};
        shapes[5] = {static_cast<int64_t>(rows),static_cast<int64_t>(dim/2),2};
        for (size_t i : {6,7,10,11}) shapes[i] = {static_cast<int64_t>(h),static_cast<int64_t>(h)};
        shapes[8] = shapes[9] = {static_cast<int64_t>(dim)};
        shapes[12] = shapes[13] = {static_cast<int64_t>(mlp),static_cast<int64_t>(h)};
        shapes[14] = {static_cast<int64_t>(h),static_cast<int64_t>(mlp)};
        std::array<Segment, 15> sources;
        uint64_t weight_bytes = 0;
        for (size_t i=0;i<sources.size();++i) {
            uint64_t bytes = i == 5 ? 4 : 2;
            for (auto d : shapes[i]) bytes *= static_cast<uint64_t>(d);
            if (bytes > 128*1024*1024) throw std::runtime_error("block source exceeds 128 MiB");
            if (i >= 6) weight_bytes += bytes;
            sources[i] = {argv[9+2*i],number(argv[10+2*i]),bytes};
        }
        const uint64_t output_bytes = rows * (18*h+3*mlp) * 2;
        if (weight_bytes > 512*1024*1024 || output_bytes > 16*1024*1024)
            throw std::runtime_error("block aggregate source/payload budget exceeded");
        c10::InferenceMode inference;
        auto& context = at::globalContext();
        if (policy == "no_reduced_precision") context.setAllowBF16ReductionCuBLAS(false);
        context.setAllowTF32CuBLAS(false);
        const auto device = c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(ordinal));
        c10::DeviceGuard guard(device);
        std::array<at::Tensor, 15> tensors;
        for (size_t i=0;i<tensors.size();++i) tensors[i] = read(sources[i], shapes[i], i == 5).to(device);
        const auto linear = [&](const at::Tensor& x, size_t weight) { return at::linear(x, tensors[6+weight], std::nullopt); };
        std::array<at::Tensor, 21> out;
        out[0] = layer_norm(tensors[0], epsilon);
        out[1] = modulation(out[0], tensors[1]);
        out[2] = b(at::tanh(f(tensors[2])));
        out[3] = linear(out[1], 0); out[4] = linear(out[1], 1);
        out[5] = head_norm(out[3], tensors[8], rows, heads, dim, epsilon);
        out[6] = head_norm(out[4], tensors[9], rows, heads, dim, epsilon);
        out[7] = rope(out[5], tensors[5], rows, heads, dim);
        out[8] = rope(out[6], tensors[5], rows, heads, dim);
        out[9] = linear(out[1], 4);
        const auto query = f(out[7]).reshape({static_cast<int64_t>(rows),static_cast<int64_t>(heads),static_cast<int64_t>(dim)}).permute({1,0,2});
        const auto key = f(out[8]).reshape({static_cast<int64_t>(rows),static_cast<int64_t>(heads),static_cast<int64_t>(dim)}).permute({1,0,2});
        const auto value = f(out[9]).reshape({static_cast<int64_t>(rows),static_cast<int64_t>(heads),static_cast<int64_t>(dim)}).permute({1,0,2});
        const auto scores = at::matmul(query, key.transpose(-2,-1)) * scale;
        const auto probabilities = at::softmax(scores, -1);
        out[10] = b(at::matmul(probabilities,value).permute({1,0,2}).contiguous().reshape({static_cast<int64_t>(rows),static_cast<int64_t>(h)}));
        out[11] = linear(out[10], 5);
        out[12] = residual(tensors[0],out[2],out[11]);
        out[13] = layer_norm(out[12],epsilon);
        out[14] = modulation(out[13],tensors[3]);
        out[15] = b(at::tanh(f(tensors[4])));
        out[16] = linear(out[14],6); out[17] = linear(out[14],7);
        out[18] = silu_mul(out[16],out[17]);
        out[19] = linear(out[18],8);
        out[20] = residual(out[12],out[15],out[19]);
        std::array<at::Tensor,21> cpu;
        for (size_t i=0;i<out.size();++i) {
            const int64_t width = i>=16 && i<=18 ? mlp : h;
            if (out[i].scalar_type()!=at::kBFloat16 || out[i].dim()!=2 || out[i].size(0)!=static_cast<int64_t>(rows) || out[i].size(1)!=width)
                throw std::runtime_error("ATen block boundary dtype/shape differs");
            cpu[i] = out[i].cpu().contiguous();
        }
        std::ostringstream metadata;
        metadata << "{\"operation\":\"aten.qwen_image21.block0\",\"arithmetic\":\"bf16-boundaries.f32-two-pass-norm.f32-dense-softmax.identity-rope.v1\""
            << ",\"torch_version\":" << quote(TORCH_VERSION) << ",\"torch_config\":" << quote(at::show_config())
            << ",\"cxx11_abi\":" << _GLIBCXX_USE_CXX11_ABI << ",\"device\":" << ordinal << ",\"policy\":" << quote(policy)
            << ",\"geometry\":{\"rows\":" << rows << ",\"heads\":" << heads << ",\"head_dim\":" << dim
            << ",\"mlp_width\":" << mlp << ",\"epsilon_f32_bits\":" << epsilon_bits << ",\"scale_f32_bits\":" << scale_bits << "},\"sources\":[";
        for (size_t i=0;i<sources.size();++i) {
            if (i) metadata << ',';
            metadata << "{\"path\":" << quote(sources[i].path) << ",\"offset\":" << sources[i].offset << ",\"bytes\":" << sources[i].bytes << '}';
        }
        metadata << "],\"boundaries\":[";
        for (size_t i=0;i<cpu.size();++i) {
            if (i) metadata << ',';
            metadata << "{\"value_id\":" << quote(OUTPUT_IDS[i]) << ",\"shape\":[" << rows << ',' << cpu[i].size(1)
                << "],\"bytes\":" << cpu[i].nbytes() << '}';
        }
        metadata << "],\"bf16_reduction_option\":" << static_cast<int>(context.allowBF16ReductionCuBLAS())
            << ",\"allow_tf32\":" << (context.allowTF32CuBLAS()?"true":"false")
            << ",\"float32_matmul_precision\":" << static_cast<int>(context.float32MatmulPrecision())
            << ",\"blas_preferred_backend\":" << static_cast<int>(context.blasPreferredBackend())
            << ",\"deterministic_algorithms\":" << (context.deterministicAlgorithms()?"true":"false")
            << ",\"cuda_runtime_version\":" << at::Context::versionCUDART() << ",\"loaded_libraries\":[";
        bool first = true;
        for (const auto& library : mapped_libraries()) { if (!first) metadata << ','; first=false; metadata << quote(library); }
        metadata << "]}";
        const auto encoded = metadata.str();
        if (encoded.size()>65536) throw std::runtime_error("block metadata exceeds 64 KiB");
        std::cout.write("FBLKOR01",8); u32(1); u32(static_cast<uint32_t>(encoded.size()));
        std::cout.write(encoded.data(),static_cast<std::streamsize>(encoded.size())); u64(output_bytes);
        for (const auto& tensor : cpu)
            std::cout.write(reinterpret_cast<const char*>(tensor.const_data_ptr<c10::BFloat16>()),static_cast<std::streamsize>(tensor.nbytes()));
        std::cout.flush(); return std::cout?0:1;
    } catch (const std::exception& error) {
        std::cerr << "Torch block oracle: " << error.what() << '\n'; return 1;
    }
}
