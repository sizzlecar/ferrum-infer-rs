// Test-only deployment oracle: dispatch the installed LibTorch CUDA kernel.
// No CUDA source, cuRAND implementation, Python API, or Ferrum kernel is used.
#include <ATen/ATen.h>
#include <ATen/Version.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/randn.h>
#include <c10/core/DeviceGuard.h>
#include <torch/version.h>
#include <bit>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
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
static std::string json_quote(std::string_view text) {
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
static std::set<std::string> loaded_libraries() {
    // Linux oracle only. Maps resolves loader names to the actual mapped paths,
    // including libraries opened lazily by the first CUDA operation.
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
int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: oracle DEVICE SEED OFFSET C H W DRAWS");
        static_assert(std::endian::native == std::endian::little,
                      "the raw BF16 protocol requires a little-endian host");
        const uint64_t device_index = number(argv[1]);
        const uint64_t seed = number(argv[2]), offset = number(argv[3]);
        const uint64_t c = number(argv[4]), h = number(argv[5]), w = number(argv[6]);
        const uint64_t draws = number(argv[7]);
        if (device_index > static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max()) ||
            !c || !h || !w || c > INT32_MAX / 2 || h > INT32_MAX / 2 / c ||
            w > INT32_MAX / 2 / c / h || (draws != 1 && draws != 2) || offset % 4 != 0)
            throw std::runtime_error("unsupported device, BF16 extent, draw count, or Torch offset");
        const uint64_t bytes = c * h * w * 2;
        const auto device = c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device_index));
        c10::DeviceGuard guard(device);
        auto generator = at::cuda::detail::createCUDAGenerator(device.index());
        {
            std::lock_guard<std::mutex> lock(generator.mutex());
            generator.set_current_seed(seed);
            generator.set_offset(offset);
        }
        for (uint64_t draw = 0; draw < draws; ++draw) {
            const uint64_t before = generator.get_offset();
            // Shape is the fixed pipeline's B,F,C,H,W with B=F=1. Contiguous
            // bytes have exactly the NCHW order consumed by Ferrum.
            auto tensor = at::randn(
                {1, 1, static_cast<int64_t>(c), static_cast<int64_t>(h), static_cast<int64_t>(w)},
                generator, at::TensorOptions().dtype(at::kBFloat16).device(device));
            auto cpu = tensor.cpu().contiguous(); // blocking, no FP32 conversion
            const uint64_t after = generator.get_offset();
            if (static_cast<uint64_t>(cpu.nbytes()) != bytes || cpu.scalar_type() != at::kBFloat16)
                throw std::runtime_error("unexpected Torch output storage");
            if (draw == 0) {
                std::ostringstream metadata;
                metadata << "{\"torch_version\":" << json_quote(TORCH_VERSION)
                         << ",\"torch_config\":" << json_quote(at::show_config())
                         << ",\"cxx11_abi\":" << _GLIBCXX_USE_CXX11_ABI
                         << ",\"device\":" << device_index << ",\"seed\":" << seed
                         << ",\"offset\":" << offset << ",\"draws\":" << draws
                         << ",\"shape\":[1,1," << c << ',' << h << ',' << w
                         << "],\"dtype\":\"BF16\",\"loaded_libraries\":[";
                bool first = true;
                for (const auto& library : loaded_libraries()) {
                    if (!first) metadata << ',';
                    first = false;
                    metadata << json_quote(library);
                }
                metadata << "]}";
                const std::string encoded = metadata.str();
                if (encoded.size() > 65536) throw std::runtime_error("oracle metadata exceeds 64KiB");
                std::cout.write("FRNGOR01", 8);
                u32(1); u32(static_cast<uint32_t>(encoded.size()));
                std::cout.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
            }
            u64(before); u64(after); u64(bytes);
            std::cout.write(reinterpret_cast<const char*>(cpu.const_data_ptr<c10::BFloat16>()),
                            static_cast<std::streamsize>(bytes));
            if (!std::cout) throw std::runtime_error("oracle output write failed");
        }
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Torch RNG oracle: " << error.what() << '\n';
        return 1;
    }
}
