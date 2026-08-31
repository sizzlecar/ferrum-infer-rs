#include <stdint.h>

// GPT-OSS stores one expert's packed E2M1 nibbles as byte-contiguous
// [N, K / 32, 16]. Treat every four bytes as one GPTQ-compatible word and
// transpose [N, K / 8] to the [K / 8, N] input expected by Marlin's native
// repacker. Values remain bit-identical; no quantized value is decoded.
extern "C" __global__ void gpt_oss_mxfp4_blocks_to_gptq_words(
    const uint32_t* __restrict__ source,
    uint32_t* __restrict__ destination,
    int rows_n,
    int columns_k) {
  const uint64_t packed_columns = static_cast<uint64_t>(columns_k) / 8;
  const uint64_t count = static_cast<uint64_t>(rows_n) * packed_columns;
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index >= count) {
    return;
  }
  const uint64_t row = index / packed_columns;
  const uint64_t packed_column = index - row * packed_columns;
  destination[packed_column * static_cast<uint64_t>(rows_n) + row] =
      source[index];
}

// Map source E8M0 bytes [N, K / 32] directly into the scale layout consumed
// by vLLM's MXFP4 Marlin MoE kernel. This fuses the source transpose, Marlin
// P64 permutation, vLLM row-pair/8-column packing, and final [0,2,1,3]
// four-byte interleave into one device pass.
extern "C" __global__ void gpt_oss_mxfp4_scales_to_marlin(
    const uint8_t* __restrict__ source,
    uint8_t* __restrict__ destination,
    int rows_n,
    int columns_k) {
  const uint64_t groups = static_cast<uint64_t>(columns_k) / 32;
  const uint64_t rows = static_cast<uint64_t>(rows_n);
  const uint64_t count = groups * rows;
  const uint64_t destination_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x +
      static_cast<uint64_t>(threadIdx.x);
  if (destination_index >= count) {
    return;
  }

  // Undo vLLM's final swap(1, 2) in each four-byte chunk.
  const uint64_t chunk_base = destination_index & ~uint64_t{3};
  const uint64_t lane4 = destination_index & uint64_t{3};
  const uint64_t first_index =
      chunk_base + (lane4 == 1 ? 2 : (lane4 == 2 ? 1 : lane4));

  // Undo the row-pair by eight-column packing.
  const uint64_t lane8 = first_index & uint64_t{7};
  uint64_t packed = first_index >> 3;
  const uint64_t pair_row = packed & uint64_t{1};
  packed >>= 1;
  const uint64_t column_blocks = rows / 8;
  const uint64_t column_block = packed % column_blocks;
  const uint64_t row_pair = packed / column_blocks;
  const uint64_t p64_index =
      (row_pair * 2 + pair_row) * rows + column_block * 8 + lane8;

  // Undo Marlin's P64 permutation. For destination-local d, P64[d] reads
  // transposed[(d / 8) + 8 * (d % 8)].
  const uint64_t p64_chunk = p64_index & ~uint64_t{63};
  const uint64_t p64_lane = p64_index & uint64_t{63};
  const uint64_t transposed_index =
      p64_chunk + (p64_lane >> 3) + 8 * (p64_lane & uint64_t{7});

  const uint64_t group = transposed_index / rows;
  const uint64_t row = transposed_index - group * rows;
  destination[destination_index] = source[row * groups + group];
}
