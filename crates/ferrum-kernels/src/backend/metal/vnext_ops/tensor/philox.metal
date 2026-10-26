// Philox4x32-10 constants/permutation from Random123 philox.h:
// https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h
/*
Copyright 2010-2011, D. E. Shaw Research. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
  this list of conditions, and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions, and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of D. E. Shaw Research nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
inline uint4 ferrum_philox4x32_10(uint4 counter, uint2 key) {
    for (uint round = 0; round < 10; ++round) {
        ulong a = ulong(counter.x) * 0xD2511F53u;
        ulong b = ulong(counter.z) * 0xCD9E8D57u;
        counter = uint4(uint(b >> 32) ^ counter.y ^ key.x, uint(b),
                        uint(a >> 32) ^ counter.w ^ key.y, uint(a));
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return counter;
}
inline float2 ferrum_normal_pair(uint a, uint b) {
    float radial = (float(a >> 9) + 0.5f) * 0x1p-23f;
    float uniform_angle = (float(b >> 9) + 0.5f) * 0x1p-23f;
    float radius = sqrt(-2.0f * log(radial));
    float angle = 6.2831853071795864769f * uniform_angle;
    return float2(radius * cos(angle), radius * sin(angle));
}
struct PhiloxParams { ulong elements, grid_width; };
kernel void ferrum_tensor_philox_normal_bf16(
    device const uint* state [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant PhiloxParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    ulong offset = ulong(state[2]) | (ulong(state[3]) << 32);
    ulong skip = offset % 4;
    ulong blocks = (p.elements + skip + 3) / 4;
    for (ulong block = gid; block < blocks; block += p.grid_width) {
        // Dividing before addition preserves all bits even at offset=u64::MAX.
        ulong counter = offset / 4 + block;
        uint4 bits = ferrum_philox4x32_10(uint4(uint(counter), uint(counter >> 32), 0, 0), uint2(state[0], state[1]));
        float4 normals = float4(ferrum_normal_pair(bits.x, bits.y), ferrum_normal_pair(bits.z, bits.w));
        for (uint lane = 0; lane < 4; ++lane) {
            ulong index = block * 4 + lane;
            if (index >= skip && index - skip < p.elements)
                output[index - skip] = encode_bf16(normals[lane]);
        }
    }
}
