// Diagnostic only. Host validation retains and bounds every independent target.
// Three little-endian u64 fields: payload offset, absolute destination, length.
__device__ unsigned long long read_u64(const unsigned char* p) {
    unsigned long long value = 0;
    for (unsigned int i = 0; i < 8; ++i) value |= (unsigned long long)p[i] << (8 * i);
    return value;
}

extern "C" __global__ void ferrum_test_input_upload_scatter(
    const unsigned char* packet, unsigned int count) {
    if (blockIdx.x >= count) return;
    const unsigned char* descriptor = packet + (unsigned long long)blockIdx.x * 24;
    const unsigned long long source = read_u64(descriptor);
    unsigned char* destination = reinterpret_cast<unsigned char*>(read_u64(descriptor + 8));
    const unsigned long long bytes = read_u64(descriptor + 16);
    for (unsigned long long i = threadIdx.x; i < bytes; i += blockDim.x)
        destination[i] = packet[source + i];
}
