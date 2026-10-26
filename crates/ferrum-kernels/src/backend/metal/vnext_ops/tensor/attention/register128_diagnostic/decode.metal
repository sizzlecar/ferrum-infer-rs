// Diagnostic only: conversion and storage, without FP32 arithmetic that could
// flush a decoded subnormal before its payload can be observed.
kernel void ferrum_diagnostic_bf16_decode(
    const device ushort* input [[buffer(0)]],
    device float* output [[buffer(3)]],
    constant ulong& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (ulong(index) < count) output[index] = decode_bf16(input[index]);
}
