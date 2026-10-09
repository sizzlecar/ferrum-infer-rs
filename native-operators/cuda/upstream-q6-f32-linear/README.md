# Q6_K F32 D4/MMQ operator

This optional, separately locked operator accepts F32 input and publishes F32
output. It uses pinned llama.cpp Q6_K MMQ with D4 activation scales for actual
rows 1 through 32, input width divisible by 256, and positive output width.
It does not extend the original F16 linear ABI or enable an automatic route.

The pack preserves the prototype's fast-math expressions. Zero groups and
padding become positive zero. Nonfinite input, a nonzero group flushed to zero,
or a zero/nonfinite reciprocal quantization factor poisons that row. In the
pinned CUDA compilation, `div.approx.ftz` makes magnitudes above 2^126 poison;
finite scale underflow to zero is allowed. A nonfinite consumed weight
coefficient poisons the whole physical leaf. Publication propagates either flag
or nonfinite raw output as canonical F32 NaN (`0x7fc00000`); finite F32 values
are retained without an F16 conversion. These rules do not reject before dot.

The native plan is reconstructed before each launch. Its packed allocation
includes every cooperative tile load, even inactive row lanes. The caller owns
the scratch, F32 output and per-row flags, and retains a separate four-byte
weight flag. Device ordering and live lease validation belong to the provider.
The artifact is built from this source definition and the pinned upstream
headers; the experimental static archive is not a production dependency.
