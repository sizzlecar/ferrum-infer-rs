//! Checked CUDA views for the exact cold Q8 coefficient representation.
use super::*;
use ferrum_runtime::ElementType;

pub(super) fn q8_f32_bytes(k: u32, n: u32) -> Result<(u64, u64), CudaDeviceRuntimeError> {
    if k == 0 || k % 32 != 0 || n == 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "Q8 F32 preparation requires positive K32/N",
        ));
    }
    let source = u64::from(n)
        .checked_mul(u64::from(k / 32))
        .and_then(|v| v.checked_mul(34));
    let decoded = u64::from(n)
        .checked_mul(u64::from(k))
        .and_then(|v| v.checked_mul(4));
    source
        .zip(decoded)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("Q8 F32 matrix bytes overflow"))
}

fn disjoint(a: &CudaBufferRegion, b: &CudaBufferRegion) -> Result<bool, CudaDeviceRuntimeError> {
    let end = |r: &CudaBufferRegion| {
        r.device_ptr()
            .checked_add(r.length_bytes())
            .ok_or_else(|| CudaDeviceRuntimeError::contract("Q8 F32 region address overflow"))
    };
    Ok(a.device_ptr() >= end(b)? || b.device_ptr() >= end(a)?)
}

fn hot_extents(
    rows: u32,
    k: u32,
    n: u32,
    stride: u32,
    offset: u32,
) -> Result<(u64, u64), CudaDeviceRuntimeError> {
    if rows == 0 || rows > u16::MAX as u32 {
        return Err(CudaDeviceRuntimeError::contract(
            "Q8 F32 row grid is invalid",
        ));
    }
    q8_f32_bytes(k, n)?;
    let end = offset
        .checked_add(n)
        .filter(|&end| end <= stride)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("Q8 F32 output window exceeds stride"))?;
    let input = u64::from(rows)
        .checked_mul(u64::from(k))
        .and_then(|v| v.checked_mul(2));
    let output = u64::from(rows - 1)
        .checked_mul(u64::from(stride))
        .and_then(|v| v.checked_add(u64::from(end)))
        .and_then(|v| v.checked_mul(2));
    input
        .zip(output)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("Q8 F32 hot extent overflows"))
}

impl CudaNativeBlockKernels {
    pub(in crate::backend::cuda::vnext_ops) fn validate_q8_0_f32_destination(
        &self,
        source: &CudaBufferRegion,
        destination: &CudaBufferRegion,
        k: u32,
        n: u32,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let (encoded, decoded) = q8_f32_bytes(k, n)?;
        if source.element_type() != ElementType::U8
            || !matches!(
                destination.element_type(),
                ElementType::U8 | ElementType::F32
            )
            || source.length_bytes() != encoded
            || destination.length_bytes() != decoded
            || source.runtime_instance() != destination.runtime_instance()
            || !Arc::ptr_eq(source.context(), &self.q8_predecode_context)
            || !Arc::ptr_eq(destination.context(), &self.q8_predecode_context)
            || destination.device_ptr() % 4 != 0
            || !disjoint(source, destination)?
        {
            return Err(CudaDeviceRuntimeError::contract(
                "Q8 F32 preparation differs from exact source/destination views",
            ));
        }
        Ok(())
    }

    pub(in crate::backend::cuda::vnext_ops) fn predecode_q8_0_f32(
        &self,
        stream: &CudaStream,
        source: &CudaBufferRegion,
        destination: &CudaBufferRegion,
        k: u32,
        n: u32,
    ) -> Result<(), CudaDeviceRuntimeError> {
        self.validate_q8_0_f32_destination(source, destination, k, n)?;
        if !Arc::ptr_eq(stream.context(), &self.q8_predecode_context) {
            return Err(CudaDeviceRuntimeError::contract(
                "Q8 F32 preparation stream belongs to another CUDA context",
            ));
        }
        let (_, bytes) = q8_f32_bytes(k, n)?;
        let grid = u32::try_from((bytes / 4).div_ceil(256))
            .ok()
            .filter(|&grid| grid <= i32::MAX as u32)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("Q8 F32 preparation grid exceeds CUDA extent")
            })?;
        let source_address = source.device_ptr();
        let destination_address = destination.device_ptr();
        let mut launch = stream.launch_builder(&self.q8_0_predecode_f32);
        launch
            .arg(&source_address)
            .arg(&destination_address)
            .arg(&k)
            .arg(&n);
        // SAFETY: exact retained byte-safe Q8 source and disjoint aligned F32
        // destination were checked; the Core command owns both through its fence.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| CudaDeviceRuntimeError::driver("Q8 F32 cold preparation", e))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::cuda::vnext_ops) fn linear_q8_0_predecoded_f32_f16(
        &self,
        stream: &CudaStream,
        input: &CudaBufferRegion,
        weights: &CudaBufferRegion,
        output: &CudaBufferRegion,
        rows: u32,
        k: u32,
        n: u32,
        output_stride: u32,
        output_offset: u32,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let (_, bytes) = q8_f32_bytes(k, n)?;
        let (input_bytes, output_bytes) = hot_extents(rows, k, n, output_stride, output_offset)?;
        if !matches!(input.element_type(), ElementType::F16 | ElementType::U8)
            || !matches!(output.element_type(), ElementType::F16 | ElementType::U8)
            || !matches!(weights.element_type(), ElementType::U8 | ElementType::F32)
            || input.device_ptr() % 2 != 0
            || output.device_ptr() % 2 != 0
            || weights.device_ptr() % 4 != 0
            || input_bytes > input.length_bytes()
            || output_bytes > output.length_bytes()
            || weights.length_bytes() != bytes
            || input.runtime_instance() != weights.runtime_instance()
            || output.runtime_instance() != weights.runtime_instance()
            || !Arc::ptr_eq(stream.context(), &self.q8_predecode_context)
            || !Arc::ptr_eq(input.context(), stream.context())
            || !Arc::ptr_eq(output.context(), stream.context())
            || !Arc::ptr_eq(weights.context(), stream.context())
            || !disjoint(input, output)?
            || !disjoint(weights, output)?
        {
            return Err(CudaDeviceRuntimeError::contract(
                "Q8 F32 hot launch differs from exact current views",
            ));
        }
        let x = input.device_ptr();
        let w = weights.device_ptr();
        let y = output.device_ptr();
        let mut launch = stream.launch_builder(&self.linear_q8_0_predecoded_f32_f16);
        launch
            .arg(&x)
            .arg(&w)
            .arg(&y)
            .arg(&rows)
            .arg(&k)
            .arg(&n)
            .arg(&output_stride)
            .arg(&output_offset);
        // SAFETY: bounded F16 current input/output and immutable aligned F32
        // coefficients remain retained by the consuming command.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (n.div_ceil(4), rows, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| CudaDeviceRuntimeError::driver("Q8 predecoded F32 linear", e))
    }
}

#[cfg(test)]
mod tests {
    use super::{hot_extents, q8_f32_bytes};
    #[test]
    fn q8_f32_preparation_byte_extents_preserve_k32_and_reject_invalid_shapes() {
        assert_eq!(q8_f32_bytes(32, 1).unwrap(), (34, 128));
        assert_eq!(q8_f32_bytes(5120, 48).unwrap(), (261120, 983040));
        for (k, n) in [(0, 48), (31, 48), (33, 48), (5120, 0)] {
            assert!(q8_f32_bytes(k, n).is_err());
        }
        assert!(q8_f32_bytes(u32::MAX - 31, u32::MAX).is_err());
    }

    #[test]
    fn q8_f32_hot_extents_preserve_shared_windows_and_row_bounds() {
        assert_eq!(
            hot_extents(8, 5120, 48, 16480, 16384).unwrap(),
            (81920, 263584)
        );
        assert_eq!(
            hot_extents(8, 5120, 48, 16480, 16432).unwrap(),
            (81920, 263680)
        );
        assert!(hot_extents(8, 5120, 48, 16480, 16433).is_err());
        assert!(hot_extents(1, 32, 1, u32::MAX, u32::MAX).is_err());
        assert!(hot_extents(0, 5120, 48, 16480, 16384).is_err());
        assert!(hot_extents(65536, 5120, 48, 16480, 16384).is_err());
    }
}
