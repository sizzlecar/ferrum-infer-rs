use super::*;
use ferrum_runtime::{BatchedF32Geometry, BatchedF32Op};

#[cfg(test)]
mod tests;

pub(crate) struct ReferenceBatchedF32Launch {
    pub(crate) inputs: Vec<ReferenceBufferRegion>,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: BatchedF32Geometry,
}
impl ReferenceBatchedF32Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        if self.inputs.len() != self.geometry.op().input_count() as usize {
            return Err(error("wrong input count"));
        }
        for (r, bytes) in self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, r)| (r, self.geometry.input_bytes(i)))
            .chain(std::iter::once((
                &self.output,
                self.geometry.output_bytes(),
            )))
        {
            r.validate_range()?;
            if r.element_type != ElementType::F32
                || r.length_bytes as u64 != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(error("requires exact aligned F32 extents"));
            }
        }
        if self.inputs.iter().any(|x| self.output.overlaps(x)) {
            return Err(error("output aliases input"));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in self.inputs.iter().chain(std::iter::once(&self.output)) {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
fn error(s: &str) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("batched F32: {s}"))
}
pub(super) fn execute(l: &ReferenceBatchedF32Launch) {
    match l.inputs.as_slice() {
        [input] => execute_locked(l.geometry, AllocationLocks::new([input, &l.output])),
        [left, right] => execute_locked(l.geometry, AllocationLocks::new([left, right, &l.output])),
        _ => unreachable!("validated fixed operand count"),
    }
}
fn execute_locked<const N: usize>(g: BatchedF32Geometry, mut storage: AllocationLocks<'_, N>) {
    let out = N - 1;
    match g.op() {
        BatchedF32Op::Matmul => {
            for batch in 0..g.batch() * g.group() {
                for m in 0..g.m() {
                    for n in 0..g.n() {
                        let mut sum = 0.0f32;
                        for k in 0..g.k() {
                            let a = storage.read_f32(0, ((batch * g.m() + m) * g.k() + k) as usize);
                            let offset = if g.transpose_rhs() {
                                (batch * g.n() + n) * g.k() + k
                            } else {
                                (batch * g.k() + k) * g.n() + n
                            };
                            let b = storage.read_f32(1, offset as usize);
                            sum = a.mul_add(b, sum);
                        }
                        storage.write_f32(out, ((batch * g.m() + m) * g.n() + n) as usize, sum);
                    }
                }
            }
        }
        BatchedF32Op::Scale => {
            for i in 0..g.output_elements() {
                let value = storage.read_f32(0, i as usize) * g.scale();
                storage.write_f32(out, i as usize, value);
            }
        }
        BatchedF32Op::Softmax => {
            for row in 0..g.rows() {
                let offset = (row * g.n()) as usize;
                let mut maximum = f32::NEG_INFINITY;
                let mut invalid = false;
                for n in 0..g.n() as usize {
                    let x = storage.read_f32(0, offset + n);
                    invalid |= x.is_nan() || x == f32::INFINITY;
                    maximum = maximum.max(x);
                }
                invalid |= maximum == f32::NEG_INFINITY;
                let mut sum = 0f32;
                if !invalid {
                    for n in 0..g.n() as usize {
                        let exp = (storage.read_f32(0, offset + n) - maximum).exp();
                        sum += exp;
                    }
                }
                for n in 0..g.n() as usize {
                    let value = if invalid {
                        f32::from_bits(0x7fc00000)
                    } else {
                        (storage.read_f32(0, offset + n) - maximum).exp() / sum
                    };
                    storage.write_f32(out, offset + n, value);
                }
            }
        }
    }
}
