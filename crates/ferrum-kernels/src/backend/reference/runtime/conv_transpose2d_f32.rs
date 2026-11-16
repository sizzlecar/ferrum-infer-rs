use super::*;
use ferrum_runtime::ConvTranspose2dF32Geometry;
pub(crate) struct ReferenceConvTranspose2dF32Launch {
    pub inputs: Vec<ReferenceBufferRegion>,
    pub output: ReferenceBufferRegion,
    pub geometry: ConvTranspose2dF32Geometry,
}
impl ReferenceConvTranspose2dF32Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        if self.inputs.len() != 3 {
            return Err(invalid("operand count differs"));
        }
        for (i, r) in self
            .inputs
            .iter()
            .enumerate()
            .chain(std::iter::once((self.inputs.len(), &self.output)))
        {
            r.validate_range()?;
            let count = if i == self.inputs.len() {
                Some(self.geometry.elements())
            } else {
                self.geometry.input_elements(i)
            };
            if r.element_type != ElementType::F32
                || count.and_then(|n| n.checked_mul(4)) != Some(r.length_bytes as u64)
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(invalid("requires exact scalar-aligned F32 ranges"));
            }
        }
        if self.inputs.iter().any(|r| self.output.overlaps(r)) {
            return Err(invalid("output overlaps input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in self.inputs.iter().chain(std::iter::once(&self.output)) {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.geometry.elements(),
            self.geometry.logical_arithmetic_operations(),
        )
        .map_err(|e| invalid(e.to_string()))
    }
}
pub(super) fn execute(l: &ReferenceConvTranspose2dF32Launch) {
    let g = l.geometry;
    let [batch, ci, ih, iw] = g.input();
    let [_, co, oh, ow] = g.output();
    let [_, co_group, kh, kw] = g.weight();
    let ci_group = ci / g.groups();
    let [ph, pw] = g.padding();
    let [sh, sw] = g.stride();
    let [dh, dw] = g.dilation();
    let mut data = AllocationLocks::new([&l.inputs[0], &l.inputs[1], &l.inputs[2], &l.output]);
    for n in 0..batch {
        for o in 0..co {
            let group = o / co_group;
            for y in 0..oh {
                for x in 0..ow {
                    let mut sum = 0.0f32;
                    for c in group * ci_group..(group + 1) * ci_group {
                        for ky in 0..kh {
                            let iy =
                                i128::from(y) + i128::from(ph) - i128::from(ky) * i128::from(dh);
                            if iy < 0
                                || iy % i128::from(sh) != 0
                                || iy / i128::from(sh) >= i128::from(ih)
                            {
                                continue;
                            }
                            for kx in 0..kw {
                                let ix = i128::from(x) + i128::from(pw)
                                    - i128::from(kx) * i128::from(dw);
                                if ix < 0
                                    || ix % i128::from(sw) != 0
                                    || ix / i128::from(sw) >= i128::from(iw)
                                {
                                    continue;
                                }
                                let input = (((n * ci + c) * ih + (iy / i128::from(sh)) as u64)
                                    * iw
                                    + (ix / i128::from(sw)) as u64)
                                    as usize;
                                let weight =
                                    (((c * co_group + o % co_group) * kh + ky) * kw + kx) as usize;
                                let product = data.read_f32(0, input) * data.read_f32(1, weight);
                                sum += product;
                            }
                        }
                    }
                    let value = sum + data.read_f32(2, o as usize);
                    data.write_f32(3, (((n * co + o) * oh + y) * ow + x) as usize, value);
                }
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!(
        "Reference F32 transposed convolution: {}",
        reason.into()
    ))
}
#[cfg(test)]
mod tests;
