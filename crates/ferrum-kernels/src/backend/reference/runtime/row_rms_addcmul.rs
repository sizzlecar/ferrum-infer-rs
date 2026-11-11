//! Independent single-round RowRMS and addcmul; no staged BF16 intermediates.
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{AddcmulBf16Geometry, RowBroadcast, RowRmsNormSingleRoundBf16Geometry};
#[derive(Clone, Copy)]
pub(crate) enum ReferenceRowRmsAddcmulGeometry {
    RowRms(RowRmsNormSingleRoundBf16Geometry),
    Addcmul(AddcmulBf16Geometry),
}
impl ReferenceRowRmsAddcmulGeometry {
    pub(crate) fn rows(self) -> u64 {
        match self {
            Self::RowRms(g) => g.rows(),
            Self::Addcmul(g) => g.rows(),
        }
    }
    pub(crate) fn width(self) -> u64 {
        match self {
            Self::RowRms(g) => g.width(),
            Self::Addcmul(g) => g.width(),
        }
    }
    pub(crate) fn parameter_elements(self) -> u64 {
        match self {
            Self::RowRms(g) => g.width(),
            Self::Addcmul(g) => {
                if g.mode() == RowBroadcast::Row {
                    g.width()
                } else {
                    g.rows() * g.width()
                }
            }
        }
    }
    pub(crate) fn logical_operations(self) -> Result<u64, ReferenceDeviceRuntimeError> {
        match self {
            Self::RowRms(g) => g.logical_arithmetic_operations(),
            Self::Addcmul(g) => g.logical_arithmetic_operations(),
        }
        .map_err(invalid)
    }
}
pub(crate) struct ReferenceRowRmsAddcmulLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) parameter: ReferenceBufferRegion,
    pub(crate) update: Option<ReferenceBufferRegion>,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: ReferenceRowRmsAddcmulGeometry,
}
impl ReferenceRowRmsAddcmulLaunch {
    pub(crate) fn operation_name(&self) -> &'static str {
        match self.geometry {
            ReferenceRowRmsAddcmulGeometry::RowRms(_) => "vnext_row_rms_single_round_bf16",
            ReferenceRowRmsAddcmulGeometry::Addcmul(_) => "vnext_addcmul_bf16",
        }
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.geometry.rows() * self.geometry.width(),
            self.geometry.logical_operations()?,
        )
        .map_err(invalid)
    }
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let bytes = self.geometry.rows() * self.geometry.width() * 2;
        let check = |region: &ReferenceBufferRegion,
                     expected: u64|
         -> Result<(), ReferenceDeviceRuntimeError> {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes as u64 != expected
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid("requires exact aligned BF16 operands"));
            }
            Ok(())
        };
        check(&self.input, bytes)?;
        check(&self.output, bytes)?;
        check(&self.parameter, self.geometry.parameter_elements() * 2)?;
        match (self.geometry, &self.update) {
            (ReferenceRowRmsAddcmulGeometry::RowRms(_), None) => {}
            (ReferenceRowRmsAddcmulGeometry::Addcmul(_), Some(update)) => check(update, bytes)?,
            _ => return Err(invalid("update presence differs from operation")),
        }
        if self.output.overlaps(&self.input)
            || self.output.overlaps(&self.parameter)
            || self
                .update
                .as_ref()
                .is_some_and(|u| self.output.overlaps(u))
        {
            return Err(invalid("output aliases a read input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.parameter, &self.output] {
            region.validate_runtime(runtime)?;
        }
        if let Some(update) = &self.update {
            update.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceRowRmsAddcmulLaunch) {
    match launch.geometry {
        ReferenceRowRmsAddcmulGeometry::RowRms(g) => {
            let mut storage =
                AllocationLocks::new([&launch.input, &launch.parameter, &launch.output]);
            let width = g.width() as usize;
            for row in 0..g.rows() as usize {
                let start = row * width;
                let mut lanes = [0.0_f32; 32];
                for (lane, sum) in lanes.iter_mut().enumerate() {
                    for d in (lane..width).step_by(32) {
                        let x = bf16::from_bits(storage.read_bits(0, start + d)).to_f32();
                        *sum = x.mul_add(x, *sum);
                    }
                }
                for offset in [16, 8, 4, 2, 1] {
                    for lane in 0..32 - offset {
                        lanes[lane] = lanes[lane] + lanes[lane + offset];
                    }
                }
                let mean = lanes[0] / width as f32;
                let inverse = 1.0_f32 / (mean + g.effective_epsilon()).sqrt();
                for d in 0..width {
                    let x = bf16::from_bits(storage.read_bits(0, start + d)).to_f32();
                    let gamma = bf16::from_bits(storage.read_bits(1, d)).to_f32();
                    storage.write_bits(
                        2,
                        start + d,
                        bf16::from_f32((x * inverse) * gamma).to_bits(),
                    );
                }
            }
        }
        ReferenceRowRmsAddcmulGeometry::Addcmul(g) => {
            let mut storage = AllocationLocks::new([
                &launch.input,
                &launch.parameter,
                launch.update.as_ref().expect("preflight checked update"),
                &launch.output,
            ]);
            for index in 0..(g.rows() * g.width()) as usize {
                let p = if g.mode() == RowBroadcast::Row {
                    index % g.width() as usize
                } else {
                    index
                };
                let x = bf16::from_bits(storage.read_bits(0, index)).to_f32();
                let gate = bf16::from_bits(storage.read_bits(1, p)).to_f32();
                let update = bf16::from_bits(storage.read_bits(2, index)).to_f32();
                let product = gate * update;
                storage.write_bits(3, index, bf16::from_f32(x + product).to_bits());
            }
        }
    }
}
fn invalid(reason: impl std::fmt::Display) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("single-round RowRMS/addcmul: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::super::{
        DeviceCommandPhase, DeviceRuntime, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    };
    use super::*;
    use ferrum_runtime::{CanonicalRational, ResolvedTensorLayout, ResolvedTensorSpec};
    fn fixture(runtime: &ReferenceDeviceRuntime, addcmul: bool) -> ReferenceRowRmsAddcmulLaunch {
        let spec = |shape| {
            ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous)
                .unwrap()
        };
        let x = spec(vec![1, 4]);
        let parameter = spec(if addcmul { vec![1, 4] } else { vec![4] });
        let region = |name, values: &[f32]| {
            let bytes: Vec<_> = values
                .iter()
                .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                .collect();
            buffer(runtime, name, &bytes, 16, ElementType::Bf16)
                .region(0..bytes.len() as u64)
                .unwrap()
        };
        ReferenceRowRmsAddcmulLaunch {
            input: region("resource.single_round.x", &[1.; 4]),
            parameter: region("resource.single_round.parameter", &[129. / 128.; 4]),
            update: if addcmul {
                Some(region("resource.single_round.u", &[132. / 128.; 4]))
            } else {
                None
            },
            output: buffer(
                runtime,
                "resource.single_round.y",
                &[0xa5; 8],
                16,
                ElementType::Bf16,
            )
            .region(0..8)
            .unwrap(),
            geometry: if addcmul {
                ReferenceRowRmsAddcmulGeometry::Addcmul(
                    AddcmulBf16Geometry::new(&x, &parameter, &x, &x, RowBroadcast::Row).unwrap(),
                )
            } else {
                ReferenceRowRmsAddcmulGeometry::RowRms(
                    RowRmsNormSingleRoundBf16Geometry::new(
                        &x,
                        &parameter,
                        &x,
                        CanonicalRational::new(1, 100000).unwrap(),
                    )
                    .unwrap(),
                )
            },
        }
    }
    #[test]
    fn both_operations_reject_dtype_extent_alignment_alias_and_wrong_update_atomically() {
        let r = ReferenceDeviceRuntime::new(config()).unwrap();
        for kind in [false, true] {
            for change in 0..6 {
                let mut a = fixture(&r, kind);
                let output = a.output.clone();
                match change {
                    0 => a.parameter.element_type = ElementType::F32,
                    1 => a.input.length_bytes -= 2,
                    2 => a.parameter.offset_bytes += 1,
                    3 => a.output = a.input.clone(),
                    4 => a.output = a.parameter.clone(),
                    5 => a.update = if kind { None } else { Some(a.input.clone()) },
                    _ => unreachable!(),
                }
                assert!(ReferenceDeviceCommand::row_rms_addcmul_tensor(a).is_err());
                assert_eq!(output.read(), [0xa5; 8]);
            }
        }
    }
    #[test]
    fn late_foreign_static_parameter_fails_before_either_dispatch() {
        let r = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        for kind in [false, true] {
            let first = fixture(&r, kind);
            let output = first.output.clone();
            let mut second = fixture(&r, kind);
            second.parameter = fixture(&foreign, kind).parameter;
            let mut stream = r.create_stream().unwrap();
            let before = r.snapshot();
            assert!(r
                .submit_entries(
                    &mut stream,
                    vec![
                        (
                            DeviceCommandPhase::Compute,
                            Some(0),
                            ReferenceDeviceCommand::row_rms_addcmul_tensor(first).unwrap()
                        ),
                        (
                            DeviceCommandPhase::Compute,
                            Some(1),
                            ReferenceDeviceCommand::row_rms_addcmul_tensor(second).unwrap()
                        )
                    ],
                    false
                )
                .is_err());
            assert_eq!(output.read(), [0xa5; 8]);
            assert_eq!(r.snapshot().submissions, before.submissions);
            assert_eq!(
                r.snapshot().row_rms_single_round_launches,
                before.row_rms_single_round_launches
            );
            assert_eq!(r.snapshot().addcmul_launches, before.addcmul_launches);
        }
    }

    #[test]
    fn old_gated_residual_still_rounds_its_product_before_the_add() {
        use super::super::{ReferenceGatedResidualLaunch, ReferencePointwiseShape};
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let new = fixture(&runtime, true);
        let new_output = new.output.clone();
        let old = fixture(&runtime, true);
        let old_output = old.output.clone();
        let old = ReferenceGatedResidualLaunch {
            input: old.input,
            gate: old.parameter,
            update: old.update.unwrap(),
            output: old.output,
            shape: ReferencePointwiseShape {
                rows: 1,
                width: 4,
                broadcast: RowBroadcast::Row,
            },
        };
        let mut stream = runtime.create_stream().unwrap();
        runtime
            .submit_entries(
                &mut stream,
                vec![
                    (
                        DeviceCommandPhase::Compute,
                        Some(0),
                        ReferenceDeviceCommand::row_rms_addcmul_tensor(new).unwrap(),
                    ),
                    (
                        DeviceCommandPhase::Compute,
                        Some(1),
                        ReferenceDeviceCommand::gated_residual_tensor(old).unwrap(),
                    ),
                ],
                false,
            )
            .unwrap();
        let expected = |value: f32| bf16::from_f32(value).to_bits().to_le_bytes().repeat(4);
        assert_eq!(new_output.read(), expected(2.046875));
        assert_eq!(old_output.read(), expected(2.03125));
    }
}
