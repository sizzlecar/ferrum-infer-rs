//! Explicit 512-lane reduction and FMA boundaries; sqrt/div is a declared CPU
//! approximation, not an emulation of CUDA's approximate reciprocal square root.
use super::*;
use ferrum_runtime::{RowSelectedRmsModulationGeometry, RowSelectedRmsModulationKind};

pub(crate) struct ReferenceRowSelectedRmsModulationLaunch {
    pub input: ReferenceBufferRegion,
    pub parameters: ReferenceBufferRegion,
    pub gamma: ReferenceBufferRegion,
    pub branch: Option<ReferenceBufferRegion>,
    pub hidden: Option<ReferenceBufferRegion>,
    pub output: ReferenceBufferRegion,
    pub geometry: RowSelectedRmsModulationGeometry,
}
impl ReferenceRowSelectedRmsModulationLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let residual = self.geometry.kind() == RowSelectedRmsModulationKind::Residual;
        if self.branch.is_some() != residual || self.hidden.is_some() != residual {
            return Err(invalid("residual port presence differs"));
        }
        let g = self.geometry.routing();
        let data = g.elements() * 2;
        for (region, bytes) in [
            (&self.input, data),
            (
                &self.parameters,
                g.parameter_count() * g.parameter_columns() * 2,
            ),
            (&self.gamma, g.columns() * 2),
            (&self.output, data),
        ]
        .into_iter()
        .chain(self.branch.iter().map(|b| (b, data)))
        .chain(self.hidden.iter().map(|b| (b, data)))
        {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes as u64 != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid("requires exact aligned BF16 physical extents"));
            }
        }
        for output in [&self.output].into_iter().chain(self.hidden.iter()) {
            if [&self.input, &self.parameters, &self.gamma]
                .into_iter()
                .chain(self.branch.iter())
                .any(|r| output.overlaps(r))
            {
                return Err(invalid("output overlaps an input"));
            }
        }
        if self
            .hidden
            .as_ref()
            .is_some_and(|h| h.overlaps(&self.output))
        {
            return Err(invalid("outputs overlap"));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.parameters, &self.gamma, &self.output]
            .into_iter()
            .chain(self.branch.iter())
            .chain(self.hidden.iter())
        {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.geometry.routing().elements(),
            self.geometry.logical_arithmetic_operations(),
        )
        .map_err(invalid)
    }
}
fn read(storage: &AllocationLocks<'_, 6>, slot: usize, index: usize) -> f32 {
    bf16::from_bits(storage.read_bits(slot, index)).to_f32()
}
fn hidden(
    storage: &AllocationLocks<'_, 6>,
    g: &RowSelectedRmsModulationGeometry,
    row: usize,
    column: usize,
    p: usize,
) -> f32 {
    let route = g.routing();
    let width = route.columns() as usize;
    if column >= width {
        return 0.;
    }
    let index = row * width + column;
    let x = read(storage, 0, index);
    if g.kind() == RowSelectedRmsModulationKind::Residual {
        let gate = read(
            storage,
            1,
            p * route.parameter_columns() as usize + g.gate_column() as usize + column,
        );
        gate.mul_add(read(storage, 3, index), x)
    } else {
        x
    }
}
fn lane_sum(h: [f32; 16], residual: bool) -> f32 {
    let mut sum = h[0].mul_add(h[0], h[1] * h[1]);
    for (j, x) in h.into_iter().enumerate().skip(2) {
        sum = if residual || j % 2 == 0 {
            x.mul_add(x, sum)
        } else {
            sum + x * x
        };
    }
    sum
}
fn xor_reduce(values: &mut [f32], offsets: &[usize]) {
    for &offset in offsets {
        // A butterfly level reads its previous level, never an updated lane.
        for lane in 0..values.len() {
            let peer = lane ^ offset;
            if peer > lane {
                let sum = values[lane] + values[peer];
                values[lane] = sum;
                values[peer] = sum;
            }
        }
    }
}
pub(super) fn execute(launch: &ReferenceRowSelectedRmsModulationLaunch) {
    let mut storage = AllocationLocks::new([
        &launch.input,
        &launch.parameters,
        &launch.gamma,
        launch.branch.as_ref().unwrap_or(&launch.input),
        launch.hidden.as_ref().unwrap_or(&launch.output),
        &launch.output,
    ]);
    let g = &launch.geometry;
    let route = g.routing();
    let width = route.columns() as usize;
    for row in 0..route.rows() as usize {
        let p = route.parameter_row(row as u64).expect("validated row") as usize;
        let mut partials = [0f32; 512];
        for (lane, sum) in partials.iter_mut().enumerate() {
            let h = std::array::from_fn(|j| {
                hidden(&storage, g, row, lane * 8 + j % 8 + (j / 8) * 4096, p)
            });
            *sum = lane_sum(h, g.kind() == RowSelectedRmsModulationKind::Residual);
        }
        let mut warps = [0f32; 16];
        for (warp, lanes) in partials.chunks_exact_mut(32).enumerate() {
            xor_reduce(lanes, &[16, 8, 4, 2, 1]);
            warps[warp] = lanes[0];
        }
        xor_reduce(&mut warps, &[8, 4, 2, 1]);
        let mean = warps[0] / width as f32;
        let inverse = 1.0f32 / libm::sqrtf(mean + g.effective_epsilon());
        for column in 0..width {
            let h = hidden(&storage, g, row, column, p);
            if launch.hidden.is_some() {
                storage.write_bits(4, row * width + column, bf16::from_f32(h).to_bits());
            }
            let normalized = h * inverse;
            let weighted = normalized * read(&storage, 2, column);
            let scale = read(
                &storage,
                1,
                p * route.parameter_columns() as usize + route.scale_column() as usize + column,
            );
            let shift = read(
                &storage,
                1,
                p * route.parameter_columns() as usize + route.shift_column() as usize + column,
            );
            storage.write_bits(
                5,
                row * width + column,
                bf16::from_f32(weighted.mul_add(1.0 + scale, shift)).to_bits(),
            );
        }
    }
}
fn invalid(s: impl std::fmt::Display) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("fused row RMS modulation: {s}"))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::*;
    fn launch(runtime: &ReferenceDeviceRuntime) -> ReferenceRowSelectedRmsModulationLaunch {
        let tensor = |shape| {
            ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous)
                .unwrap()
        };
        let x = tensor(vec![1, 2]);
        let table = tensor(vec![1, 12]);
        let gamma = tensor(vec![2]);
        let mut attrs = RowSelectedRmsModulationProfile::Block8192Vec8Warp16Fma
            .attributes()
            .unwrap();
        for (key, value) in [
            (
                "epsilon",
                SemanticValue::Rational(CanonicalRational::new(1, 100000).unwrap()),
            ),
            ("run_lengths", SemanticValue::Integers(vec![1])),
            ("parameter_rows", SemanticValue::Integers(vec![0])),
            ("shift_column", SemanticValue::Unsigned(0)),
            ("scale_column", SemanticValue::Unsigned(2)),
            ("gate_column", SemanticValue::Unsigned(4)),
        ] {
            attrs.insert(AttributeId::new(key).unwrap(), value);
        }
        let region = |name, len| {
            buffer(runtime, name, &vec![0xa5; len], 16, ElementType::Bf16)
                .region(0..len as u64)
                .unwrap()
        };
        ReferenceRowSelectedRmsModulationLaunch {
            input: region("resource.rms.x", 4),
            parameters: region("resource.rms.table", 24),
            gamma: region("resource.rms.gamma", 4),
            branch: Some(region("resource.rms.branch", 4)),
            hidden: Some(region("resource.rms.hidden", 4)),
            output: region("resource.rms.output", 4),
            geometry: RowSelectedRmsModulationGeometry::new(
                RowSelectedRmsModulationKind::Residual,
                &[&x, &table, &gamma, &x],
                &[&x, &x],
                &attrs,
            )
            .unwrap(),
        }
    }
    #[test]
    fn dual_outputs_reject_aliases_malformed_storage_and_foreign_owners_before_write() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        for change in 0..5 {
            let mut l = launch(&runtime);
            let original = l.output.clone();
            match change {
                0 => l.hidden = Some(l.output.clone()),
                1 => l.hidden = Some(l.input.clone()),
                2 => l.gamma.element_type = ElementType::F32,
                3 => l.parameters.length_bytes -= 2,
                4 => l.branch = None,
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::row_selected_rms_modulation_tensor(l).is_err());
            assert_eq!(original.read(), [0xa5; 4]);
        }
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        let mut l = launch(&runtime);
        let original = l.output.clone();
        l.branch = launch(&foreign).branch;
        let command = ReferenceDeviceCommand::row_selected_rms_modulation_tensor(l).unwrap();
        let mut stream = runtime.create_stream().unwrap();
        assert!(runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false
            )
            .is_err());
        assert_eq!(original.read(), [0xa5; 4]);
        assert_eq!(runtime.snapshot().submissions, 0);
    }
}
