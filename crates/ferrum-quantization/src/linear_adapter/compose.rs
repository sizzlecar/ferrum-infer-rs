use super::{invalid, ImportedLinearAdapter, LowRankMatrix, Result};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, HostMemoryReservation, ManagedHostBuffer,
    ManagedHostBytes, WeightComponentSpec,
};
use half::{bf16, f16};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::Arc,
};

/// A model-declared route from one complete adapter matrix to a base matrix.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LinearAdapterBinding {
    pub module: String,
    pub external_weight: String,
    pub base_dimensions: [u64; 2],
    pub adapter_output_rows: Range<u64>,
}

pub struct PreparedLinearAdapter {
    imported: Arc<ImportedLinearAdapter>,
    bindings: BTreeMap<String, LinearAdapterBinding>,
    strength: f32,
    workers: usize,
}
pub struct ManagedLinearAdapter {
    prepared: Arc<PreparedLinearAdapter>,
    host: HostMemoryAccount,
    identity: String,
    // Immutable mapped adapter storage is charged before any payload scan.
    // This reservation remains alive for every materializer that can read it.
    _mapped_claim: HostMemoryReservation,
}
impl PreparedLinearAdapter {
    pub fn new(
        imported: Arc<ImportedLinearAdapter>,
        bindings: Vec<LinearAdapterBinding>,
        strength: f32,
    ) -> Result<Self> {
        if !strength.is_finite() {
            return Err(invalid("strength must be finite"));
        }
        let mut mapped = BTreeMap::new();
        let mut modules = BTreeSet::new();
        let mut routes: BTreeMap<String, Vec<Range<u64>>> = BTreeMap::new();
        for binding in bindings {
            let matrix = imported
                .matrices
                .get(&binding.module)
                .ok_or_else(|| invalid("binding names an absent adapter target"))?;
            let rows = &binding.adapter_output_rows;
            if binding.external_weight.is_empty()
                || rows.start >= rows.end
                || rows.end > matrix.output
                || binding.base_dimensions != [rows.end - rows.start, matrix.input]
            {
                return Err(invalid(format!(
                    "adapter/base shape or row mapping differs for {:?}",
                    binding.module
                )));
            }
            if !(strength * matrix.multiplier).is_finite() {
                return Err(invalid("strength times alpha/r is nonfinite"));
            }
            modules.insert(binding.module.clone());
            routes
                .entry(binding.module.clone())
                .or_default()
                .push(rows.clone());
            if mapped
                .insert(binding.external_weight.clone(), binding)
                .is_some()
            {
                return Err(invalid(
                    "multiple adapter targets write the same base matrix",
                ));
            }
        }
        if modules != imported.matrices.keys().cloned().collect() {
            return Err(invalid(
                "every adapter target must have a complete model binding",
            ));
        }
        for (module, mut ranges) in routes {
            ranges.sort_by_key(|range| range.start);
            let mut next = 0;
            for range in ranges {
                if range.start != next {
                    return Err(invalid("adapter row routes overlap or have gaps"));
                }
                next = range.end;
            }
            if next != imported.matrices[&module].output {
                return Err(invalid("adapter row routing is incomplete"));
            }
        }
        Ok(Self {
            imported,
            bindings: mapped,
            strength,
            workers: std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(8),
        })
    }

    fn mapped_owners(&self) -> BTreeMap<usize, &[u8]> {
        let mut owners = BTreeMap::new();
        for matrix in self.imported.matrices.values() {
            for region in [&matrix.a, &matrix.b] {
                let bytes = region.owner_bytes();
                owners.entry(bytes.as_ptr() as usize).or_insert(bytes);
            }
        }
        owners
    }

    /// Metadata-only charge for the unique mapped adapter files. This does not
    /// read tensor values, hash payloads, or reserve any Host memory.
    pub fn mapped_backing_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        let bytes = self
            .mapped_owners()
            .values()
            .try_fold(0usize, |total, bytes| total.checked_add(bytes.len()))
            .ok_or_else(|| invalid("mapped adapter size overflows"))?;
        host.backing_bytes(bytes).map_err(invalid)
    }

    /// Acquire the mapped-weight Host claim, then authenticate and validate all
    /// adapter values. No base payload, device allocation or inference is used.
    pub fn manage(self: &Arc<Self>, host: &HostMemoryAccount) -> Result<Arc<ManagedLinearAdapter>> {
        let owners = self.mapped_owners();
        let bytes = owners
            .values()
            .try_fold(0usize, |n, bytes| n.checked_add(bytes.len()))
            .ok_or_else(|| invalid("mapped adapter size overflows"))?;
        let claim = host
            .reserve(HostMemoryClass::WeightBacking, bytes)
            .map_err(invalid)?;
        let mut hash = Sha256::new();
        hash.update(b"ordinary-linear-lora/f32-ba/f32-scale-add/bf16-rne/v1");
        hash.update(Sha256::digest(include_bytes!("compose.rs")));
        hash.update((self.imported.base_model.len() as u64).to_le_bytes());
        hash.update(self.imported.base_model.as_bytes());
        hash.update(self.strength.to_bits().to_le_bytes());
        // Sort content hashes rather than memory addresses: identity is stable
        // across processes; retained owners only deduplicate the live mappings.
        let mut digests = owners
            .values()
            .map(|bytes| Sha256::digest(bytes).to_vec())
            .collect::<Vec<_>>();
        digests.sort();
        for digest in digests {
            hash.update(digest);
        }
        for (name, binding) in &self.bindings {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(serde_json::to_vec(binding).map_err(invalid)?);
            let matrix = &self.imported.matrices[&binding.module];
            hash.update(matrix.multiplier.to_bits().to_le_bytes());
        }
        for matrix in self.imported.matrices.values() {
            for (region, kind) in [(&matrix.a, matrix.a_type), (&matrix.b, matrix.b_type)] {
                for chunk in region.bytes().chunks_exact(kind.size_bytes() as usize) {
                    if !decode(chunk, kind).is_finite() {
                        return Err(invalid("nonfinite adapter matrix value"));
                    }
                }
            }
        }
        Ok(Arc::new(ManagedLinearAdapter {
            prepared: self.clone(),
            host: host.clone(),
            identity: format!("{:x}", hash.finalize()),
            _mapped_claim: claim,
        }))
    }
}

impl ManagedLinearAdapter {
    pub fn same_host_parent(&self, host: &HostMemoryAccount) -> bool {
        self.host.same_parent(host)
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn strength(&self) -> f32 {
        self.prepared.strength
    }
    pub fn mapped_backing_bytes(&self) -> usize {
        self._mapped_claim.backing_bytes()
    }
    pub fn prepared(&self) -> &PreparedLinearAdapter {
        &self.prepared
    }
    pub fn affects(&self, component: &WeightComponentSpec) -> bool {
        self.prepared.affects(component)
    }
    pub fn workspace_bytes(
        &self,
        host: &HostMemoryAccount,
        component: &WeightComponentSpec,
    ) -> Result<usize> {
        self.prepared.workspace_bytes(host, component)
    }
    pub fn output_backing_bytes(
        &self,
        host: &HostMemoryAccount,
        component: &WeightComponentSpec,
    ) -> Result<usize> {
        self.prepared.output_backing_bytes(host, component)
    }
}

impl PreparedLinearAdapter {
    pub fn affects(&self, component: &WeightComponentSpec) -> bool {
        self.strength != 0.0
            && component.external_names.len() == 1
            && self.bindings.contains_key(&component.external_names[0])
    }
    fn binding(
        &self,
        component: &WeightComponentSpec,
    ) -> Result<(&LinearAdapterBinding, &LowRankMatrix)> {
        let binding = component
            .external_names
            .first()
            .filter(|_| component.external_names.len() == 1)
            .and_then(|name| self.bindings.get(name))
            .ok_or_else(|| invalid("component has no adapter binding"))?;
        if component.dimensions != binding.base_dimensions
            || component.physical_element_type() != ElementType::Bf16
        {
            return Err(invalid(
                "composition requires the exact bound BF16 source matrix",
            ));
        }
        Ok((binding, &self.imported.matrices[&binding.module]))
    }
    /// Additional physical scratch, excluding the composed BF16 output itself.
    /// One F32 A and a single F32 output row per bounded worker are sufficient.
    pub fn workspace_bytes(
        &self,
        host: &HostMemoryAccount,
        component: &WeightComponentSpec,
    ) -> Result<usize> {
        if !self.affects(component) {
            return Ok(0);
        }
        let (binding, matrix) = self.binding(component)?;
        let input = usize::try_from(matrix.input).map_err(invalid)?;
        let rank = usize::try_from(matrix.rank).map_err(invalid)?;
        let rows = usize::try_from(binding.base_dimensions[0]).map_err(invalid)?;
        let a = host
            .backing_bytes(checked_bytes(rank, input, 4)?)
            .map_err(invalid)?;
        let row = host
            .backing_bytes(checked_bytes(1, input, 4)?)
            .map_err(invalid)?;
        a.checked_add(
            row.checked_mul(self.workers.min(rows))
                .ok_or_else(|| invalid("workspace overflows"))?,
        )
        .ok_or_else(|| invalid("workspace overflows"))
    }
    pub fn output_backing_bytes(
        &self,
        host: &HostMemoryAccount,
        component: &WeightComponentSpec,
    ) -> Result<usize> {
        if !self.affects(component) {
            return Ok(0);
        }
        self.binding(component)?;
        host.backing_bytes(
            usize::try_from(component.physical_bytes().map_err(invalid)?).map_err(invalid)?,
        )
        .map_err(invalid)
    }
}

impl ManagedLinearAdapter {
    /// Explicit cold-path composition. Never call from TensorArchive::tensor or
    /// WeightComponentSource::component: preparation must remain metadata-only.
    pub fn compose(
        &self,
        host: &HostMemoryAccount,
        component: &WeightComponentSpec,
        base: &[u8],
    ) -> Result<ManagedHostBytes> {
        if !self.host.same_parent(host) {
            return Err(invalid("composition Host account has a different parent"));
        }
        let (binding, matrix) = self.prepared.binding(component)?;
        let input = usize::try_from(matrix.input).map_err(invalid)?;
        let rank = usize::try_from(matrix.rank).map_err(invalid)?;
        let rows = usize::try_from(binding.base_dimensions[0]).map_err(invalid)?;
        let output_bytes = checked_bytes(rows, input, 2)?;
        if base.len() != output_bytes {
            return Err(invalid("base payload length differs from binding"));
        }
        let mut a = host
            .allocate_zeroed(
                HostMemoryClass::WeightBacking,
                checked_bytes(rank, input, 4)?,
            )
            .map_err(invalid)?;
        let a_values = floats_mut(&mut a)?;
        for (value, bytes) in a_values.iter_mut().zip(
            matrix
                .a
                .bytes()
                .chunks_exact(matrix.a_type.size_bytes() as usize),
        ) {
            *value = decode(bytes, matrix.a_type);
        }
        let mut output = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, output_bytes)
            .map_err(invalid)?;
        let workers = self.prepared.workers.min(rows);
        let rows_per_worker = rows.div_ceil(workers);
        let scale = self.prepared.strength * matrix.multiplier;
        let a_values: &[f32] = a_values;
        std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::new();
            for (worker, out) in output
                .as_mut()
                .chunks_mut(checked_bytes(rows_per_worker, input, 2)?)
                .enumerate()
            {
                let start = worker * rows_per_worker;
                let host = host.clone();
                let matrix = matrix;
                handles.push(scope.spawn(move || -> Result<()> {
                    let mut row = host
                        .allocate_zeroed(
                            HostMemoryClass::WeightBacking,
                            checked_bytes(1, input, 4)?,
                        )
                        .map_err(invalid)?;
                    let row = floats_mut(&mut row)?;
                    for (local, destination) in out.chunks_exact_mut(input * 2).enumerate() {
                        row.fill(0.0);
                        let b_row = usize::try_from(binding.adapter_output_rows.start)
                            .map_err(invalid)?
                            + start
                            + local;
                        for k in 0..rank {
                            let offset = (b_row * rank + k) * matrix.b_type.size_bytes() as usize;
                            let coefficient = decode(&matrix.b.bytes()[offset..], matrix.b_type);
                            for (delta, a) in
                                row.iter_mut().zip(&a_values[k * input..(k + 1) * input])
                            {
                                *delta += coefficient * a;
                            }
                        }
                        let base_start = (start + local) * input * 2;
                        for ((destination, original), delta) in destination
                            .chunks_exact_mut(2)
                            .zip(base[base_start..base_start + input * 2].chunks_exact(2))
                            .zip(row.iter())
                        {
                            let original =
                                bf16::from_bits(u16::from_le_bytes([original[0], original[1]]))
                                    .to_f32();
                            let value = original + scale * delta;
                            let encoded = bf16::from_f32(value);
                            if !value.is_finite() || !encoded.is_finite() {
                                return Err(invalid("nonfinite composed BF16 weight"));
                            }
                            destination.copy_from_slice(&encoded.to_bits().to_le_bytes());
                        }
                    }
                    Ok(())
                }));
            }
            // Join all workers even after an error; no partial output escapes.
            let mut failure = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(())) => (),
                    Ok(Err(error)) => {
                        failure.get_or_insert(error);
                    }
                    Err(_) => {
                        failure.get_or_insert_with(|| invalid("composition worker panicked"));
                    }
                }
            }
            failure.map_or(Ok(()), Err)
        })?;
        Ok(output.freeze())
    }
}
fn checked_bytes(rows: usize, columns: usize, element: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .and_then(|n| n.checked_mul(element))
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| invalid("matrix workspace size overflows"))
}
fn floats_mut(buffer: &mut ManagedHostBuffer) -> Result<&mut [f32]> {
    // SAFETY: all f32 bit patterns are valid; this is the sole mutable borrow.
    // align_to_mut supplies the alignment proof rather than assuming a quota's
    // byte alignment implies native F32 alignment.
    let (prefix, values, suffix) = unsafe { buffer.as_mut().align_to_mut::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(invalid("F32 workspace is unaligned"));
    }
    Ok(values)
}
fn decode(bytes: &[u8], kind: ElementType) -> f32 {
    match kind {
        ElementType::Bf16 => bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
        ElementType::F16 => f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
        ElementType::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => unreachable!("import validates floating-point storage"),
    }
}
