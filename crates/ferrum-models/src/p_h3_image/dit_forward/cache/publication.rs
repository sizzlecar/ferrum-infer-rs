//! Private pending publication, owned by the session consumer from admission
//! through its actual completed readback. Each destination is a request scope.
use super::super::refine_only::PH3ImageRefinementDestination;
use super::*;
struct Pending<'a> {
    key: PH3ImageDiTConditioningKey,
    allocation: ManagedHostBuffer,
    _scope: &'a HostMemoryScope,
}
pub(crate) struct PreparedRefinedTextCaches<'a> {
    pending: Vec<Pending<'a>>,
    source_host: HostMemoryAccount,
}
impl<'a> PreparedRefinedTextCaches<'a> {
    pub fn new(
        c: &PH3ImageDiTConfig,
        source_host: &HostMemoryAccount,
        keys: Vec<PH3ImageDiTConditioningKey>,
        destinations: &[PH3ImageRefinementDestination<'a>],
        metadata_capacity: usize,
    ) -> Result<Self> {
        validate_keys(c, &keys)?;
        ensure!(
            destinations.len() == keys.len(),
            "cache destinations differ from submitted manifest"
        );
        ensure!(
            destinations
                .len()
                .checked_mul(std::mem::size_of::<Pending<'a>>())
                .is_some_and(|n| n <= metadata_capacity),
            "pending cache records exceed metadata capacity"
        );
        ensure!(
            destinations
                .iter()
                .enumerate()
                .all(|(i, d)| destinations[..i]
                    .iter()
                    .all(|old| !std::ptr::eq(old.scope, d.scope))),
            "cache destinations require separate request scope guards"
        );
        for (key, destination) in keys.iter().zip(destinations) {
            ensure!(
                &key.request == destination.request
                    && source_host.same_parent(&destination.scope.account()),
                "cache request identity or destination Host parent differs"
            );
        }
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(keys.len())
            .context("pending cache record allocation failed")?;
        for ((r, key), destination) in c.requests.iter().zip(keys).zip(destinations) {
            let bytes = usize::try_from(
                r.text_rows
                    .checked_mul(c.hidden)
                    .and_then(|n| n.checked_mul(2))
                    .context("request cache extent overflow")?,
            )?;
            let allocation = destination
                .scope
                .account()
                .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
            pending.push(Pending {
                key,
                allocation,
                _scope: destination.scope,
            });
        }
        Ok(Self {
            pending,
            source_host: source_host.clone(),
        })
    }
    pub fn publish(
        self,
        c: &PH3ImageDiTConfig,
        host: &HostMemoryAccount,
        declared: &CompiledTensorPort,
        output: &TensorExecutionOutput,
    ) -> Result<Vec<PH3ImageRefinedTextCache>> {
        ensure!(
            self.pending.len() == c.requests.len(),
            "cache publication membership changed"
        );
        for (p, r) in self.pending.iter().zip(&c.requests) {
            ensure!(
                p.key.rows == r.text_rows
                    && p.key.refiner_digest == current_refiner_digest(c, r.text_rows)?,
                "cache publication refiner geometry or semantics changed"
            );
        }
        ensure!(
            host.same_parent(&self.source_host)
                && host.owns(output.bytes())
                && output.bytes().class() == HostMemoryClass::Output
                && declared.value_id().as_str() == PH3ImageDiTFamily::REFINED_TEXT_ID
                && declared.value_id() == output.value_id()
                && declared.tensor() == output.tensor()
                && output.tensor().dimensions() == [c.text_rows()?, c.hidden]
                && output.tensor().element_type() == ElementType::Bf16
                && output.tensor().layout() == &ResolvedTensorLayout::Contiguous,
            "cache readback differs from the submitted session output or Host parent"
        );
        ensure!(
            output
                .bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()),
            "cache readback contains nonfinite refinement"
        );
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.pending.len())
            .context("published cache record allocation failed")?;
        let mut offset = 0usize;
        for (r, mut pending) in c.requests.iter().zip(self.pending) {
            let length = pending.allocation.len();
            let end = offset
                .checked_add(length)
                .context("cache readback range overflow")?;
            pending.allocation.as_mut().copy_from_slice(
                output
                    .bytes()
                    .bytes()
                    .get(offset..end)
                    .context("cache readback range exceeds output")?,
            );
            offset = end;
            result.push(PH3ImageRefinedTextCache {
                key: pending.key,
                tensor: inputs::managed(
                    PH3ImageDiTFamily::REFINED_TEXT_ID,
                    config::tensor(vec![r.text_rows, c.hidden], ElementType::Bf16),
                    pending.allocation.freeze(),
                )?,
            });
        }
        ensure!(
            offset == output.bytes().len(),
            "cache readback has undeclared rows"
        );
        Ok(result)
    }
}
