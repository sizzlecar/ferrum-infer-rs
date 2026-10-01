//! Frozen, finite numerical subset of checked selected algorithms. This is not
//! a provider capability declaration, execution authority, or measured support.
use super::*;
use ferrum_interfaces::execution_cost::AlgorithmWorkKindV1;
use sha2::{Digest, Sha256};

const MAX_AXES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AlgorithmAxisV1 {
    signature: [u8; 32],
    kind: u8,
}
impl AlgorithmAxisV1 {
    pub(super) fn new(signature: [u8; 32], kind: AlgorithmWorkKindV1) -> Self {
        Self {
            signature,
            kind: kind as u8,
        }
    }
    fn width(self) -> Result<usize> {
        match self.kind {
            0 => Ok(4),
            1..=4 => Ok(2),
            5 => Ok(3),
            _ => Err(StructuredUnknown::InvalidInput),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct UniverseWire {
    revision: u32,
    workload_domain: [u8; 32],
    #[serde(deserialize_with = "bounded_algorithms")]
    algorithms: Vec<AlgorithmAxisV1>,
}
fn bounded_algorithms<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<AlgorithmAxisV1>, D::Error> {
    struct Bounded;
    impl<'de> serde::de::Visitor<'de> for Bounded {
        type Value = Vec<AlgorithmAxisV1>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a bounded selected-algorithm subset")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            use serde::de::Error;
            if seq.size_hint().is_some_and(|n| n > MAX_AXES / 11) {
                return Err(A::Error::custom("algorithm capacity"));
            }
            let mut out = Vec::new();
            while let Some(value) = seq.next_element()? {
                if out.len() == MAX_AXES / 11 {
                    return Err(A::Error::custom("algorithm capacity"));
                }
                out.try_reserve_exact(1).map_err(A::Error::custom)?;
                out.push(value);
            }
            Ok(out)
        }
    }
    d.deserialize_seq(Bounded)
}
#[derive(Debug, PartialEq, Eq)]
struct UniverseInner {
    wire: UniverseWire,
    signature: [u8; 32],
    basis_axes: usize,
}

/// A digest is recomputed from the typed declaration on import. Listing an
/// algorithm permits only numerical projection; Fit and both heldout phases
/// must independently observe and qualify the required work directions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(into = "UniverseWire")]
pub struct DeclaredAlgorithmUniverseV1(Arc<UniverseInner>);

impl<'de> Deserialize<'de> for DeclaredAlgorithmUniverseV1 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        UniverseWire::deserialize(d)?
            .try_into()
            .map_err(|e| serde::de::Error::custom(format!("{e:?}")))
    }
}

impl TryFrom<UniverseWire> for DeclaredAlgorithmUniverseV1 {
    type Error = StructuredUnknownV2;
    fn try_from(wire: UniverseWire) -> Result<Self> {
        if wire.revision != 1
            || wire.workload_domain == [0; 32]
            || wire.algorithms.is_empty()
            || wire.algorithms.len() > MAX_AXES / 2
            || wire.algorithms.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        let mut axes = 0usize;
        let mut hash = Sha256::new();
        hash.update(b"ferrum.declared-algorithm-universe.v1\0");
        hash.update(wire.workload_domain);
        hash.update((wire.algorithms.len() as u64).to_le_bytes());
        for a in &wire.algorithms {
            if a.signature == [0; 32] {
                return Err(StructuredUnknown::InvalidInput);
            }
            axes = axes
                .checked_add(a.width()?)
                .ok_or(StructuredUnknown::Capacity)?;
            hash.update(a.signature);
            hash.update([a.kind]);
        }
        if axes >= MAX_AXES || wire.algorithms.len() * 11 >= MAX_AXES {
            return Err(StructuredUnknown::Capacity);
        }
        Ok(Self(Arc::new(UniverseInner {
            wire,
            signature: hash.finalize().into(),
            basis_axes: axes,
        })))
    }
}
impl From<DeclaredAlgorithmUniverseV1> for UniverseWire {
    fn from(value: DeclaredAlgorithmUniverseV1) -> Self {
        value.0.wire.clone()
    }
}
impl DeclaredAlgorithmUniverseV1 {
    /// Bounded declaration capacity, independent of observed numeric support.
    /// The first real input also checks its complete host/replay tail dimensions.
    pub fn validate_budget(&self, maximum_axes: usize, maximum_bytes: usize) -> Result<()> {
        if maximum_axes == 0
            || self.0.basis_axes > maximum_axes
            || self.0.wire.algorithms.len() * 11 > maximum_axes
            || self
                .retained_payload_bytes()
                .is_none_or(|n| n > maximum_bytes)
        {
            return Err(StructuredUnknown::Capacity);
        }
        Ok(())
    }
    /// Checked finite subset inclusion; no numeric coverage or execution permission.
    pub fn contains_universe(&self, other: &Self) -> bool {
        self.workload_domain_signature() == other.workload_domain_signature()
            && other
                .0
                .wire
                .algorithms
                .iter()
                .all(|a| self.0.wire.algorithms.binary_search(a).is_ok())
    }
    pub fn from_inputs<'a>(
        inputs: impl IntoIterator<Item = &'a StructuredInputV2>,
        maximum_axes: usize,
    ) -> Result<Self> {
        let mut builder = DeclaredAlgorithmUniverseBuilderV1::new(maximum_axes, usize::MAX)?;
        for input in inputs {
            builder.observe(input)?;
        }
        builder.finish()
    }
    /// Input-only distinction for catalogue feedback. False means a valid
    /// original input from this physical domain contains an undeclared
    /// algorithm. Missing/wrong domains and unsupported population shapes stay
    /// errors; this does not establish measured support or successful execution.
    pub fn contains_checked_algorithms(&self, input: &StructuredInputV2) -> Result<bool> {
        input.numerical_family_key()?;
        if input.physical_domain.as_ref() != Some(self.workload_domain_signature()) {
            return Err(StructuredUnknown::WrongDomain);
        }
        if input
            .algorithm_universe
            .as_ref()
            .is_some_and(|current| current != self)
        {
            return Err(StructuredUnknown::WrongDomain);
        }
        Ok(input
            .algorithm_axes
            .iter()
            .all(|a| self.0.wire.algorithms.binary_search(a).is_ok()))
    }

    pub(super) fn checked_axis_counts(&self, input: &StructuredInputV2) -> Result<(usize, usize)> {
        if input.physical_domain.as_ref() != Some(self.workload_domain_signature()) {
            return Err(StructuredUnknown::WrongDomain);
        }
        if let Some(current) = &input.algorithm_universe {
            return if current == self {
                Ok((input.basis.len(), input.support.len()))
            } else {
                Err(StructuredUnknown::WrongDomain)
            };
        }
        if !self.contains_checked_algorithms(input)? {
            return Err(StructuredUnknown::WrongDomain);
        }
        let local = input.algorithm_axes.iter().try_fold(0usize, |n, a| {
            n.checked_add(a.width()?).ok_or(StructuredUnknown::Capacity)
        })?;
        let basis = input
            .basis
            .len()
            .checked_sub(local)
            .and_then(|n| n.checked_add(self.0.basis_axes))
            .ok_or(StructuredUnknown::InvalidInput)?;
        let support = input
            .support
            .len()
            .checked_sub(input.algorithm_axes.len() * 11)
            .and_then(|n| n.checked_add(self.0.wire.algorithms.len() * 11))
            .ok_or(StructuredUnknown::InvalidInput)?;
        if basis > MAX_AXES || support > MAX_AXES {
            return Err(StructuredUnknown::Capacity);
        }
        Ok((basis, support))
    }
    /// Final retained input payload at exact requested vector capacities. Caller
    /// also reserves the still-live original input when cloning for projection.
    pub fn projected_input_retained_bytes(&self, input: &StructuredInputV2) -> Option<usize> {
        if input.algorithm_universe.is_some() {
            return input.retained_payload_bytes();
        }
        let local = input
            .algorithm_axes
            .iter()
            .try_fold(0usize, |n, a| n.checked_add(a.width().ok()?))?;
        let basis = input
            .basis
            .len()
            .checked_sub(local)?
            .checked_add(self.0.basis_axes)?;
        let support = input
            .support
            .len()
            .checked_sub(input.algorithm_axes.len().checked_mul(11)?)?
            .checked_add(self.0.wire.algorithms.len().checked_mul(11)?)?;
        input
            .retained_payload_bytes()?
            .checked_sub(
                input
                    .basis
                    .capacity()
                    .checked_mul(std::mem::size_of::<f64>())?,
            )?
            .checked_sub(
                input
                    .support
                    .capacity()
                    .checked_mul(std::mem::size_of::<u64>())?,
            )?
            .checked_add(basis.checked_mul(std::mem::size_of::<f64>())?)?
            .checked_add(support.checked_mul(std::mem::size_of::<u64>())?)?
            .checked_add(self.retained_payload_bytes()?)
    }
    /// Same stable basis placement as numeric projection, for discovery's
    /// input-only union. No completion result or measured duration is read.
    pub(in crate::implementations::continuous) fn project_basis_flags(
        &self,
        input: &StructuredInputV2,
        flags: &[bool],
    ) -> Result<Vec<bool>> {
        let (count, _) = self.checked_axis_counts(input)?;
        if flags.len() != input.basis.len() || input.algorithm_universe.is_some() {
            return Err(StructuredUnknown::WrongDomain);
        }
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| StructuredUnknown::Capacity)?;
        out.resize(count, false);
        out[0] = flags[0];
        let (mut source, mut old, mut new) = (0usize, 1usize, 1usize);
        for a in &self.0.wire.algorithms {
            let width = a.width()?;
            if input.algorithm_axes.get(source) == Some(a) {
                out[new..new + width].copy_from_slice(&flags[old..old + width]);
                source += 1;
                old += width;
            }
            new += width;
        }
        if source != input.algorithm_axes.len() || flags.len() - old != count - new {
            return Err(StructuredUnknown::WrongDomain);
        }
        out[new..].copy_from_slice(&flags[old..]);
        Ok(out)
    }

    pub fn signature(&self) -> &[u8; 32] {
        &self.0.signature
    }
    pub fn workload_domain_signature(&self) -> &[u8; 32] {
        &self.0.wire.workload_domain
    }
    pub fn algorithm_count(&self) -> usize {
        self.0.wire.algorithms.len()
    }
    pub(in crate::implementations::continuous) fn shares_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// The consuming projection keeps the original input and allocates only
    /// its two replacement vectors. The universe itself is an Arc clone.
    pub(in crate::implementations::continuous) fn projection_vector_bytes(
        &self,
        input: &StructuredInputV2,
    ) -> Result<usize> {
        let (basis, support) = self.checked_axis_counts(input)?;
        basis
            .checked_add(support)
            .and_then(|n| n.checked_mul(8))
            .ok_or(StructuredUnknown::Capacity)
    }
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<UniverseInner>())?
            .checked_add(2 * std::mem::size_of::<usize>())?
            .checked_add(
                self.0
                    .wire
                    .algorithms
                    .capacity()
                    .checked_mul(std::mem::size_of::<AlgorithmAxisV1>())?,
            )
    }
}

/// Bounded cold inventory accumulation. Neither timing nor outcomes are input.
pub struct DeclaredAlgorithmUniverseBuilderV1 {
    domain: Option<[u8; 32]>,
    algorithms: Vec<AlgorithmAxisV1>,
    maximum_axes: usize,
    maximum_retained_bytes: usize,
    tail_basis: usize,
    tail_support: usize,
}
impl DeclaredAlgorithmUniverseBuilderV1 {
    pub fn new(maximum_axes: usize, maximum_retained_bytes: usize) -> Result<Self> {
        if maximum_axes == 0
            || maximum_axes > MAX_AXES
            || maximum_retained_bytes < std::mem::size_of::<Self>()
        {
            return Err(StructuredUnknown::Capacity);
        }
        Ok(Self {
            domain: None,
            algorithms: Vec::new(),
            maximum_axes,
            maximum_retained_bytes,
            tail_basis: 0,
            tail_support: 0,
        })
    }
    pub(in crate::implementations::continuous) fn retained_upper_bound(
        maximum_axes: usize,
    ) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<UniverseInner>())?
            .checked_add(2 * std::mem::size_of::<usize>())?
            .checked_add((maximum_axes / 11).checked_mul(std::mem::size_of::<AlgorithmAxisV1>())?)
    }
    /// Import a previously checked declaration without inventing observation rows.
    pub(in crate::implementations::continuous) fn observe_universe(
        &mut self,
        universe: &DeclaredAlgorithmUniverseV1,
    ) -> Result<()> {
        let domain = *universe.workload_domain_signature();
        if self.domain.is_some_and(|old| old != domain) {
            return Err(StructuredUnknown::WrongDomain);
        }
        let additional = universe
            .0
            .wire
            .algorithms
            .iter()
            .filter(|a| self.algorithms.binary_search(a).is_err())
            .count();
        let count = self
            .algorithms
            .len()
            .checked_add(additional)
            .ok_or(StructuredUnknown::Capacity)?;
        let axes = universe
            .0
            .wire
            .algorithms
            .iter()
            .filter(|a| self.algorithms.binary_search(a).is_err())
            .chain(self.algorithms.iter())
            .try_fold(0usize, |n, a| {
                n.checked_add(a.width()?).ok_or(StructuredUnknown::Capacity)
            })?;
        let peak = self
            .algorithms
            .capacity()
            .checked_add(count)
            .and_then(|n| n.checked_mul(std::mem::size_of::<AlgorithmAxisV1>()))
            .and_then(|n| {
                n.checked_add(
                    std::mem::size_of::<Self>()
                        + std::mem::size_of::<UniverseInner>()
                        + 2 * std::mem::size_of::<usize>(),
                )
            })
            .ok_or(StructuredUnknown::Capacity)?;
        if axes
            .checked_add(self.tail_basis)
            .is_none_or(|n| n > self.maximum_axes)
            || count
                .checked_mul(11)
                .and_then(|n| n.checked_add(self.tail_support))
                .is_none_or(|n| n > self.maximum_axes)
            || peak > self.maximum_retained_bytes
        {
            return Err(StructuredUnknown::Capacity);
        }
        self.algorithms
            .try_reserve_exact(additional)
            .map_err(|_| StructuredUnknown::Capacity)?;
        for a in &universe.0.wire.algorithms {
            if let Err(i) = self.algorithms.binary_search(a) {
                self.algorithms.insert(i, *a);
            }
        }
        self.domain = Some(domain);
        self.check_capacity()
    }
    pub fn is_empty(&self) -> bool {
        self.algorithms.is_empty()
    }
    pub fn observe(&mut self, input: &StructuredInputV2) -> Result<()> {
        input.numerical_family_key()?;
        if input.algorithm_universe.is_some() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        let domain = input
            .physical_domain
            .ok_or(StructuredUnknown::WrongDomain)?;
        if self.domain.is_some_and(|d| d != domain) {
            return Err(StructuredUnknown::WrongDomain);
        }
        self.domain = Some(domain);
        let local_axes = input.algorithm_axes.iter().try_fold(0usize, |n, a| {
            n.checked_add(a.width()?).ok_or(StructuredUnknown::Capacity)
        })?;
        self.tail_basis = self.tail_basis.max(
            input
                .basis
                .len()
                .checked_sub(local_axes)
                .ok_or(StructuredUnknown::InvalidInput)?,
        );
        self.tail_support = self.tail_support.max(
            input
                .support
                .len()
                .checked_sub(input.algorithm_axes.len() * 11)
                .ok_or(StructuredUnknown::InvalidInput)?,
        );
        for a in &input.algorithm_axes {
            if let Err(index) = self.algorithms.binary_search(a) {
                let count = self
                    .algorithms
                    .len()
                    .checked_add(1)
                    .ok_or(StructuredUnknown::Capacity)?;
                let numeric = self.algorithms.iter().try_fold(a.width()?, |n, a| {
                    n.checked_add(a.width()?).ok_or(StructuredUnknown::Capacity)
                })?;
                let retained = std::mem::size_of::<Self>()
                    .checked_add(std::mem::size_of::<UniverseInner>())
                    .and_then(|n| n.checked_add(2 * std::mem::size_of::<usize>()))
                    .and_then(|n| {
                        n.checked_add(count.checked_mul(std::mem::size_of::<AlgorithmAxisV1>())?)
                    })
                    .ok_or(StructuredUnknown::Capacity)?;
                if retained > self.maximum_retained_bytes
                    || numeric + self.tail_basis > self.maximum_axes
                    || count * 11 + self.tail_support > self.maximum_axes
                {
                    return Err(StructuredUnknown::Capacity);
                }
                self.algorithms
                    .try_reserve_exact(1)
                    .map_err(|_| StructuredUnknown::Capacity)?;
                self.algorithms.insert(index, *a);
            }
        }
        self.check_capacity()
    }
    fn check_capacity(&self) -> Result<()> {
        let axes = self.algorithms.iter().try_fold(0usize, |n, a| {
            n.checked_add(a.width()?).ok_or(StructuredUnknown::Capacity)
        })?;
        let retained = std::mem::size_of::<Self>()
            + std::mem::size_of::<UniverseInner>()
            + 2 * std::mem::size_of::<usize>()
            + self.algorithms.capacity() * std::mem::size_of::<AlgorithmAxisV1>();
        if retained > self.maximum_retained_bytes
            || axes + self.tail_basis > self.maximum_axes
            || self.algorithms.len() * 11 + self.tail_support > self.maximum_axes
        {
            return Err(StructuredUnknown::Capacity);
        }
        Ok(())
    }
    pub fn finish(self) -> Result<DeclaredAlgorithmUniverseV1> {
        self.check_capacity()?;
        UniverseWire {
            revision: 1,
            workload_domain: self.domain.ok_or(StructuredUnknown::MissingEvidence)?,
            algorithms: self.algorithms,
        }
        .try_into()
    }
}

impl StructuredInputV2 {
    pub fn algorithm_universe_signature(&self) -> Option<&[u8; 32]> {
        self.algorithm_universe
            .as_ref()
            .map(DeclaredAlgorithmUniverseV1::signature)
    }
    /// Only checked complete original work is aligned. Missing known primitives
    /// receive zero; a present primitive outside the declaration is an error.
    /// Original owner, exact domain, host rows and physical route stay intact.
    pub fn with_algorithm_universe(
        mut self,
        universe: &DeclaredAlgorithmUniverseV1,
    ) -> Result<Self> {
        self.numerical_family_key()?;
        if self.physical_domain.as_ref() != Some(universe.workload_domain_signature()) {
            return Err(StructuredUnknown::WrongDomain);
        }
        if let Some(current) = &self.algorithm_universe {
            return if current == universe {
                Ok(self)
            } else {
                Err(StructuredUnknown::WrongDomain)
            };
        }
        let local = self.algorithm_axes.iter().try_fold(0usize, |n, a| {
            n.checked_add(a.width()?).ok_or(StructuredUnknown::Capacity)
        })?;
        let old_basis_end = 1 + local;
        let old_support_end = self.algorithm_axes.len() * 11;
        let new_basis_end = 1 + universe.0.basis_axes;
        let new_support_end = universe.0.wire.algorithms.len() * 11;
        let basis_len = new_basis_end
            .checked_add(
                self.basis
                    .len()
                    .checked_sub(old_basis_end)
                    .ok_or(StructuredUnknown::InvalidInput)?,
            )
            .ok_or(StructuredUnknown::Capacity)?;
        let support_len = new_support_end
            .checked_add(
                self.support
                    .len()
                    .checked_sub(old_support_end)
                    .ok_or(StructuredUnknown::InvalidInput)?,
            )
            .ok_or(StructuredUnknown::Capacity)?;
        if basis_len > MAX_AXES || support_len > MAX_AXES {
            return Err(StructuredUnknown::Capacity);
        }
        // Verify subset before allocating any replacement arrays.
        if self
            .algorithm_axes
            .iter()
            .any(|a| universe.0.wire.algorithms.binary_search(a).is_err())
        {
            return Err(StructuredUnknown::WrongDomain);
        }
        let mut basis = Vec::new();
        basis
            .try_reserve_exact(basis_len)
            .map_err(|_| StructuredUnknown::Capacity)?;
        if basis.capacity() != basis_len {
            return Err(StructuredUnknown::Capacity);
        }
        basis.resize(new_basis_end, 0.);
        basis[0] = 1.;
        let mut support = Vec::new();
        support
            .try_reserve_exact(support_len)
            .map_err(|_| StructuredUnknown::Capacity)?;
        if support.capacity() != support_len {
            return Err(StructuredUnknown::Capacity);
        }
        support.resize(new_support_end, 0);
        let (mut source, mut source_basis, mut target_basis) = (0usize, 1usize, 1usize);
        for (target, a) in universe.0.wire.algorithms.iter().enumerate() {
            let width = a.width()?;
            if self.algorithm_axes.get(source) == Some(a) {
                basis[target_basis..target_basis + width]
                    .copy_from_slice(&self.basis[source_basis..source_basis + width]);
                support[target * 11..(target + 1) * 11]
                    .copy_from_slice(&self.support[source * 11..(source + 1) * 11]);
                source += 1;
                source_basis += width;
            }
            target_basis += width;
        }
        if source != self.algorithm_axes.len() {
            return Err(StructuredUnknown::WrongDomain);
        }
        basis.extend_from_slice(&self.basis[old_basis_end..]);
        support.extend_from_slice(&self.support[old_support_end..]);
        let basis_delta = new_basis_end - old_basis_end;
        let support_delta = new_support_end - old_support_end;
        self.pending_basis_offset += basis_delta;
        self.pending_support_offset += support_delta;
        if let Some(c) = &mut self.completion {
            c.basis_offset += basis_delta;
            c.support_offset += support_delta;
        }
        if let Some((b, s)) = &mut self.repetition_offsets {
            *b += basis_delta;
            *s += support_delta;
        }
        self.basis = basis;
        self.support = support;
        self.algorithm_universe = Some(universe.clone());
        Ok(self)
    }
}
impl StructuredQueryV2 {
    pub fn with_algorithm_universe(
        mut self,
        universe: &DeclaredAlgorithmUniverseV1,
    ) -> Result<Self> {
        self.input = self.input.with_algorithm_universe(universe)?;
        Ok(self)
    }
}
