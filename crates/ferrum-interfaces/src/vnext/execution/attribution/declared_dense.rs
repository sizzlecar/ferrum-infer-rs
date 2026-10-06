//! Explicit projection conversion is source attribution, never a claim that
//! quantized kernels execute the converted values.
use super::{invalid_attribution, is_quantized_values};
use crate::vnext::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct DeclaredDenseAttribution {
    basis: &'static str,
    source_quant_tensor_count: u64,
    retained_quantized_source_tensor_count: u64,
    source_quant_component_count: u64,
    source_quant_component_bytes: u64,
    dense_f16_component_count: u64,
    dense_f16_component_bytes: u64,
    mapping_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    rn_fragment_execution: Option<DeclaredFragmentAttribution>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct DeclaredFragmentAttribution {
    component_count: u64,
    component_bytes: u64,
    mapping_sha256: String,
}
impl DeclaredDenseAttribution {
    pub(super) fn has_fragment(&self) -> bool {
        self.rn_fragment_execution.is_some()
    }
    pub(super) fn witness_schema(&self) -> &'static str {
        if self.has_fragment() {
            "ferrum.vnext.provider-attribution.v3"
        } else {
            "ferrum.vnext.provider-attribution.v2"
        }
    }
}

#[derive(Default)]
pub(super) struct DenseAttributionBuilder {
    tensors: BTreeSet<String>,
    quantized_tensors: BTreeSet<String>,
    source_bytes: BTreeMap<WeightId, u64>,
    dense_bytes: BTreeMap<WeightId, u64>,
    mappings: BTreeSet<(WeightId, WeightId, String, u32)>,
    fragment_bytes: BTreeMap<WeightId, u64>,
    fragment_mappings: BTreeSet<(WeightId, WeightId, String, u32)>,
}

fn validate_binding(
    operation: &OperationId,
    version: ContractVersion,
    role: ResolvedValueRole,
    ordinal: u32,
    layout: &PhysicalWeightLayout,
    component: &ResolvedWeightComponentLayout,
) -> Result<(), VNextError> {
    let authorized_layout = match layout {
        PhysicalWeightLayout::Dense { component_id } => {
            gguf_f16_projection_role_v1(operation, ordinal).is_some()
                && component_id == component.component_id()
        }
        PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
            dense_values,
            fragment_values,
            ..
        } => {
            gguf_rn_f16_fragment_role_v1(operation, ordinal).is_some()
                && dense_values.component_id != fragment_values.component_id
                && &dense_values.component_id == component.component_id()
                && dense_values.storage == PhysicalStorageLayout::exact_contiguous()
                && fragment_values.storage == PhysicalStorageLayout::exact_contiguous()
        }
        _ => false,
    };
    if version != ContractVersion::new(1, 0)
        || role != ResolvedValueRole::Input
        || !authorized_layout
        || component.role() != WeightComponentRole::Values
        || component.encoding()
            != &(WeightEncoding::Dense {
                element_type: ElementType::F16,
            })
    {
        return Err(invalid_attribution(format!(
            "denominator source components map to undeclared dense execution values `{}` at `{operation}`/{role:?}/{ordinal}",
            component.component_id()
        )));
    }
    Ok(())
}

fn validate_approval(
    approval: Option<&ApproximateWeightQualityApprovalRecord>,
    source_schema: &str,
    execution_schema: &str,
) -> Result<(), VNextError> {
    let approval = approval.ok_or_else(|| {
        invalid_attribution(
        "declared dense F16 source conversion requires its validated approximate quality approval"
    )
    })?;
    if approval.source_schema_fingerprint() != source_schema
        || approval.execution_schema_fingerprint() != execution_schema
    {
        return Err(invalid_attribution(
            "declared dense F16 approval does not bind the live source and execution schemas",
        ));
    }
    Ok(())
}

fn validate_fragment_source_group(
    dense: &[WeightId],
    packet: &[WeightId],
    denominator: &[&WeightId],
) -> Result<(), VNextError> {
    if dense != packet
        || packet.is_empty()
        || packet.iter().collect::<BTreeSet<_>>().len() != packet.len()
        || packet.iter().ne(denominator.iter().copied())
    {
        return Err(invalid_attribution(
            "RN dual representations must retain the same ordered complete source group",
        ));
    }
    Ok(())
}

impl DenseAttributionBuilder {
    pub(super) fn record_quantized(&mut self, tensors: &BTreeSet<String>) {
        self.quantized_tensors.extend(tensors.iter().cloned());
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &mut self,
        family: &PreparedModelFamily,
        execution: &ExecutionWeightPlan,
        node: &PlanNode,
        value: &ResolvedValueBinding,
        weight: &ResolvedWeightBinding,
        component: &ResolvedWeightComponentLayout,
        source_ids: &[&WeightId],
        source_tensors: &BTreeMap<WeightId, BTreeSet<String>>,
    ) -> Result<(), VNextError> {
        validate_binding(
            node.operation_id(),
            node.operation_version(),
            value.role(),
            value.ordinal(),
            weight.physical_layout(),
            component,
        )?;
        let source_fingerprint = family.weight_schema().fingerprint()?;
        let execution_fingerprint = execution.schema().fingerprint()?;
        if execution.source_schema_fingerprint() != source_fingerprint {
            return Err(invalid_attribution(
                "dense conversion source schema changed",
            ));
        }
        validate_approval(
            execution.approximate_quality_approval(),
            &source_fingerprint,
            &execution_fingerprint,
        )?;
        let actual = execution
            .schema()
            .components
            .iter()
            .find(|c| &c.id == component.component_id())
            .ok_or_else(|| {
                invalid_attribution("dense execution component is absent from approved schema")
            })?;
        if actual.role != component.role()
            || &actual.encoding != component.encoding()
            || actual.dimensions != component.physical_dimensions()
        {
            return Err(invalid_attribution(
                "dense component differs from approved execution schema",
            ));
        }
        for source_id in source_ids {
            let source = family
                .weight_schema()
                .components
                .iter()
                .find(|c| &c.id == *source_id)
                .ok_or_else(|| invalid_attribution("dense provenance source is absent"))?;
            if !is_quantized_values(source.role, &source.encoding) {
                return Err(invalid_attribution(
                    "dense denominator provenance is not quantized",
                ));
            }
            self.source_bytes
                .insert(source.id.clone(), source.physical_bytes()?);
            self.tensors
                .extend(source_tensors[*source_id].iter().cloned());
            self.mappings.insert((
                source.id.clone(),
                component.component_id().clone(),
                node.operation_id().to_string(),
                value.ordinal(),
            ));
        }
        self.dense_bytes.insert(
            component.component_id().clone(),
            component.physical_bytes()?,
        );
        Ok(())
    }

    /// The packet is derived storage for the same converted source, not a
    /// second source tensor or proof that a particular wave used this route.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_fragment(
        &mut self,
        family: &PreparedModelFamily,
        execution: &ExecutionWeightPlan,
        node: &PlanNode,
        value: &ResolvedValueBinding,
        weight: &ResolvedWeightBinding,
        component: &ResolvedWeightComponentLayout,
        source_ids: &[&WeightId],
        source_tensors: &BTreeMap<WeightId, BTreeSet<String>>,
    ) -> Result<(), VNextError> {
        let PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
            dense_values,
            fragment_values,
            source_format,
        } = weight.physical_layout()
        else {
            return Err(invalid_attribution(
                "fragment attribution requires typed dual layout",
            ));
        };
        if node.operation_version() != ContractVersion::new(1, 0)
            || value.role() != ResolvedValueRole::Input
            || gguf_rn_f16_fragment_role_v1(node.operation_id(), value.ordinal()).is_none()
            || &fragment_values.component_id != component.component_id()
        {
            return Err(invalid_attribution(
                "fragment attribution has no exact authorized weight slot",
            ));
        }
        weight.validate_logical(value.tensor().dimensions(), value.tensor().element_type())?;
        let plan =
            RnF16FragmentPlanV1::from_dimensions(*source_format, value.tensor().dimensions())?;
        let source_fingerprint = family.weight_schema().fingerprint()?;
        if execution.source_schema_fingerprint() != source_fingerprint {
            return Err(invalid_attribution("fragment source schema changed"));
        }
        validate_approval(
            execution.approximate_quality_approval(),
            &source_fingerprint,
            &execution.schema().fingerprint()?,
        )?;
        let actual = execution
            .schema()
            .components
            .iter()
            .find(|c| &c.id == component.component_id())
            .ok_or_else(|| invalid_attribution("fragment absent from approved execution schema"))?;
        if actual.role != component.role()
            || &actual.encoding != component.encoding()
            || actual.dimensions != component.physical_dimensions()
            || component.encoding() != &plan.packed_encoding()
        {
            return Err(invalid_attribution(
                "fragment differs from approved typed representation",
            ));
        }
        let dense_sources = execution
            .component_sources()
            .get(&dense_values.component_id)
            .ok_or_else(|| invalid_attribution("dual dense source group is absent"))?;
        let packet_sources = execution
            .component_sources()
            .get(&fragment_values.component_id)
            .ok_or_else(|| invalid_attribution("dual fragment source group is absent"))?;
        validate_fragment_source_group(dense_sources, packet_sources, source_ids)?;
        let mut rows = 0_u64;
        for source_id in source_ids {
            let source = family
                .weight_schema()
                .components
                .iter()
                .find(|c| &c.id == *source_id)
                .ok_or_else(|| invalid_attribution("fragment provenance source absent"))?;
            if source.role != WeightComponentRole::PackedValues
                || source.encoding != WeightEncoding::BlockQuantized(plan.source_block_spec())
                || !(2..=3).contains(&source.dimensions.len())
                || source.dimensions.last().copied() != Some(plan.k() / 256)
            {
                return Err(invalid_attribution(
                    "fragment source shape or encoding differs from typed plan",
                ));
            }
            rows = source.dimensions[..source.dimensions.len() - 1]
                .iter()
                .try_fold(1_u64, |n, d| n.checked_mul(*d))
                .and_then(|n| rows.checked_add(n))
                .ok_or_else(|| invalid_attribution("fragment source rows overflow"))?;
            self.source_bytes
                .insert(source.id.clone(), source.physical_bytes()?);
            self.tensors
                .extend(source_tensors[*source_id].iter().cloned());
            self.fragment_mappings.insert((
                source.id.clone(),
                component.component_id().clone(),
                node.operation_id().to_string(),
                value.ordinal(),
            ));
        }
        if rows != plan.n() {
            return Err(invalid_attribution(
                "fragment source group does not cover whole logical N",
            ));
        }
        self.fragment_bytes.insert(
            component.component_id().clone(),
            component.physical_bytes()?,
        );
        Ok(())
    }

    pub(super) fn finish(
        self,
        all_tensors: &BTreeSet<String>,
    ) -> Result<Option<DeclaredDenseAttribution>, VNextError> {
        if self.tensors.is_empty() {
            return Ok(None);
        }
        if !self.tensors.is_subset(all_tensors)
            || !self.tensors.is_disjoint(&self.quantized_tensors)
            || self
                .tensors
                .union(&self.quantized_tensors)
                .ne(all_tensors.iter())
        {
            return Err(invalid_attribution(
                "declared conversion does not partition the original source denominator",
            ));
        }
        let count = |n: usize| {
            u64::try_from(n).map_err(|_| invalid_attribution("inventory count exceeds u64"))
        };
        let bytes = |map: &BTreeMap<WeightId, u64>| {
            map.values().try_fold(0_u64, |sum, n| {
                sum.checked_add(*n)
                    .ok_or_else(|| invalid_attribution("inventory bytes exceed u64"))
            })
        };
        Ok(Some(DeclaredDenseAttribution {
            basis: "unique_mapped_component_inventory_bytes_excluding_placement_alignment_and_peak_memory",
            source_quant_tensor_count: count(self.tensors.len())?,
            retained_quantized_source_tensor_count: count(all_tensors.difference(&self.tensors).count())?,
            source_quant_component_count: count(self.source_bytes.len())?,
            source_quant_component_bytes: bytes(&self.source_bytes)?,
            dense_f16_component_count: count(self.dense_bytes.len())?,
            dense_f16_component_bytes: bytes(&self.dense_bytes)?,
            mapping_sha256: super::canonical_fingerprint(&self.mappings, "fingerprint declared dense F16 mappings")?,
            rn_fragment_execution: if self.fragment_bytes.is_empty() { None } else { Some(DeclaredFragmentAttribution {
                component_count: count(self.fragment_bytes.len())?,
                component_bytes: bytes(&self.fragment_bytes)?,
                mapping_sha256: super::canonical_fingerprint(&self.fragment_mappings,"fingerprint declared RN fragment mappings")?,
            }) },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_dense_attribution_requires_exact_typed_slot_and_f16_layout() {
        let id = WeightId::new("fixture.execution.component").unwrap();
        let component = ResolvedWeightComponentLayout::from_parts(
            id.clone(),
            WeightComponentRole::Values,
            vec![4, 256],
            WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
        );
        let layout = PhysicalWeightLayout::Dense {
            component_id: id.clone(),
        };
        let op = OperationId::new(DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID).unwrap();
        // FFN gate/up is input one; activation and output have no weight permission.
        assert!(gguf_f16_projection_role_v1(&op, 1).is_some());
        validate_binding(
            &op,
            ContractVersion::new(1, 0),
            ResolvedValueRole::Input,
            1,
            &layout,
            &component,
        )
        .unwrap();
        for (operation, role, ordinal) in [
            (
                OperationId::new(DENSE_SWIGLU_OPERATION_ID).unwrap(),
                ResolvedValueRole::Input,
                1,
            ),
            (op.clone(), ResolvedValueRole::Output, 1),
            (op.clone(), ResolvedValueRole::Input, 0),
        ] {
            assert!(validate_binding(
                &operation,
                ContractVersion::new(1, 0),
                role,
                ordinal,
                &layout,
                &component
            )
            .is_err());
        }
        let f32 = ResolvedWeightComponentLayout::from_parts(
            id.clone(),
            WeightComponentRole::Values,
            vec![4, 256],
            WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
        );
        assert!(validate_binding(
            &op,
            ContractVersion::new(1, 0),
            ResolvedValueRole::Input,
            1,
            &layout,
            &f32
        )
        .is_err());
        let unrelated = PhysicalWeightLayout::Dense {
            component_id: WeightId::new("fixture.other").unwrap(),
        };
        assert!(validate_binding(
            &op,
            ContractVersion::new(1, 0),
            ResolvedValueRole::Input,
            1,
            &unrelated,
            &component
        )
        .is_err());
        assert!(validate_approval(None, "source", "execution").is_err());
    }

    #[test]
    fn declared_dense_attribution_inventory_deduplicates_fused_sources_and_checks_overflow() {
        let source = WeightId::new("fixture.source").unwrap();
        let dense = WeightId::new("fixture.execution").unwrap();
        let mut b = DenseAttributionBuilder::default();
        b.tensors.insert("gate.weight".into());
        b.quantized_tensors.insert("head.weight".into());
        b.source_bytes.insert(source.clone(), 144);
        b.source_bytes.insert(source, 144);
        b.dense_bytes.insert(dense, 512);
        let report = b
            .finish(&BTreeSet::from([
                "gate.weight".into(),
                "head.weight".into(),
            ]))
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                report.source_quant_tensor_count,
                report.retained_quantized_source_tensor_count
            ),
            (1, 1)
        );
        assert_eq!(
            (
                report.source_quant_component_bytes,
                report.dense_f16_component_bytes
            ),
            (144, 512)
        );
        let mut b = DenseAttributionBuilder::default();
        b.tensors.insert("gate.weight".into());
        b.source_bytes
            .insert(WeightId::new("fixture.a").unwrap(), u64::MAX);
        b.source_bytes
            .insert(WeightId::new("fixture.b").unwrap(), 1);
        assert!(b.finish(&BTreeSet::from(["gate.weight".into()])).is_err());
    }
}

#[cfg(test)]
mod rn_fragment_tests {
    use super::*;
    #[test]
    fn rn_fragment_attribution_requires_dual_slot_and_complete_ordered_sources() {
        let dense = WeightId::new("fixture.dense").unwrap();
        let packet = WeightId::new("fixture.packet").unwrap();
        let layout = PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
            dense_values: PhysicalWeightComponentBinding::exact_contiguous(dense.clone()),
            fragment_values: PhysicalWeightComponentBinding::exact_contiguous(packet),
            source_format: RnF16FragmentSourceFormatV1::Q4K,
        };
        let component = ResolvedWeightComponentLayout::from_parts(
            dense.clone(),
            WeightComponentRole::Values,
            vec![2, 9, 256],
            WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
        );
        let op = OperationId::new(DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID).unwrap();
        for ordinal in [1, 2] {
            validate_binding(
                &op,
                ContractVersion::new(1, 0),
                ResolvedValueRole::Input,
                ordinal,
                &layout,
                &component,
            )
            .unwrap();
        }
        for (operation, version, role, ordinal) in [
            (
                OperationId::new(DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID).unwrap(),
                ContractVersion::new(1, 0),
                ResolvedValueRole::Input,
                1,
            ),
            (
                op.clone(),
                ContractVersion::new(2, 0),
                ResolvedValueRole::Input,
                1,
            ),
            (
                op.clone(),
                ContractVersion::new(1, 0),
                ResolvedValueRole::Output,
                1,
            ),
            (
                op.clone(),
                ContractVersion::new(1, 0),
                ResolvedValueRole::Input,
                0,
            ),
        ] {
            assert!(
                validate_binding(&operation, version, role, ordinal, &layout, &component).is_err()
            );
        }
        assert!(validate_binding(
            &op,
            ContractVersion::new(1, 0),
            ResolvedValueRole::Input,
            1,
            &PhysicalWeightLayout::Dense {
                component_id: dense
            },
            &component
        )
        .is_err());
        let a = WeightId::new("source.gate").unwrap();
        let b = WeightId::new("source.up").unwrap();
        let ordered = [a.clone(), b.clone()];
        validate_fragment_source_group(&ordered, &ordered, &[&a, &b]).unwrap();
        assert!(
            validate_fragment_source_group(&ordered, &[b.clone(), a.clone()], &[&b, &a]).is_err()
        );
        assert!(validate_fragment_source_group(&ordered, &ordered, &[&a]).is_err());
        assert!(validate_fragment_source_group(
            &[a.clone(), a.clone()],
            &[a.clone(), a.clone()],
            &[&a, &a]
        )
        .is_err());
        assert!(validate_fragment_source_group(&[], &[], &[]).is_err());
    }
    fn inventory(packet: bool) -> DeclaredDenseAttribution {
        let mut b = DenseAttributionBuilder::default();
        b.tensors.extend(["gate.weight".into(), "up.weight".into()]);
        b.quantized_tensors.insert("head.weight".into());
        b.source_bytes
            .insert(WeightId::new("source.gate").unwrap(), 9 * 144);
        b.source_bytes
            .insert(WeightId::new("source.up").unwrap(), 9 * 144);
        b.dense_bytes
            .insert(WeightId::new("execution.dense").unwrap(), 18 * 256 * 2);
        if packet {
            // Reusing the source by the second representation does not add to its denominator.
            b.source_bytes
                .insert(WeightId::new("source.gate").unwrap(), 9 * 144);
            b.tensors.insert("gate.weight".into());
            b.fragment_bytes
                .insert(WeightId::new("execution.packet").unwrap(), 2 * 8 * 384);
        }
        b.finish(&BTreeSet::from([
            "gate.weight".into(),
            "up.weight".into(),
            "head.weight".into(),
        ]))
        .unwrap()
        .unwrap()
    }
    #[test]
    fn rn_fragment_attribution_v3_counts_storage_without_recounting_source_and_keeps_v2_wire() {
        let dense = inventory(false);
        let dual = inventory(true);
        assert_eq!(
            dense.witness_schema(),
            "ferrum.vnext.provider-attribution.v2"
        );
        assert_eq!(
            dual.witness_schema(),
            "ferrum.vnext.provider-attribution.v3"
        );
        let before = serde_json::to_value(&dense).unwrap();
        assert!(before.get("rn_fragment_execution").is_none());
        let mut after = serde_json::to_value(&dual).unwrap();
        let fragment = after
            .as_object_mut()
            .unwrap()
            .remove("rn_fragment_execution")
            .unwrap();
        assert_eq!(
            after, before,
            "all existing v2 fields keep their meaning and bytes"
        );
        assert_eq!(fragment["component_count"], 1);
        assert_eq!(fragment["component_bytes"], 6144);
        assert_eq!(fragment["mapping_sha256"].as_str().unwrap().len(), 64);
        assert_eq!(dual.source_quant_tensor_count, 2);
        assert_eq!(dual.retained_quantized_source_tensor_count, 1);
        assert_eq!(dual.source_quant_component_count, 2);
        assert_eq!(dual.dense_f16_component_bytes, 9216);
        assert_eq!(dual.source_quant_component_bytes, 2592);
    }
}
