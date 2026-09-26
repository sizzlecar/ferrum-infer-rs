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
}

#[derive(Default)]
pub(super) struct DenseAttributionBuilder {
    tensors: BTreeSet<String>,
    quantized_tensors: BTreeSet<String>,
    source_bytes: BTreeMap<WeightId, u64>,
    dense_bytes: BTreeMap<WeightId, u64>,
    mappings: BTreeSet<(WeightId, WeightId, String, u32)>,
}

fn validate_binding(
    operation: &OperationId,
    version: ContractVersion,
    role: ResolvedValueRole,
    ordinal: u32,
    layout: &PhysicalWeightLayout,
    component: &ResolvedWeightComponentLayout,
) -> Result<(), VNextError> {
    if version != ContractVersion::new(1, 0)
        || role != ResolvedValueRole::Input
        || gguf_f16_projection_role_v1(operation, ordinal).is_none()
        || component.role() != WeightComponentRole::Values
        || component.encoding()
            != &(WeightEncoding::Dense {
                element_type: ElementType::F16,
            })
        || !matches!(layout, PhysicalWeightLayout::Dense { component_id }
            if component_id == component.component_id())
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
