use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

use crate::vnext::{
    PreparedModelFamily, ResolvedModelPlan, StaticInitializationReceipt, VNextError,
    WeightComponentRole, WeightEncoding, WeightId, IDENTITY_WEIGHT_MATERIALIZER_ID,
};

use super::foundation::canonical_fingerprint;

pub const PROVIDER_ATTRIBUTION_WITNESS_SCHEMA: &str = "ferrum.vnext.provider-attribution.v1";
pub const PROVIDER_ATTRIBUTION_STATIC_BASIS: &str =
    "resolved_plan_and_completed_static_initialization";
pub const PROVIDER_ATTRIBUTION_STATIC_FALLBACK_BASIS: &str =
    "fail_closed_quantized_execution_component_and_vnext_plan_binding";

fn invalid_attribution(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("provider attribution: {}", reason.into()),
    }
}

fn is_quantized_values(role: WeightComponentRole, encoding: &WeightEncoding) -> bool {
    role == WeightComponentRole::PackedValues
        && matches!(
            encoding,
            WeightEncoding::Quantized(_) | WeightEncoding::BlockQuantized(_)
        )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ProviderAttributionDenominatorMaterial<'a> {
    operations: &'a BTreeSet<String>,
    quant_tensors: &'a BTreeSet<String>,
}

/// Canonical M0-compatible denominator derived from the prepared semantic
/// family. Tensor names remain private; the public witness exposes only this
/// commitment and its exact tensor/operation cardinalities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedProviderAttributionDenominator {
    quant_tensors: BTreeSet<String>,
    operations: BTreeSet<String>,
    tensor_operations: BTreeMap<String, String>,
    source_component_tensors: BTreeMap<WeightId, BTreeSet<String>>,
    source_quantization_format_ids: BTreeSet<String>,
    sha256: String,
}

impl QuantizedProviderAttributionDenominator {
    pub fn from_prepared_family(family: &PreparedModelFamily) -> Result<Option<Self>, VNextError> {
        let schema = family.weight_schema();
        let weights_by_value = family
            .program()
            .weights()
            .iter()
            .map(|weight| (&weight.value_id, &weight.weight_id))
            .collect::<BTreeMap<_, _>>();
        let components_by_id = schema
            .components
            .iter()
            .map(|component| (&component.id, component))
            .collect::<BTreeMap<_, _>>();

        let mut tensor_operation_sets = BTreeMap::<String, BTreeSet<String>>::new();
        let mut source_component_tensors = BTreeMap::<WeightId, BTreeSet<String>>::new();
        let mut source_quantization_format_ids = BTreeSet::new();
        for node in family
            .program()
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
        {
            for input in &node.inputs {
                let Some(weight_id) = weights_by_value.get(input) else {
                    continue;
                };
                for component in schema.physical_component_refs(weight_id)? {
                    if !is_quantized_values(component.role, &component.encoding) {
                        continue;
                    }
                    let format_id = match &component.encoding {
                        WeightEncoding::Quantized(spec) => spec.format_id.as_str(),
                        WeightEncoding::BlockQuantized(spec) => spec.format_id.as_str(),
                        WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. } => {
                            unreachable!("filtered quantized source component")
                        }
                    };
                    source_quantization_format_ids.insert(format_id.to_owned());
                    let names = source_component_tensors
                        .entry(component.id.clone())
                        .or_default();
                    for external_name in &component.external_names {
                        names.insert(external_name.clone());
                        tensor_operation_sets
                            .entry(external_name.clone())
                            .or_default()
                            .insert(node.operation_id.to_string());
                    }
                }
            }
        }
        if tensor_operation_sets.is_empty() {
            return Ok(None);
        }

        let referenced_source_components = source_component_tensors.keys().collect::<BTreeSet<_>>();
        let unowned_quantized_component = components_by_id.values().find(|component| {
            component.required
                && is_quantized_values(component.role, &component.encoding)
                && !referenced_source_components.contains(&component.id)
        });
        if let Some(component) = unowned_quantized_component {
            return Err(invalid_attribution(format!(
                "required quantized source component `{}` has no semantic operation owner",
                component.id
            )));
        }

        let mut tensor_operations = BTreeMap::new();
        for (tensor, operations) in tensor_operation_sets {
            if operations.len() != 1 {
                return Err(invalid_attribution(format!(
                    "quantized source tensor `{tensor}` does not have exactly one operation owner"
                )));
            }
            tensor_operations.insert(
                tensor,
                operations
                    .into_iter()
                    .next()
                    .expect("one checked operation owner"),
            );
        }
        let quant_tensors = tensor_operations.keys().cloned().collect::<BTreeSet<_>>();
        let operations = tensor_operations.values().cloned().collect::<BTreeSet<_>>();
        let sha256 = canonical_fingerprint(
            &ProviderAttributionDenominatorMaterial {
                operations: &operations,
                quant_tensors: &quant_tensors,
            },
            "fingerprint quantized provider attribution denominator",
        )?;
        Ok(Some(Self {
            quant_tensors,
            operations,
            tensor_operations,
            source_component_tensors,
            source_quantization_format_ids,
            sha256,
        }))
    }

    pub fn quant_tensor_count(&self) -> usize {
        self.quant_tensors.len()
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub fn item_count(&self) -> usize {
        self.quant_tensor_count() + self.operation_count()
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn source_quantization_format_ids(&self) -> &BTreeSet<String> {
        &self.source_quantization_format_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct ProviderMappingRow {
    quant_tensor: String,
    node_id: String,
    operation_id: String,
    provider_id: String,
    provider_implementation_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationProviderAttributionBinding {
    operation_id: String,
    provider_id: String,
    provider_implementation_fingerprint: String,
    quant_tensor_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderAttributionCounts {
    expected_quant_tensor_count: u64,
    attributed_quant_tensor_count: u64,
    expected_operation_count: u64,
    attributed_operation_count: u64,
    expected_item_count: u64,
    attributed_item_count: u64,
    percent: f64,
    denominator_sha256: String,
}

impl ProviderAttributionCounts {
    pub const fn expected_quant_tensor_count(&self) -> u64 {
        self.expected_quant_tensor_count
    }

    pub const fn attributed_quant_tensor_count(&self) -> u64 {
        self.attributed_quant_tensor_count
    }

    pub const fn expected_operation_count(&self) -> u64 {
        self.expected_operation_count
    }

    pub const fn attributed_operation_count(&self) -> u64 {
        self.attributed_operation_count
    }

    pub const fn expected_item_count(&self) -> u64 {
        self.expected_item_count
    }

    pub const fn attributed_item_count(&self) -> u64 {
        self.attributed_item_count
    }

    pub const fn percent(&self) -> f64 {
        self.percent
    }

    pub fn denominator_sha256(&self) -> &str {
        &self.denominator_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderAttributionFallbackCounts {
    silent: u64,
    dense: u64,
    legacy: u64,
}

impl ProviderAttributionFallbackCounts {
    pub const fn silent(&self) -> u64 {
        self.silent
    }

    pub const fn dense(&self) -> u64 {
        self.dense
    }

    pub const fn legacy(&self) -> u64 {
        self.legacy
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderAttributionBinding {
    prepared_family_fingerprint: String,
    source_schema_fingerprint: String,
    source_weight_format_id: String,
    source_weight_layout_id: String,
    source_quantization_format_ids: BTreeSet<String>,
    execution_weight_plan_fingerprint: String,
    execution_schema_fingerprint: String,
    execution_weight_format_id: String,
    execution_weight_layout_id: String,
    execution_quantization_format_ids: BTreeSet<String>,
    materializer_id: String,
    materializer_version: super::ContractVersion,
    materializer_implementation_fingerprint: String,
    execution_contract_fingerprint: Option<String>,
    quality_vector_digest: Option<String>,
    quality_artifact_sha256: Option<String>,
    resolved_plan_fingerprint: String,
    plan_id: String,
    plan_hash: String,
    operation_provider_bindings: Vec<OperationProviderAttributionBinding>,
    provider_mapping_sha256: String,
    static_initialized_resource_count: u64,
    static_uploaded_component_count: u64,
    static_imported_component_count: u64,
}

impl ProviderAttributionBinding {
    pub fn execution_contract_fingerprint(&self) -> Option<&str> {
        self.execution_contract_fingerprint.as_deref()
    }

    pub fn quality_vector_digest(&self) -> Option<&str> {
        self.quality_vector_digest.as_deref()
    }

    pub fn plan_hash(&self) -> &str {
        &self.plan_hash
    }

    pub fn provider_mapping_sha256(&self) -> &str {
        &self.provider_mapping_sha256
    }
}

/// Compact, fail-closed witness for quantized provider attribution.
///
/// This first version is intentionally plan-static: construction requires the
/// receipt that only exists after atomic static initialization completes. It
/// proves every M0 denominator tensor is mapped through a quantized execution
/// component to one selected vNext plan node/provider. It does not claim
/// per-request normal-retirement coverage.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StaticProviderAttributionWitness {
    schema: &'static str,
    attribution_basis: &'static str,
    fallback_basis: &'static str,
    provider_attribution: ProviderAttributionCounts,
    fallback_counts: ProviderAttributionFallbackCounts,
    binding: ProviderAttributionBinding,
}

impl StaticProviderAttributionWitness {
    pub fn from_completed_static_initialization(
        resolved: &ResolvedModelPlan,
        receipt: &StaticInitializationReceipt,
    ) -> Result<Option<Self>, VNextError> {
        let family = &resolved.parts().prepared_family;
        let plan = resolved.execution_plan();
        let payload = plan.payload();
        let execution_weights = payload.execution_weights();
        // The v1 witness assumes one operation owner per source tensor. The
        // identity path can legally reuse a tied quantized embedding for both
        // token embedding and logits, so this optional witness must not block
        // startup on that path. Other exact materializers still emit it.
        if execution_weights.materializer_id().as_str() == IDENTITY_WEIGHT_MATERIALIZER_ID
            && execution_weights.approximate_quality_approval().is_none()
        {
            return Ok(None);
        }
        let Some(denominator) =
            QuantizedProviderAttributionDenominator::from_prepared_family(family)?
        else {
            return Ok(None);
        };
        let source_components = family
            .weight_schema()
            .components
            .iter()
            .map(|component| (&component.id, component))
            .collect::<BTreeMap<_, _>>();
        let mut mapping_rows = BTreeSet::new();
        let mut attributed_tensors = BTreeSet::new();
        let mut attributed_operations = BTreeSet::new();

        for node in payload.nodes() {
            for value in node.values() {
                let Some(weight) = value.weight() else {
                    continue;
                };
                for execution_component in weight.components() {
                    let source_ids = execution_weights
                        .component_sources()
                        .get(execution_component.component_id())
                        .ok_or_else(|| {
                            invalid_attribution(format!(
                                "execution component `{}` has no source provenance",
                                execution_component.component_id()
                            ))
                        })?;
                    let denominator_source_ids = source_ids
                        .iter()
                        .filter(|source_id| {
                            denominator
                                .source_component_tensors
                                .contains_key(*source_id)
                        })
                        .collect::<Vec<_>>();
                    if denominator_source_ids.is_empty() {
                        continue;
                    }
                    if !is_quantized_values(
                        execution_component.role(),
                        execution_component.encoding(),
                    ) {
                        if matches!(
                            execution_component.role(),
                            WeightComponentRole::Values | WeightComponentRole::PackedValues
                        ) {
                            return Err(invalid_attribution(format!(
                                "denominator source components map to dense execution values `{}`",
                                execution_component.component_id()
                            )));
                        }
                        continue;
                    }
                    for source_id in denominator_source_ids {
                        let Some(tensors) = denominator.source_component_tensors.get(source_id)
                        else {
                            unreachable!("filtered denominator source component")
                        };
                        let source_component =
                            source_components.get(source_id).ok_or_else(|| {
                                invalid_attribution(format!(
                                "source component `{source_id}` is absent from the prepared family"
                            ))
                            })?;
                        if !is_quantized_values(source_component.role, &source_component.encoding) {
                            return Err(invalid_attribution(format!(
                                "quantized execution component `{}` maps a denominator tensor through non-quantized source `{source_id}`",
                                execution_component.component_id()
                            )));
                        }
                        for tensor in tensors {
                            let expected_operation = denominator
                                .tensor_operations
                                .get(tensor)
                                .expect("denominator tensor has one operation owner");
                            if node.operation_id().as_str() != expected_operation {
                                return Err(invalid_attribution(format!(
                                    "quantized source tensor `{tensor}` is bound to operation `{}` instead of `{expected_operation}`",
                                    node.operation_id()
                                )));
                            }
                            attributed_tensors.insert(tensor.clone());
                            attributed_operations.insert(expected_operation.clone());
                            mapping_rows.insert(ProviderMappingRow {
                                quant_tensor: tensor.clone(),
                                node_id: node.id().to_string(),
                                operation_id: expected_operation.clone(),
                                provider_id: node.selection().selected_provider().to_string(),
                                provider_implementation_fingerprint: node
                                    .provider_implementation_fingerprint()
                                    .to_owned(),
                            });
                        }
                    }
                }
            }
        }
        if attributed_tensors != denominator.quant_tensors
            || attributed_operations != denominator.operations
        {
            return Err(invalid_attribution(format!(
                "resolved quantized provider mapping covers {}/{} tensors and {}/{} operations",
                attributed_tensors.len(),
                denominator.quant_tensors.len(),
                attributed_operations.len(),
                denominator.operations.len()
            )));
        }
        if mapping_rows.len() != denominator.quant_tensors.len() {
            return Err(invalid_attribution(format!(
                "resolved provider mapping has {} rows for {} denominator tensors",
                mapping_rows.len(),
                denominator.quant_tensors.len()
            )));
        }

        let provider_mapping_sha256 =
            canonical_fingerprint(&mapping_rows, "fingerprint quantized provider mapping")?;
        let mut operation_provider_tensors =
            BTreeMap::<(String, String, String), BTreeSet<String>>::new();
        for row in &mapping_rows {
            operation_provider_tensors
                .entry((
                    row.operation_id.clone(),
                    row.provider_id.clone(),
                    row.provider_implementation_fingerprint.clone(),
                ))
                .or_default()
                .insert(row.quant_tensor.clone());
        }
        let operation_provider_bindings = operation_provider_tensors
            .into_iter()
            .map(
                |((operation_id, provider_id, provider_implementation_fingerprint), tensors)| {
                    Ok(OperationProviderAttributionBinding {
                        operation_id,
                        provider_id,
                        provider_implementation_fingerprint,
                        quant_tensor_count: u64::try_from(tensors.len()).map_err(|_| {
                            invalid_attribution("operation quant tensor count exceeds u64")
                        })?,
                    })
                },
            )
            .collect::<Result<Vec<_>, VNextError>>()?;
        if operation_provider_bindings.len() != denominator.operations.len() {
            return Err(invalid_attribution(format!(
                "{} denominator operations resolve to {} provider bindings",
                denominator.operations.len(),
                operation_provider_bindings.len()
            )));
        }
        let family_fingerprint = family.fingerprint()?;
        let source_schema_fingerprint = family.weight_schema().fingerprint()?;
        if execution_weights.source_schema_fingerprint() != source_schema_fingerprint {
            return Err(invalid_attribution(
                "execution weight plan source schema differs from the live prepared family",
            ));
        }
        let approval = execution_weights.approximate_quality_approval();
        let expected_quant_tensor_count = u64::try_from(denominator.quant_tensors.len())
            .map_err(|_| invalid_attribution("quant tensor count exceeds u64"))?;
        let expected_operation_count = u64::try_from(denominator.operations.len())
            .map_err(|_| invalid_attribution("operation count exceeds u64"))?;
        let expected_item_count = expected_quant_tensor_count
            .checked_add(expected_operation_count)
            .ok_or_else(|| invalid_attribution("denominator item count exceeds u64"))?;
        let provider_attribution = ProviderAttributionCounts {
            expected_quant_tensor_count,
            attributed_quant_tensor_count: expected_quant_tensor_count,
            expected_operation_count,
            attributed_operation_count: expected_operation_count,
            expected_item_count,
            attributed_item_count: expected_item_count,
            percent: 100.0,
            denominator_sha256: denominator.sha256,
        };
        let binding = ProviderAttributionBinding {
            prepared_family_fingerprint: family_fingerprint,
            source_schema_fingerprint,
            source_weight_format_id: family.weight_schema().format_id.to_string(),
            source_weight_layout_id: family.weight_schema().layout_id.to_string(),
            source_quantization_format_ids: denominator.source_quantization_format_ids,
            execution_weight_plan_fingerprint: execution_weights.fingerprint()?,
            execution_schema_fingerprint: execution_weights.schema().fingerprint()?,
            execution_weight_format_id: execution_weights.schema().format_id.to_string(),
            execution_weight_layout_id: execution_weights.schema().layout_id.to_string(),
            execution_quantization_format_ids: execution_weights
                .schema()
                .quantization_formats()
                .into_iter()
                .map(|format| format.to_string())
                .collect(),
            materializer_id: execution_weights.materializer_id().to_string(),
            materializer_version: execution_weights.materializer_version(),
            materializer_implementation_fingerprint: execution_weights
                .materializer_implementation_fingerprint()
                .to_owned(),
            execution_contract_fingerprint: approval
                .map(|approval| approval.execution_contract_fingerprint().to_owned()),
            quality_vector_digest: approval
                .map(|approval| approval.quality_vector_digest().to_owned()),
            quality_artifact_sha256: approval.map(|approval| approval.artifact_sha256().to_owned()),
            resolved_plan_fingerprint: resolved.fingerprint().to_owned(),
            plan_id: payload.plan_id().to_string(),
            plan_hash: plan.plan_hash().to_string(),
            operation_provider_bindings,
            provider_mapping_sha256,
            static_initialized_resource_count: u64::try_from(receipt.initialized_resource_count())
                .map_err(|_| invalid_attribution("initialized resource count exceeds u64"))?,
            static_uploaded_component_count: u64::try_from(receipt.uploaded_component_count())
                .map_err(|_| invalid_attribution("uploaded component count exceeds u64"))?,
            static_imported_component_count: u64::try_from(receipt.imported_component_count())
                .map_err(|_| invalid_attribution("imported component count exceeds u64"))?,
        };
        Ok(Some(Self {
            schema: PROVIDER_ATTRIBUTION_WITNESS_SCHEMA,
            attribution_basis: PROVIDER_ATTRIBUTION_STATIC_BASIS,
            fallback_basis: PROVIDER_ATTRIBUTION_STATIC_FALLBACK_BASIS,
            provider_attribution,
            fallback_counts: ProviderAttributionFallbackCounts {
                silent: 0,
                dense: 0,
                legacy: 0,
            },
            binding,
        }))
    }

    pub fn provider_attribution(&self) -> &ProviderAttributionCounts {
        &self.provider_attribution
    }

    pub fn fallback_counts(&self) -> &ProviderAttributionFallbackCounts {
        &self.fallback_counts
    }

    pub fn binding(&self) -> &ProviderAttributionBinding {
        &self.binding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn denominator_fingerprint_matches_the_release_canonical_json_contract() {
        let operations = BTreeSet::from([
            "operation.causal_paged_attention".to_owned(),
            "operation.dense_swiglu".to_owned(),
            "operation.gated_delta_recurrent_attention".to_owned(),
        ]);
        let quant_tensors = BTreeSet::from([
            "model.layers.0.mlp.down_proj.weight".to_owned(),
            "model.layers.0.mlp.gate_proj.weight".to_owned(),
        ]);
        let digest = canonical_fingerprint(
            &ProviderAttributionDenominatorMaterial {
                operations: &operations,
                quant_tensors: &quant_tensors,
            },
            "test provider attribution denominator",
        )
        .unwrap();
        let expected = Sha256::digest(
            br#"{"operations":["operation.causal_paged_attention","operation.dense_swiglu","operation.gated_delta_recurrent_attention"],"quant_tensors":["model.layers.0.mlp.down_proj.weight","model.layers.0.mlp.gate_proj.weight"]}"#,
        );
        assert_eq!(digest, format!("{expected:x}"));
    }
}
