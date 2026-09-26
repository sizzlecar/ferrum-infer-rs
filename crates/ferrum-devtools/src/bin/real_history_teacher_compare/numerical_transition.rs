//! Explicit numerical qualification, never implementation equivalence or release approval.
use super::*;
use ferrum_interfaces::vnext::{
    ProviderExecutionRepeatability, ProviderExecutionSemantics, ProviderReplayEquivalence,
};
use ferrum_types::{NumericalExecutionPolicy, NumericalProfileId};
use transition::ExecutionPin;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum NumericalScope {
    /// Qwen's declared FFN-only Q8 activation policy; GDN/head/KV remain unchanged.
    DenseSwiGluQ8F32ScaleV1,
    /// Separately declared Q8 policy with the original input's F32 sum.
    DenseSwiGluQ8F32ScaleInputSumV1,
    /// B8 physical gate/up-only Stream-MMQ, with strict fallback and down.
    DenseSwiGluQ8GateUpStreamMmqV1,
    /// All FFN, GDN and causal projections; untouched nodes retain strict signatures.
    Qwen35GgufF16ProjectionsV1,
    /// RN-F16 attention plus compressed whole-M2..8 residual2 FFN; separately qualified.
    Qwen35GgufF16AttentionQ8Residual2FfnM2to8V1,
}

impl NumericalScope {
    fn candidate_profile(self) -> &'static str {
        match self {
            Self::Qwen35GgufF16AttentionQ8Residual2FfnM2to8V1 => {
                "qwen3_5.f32-master.gguf-f16-attention.q8-residual2-ffn-m2to8"
            }
            Self::Qwen35GgufF16ProjectionsV1 => "qwen3_5.f32-master.gguf-f16-projections",
            Self::DenseSwiGluQ8GateUpStreamMmqV1 => "qwen3_5.f32-master.q8-gate-up-stream-mmq",
            Self::DenseSwiGluQ8F32ScaleV1 => "qwen3_5.f32-master.q8-swiglu",
            Self::DenseSwiGluQ8F32ScaleInputSumV1 => "qwen3_5.f32-master.q8-swiglu-input-sum",
        }
    }

    fn candidate_operation(self) -> Option<&'static str> {
        match self {
            Self::Qwen35GgufF16ProjectionsV1
            | Self::Qwen35GgufF16AttentionQ8Residual2FfnM2to8V1 => None,
            Self::DenseSwiGluQ8GateUpStreamMmqV1 => {
                Some("operation.dense_swiglu.q8-gate-up-stream-mmq-f32scale")
            }
            Self::DenseSwiGluQ8F32ScaleV1 => Some("operation.dense_swiglu.q8-f32scale"),
            Self::DenseSwiGluQ8F32ScaleInputSumV1 => {
                Some("operation.dense_swiglu.q8-f32scale-input-sum")
            }
        }
    }
    fn all_projections(self) -> bool {
        matches!(
            self,
            Self::Qwen35GgufF16ProjectionsV1 | Self::Qwen35GgufF16AttentionQ8Residual2FfnM2to8V1
        )
    }

    fn projection_operation(self, reference: &str) -> Option<&'static str> {
        use ferrum_interfaces::vnext::{
            DENSE_SWIGLU_OPERATION_ID, DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID,
        };
        if matches!(self, Self::Qwen35GgufF16AttentionQ8Residual2FfnM2to8V1)
            && reference == DENSE_SWIGLU_OPERATION_ID
        {
            Some(DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID)
        } else {
            gguf_f16_operation(reference)
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Qwen35GgufF16AttentionQ8Residual2FfnM2to8V1 => {
                "declared_rn_f16_attention_residual2_ffn_fixed_history_full_vocabulary_quality_only"
            }
            Self::Qwen35GgufF16ProjectionsV1 => {
                "declared_all_projections_rn_f16_fixed_history_full_vocabulary_quality_only"
            }
            _ => "declared_ffn_q8_numerical_policy_fixed_history_full_vocabulary_quality_only",
        }
    }

    fn operation_pair(self, reference: &str, candidate: &str) -> bool {
        if self.all_projections() {
            self.projection_operation(reference) == Some(candidate)
        } else {
            reference == "operation.dense_swiglu" && Some(candidate) == self.candidate_operation()
        }
    }
}

fn gguf_f16_operation(reference: &str) -> Option<&'static str> {
    use ferrum_interfaces::vnext::*;
    match reference {
        DENSE_SWIGLU_OPERATION_ID => Some(DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID),
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID => {
            Some(GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID)
        }
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID => {
            Some(CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID)
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NumericalPin {
    pub numerical_profile: NumericalProfileId,
    pub family_fingerprint: String,
    pub program_fingerprint: String,
    pub execution: ExecutionPin,
}

impl NumericalPin {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.numerical_profile.as_str() != "auto"
                && files::canonical_sha(&self.family_fingerprint)
                && files::canonical_sha(&self.program_fingerprint),
            "numerical transition profile/family/program pin is malformed"
        );
        self.execution.validate()
    }

    fn matches(
        &self,
        capture: &capture::CheckedCapture,
        receipt: &receipt::CheckedReceipt,
    ) -> Result<bool> {
        let identity = capture
            .manifest
            .identity
            .as_ref()
            .context("capture identity missing")?;
        Ok(
            identity.numerical_profile == self.numerical_profile.as_str()
                && identity.family_fingerprint == self.family_fingerprint
                && identity.program_fingerprint == self.program_fingerprint
                && ExecutionPin::observed(capture, receipt)? == self.execution,
        )
    }
}

// ContractVersion's general wire accepts extra fields. This declaration must not:
// keep a strict local wire, then use the real semantics parser to verify its hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SemanticsVersion {
    pub major: u16,
    pub minor: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SemanticsPin {
    pub contract_version: SemanticsVersion,
    pub contract_fingerprint: String,
    pub repeatability: ProviderExecutionRepeatability,
    pub replay_equivalence: ProviderReplayEquivalence,
}

impl SemanticsPin {
    fn checked(&self) -> Result<ProviderExecutionSemantics> {
        serde_json::from_value(serde_json::to_value(self)?)
            .context("numerical transition execution semantics are invalid")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct NodeImplementation {
    pub operation_id: String,
    pub provider_id: String,
    pub implementation_fingerprint: String,
}

impl NodeImplementation {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.operation_id.trim().is_empty()
                && !self.provider_id.trim().is_empty()
                && files::canonical_sha(&self.implementation_fingerprint),
            "numerical transition node implementation is malformed"
        );
        Ok(())
    }

    fn matches(&self, node: &receipt::NodeSignature) -> bool {
        self.operation_id == node.operation_id
            && self.provider_id == node.provider_id
            && self.implementation_fingerprint == node.implementation_fingerprint
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NodeChange {
    pub node_index: usize,
    pub node_id: String,
    pub execution_semantics: SemanticsPin,
    pub reference: NodeImplementation,
    pub candidate: NodeImplementation,
}

/// Program-derived names may change with a numerical profile. This declaration
/// maps names only; it grants no new output range, node or readback authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OutputResourceMapping {
    pub reference_resource_id: String,
    pub candidate_resource_id: String,
}

impl OutputResourceMapping {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.reference_resource_id.trim().is_empty()
                && !self.candidate_resource_id.trim().is_empty()
                && self.reference_resource_id != self.candidate_resource_id,
            "numerical output resource mapping requires two distinct nonempty resource IDs"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NumericalTransition {
    pub schema_version: u32,
    pub scope: NumericalScope,
    pub reference: NumericalPin,
    pub candidate: NumericalPin,
    pub ordered_node_count: usize,
    /// Strictly increasing physical program indices; all other signatures must match.
    pub node_changes: Vec<NodeChange>,
    /// Schema 1 remains exact. Schema 2 requires this explicit two-arm pin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_resource_mapping: Option<OutputResourceMapping>,
}

pub(super) struct DeclaredNumericalTransition {
    declaration: NumericalTransition,
    path: PathBuf,
    sha256: String,
}

impl DeclaredNumericalTransition {
    pub(super) fn read(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).context("read numerical profile transition declaration")?;
        let declaration: NumericalTransition = serde_json::from_slice(&bytes)
            .context("parse numerical profile transition declaration")?;
        match (
            declaration.schema_version,
            &declaration.output_resource_mapping,
        ) {
            (1, None) => {}
            (2, Some(mapping)) => mapping.validate()?,
            (1, Some(_)) => anyhow::bail!("numerical output resource mapping requires schema 2"),
            (2, None) => {
                anyhow::bail!("numerical transition schema 2 requires output resource mapping")
            }
            _ => anyhow::bail!("unsupported numerical transition schema"),
        }
        declaration.reference.validate()?;
        declaration.candidate.validate()?;
        ensure!(
            declaration.reference.numerical_profile != declaration.candidate.numerical_profile,
            "numerical transition requires distinct explicit profiles"
        );
        ensure!(
            declaration.reference.numerical_profile.as_str() == "qwen3_5.f32-master"
                && declaration.candidate.numerical_profile.as_str()
                    == declaration.scope.candidate_profile(),
            "numerical profile pair is outside the declared numerical scope"
        );
        ensure!(
            !declaration.node_changes.is_empty(),
            "numerical transition has no node changes"
        );
        let mut previous = None;
        let mut ids = BTreeSet::new();
        for change in &declaration.node_changes {
            ensure!(
                previous.is_none_or(|index| index < change.node_index)
                    && change.node_index < declaration.ordered_node_count
                    && !change.node_id.trim().is_empty()
                    && ids.insert(&change.node_id),
                "numerical transition node changes are unordered, duplicated or out of range"
            );
            previous = Some(change.node_index);
            change.reference.validate()?;
            change.candidate.validate()?;
            change.execution_semantics.checked()?;
            ensure!(
                declaration.scope.operation_pair(&change.reference.operation_id, &change.candidate.operation_id),
                "node transition is outside dense SwiGLU Q8 scope or the explicit all-projection scope"
            );
        }
        if declaration.scope.all_projections() {
            ensure!(
                declaration.reference.execution.binary_sha256
                    == declaration.candidate.execution.binary_sha256,
                "GGUF RN-F16 qualification requires the same executable for strict and candidate"
            );
            let kinds: BTreeSet<_> = declaration
                .node_changes
                .iter()
                .map(|c| c.reference.operation_id.as_str())
                .collect();
            for required in [
                "operation.dense_swiglu",
                "operation.gated_delta_recurrent_attention.f32-master",
                "operation.causal_paged_attention.f32-master",
            ] {
                ensure!(
                    kinds.contains(required),
                    "all-projection scope requires FFN, GDN and causal node inventory"
                );
            }
        }
        Ok(Self {
            declaration,
            path: path.to_path_buf(),
            sha256: sha256(&bytes),
        })
    }

    pub(super) fn normalize_configuration(
        &self,
        reference: &mut Value,
        candidate: &mut Value,
    ) -> Result<()> {
        for (configuration, pin) in [
            (reference, &self.declaration.reference),
            (candidate, &self.declaration.candidate),
        ] {
            let policy = configuration.get("numerical_execution").context(
                "numerical transition requires explicit configuration numerical_execution.require",
            )?;
            let expected = serde_json::to_value(NumericalExecutionPolicy::Require(
                pin.numerical_profile.clone(),
            ))?;
            // Exact typed wire: Auto, extra policy fields and absent Require cannot be normalized.
            ensure!(policy == &expected, "configuration numerical_execution.require differs from explicit numerical transition pin");
            *configuration
                .pointer_mut("/numerical_execution/require")
                .expect("validated Require") =
                json!({"exact_declared_numerical_profile_transition":true});
        }
        Ok(())
    }

    pub(super) fn qualify(
        &self,
        reference: &capture::CheckedCapture,
        candidate: &capture::CheckedCapture,
        r: &receipt::CheckedReceipt,
        c: &receipt::CheckedReceipt,
    ) -> Result<Value> {
        let d = &self.declaration;
        ensure!(
            reference.directory.canonicalize()? != candidate.directory.canonicalize()?
                && reference.manifest_sha256 != candidate.manifest_sha256,
            "numerical transition requires distinct captures"
        );
        ensure!(
            d.reference.matches(reference, r)?,
            "reference identity differs from exact numerical transition pin"
        );
        ensure!(
            d.candidate.matches(candidate, c)?,
            "candidate identity differs from exact numerical transition pin"
        );
        ensure!(
            r.node_signature.len() == d.ordered_node_count
                && c.node_signature.len() == d.ordered_node_count,
            "numerical transition changes ordered node inventory"
        );
        let mut changes = d.node_changes.iter().peekable();
        for (index, (a, b)) in r.node_signature.iter().zip(&c.node_signature).enumerate() {
            ensure!(a.node_id == b.node_id && a.execution_semantics == b.execution_semantics,
                "numerical transition changes ordered node topology or execution semantics at {index}");
            if d.scope.all_projections() {
                if let Some(expected) = d.scope.projection_operation(&a.operation_id) {
                    ensure!(b.operation_id == expected && changes.peek().is_some_and(|change|change.node_index==index),
                        "all-projection scope omitted or changed a strict projection at node {index}");
                }
            }
            if changes
                .peek()
                .is_some_and(|change| change.node_index == index)
            {
                let change = changes.next().expect("peeked node");
                ensure!(
                    a.node_id == change.node_id
                        && a.execution_semantics == change.execution_semantics.checked()?
                        && change.reference.matches(a)
                        && change.candidate.matches(b),
                    "node {index} differs from exact declared numerical transition"
                );
            } else {
                ensure!(a == b, "undeclared numerical transition at node {index}");
            }
        }
        ensure!(
            changes.next().is_none(),
            "numerical transition has unused node changes"
        );
        if let Some(mapping) = &d.output_resource_mapping {
            ensure!(
                r.output_binding.resource_id() == mapping.reference_resource_id
                    && c.output_binding.resource_id() == mapping.candidate_resource_id,
                "product output resource differs from exact numerical transition mapping"
            );
            ensure!(
                r.output_binding.same_product_range(&c.output_binding),
                "product output binding differs beyond declared resource ID mapping"
            );
        } else {
            ensure!(
                r.output_binding == c.output_binding,
                "product output binding differs between arms"
            );
        }
        Ok(json!({
            "mode":"pinned_numerical_profile_transition",
            "declaration":{"path":self.path,"sha256":self.sha256,"content":d},
            "numerical_profiles":{"reference":d.reference.numerical_profile,"candidate":d.candidate.numerical_profile},
            "ordered_node_count":d.ordered_node_count,"changed_node_count":d.node_changes.len(),
            "ordered_node_ids_and_execution_semantics_matched":true,
            "output_resource_mapping_verified":d.output_resource_mapping.is_some(),
            "product_output_node_usage_offset_and_layout_matched":true,
            "same_physical_plan_claimed":false,"physical_plan_equivalence_independently_recomputed":false,
            "numerical_equivalence_claimed":false,"release_approved":false,
            "actual_route_coverage":if d.scope.all_projections() || matches!(d.scope, NumericalScope::DenseSwiGluQ8GateUpStreamMmqV1) {
                "requires_independent_completed_wave_node_evidence"
            } else { "not_established_by_profile_transition" },
            "scope":d.scope.description()
        }))
    }
}
