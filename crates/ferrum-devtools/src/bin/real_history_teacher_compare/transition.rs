//! An exact implementation migration declaration, never a blanket identity exemption.
use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ExecutionPin {
    pub binary_sha256: String,
    pub resolved_plan_fingerprint: String,
    pub plan_id: String,
    pub plan_hash: String,
    pub runtime_implementation_fingerprint: String,
}

impl ExecutionPin {
    pub(super) fn observed(
        capture: &capture::CheckedCapture,
        receipt: &receipt::CheckedReceipt,
    ) -> Result<Self> {
        let identity = capture
            .manifest
            .identity
            .as_ref()
            .context("capture identity missing")?;
        Ok(Self {
            binary_sha256: identity.binary.sha256.clone(),
            resolved_plan_fingerprint: identity.resolved_plan_fingerprint.clone(),
            plan_id: receipt.plan_id.clone(),
            plan_hash: receipt.plan_hash.clone(),
            runtime_implementation_fingerprint: receipt.runtime.clone(),
        })
    }
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            [
                &self.binary_sha256,
                &self.resolved_plan_fingerprint,
                &self.plan_hash,
                &self.runtime_implementation_fingerprint
            ]
            .into_iter()
            .all(|s| files::canonical_sha(s))
                && !self.plan_id.trim().is_empty(),
            "transition execution pin is not a complete canonical identity"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProviderChange {
    pub provider_id: String,
    pub reference_fingerprint: String,
    pub candidate_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ImplementationTransition {
    pub schema_version: u32,
    pub reference: ExecutionPin,
    pub candidate: ExecutionPin,
    pub provider_changes: Vec<ProviderChange>,
}

pub(super) struct DeclaredTransition {
    declaration: ImplementationTransition,
    path: PathBuf,
    sha256: String,
}

impl DeclaredTransition {
    pub(super) fn read(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).context("read implementation transition declaration")?;
        let declaration: ImplementationTransition = serde_json::from_slice(&bytes)
            .context("parse implementation transition declaration")?;
        ensure!(
            declaration.schema_version == 1,
            "unsupported transition schema"
        );
        declaration.reference.validate()?;
        declaration.candidate.validate()?;
        ensure!(
            declaration.reference.binary_sha256 != declaration.candidate.binary_sha256,
            "implementation transition requires distinct verified binaries"
        );
        let mut seen = BTreeSet::new();
        ensure!(
            !declaration.provider_changes.is_empty(),
            "transition has no provider changes"
        );
        for change in &declaration.provider_changes {
            ensure!(
                !change.provider_id.trim().is_empty()
                    && seen.insert(&change.provider_id)
                    && files::canonical_sha(&change.reference_fingerprint)
                    && files::canonical_sha(&change.candidate_fingerprint)
                    && change.reference_fingerprint != change.candidate_fingerprint,
                "transition provider declaration is duplicated, unchanged or malformed"
            );
        }
        Ok(Self {
            declaration,
            path: path.to_path_buf(),
            sha256: sha256(&bytes),
        })
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
            "transition requires distinct capture directories and manifest hashes"
        );
        ensure!(
            ExecutionPin::observed(reference, r)? == d.reference,
            "reference execution identity does not match exact transition pin"
        );
        ensure!(
            ExecutionPin::observed(candidate, c)? == d.candidate,
            "candidate execution identity does not match exact transition pin"
        );
        ensure!(
            r.node_signature.len() == c.node_signature.len(),
            "transition changes ordered node inventory"
        );
        let changes: BTreeMap<_, _> = d
            .provider_changes
            .iter()
            .map(|change| (change.provider_id.as_str(), change))
            .collect();
        let mut matched = BTreeMap::<&str, usize>::new();
        for (a, b) in r.node_signature.iter().zip(&c.node_signature) {
            ensure!(
                a.node_id == b.node_id
                    && a.operation_id == b.operation_id
                    && a.provider_id == b.provider_id
                    && a.execution_semantics == b.execution_semantics,
                "transition changes ordered node topology or execution semantics at {}",
                a.node_id
            );
            if let Some(change) = changes.get(a.provider_id.as_str()) {
                ensure!(
                    a.implementation_fingerprint == change.reference_fingerprint
                        && b.implementation_fingerprint == change.candidate_fingerprint,
                    "provider {} differs from exact declared transition",
                    a.provider_id
                );
                *matched.entry(a.provider_id.as_str()).or_default() += 1;
            } else {
                ensure!(
                    a.implementation_fingerprint == b.implementation_fingerprint,
                    "undeclared provider implementation change: {}",
                    a.provider_id
                );
            }
        }
        ensure!(
            matched.len() == changes.len(),
            "transition contains an unused provider exemption"
        );
        Ok(json!({
            "mode":"pinned_implementation_transition",
            "declaration":{"path":self.path,"sha256":self.sha256,"content":d},
            "matched_provider_node_counts":matched,
            "ordered_node_count":r.node_signature.len(),
            "ordered_nodes_and_execution_semantics_matched":true,
            "same_physical_plan_claimed":false,
            "physical_plan_equivalence_independently_recomputed":false,
            "scope":"declared_implementation_migration_same_logical_semantics_fixed_history_full_vocabulary"
        }))
    }
}
