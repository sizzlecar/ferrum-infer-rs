use super::{
    canonical_fingerprint, is_canonical_sha256, ContractVersion, Deserialize, ModelFamilyId,
    PreparedModelFamily, Serialize, VNextError, WeightMaterializerId, WeightSchema,
};

const IDENTITY_MATERIALIZER_ID: &str = "weight-materializer.identity";
const IDENTITY_MATERIALIZER_VERSION: ContractVersion = ContractVersion::new(1, 0);

#[derive(Serialize)]
struct IdentityMaterializerFingerprint<'a> {
    id: &'a str,
    version: ContractVersion,
    contract: &'a str,
}

fn identity_materializer_fingerprint() -> Result<String, VNextError> {
    canonical_fingerprint(
        &IdentityMaterializerFingerprint {
            id: IDENTITY_MATERIALIZER_ID,
            version: IDENTITY_MATERIALIZER_VERSION,
            contract: "execution-weight-plan.identity.v1",
        },
        "fingerprint identity weight materializer",
    )
}

/// Trusted physical weight contract selected for one immutable execution plan.
///
/// The prepared family remains the source/checkpoint contract. This plan owns
/// the physical schema consumed by providers, static memory planning, and
/// initialization. Keeping those contracts separate allows a backend to
/// select a prepared layout without changing model semantics or hiding an
/// allocation behind a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionWeightPlan {
    source_schema_fingerprint: String,
    materializer_id: WeightMaterializerId,
    materializer_version: ContractVersion,
    materializer_implementation_fingerprint: String,
    schema: WeightSchema,
}

impl ExecutionWeightPlan {
    pub fn identity(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        let plan = Self {
            source_schema_fingerprint: family.weight_schema().fingerprint()?,
            materializer_id: WeightMaterializerId::new(IDENTITY_MATERIALIZER_ID)?,
            materializer_version: IDENTITY_MATERIALIZER_VERSION,
            materializer_implementation_fingerprint: identity_materializer_fingerprint()?,
            schema: family.weight_schema().clone(),
        };
        plan.validate_against_family(family)?;
        Ok(plan)
    }

    pub fn source_schema_fingerprint(&self) -> &str {
        &self.source_schema_fingerprint
    }

    pub fn materializer_id(&self) -> &WeightMaterializerId {
        &self.materializer_id
    }

    pub const fn materializer_version(&self) -> ContractVersion {
        self.materializer_version
    }

    pub fn materializer_implementation_fingerprint(&self) -> &str {
        &self.materializer_implementation_fingerprint
    }

    pub fn schema(&self) -> &WeightSchema {
        &self.schema
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(self, "fingerprint execution weight plan")
    }

    pub(super) fn validate_structure(&self, family_id: &ModelFamilyId) -> Result<(), VNextError> {
        if !is_canonical_sha256(&self.source_schema_fingerprint)
            || !is_canonical_sha256(&self.materializer_implementation_fingerprint)
            || self.materializer_version.major == 0
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "execution weight plan provenance is invalid".to_owned(),
            });
        }
        self.schema.validate(family_id)
    }

    /// Identity is the only admitted materializer in this first production
    /// slice. Later materializers must be reconstructed through a trusted
    /// catalog; an untrusted serialized plan may never authorize itself.
    pub(crate) fn validate_against_family(
        &self,
        family: &PreparedModelFamily,
    ) -> Result<(), VNextError> {
        self.validate_structure(family.family_id())?;
        if self.source_schema_fingerprint != family.weight_schema().fingerprint()?
            || self.materializer_id.as_str() != IDENTITY_MATERIALIZER_ID
            || self.materializer_version != IDENTITY_MATERIALIZER_VERSION
            || self.materializer_implementation_fingerprint != identity_materializer_fingerprint()?
            || &self.schema != family.weight_schema()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason:
                    "execution weight plan is not the trusted identity materialization for its prepared family"
                        .to_owned(),
            });
        }
        Ok(())
    }
}
