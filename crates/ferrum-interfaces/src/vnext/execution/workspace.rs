use super::{
    align_up, invalid_plan, Deserialize, Deserializer, DynamicResourceDemand, DynamicResourceShape,
    DynamicResourceShapeBucket, DynamicStorageRequirement, ResourceWorkShape, Serialize,
    VNextError, MAX_PROVIDER_WORKSPACE_SHAPE_BUCKETS,
};

/// Provider-owned unit sizing formula. Scheduler and admission ceilings are
/// intentionally absent so one implementation estimate remains reusable
/// across runtime policies. Core binds those ceilings when it builds the
/// executable memory plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderWorkspaceSizeFormula {
    Fixed {
        bytes: u64,
    },
    ActualSequences {
        bytes_per_sequence: u64,
    },
    Tokens {
        bytes_per_token: u64,
    },
    Pages {
        bytes_per_page: u64,
        maximum_pages: u64,
    },
    BoundedShapeBuckets {
        buckets: Vec<DynamicResourceShapeBucket>,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum ProviderWorkspaceSizeFormulaWire {
    Fixed {
        bytes: u64,
    },
    ActualSequences {
        bytes_per_sequence: u64,
    },
    Tokens {
        bytes_per_token: u64,
    },
    Pages {
        bytes_per_page: u64,
        maximum_pages: u64,
    },
    BoundedShapeBuckets {
        buckets: Vec<DynamicResourceShapeBucket>,
    },
}

impl ProviderWorkspaceSizeFormula {
    pub fn fixed(bytes: u64) -> Result<Self, VNextError> {
        Self::validated(Self::Fixed { bytes })
    }

    pub fn actual_sequences(bytes_per_sequence: u64) -> Result<Self, VNextError> {
        Self::validated(Self::ActualSequences { bytes_per_sequence })
    }

    pub fn tokens(bytes_per_token: u64) -> Result<Self, VNextError> {
        Self::validated(Self::Tokens { bytes_per_token })
    }

    pub fn pages(bytes_per_page: u64, maximum_pages: u64) -> Result<Self, VNextError> {
        Self::validated(Self::Pages {
            bytes_per_page,
            maximum_pages,
        })
    }

    pub fn bounded_shape_buckets(
        buckets: Vec<DynamicResourceShapeBucket>,
    ) -> Result<Self, VNextError> {
        Self::validated(Self::BoundedShapeBuckets { buckets })
    }

    fn validated(formula: Self) -> Result<Self, VNextError> {
        formula.validate()?;
        Ok(formula)
    }

    fn validate(&self) -> Result<(), VNextError> {
        let valid = match self {
            Self::Fixed { bytes } => *bytes > 0,
            Self::ActualSequences { bytes_per_sequence }
            | Self::Tokens {
                bytes_per_token: bytes_per_sequence,
            } => *bytes_per_sequence > 0,
            Self::Pages {
                bytes_per_page,
                maximum_pages,
            } => {
                *bytes_per_page > 0
                    && *maximum_pages > 0
                    && bytes_per_page.checked_mul(*maximum_pages).is_some()
            }
            Self::BoundedShapeBuckets { buckets } => {
                !buckets.is_empty()
                    && buckets.len() <= MAX_PROVIDER_WORKSPACE_SHAPE_BUCKETS
                    && buckets.windows(2).all(|pair| {
                        let previous = &pair[0];
                        let next = &pair[1];
                        next.maximum_sequences() >= previous.maximum_sequences()
                            && next.maximum_tokens() >= previous.maximum_tokens()
                            && next.maximum_pages() >= previous.maximum_pages()
                            && (next.maximum_sequences() > previous.maximum_sequences()
                                || next.maximum_tokens() > previous.maximum_tokens()
                                || next.maximum_pages() > previous.maximum_pages())
                            && next.bytes() >= previous.bytes()
                    })
            }
        };
        if !valid {
            return Err(invalid_plan(
                "provider workspace formula is zero, overflowing, or non-canonical",
            ));
        }
        Ok(())
    }

    fn evaluate_shape_bytes(&self, shape: DynamicResourceShape) -> Result<u64, VNextError> {
        self.validate()?;
        let bytes = match self {
            Self::Fixed { bytes } => *bytes,
            Self::ActualSequences { bytes_per_sequence } => bytes_per_sequence
                .checked_mul(u64::from(shape.sequences))
                .ok_or_else(|| invalid_plan("provider sequence workspace overflows u64"))?,
            Self::Tokens { bytes_per_token } => bytes_per_token
                .checked_mul(shape.tokens)
                .ok_or_else(|| invalid_plan("provider token workspace overflows u64"))?,
            Self::Pages {
                bytes_per_page,
                maximum_pages,
            } if shape.pages <= *maximum_pages => bytes_per_page
                .checked_mul(shape.pages)
                .ok_or_else(|| invalid_plan("provider page workspace overflows u64"))?,
            Self::BoundedShapeBuckets { buckets } => buckets
                .iter()
                .find(|bucket| bucket.covers(shape))
                .map(DynamicResourceShapeBucket::bytes)
                .ok_or_else(|| invalid_plan("actual invocation shape exceeds provider buckets"))?,
            Self::Pages { .. } => {
                return Err(invalid_plan(
                    "actual invocation pages exceed the provider implementation bound",
                ))
            }
        };
        if bytes == 0 {
            return Err(invalid_plan("provider workspace evaluates to zero bytes"));
        }
        Ok(bytes)
    }

    pub(super) fn bind_runtime_limits(
        &self,
        maximum_sequences: u32,
        maximum_tokens: u64,
    ) -> Result<DynamicResourceDemand, VNextError> {
        self.validate()?;
        match self {
            Self::Fixed { bytes } => DynamicResourceDemand::fixed(*bytes),
            Self::ActualSequences { bytes_per_sequence } => {
                DynamicResourceDemand::actual_sequences(*bytes_per_sequence, maximum_sequences)
            }
            Self::Tokens { bytes_per_token } => {
                DynamicResourceDemand::tokens(*bytes_per_token, maximum_tokens)
            }
            Self::Pages {
                bytes_per_page,
                maximum_pages,
            } => DynamicResourceDemand::pages(*bytes_per_page, *maximum_pages),
            Self::BoundedShapeBuckets { buckets } => {
                DynamicResourceDemand::bounded_shape_buckets(buckets.clone())
            }
        }
    }

    fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed { .. })
    }

    fn is_valid_for_sequence_scope(&self) -> bool {
        match self {
            Self::Fixed { .. } | Self::Tokens { .. } | Self::Pages { .. } => true,
            Self::BoundedShapeBuckets { buckets } => {
                buckets.iter().all(|bucket| bucket.maximum_sequences() == 1)
            }
            Self::ActualSequences { .. } => false,
        }
    }
}

impl<'de> Deserialize<'de> for ProviderWorkspaceSizeFormula {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let formula = match ProviderWorkspaceSizeFormulaWire::deserialize(deserializer)? {
            ProviderWorkspaceSizeFormulaWire::Fixed { bytes } => Self::Fixed { bytes },
            ProviderWorkspaceSizeFormulaWire::ActualSequences { bytes_per_sequence } => {
                Self::ActualSequences { bytes_per_sequence }
            }
            ProviderWorkspaceSizeFormulaWire::Tokens { bytes_per_token } => {
                Self::Tokens { bytes_per_token }
            }
            ProviderWorkspaceSizeFormulaWire::Pages {
                bytes_per_page,
                maximum_pages,
            } => Self::Pages {
                bytes_per_page,
                maximum_pages,
            },
            ProviderWorkspaceSizeFormulaWire::BoundedShapeBuckets { buckets } => {
                Self::BoundedShapeBuckets { buckets }
            }
        };
        Self::validated(formula).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderWorkspaceScope {
    Plan,
    Request,
    Sequence,
    Step,
    Invocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderWorkspaceRequirement {
    pub(super) size_formula: ProviderWorkspaceSizeFormula,
    pub(super) alignment_bytes: u64,
    pub(super) scope: ProviderWorkspaceScope,
    pub(super) storage: DynamicStorageRequirement,
}

impl ProviderWorkspaceRequirement {
    /// Convenience constructor for a fixed-size workspace. Shape-dependent
    /// providers must use [`Self::from_formula`].
    pub fn new(
        fixed_bytes: u64,
        alignment_bytes: u64,
        scope: ProviderWorkspaceScope,
        storage: DynamicStorageRequirement,
    ) -> Result<Self, VNextError> {
        Self::from_formula(
            ProviderWorkspaceSizeFormula::fixed(fixed_bytes)?,
            alignment_bytes,
            scope,
            storage,
        )
    }

    pub fn from_formula(
        size_formula: ProviderWorkspaceSizeFormula,
        alignment_bytes: u64,
        scope: ProviderWorkspaceScope,
        storage: DynamicStorageRequirement,
    ) -> Result<Self, VNextError> {
        size_formula.validate()?;
        if alignment_bytes == 0
            || !alignment_bytes.is_power_of_two()
            || (scope == ProviderWorkspaceScope::Plan && !size_formula.is_fixed())
            || (scope == ProviderWorkspaceScope::Sequence
                && !size_formula.is_valid_for_sequence_scope())
        {
            return Err(invalid_plan(
                "provider workspace has invalid formula, alignment, or scope",
            ));
        }
        let requirement = Self {
            size_formula,
            alignment_bytes,
            scope,
            storage,
        };
        requirement.minimum_bytes()?;
        Ok(requirement)
    }

    pub fn size_formula(&self) -> &ProviderWorkspaceSizeFormula {
        &self.size_formula
    }

    pub fn evaluate_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(work.immediate_shape())
    }

    pub fn evaluate_fit_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(work.fit_shape())
    }

    pub(crate) fn evaluate_shape_bytes(
        &self,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        align_up(
            self.size_formula.evaluate_shape_bytes(shape)?,
            self.alignment_bytes,
        )
    }

    pub fn minimum_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(DynamicResourceShape::from_validated(1, 1, 1))
    }

    pub fn fixed_bytes(&self) -> Option<u64> {
        match &self.size_formula {
            ProviderWorkspaceSizeFormula::Fixed { bytes } => Some(*bytes),
            _ => None,
        }
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    pub const fn scope(&self) -> ProviderWorkspaceScope {
        self.scope
    }

    pub fn storage(&self) -> &DynamicStorageRequirement {
        &self.storage
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProviderWorkspaceRequirementWire {
    pub(super) size_formula: ProviderWorkspaceSizeFormula,
    pub(super) alignment_bytes: u64,
    pub(super) scope: ProviderWorkspaceScope,
    pub(super) storage: DynamicStorageRequirement,
}

impl<'de> Deserialize<'de> for ProviderWorkspaceRequirement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderWorkspaceRequirementWire::deserialize(deserializer)?;
        Self::from_formula(
            wire.size_formula,
            wire.alignment_bytes,
            wire.scope,
            wire.storage,
        )
        .map_err(serde::de::Error::custom)
    }
}
