use super::contracts::invalid_plan;
use super::{Deserialize, Deserializer, Digest, Range, Serialize, Sha256, VNextError};

pub const MAX_PROVIDER_WORKSPACE_SHAPE_BUCKETS: usize = 64;

/// Internal evaluator dimensions. Public callers supply token/page evidence;
/// only core lowers that evidence into aggregate formula inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DynamicResourceShape {
    pub(super) sequences: u32,
    pub(super) tokens: u64,
    pub(super) pages: u64,
}

impl DynamicResourceShape {
    pub(crate) fn new(sequences: u32, tokens: u64, pages: u64) -> Result<Self, VNextError> {
        if sequences == 0 || tokens == 0 || pages == 0 {
            return Err(invalid_plan(
                "dynamic resource shape dimensions must be non-zero",
            ));
        }
        Ok(Self {
            sequences,
            tokens,
            pages,
        })
    }

    pub(crate) const fn sequences(self) -> u32 {
        self.sequences
    }

    pub(crate) const fn tokens(self) -> u64 {
        self.tokens
    }

    pub(crate) const fn pages(self) -> u64 {
        self.pages
    }

    pub(crate) const fn from_validated(sequences: u32, tokens: u64, pages: u64) -> Self {
        Self {
            sequences,
            tokens,
            pages,
        }
    }
}

/// Evidence for one non-empty immediate token span inside an exact full input.
/// Counts are derived from the supplied token slice and private range rather
/// than accepted as caller-provided aggregate dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokenSpanWork {
    pub(super) immediate_tokens: u64,
    pub(super) full_input_tokens: u64,
    pub(super) fingerprint: String,
}

impl TokenSpanWork {
    pub fn from_token_ids(
        full_input: &[u32],
        immediate_range: Range<usize>,
    ) -> Result<Self, VNextError> {
        if full_input.is_empty()
            || immediate_range.start >= immediate_range.end
            || immediate_range.end > full_input.len()
        {
            return Err(invalid_plan(
                "token work requires a non-empty in-bounds immediate span",
            ));
        }
        let immediate_tokens = u64::try_from(immediate_range.len())
            .map_err(|_| invalid_plan("immediate token span exceeds u64"))?;
        let full_input_tokens = u64::try_from(full_input.len())
            .map_err(|_| invalid_plan("full token input exceeds u64"))?;
        let mut digest = Sha256::new();
        digest.update(b"ferrum.runtime-vnext.token-span-work.v1\0");
        digest.update(full_input_tokens.to_le_bytes());
        digest.update(
            u64::try_from(immediate_range.start)
                .map_err(|_| invalid_plan("token span start exceeds u64"))?
                .to_le_bytes(),
        );
        digest.update(
            u64::try_from(immediate_range.end)
                .map_err(|_| invalid_plan("token span end exceeds u64"))?
                .to_le_bytes(),
        );
        for token in full_input {
            digest.update(token.to_le_bytes());
        }
        Ok(Self {
            immediate_tokens,
            full_input_tokens,
            fingerprint: format!("{:x}", digest.finalize()),
        })
    }

    pub const fn immediate_tokens(&self) -> u64 {
        self.immediate_tokens
    }

    pub const fn full_input_tokens(&self) -> u64 {
        self.full_input_tokens
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CommittedPageWork {
    pub(super) pages: u64,
    pub(super) fingerprint: String,
}

impl CommittedPageWork {
    pub(crate) fn new(pages: u64, fingerprint: String) -> Result<Self, VNextError> {
        if pages == 0
            || fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid_plan("committed page work evidence is invalid"));
        }
        Ok(Self { pages, fingerprint })
    }

    pub(crate) const fn pages(&self) -> u64 {
        self.pages
    }
}

/// Typed shape evidence shared by scoped admission and provider formula
/// evaluation. The aggregate dimensions and fingerprint are core-derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceWorkShape {
    pub(super) token_spans: Vec<TokenSpanWork>,
    pub(super) committed_pages: Vec<CommittedPageWork>,
    pub(super) immediate_sequences: u32,
    pub(super) immediate_tokens: u64,
    pub(super) immediate_pages: u64,
    pub(super) fit_sequences: u32,
    pub(super) fit_tokens: u64,
    pub(super) fit_pages: u64,
    pub(super) fingerprint: String,
}

impl ResourceWorkShape {
    pub fn from_token_spans(token_spans: Vec<TokenSpanWork>) -> Result<Self, VNextError> {
        Self::from_sources(token_spans, Vec::new())
    }

    pub fn single(token_span: TokenSpanWork) -> Result<Self, VNextError> {
        Self::from_token_spans(vec![token_span])
    }

    pub(crate) fn from_sources(
        token_spans: Vec<TokenSpanWork>,
        committed_pages: Vec<CommittedPageWork>,
    ) -> Result<Self, VNextError> {
        if token_spans.is_empty() {
            return Err(invalid_plan("resource work requires token evidence"));
        }
        let immediate_sequences = u32::try_from(token_spans.len())
            .map_err(|_| invalid_plan("resource work sequence count exceeds u32"))?;
        let immediate_tokens = token_spans.iter().try_fold(0_u64, |total, span| {
            total
                .checked_add(span.immediate_tokens())
                .ok_or_else(|| invalid_plan("resource work immediate tokens overflow u64"))
        })?;
        let fit_tokens = token_spans.iter().try_fold(0_u64, |total, span| {
            total
                .checked_add(span.full_input_tokens())
                .ok_or_else(|| invalid_plan("resource work full-input tokens overflow u64"))
        })?;
        let pages = committed_pages.iter().try_fold(0_u64, |total, page_work| {
            total
                .checked_add(page_work.pages())
                .ok_or_else(|| invalid_plan("resource work pages overflow u64"))
        })?;
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            token_spans: &'a [TokenSpanWork],
            committed_pages: &'a [CommittedPageWork],
        }
        let bytes = serde_json::to_vec(&FingerprintInput {
            domain: "ferrum.runtime-vnext.resource-work-shape.v1",
            token_spans: &token_spans,
            committed_pages: &committed_pages,
        })
        .map_err(|error| invalid_plan(format!("resource work encode failed: {error}")))?;
        Ok(Self {
            token_spans,
            committed_pages,
            immediate_sequences,
            immediate_tokens,
            immediate_pages: pages,
            fit_sequences: immediate_sequences,
            fit_tokens,
            fit_pages: pages,
            fingerprint: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    pub const fn immediate_sequences(&self) -> u32 {
        self.immediate_sequences
    }

    pub const fn immediate_tokens(&self) -> u64 {
        self.immediate_tokens
    }

    pub const fn immediate_pages(&self) -> u64 {
        self.immediate_pages
    }

    pub const fn fit_sequences(&self) -> u32 {
        self.fit_sequences
    }

    pub const fn fit_tokens(&self) -> u64 {
        self.fit_tokens
    }

    pub const fn fit_pages(&self) -> u64 {
        self.fit_pages
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(crate) const fn immediate_shape(&self) -> DynamicResourceShape {
        DynamicResourceShape::from_validated(
            self.immediate_sequences,
            self.immediate_tokens,
            self.immediate_pages,
        )
    }

    pub(crate) const fn fit_shape(&self) -> DynamicResourceShape {
        DynamicResourceShape::from_validated(self.fit_sequences, self.fit_tokens, self.fit_pages)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicResourceShapeBucket {
    pub(super) maximum_sequences: u32,
    pub(super) maximum_tokens: u64,
    pub(super) maximum_pages: u64,
    pub(super) bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DynamicResourceShapeBucketWire {
    pub(super) maximum_sequences: u32,
    pub(super) maximum_tokens: u64,
    pub(super) maximum_pages: u64,
    pub(super) bytes: u64,
}

impl DynamicResourceShapeBucket {
    pub fn new(
        maximum_sequences: u32,
        maximum_tokens: u64,
        maximum_pages: u64,
        bytes: u64,
    ) -> Result<Self, VNextError> {
        if maximum_sequences == 0 || maximum_tokens == 0 || maximum_pages == 0 || bytes == 0 {
            return Err(invalid_plan(
                "workspace shape bucket bounds and bytes must be non-zero",
            ));
        }
        Ok(Self {
            maximum_sequences,
            maximum_tokens,
            maximum_pages,
            bytes,
        })
    }

    pub(super) fn covers(&self, shape: DynamicResourceShape) -> bool {
        shape.sequences <= self.maximum_sequences
            && shape.tokens <= self.maximum_tokens
            && shape.pages <= self.maximum_pages
    }

    pub const fn maximum_sequences(&self) -> u32 {
        self.maximum_sequences
    }

    pub const fn maximum_tokens(&self) -> u64 {
        self.maximum_tokens
    }

    pub const fn maximum_pages(&self) -> u64 {
        self.maximum_pages
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl<'de> Deserialize<'de> for DynamicResourceShapeBucket {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicResourceShapeBucketWire::deserialize(deserializer)?;
        Self::new(
            wire.maximum_sequences,
            wire.maximum_tokens,
            wire.maximum_pages,
            wire.bytes,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Core-validated sizing formula supplied by a provider. Its maximum shape is
/// the boundary of one provider invocation, not the scheduler's global active
/// sequence ceiling; a scheduler may split larger ready sets into batches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicResourceDemand {
    Fixed {
        bytes: u64,
    },
    ActualSequences {
        bytes_per_sequence: u64,
        maximum_sequences: u32,
    },
    Tokens {
        bytes_per_token: u64,
        maximum_tokens: u64,
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
pub(super) enum DynamicResourceDemandWire {
    Fixed {
        bytes: u64,
    },
    ActualSequences {
        bytes_per_sequence: u64,
        maximum_sequences: u32,
    },
    Tokens {
        bytes_per_token: u64,
        maximum_tokens: u64,
    },
    Pages {
        bytes_per_page: u64,
        maximum_pages: u64,
    },
    BoundedShapeBuckets {
        buckets: Vec<DynamicResourceShapeBucket>,
    },
}

impl DynamicResourceDemand {
    pub fn fixed(bytes: u64) -> Result<Self, VNextError> {
        Self::validated(Self::Fixed { bytes })
    }

    pub fn actual_sequences(
        bytes_per_sequence: u64,
        maximum_sequences: u32,
    ) -> Result<Self, VNextError> {
        Self::validated(Self::ActualSequences {
            bytes_per_sequence,
            maximum_sequences,
        })
    }

    pub fn tokens(bytes_per_token: u64, maximum_tokens: u64) -> Result<Self, VNextError> {
        Self::validated(Self::Tokens {
            bytes_per_token,
            maximum_tokens,
        })
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

    pub(super) fn validated(demand: Self) -> Result<Self, VNextError> {
        demand.validate()?;
        Ok(demand)
    }

    pub(super) fn validate(&self) -> Result<(), VNextError> {
        let valid = match self {
            Self::Fixed { bytes } => *bytes > 0,
            Self::ActualSequences {
                bytes_per_sequence,
                maximum_sequences,
            } => {
                *bytes_per_sequence > 0
                    && *maximum_sequences > 0
                    && bytes_per_sequence
                        .checked_mul(u64::from(*maximum_sequences))
                        .is_some()
            }
            Self::Tokens {
                bytes_per_token,
                maximum_tokens,
            } => {
                *bytes_per_token > 0
                    && *maximum_tokens > 0
                    && bytes_per_token.checked_mul(*maximum_tokens).is_some()
            }
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
                        next.maximum_sequences >= previous.maximum_sequences
                            && next.maximum_tokens >= previous.maximum_tokens
                            && next.maximum_pages >= previous.maximum_pages
                            && (next.maximum_sequences > previous.maximum_sequences
                                || next.maximum_tokens > previous.maximum_tokens
                                || next.maximum_pages > previous.maximum_pages)
                            && next.bytes >= previous.bytes
                    })
            }
        };
        if !valid {
            return Err(invalid_plan(
                "dynamic resource formula is zero, overflowing, or non-canonical",
            ));
        }
        Ok(())
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
        self.validate()?;
        let bytes = match self {
            Self::Fixed { bytes } => *bytes,
            Self::ActualSequences {
                bytes_per_sequence,
                maximum_sequences,
            } if shape.sequences <= *maximum_sequences => bytes_per_sequence
                .checked_mul(u64::from(shape.sequences))
                .ok_or_else(|| invalid_plan("sequence-scaled resource request overflows u64"))?,
            Self::Tokens {
                bytes_per_token,
                maximum_tokens,
            } if shape.tokens <= *maximum_tokens => bytes_per_token
                .checked_mul(shape.tokens)
                .ok_or_else(|| invalid_plan("token-scaled resource request overflows u64"))?,
            Self::Pages {
                bytes_per_page,
                maximum_pages,
            } if shape.pages <= *maximum_pages => bytes_per_page
                .checked_mul(shape.pages)
                .ok_or_else(|| invalid_plan("page-scaled resource request overflows u64"))?,
            Self::BoundedShapeBuckets { buckets } => buckets
                .iter()
                .find(|bucket| bucket.covers(shape))
                .map(DynamicResourceShapeBucket::bytes)
                .ok_or_else(|| invalid_plan("actual invocation shape exceeds workspace buckets"))?,
            _ => {
                return Err(invalid_plan(
                    "actual invocation shape exceeds its bounded resource formula",
                ))
            }
        };
        if bytes == 0 {
            return Err(invalid_plan(
                "dynamic resource request evaluates to zero bytes",
            ));
        }
        Ok(bytes)
    }

    pub(crate) fn minimum_shape(&self) -> DynamicResourceShape {
        DynamicResourceShape {
            sequences: 1,
            tokens: 1,
            pages: 1,
        }
    }

    pub(crate) fn theoretical_maximum_shape(&self) -> DynamicResourceShape {
        match self {
            Self::Fixed { .. } => self.minimum_shape(),
            Self::ActualSequences {
                maximum_sequences, ..
            } => DynamicResourceShape {
                sequences: *maximum_sequences,
                tokens: 1,
                pages: 1,
            },
            Self::Tokens { maximum_tokens, .. } => DynamicResourceShape {
                sequences: 1,
                tokens: *maximum_tokens,
                pages: 1,
            },
            Self::Pages { maximum_pages, .. } => DynamicResourceShape {
                sequences: 1,
                tokens: 1,
                pages: *maximum_pages,
            },
            Self::BoundedShapeBuckets { buckets } => {
                let bucket = buckets
                    .last()
                    .expect("validated bounded formula has at least one bucket");
                DynamicResourceShape {
                    sequences: bucket.maximum_sequences,
                    tokens: bucket.maximum_tokens,
                    pages: bucket.maximum_pages,
                }
            }
        }
    }

    pub(super) fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed { .. })
    }

    pub(super) fn is_valid_for_sequence_scope(&self) -> bool {
        match self {
            Self::Fixed { .. } | Self::Tokens { .. } | Self::Pages { .. } => true,
            Self::BoundedShapeBuckets { buckets } => {
                buckets.iter().all(|bucket| bucket.maximum_sequences == 1)
            }
            Self::ActualSequences { .. } => false,
        }
    }
}

impl<'de> Deserialize<'de> for DynamicResourceDemand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let demand = match DynamicResourceDemandWire::deserialize(deserializer)? {
            DynamicResourceDemandWire::Fixed { bytes } => Self::Fixed { bytes },
            DynamicResourceDemandWire::ActualSequences {
                bytes_per_sequence,
                maximum_sequences,
            } => Self::ActualSequences {
                bytes_per_sequence,
                maximum_sequences,
            },
            DynamicResourceDemandWire::Tokens {
                bytes_per_token,
                maximum_tokens,
            } => Self::Tokens {
                bytes_per_token,
                maximum_tokens,
            },
            DynamicResourceDemandWire::Pages {
                bytes_per_page,
                maximum_pages,
            } => Self::Pages {
                bytes_per_page,
                maximum_pages,
            },
            DynamicResourceDemandWire::BoundedShapeBuckets { buckets } => {
                Self::BoundedShapeBuckets { buckets }
            }
        };
        Self::validated(demand).map_err(serde::de::Error::custom)
    }
}

pub type ProviderWorkspaceSizeFormula = DynamicResourceDemand;
