use super::super::slo_planner::{interpolate_reference_work, ReferenceWorkEvaluation};
use super::*;

/// An explicit reference partition, not a list of permitted executor shapes.
/// The longest real input supplies B; each real input supplies its final F.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PiecewiseReferenceSpec {
    pub minimum_prompt_tokens: NonZeroU32,
    pub maximum_prompt_tokens: NonZeroU32,
    #[serde(deserialize_with = "endpoints")]
    pub body_endpoints: Vec<NonZeroU32>,
}
fn endpoints<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<NonZeroU32>, D::Error> {
    bounded_vec::<D, NonZeroU32, PREFILL_REFERENCE_MAX_POINTS_PER_CURVE>(d)
}
impl PiecewiseReferenceSpec {
    pub fn validate(&self) -> Result<(), ReferenceError> {
        let maximum = self.maximum_prompt_tokens.get();
        if self.minimum_prompt_tokens > self.maximum_prompt_tokens
            || self.body_endpoints.len() + 2 > PREFILL_REFERENCE_MAX_POINTS_PER_CURVE
            || self.body_endpoints.windows(2).any(|p| p[0] >= p[1])
            || (maximum == 1 && !self.body_endpoints.is_empty())
            || (maximum > 1 && self.body_endpoints.last().map(|p| p.get()) != Some(maximum - 1))
        {
            return Err(ReferenceError::Evidence(
                "invalid piecewise domain or body partition",
            ));
        }
        Ok(())
    }
    /// Every declared curve uses the same absolute prefix partition, truncated
    /// at N-1, followed by a real final one-token wave. Never narrows a receipt.
    pub fn next_count(&self, total: u32, offset: u32) -> Result<NonZeroU32, ReferenceError> {
        if total < self.minimum_prompt_tokens.get()
            || total > self.maximum_prompt_tokens.get()
            || offset >= total
        {
            return Err(ReferenceError::Evidence(
                "piecewise input outside declared domain",
            ));
        }
        if offset == total - 1 {
            return Ok(NonZeroU32::MIN);
        }
        let index = self.body_endpoints.partition_point(|p| p.get() <= offset);
        let end = self
            .body_endpoints
            .get(index)
            .map(|p| p.get())
            .unwrap_or(total - 1)
            .min(total - 1);
        NonZeroU32::new(end.checked_sub(offset).ok_or(ReferenceError::Overflow)?)
            .ok_or(ReferenceError::Evidence("empty piecewise segment"))
    }
    pub fn segment_count(&self, total: u32) -> Result<usize, ReferenceError> {
        self.next_count(total, 0)?;
        Ok(if total == 1 {
            1
        } else {
            self.body_endpoints.partition_point(|p| p.get() < total - 1) + 2
        })
    }
    pub fn protocol_sha256(
        &self,
        protocol: &ReferenceProtocolV1,
    ) -> Result<[u8; 32], ReferenceError> {
        let mut hash = Sha256::new();
        hash.update(b"ferrum.prefill-reference-protocol.v2.prefix-final-integer-linear\0");
        hash.update(serde_json::to_vec(&(protocol, self))?);
        Ok(hash.finalize().into())
    }
}

/// V2 has a distinct wire schema. The shared protocol declares measurements;
/// piecewise declares how their frozen scoring units cover unmeasured N/p.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCalibrationV2 {
    pub schema_version: u32,
    pub reference_revision: NonZeroU64,
    pub fingerprint: ProfileFingerprint,
    pub generated_unix_ns: NonZeroU64,
    pub protocol: ReferenceProtocolV1,
    pub piecewise: PiecewiseReferenceSpec,
    #[serde(deserialize_with = "super::wire::repetitions")]
    pub decode_samples: Vec<ReferenceObservedSample>,
    #[serde(deserialize_with = "super::wire::curves")]
    pub curves: Vec<ReferenceCurveInput>,
}
impl ReferenceCalibrationV2 {
    pub(super) fn from_evidence(
        value: ReferenceCalibrationV1,
        piecewise: PiecewiseReferenceSpec,
    ) -> Self {
        Self {
            schema_version: PREFILL_REFERENCE_SCHEMA_V2,
            reference_revision: value.reference_revision,
            fingerprint: value.fingerprint,
            generated_unix_ns: value.generated_unix_ns,
            protocol: value.protocol,
            piecewise,
            decode_samples: value.decode_samples,
            curves: value.curves,
        }
    }
    pub(super) fn into_evidence(
        self,
    ) -> Result<(ReferenceCalibrationV1, PiecewiseReferenceSpec), ReferenceError> {
        if self.schema_version != PREFILL_REFERENCE_SCHEMA_V2 {
            return Err(ReferenceError::Schema(self.schema_version));
        }
        Ok((
            ReferenceCalibrationV1 {
                schema_version: PREFILL_REFERENCE_SCHEMA_V1,
                reference_revision: self.reference_revision,
                fingerprint: self.fingerprint,
                generated_unix_ns: self.generated_unix_ns,
                protocol: self.protocol,
                decode_samples: self.decode_samples,
                curves: self.curves,
            },
            self.piecewise,
        ))
    }
}

#[derive(Debug)]
pub(super) struct PiecewiseDefinition {
    pub spec: PiecewiseReferenceSpec,
    body: Arc<[ReferenceWorkPoint]>,
    final_work: Vec<ReferenceWorkPoint>,
}
impl PiecewiseDefinition {
    pub fn compile(
        spec: &PiecewiseReferenceSpec,
        curves: &BTreeMap<u32, Arc<PrefillReferenceWork>>,
    ) -> Result<Self, ReferenceError> {
        spec.validate()?;
        if curves.keys().next().copied() != Some(spec.minimum_prompt_tokens.get())
            || curves.keys().next_back().copied() != Some(spec.maximum_prompt_tokens.get())
        {
            return Err(ReferenceError::Evidence(
                "piecewise domain lacks actual terminal boundary anchors",
            ));
        }
        let longest = &curves[&spec.maximum_prompt_tokens.get()];
        let body: Arc<[ReferenceWorkPoint]> =
            Arc::from(&longest.points[..longest.points.len() - 1]);
        // A nanosecond score cannot represent strictly useful integer progress
        // at a finer resolution. Reject it; never insert invented 1ns work.
        if body.windows(2).any(|p| {
            p[1].cumulative_work_ns - p[0].cumulative_work_ns
                < u64::from(p[1].prompt_tokens - p[0].prompt_tokens)
        }) {
            return Err(ReferenceError::Evidence(
                "piecewise work resolution is insufficient",
            ));
        }
        let mut final_work = Vec::with_capacity(curves.len());
        for (&n, curve) in curves {
            let tail = &curve.points[curve.points.len() - 2..];
            if tail[0].prompt_tokens != n - 1 {
                return Err(ReferenceError::Evidence(
                    "terminal anchor is not final one-token work",
                ));
            }
            final_work.push(ReferenceWorkPoint {
                prompt_tokens: n,
                cumulative_work_ns: tail[1]
                    .cumulative_work_ns
                    .checked_sub(tail[0].cumulative_work_ns)
                    .filter(|v| *v > 0)
                    .ok_or(ReferenceError::Overflow)?,
            });
        }
        Ok(Self {
            spec: spec.clone(),
            body,
            final_work,
        })
    }
    pub fn bind(
        &self,
        version: u64,
        total: NonZeroU32,
    ) -> Result<Arc<PrefillReferenceWork>, ReferenceUnknown> {
        let n = total.get();
        if total < self.spec.minimum_prompt_tokens || total > self.spec.maximum_prompt_tokens {
            return Err(ReferenceUnknown::LengthNotCalibrated);
        }
        let prefix = interpolate_reference_work(&self.body, n - 1)
            .ok_or(ReferenceUnknown::MissingEndpoint)?;
        let final_work = interpolate_reference_work(&self.final_work, n)
            .ok_or(ReferenceUnknown::MissingEndpoint)?;
        let mut points: Vec<_> = self
            .body
            .iter()
            .copied()
            .take_while(|p| p.prompt_tokens < n - 1)
            .collect();
        points.push(ReferenceWorkPoint {
            prompt_tokens: n - 1,
            cumulative_work_ns: prefix,
        });
        points.push(ReferenceWorkPoint {
            prompt_tokens: n,
            cumulative_work_ns: prefix
                .checked_add(final_work)
                .ok_or(ReferenceUnknown::ArithmeticOverflow)?,
        });
        Ok(Arc::new(PrefillReferenceWork {
            version,
            points,
            evaluation: ReferenceWorkEvaluation::PiecewiseV2 {
                body: Arc::clone(&self.body),
            },
        }))
    }
}
