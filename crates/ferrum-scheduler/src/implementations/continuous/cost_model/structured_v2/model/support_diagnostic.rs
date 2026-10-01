//! Optional input-only diagnostics. These values carry no membership authority
//! and are never included in source/profile records or model signatures.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredFitSupportReasonV1 {
    /// `support_axis` indexes StructuredInputV2::joint_support_coordinates().
    BelowMinimum {
        support_axis: usize,
        query: u64,
        minimum: u64,
        maximum: u64,
    },
    AboveAllFitMax {
        support_axis: usize,
        query: u64,
        minimum: u64,
        maximum: u64,
    },
    /// Every coordinate is inside its marginal range, but no one complete Fit
    /// point dominates the query. This bounded witness is the first blocking
    /// axis of the first original Fit point, not a global single-axis failure.
    NoJointDominator {
        support_axis: usize,
        query: u64,
        minimum: u64,
        maximum: u64,
        first_fit_point_upper: u64,
    },
    /// A row-space failure has no generally meaningful single basis axis.
    UnidentifiedDirection {
        basis_axes: usize,
        identified_rank: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredFitSupportDiagnosticV1 {
    /// First eight bytes of the original input domain digest, for log correlation
    /// only. This shortened value is never used for identity validation.
    pub input_domain_prefix: u64,
    /// Layout metadata to map support indices back to the frozen projection,
    /// without logging the full vector or per-algorithm catalogue.
    pub support_axes: usize,
    pub completion_support_offset: Option<usize>,
    pub reason: StructuredFitSupportReasonV1,
}

impl FittedStructuredModelV2 {
    /// Debug/worker use only: repeats the input-only membership checks without
    /// changing their result. It borrows the input and Fit support; only the
    /// existing row-space identification routine may allocate its scratch row.
    pub fn diagnose_service_fit_support(
        &self,
        input: &StructuredInputV2,
    ) -> Result<Option<StructuredFitSupportDiagnosticV1>> {
        if self.service_window.as_ref().is_none_or(|window| {
            window.contract.domain_policy != StructuredServiceDomainPolicyV1::FrozenFitSupportV1
        }) {
            return Err(StructuredUnknown::WrongProtocol);
        }
        self.exemplar.same_domain(input)?;
        input.validate(&self.settings)?;
        let (fit, support, _) = self.numerical.legacy()?;
        let reason = if let Some(reason) = support.diagnose(&input.support) {
            reason
        } else {
            match fit.identify(&input.basis) {
                Ok(()) => return Ok(None),
                Err(StructuredUnknown::UnidentifiedDirection) => {
                    StructuredFitSupportReasonV1::UnidentifiedDirection {
                        basis_axes: input.basis.len(),
                        identified_rank: fit.rank(),
                    }
                }
                Err(error) => return Err(error),
            }
        };
        Ok(Some(StructuredFitSupportDiagnosticV1 {
            input_domain_prefix: u64::from_be_bytes(input.domain[..8].try_into().unwrap()),
            support_axes: input.support.len(),
            completion_support_offset: input.completion.as_ref().map(|value| value.support_offset),
            reason,
        }))
    }
}

impl CalibratedStructuredModelV2 {
    /// Reports only the frozen Fit rejection. Residual-only support failures
    /// remain distinct and are not mislabelled OutsideFitSupport.
    pub fn diagnose_service_fit_support(
        &self,
        input: &StructuredInputV2,
    ) -> Result<Option<StructuredFitSupportDiagnosticV1>> {
        self.fitted.diagnose_service_fit_support(input)
    }
}
