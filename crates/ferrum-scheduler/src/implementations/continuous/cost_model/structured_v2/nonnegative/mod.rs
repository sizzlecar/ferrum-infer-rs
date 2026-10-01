//! Nonnegative empirical cost set, separate from the legacy row-space model.
//! Pure numerical evidence only: callers must validate the physical workload
//! domain and original phase population before giving these inputs authority.
use super::{
    QueryResult, Result, StructuredQueryFailureV2, StructuredSettingsV2, StructuredUnknown,
};
use serde::{Deserialize, Serialize};

mod certificates;
mod fit;
mod signed_certificates;
pub use signed_certificates::SignedBasisCertificateV1;
#[cfg(test)]
mod query_outcome_tests;
#[cfg(test)]
mod tests;

const FRACTION_BITS: u32 = 20;
const DENOMINATOR: u128 = 1 << FRACTION_BITS;
const SWEEPS: usize = 64;
const EXACT_INTEGER_LIMIT: u64 = 1 << 53;
const MAX_CERTIFICATES: usize = 8;
const REVISION: &[u8] = b"ferrum.nonnegative-physical-envelope.numeric.v1\0";

/// A visit is one row in one coordinate update; every fit does exactly 64
/// sweeps. Exceeding the declared budget rejects the whole population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeSettings {
    pub maximum_coordinate_visits: u64,
    pub maximum_query_certificates: usize,
}
impl Default for EnvelopeSettings {
    fn default() -> Self {
        Self {
            maximum_coordinate_visits: 32_000_000,
            maximum_query_certificates: MAX_CERTIFICATES,
        }
    }
}
impl EnvelopeSettings {
    pub(super) fn validate(self) -> Result<()> {
        if self.maximum_coordinate_visits == 0
            || self.maximum_coordinate_visits > 128_000_000
            || !(1..=MAX_CERTIFICATES).contains(&self.maximum_query_certificates)
        {
            return Err(StructuredUnknown::InvalidSettings);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FitSample<'a> {
    pub axes: &'a [u64],
    pub wall_ns: u64,
}

/// A wire representation, never trusted without replay against all original
/// Fit rows. u128 coefficients are represented as [low, high] u64 words.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitCertificate {
    pub input_digest: [u8; 32],
    pub column_maxima: Vec<u64>,
    pub coefficient_words: Vec<[u64; 2]>,
    pub epsilon_ns: u64,
    pub geometry_rank: usize,
    /// Present only for the explicitly declared identified envelope. The
    /// floating map proposes weights; exact integer dual checks authorize them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signed_basis: Option<SignedBasisCertificateV1>,
}

#[derive(Debug)]
struct UpperCertificate {
    axes: Vec<u128>,
    bound_ns: u128,
}
#[derive(Debug)]
pub(super) struct NonNegativeFit {
    frozen: FitCertificate,
    coefficients: Vec<u128>,
    certificates: Vec<UpperCertificate>,
    maximum_wave_ns: u64,
    signed_basis: Option<signed_certificates::SignedBasis>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CostEnvelope {
    /// A feasible model's rounded typical estimate, never a lower bound.
    pub typical_ns: u64,
    pub lower_ns: u64,
    /// Upper bound on the declared empirical coefficient set, not on hardware.
    pub upper_ns: u64,
}

/// Passive explanation of the same selected numerical certificate. It carries
/// no fitted model, source population, or prediction authority.
#[derive(Debug)]
pub(super) struct PredictionBoundDiagnostic<'a> {
    pub certificate_index: usize,
    pub upper_ns: u64,
    pub limiting_axis: usize,
    pub input_value: u64,
    pub fit_maximum: u64,
    pub certificate_axis: u128,
    pub certificate_bound_ns: u128,
    pub certificate_axes: &'a [u128],
}

impl NonNegativeFit {
    pub(super) fn fit(
        samples: &[FitSample<'_>],
        settings: &StructuredSettingsV2,
        envelope: EnvelopeSettings,
    ) -> Result<Self> {
        Self::fit_inner(samples, settings, envelope, false)
    }

    pub(super) fn fit_identified(
        samples: &[FitSample<'_>],
        settings: &StructuredSettingsV2,
        envelope: EnvelopeSettings,
    ) -> Result<Self> {
        Self::fit_inner(samples, settings, envelope, true)
    }

    fn fit_inner(
        samples: &[FitSample<'_>],
        settings: &StructuredSettingsV2,
        envelope: EnvelopeSettings,
        identified: bool,
    ) -> Result<Self> {
        let (maxima, geometry, digest) = fit::validate_population(samples, settings, envelope)?;
        let rank = geometry.basis.len();
        // Old contracts do not retain geometry while solving or allocate the
        // optional map/cache. New maps depend only on original Fit inputs.
        let signed_basis = if identified {
            signed_certificates::check_work(
                samples.len(),
                maxima.len(),
                rank,
                envelope.maximum_coordinate_visits,
            )?;
            Some(signed_certificates::propose(&geometry)?)
        } else {
            None
        };
        drop(geometry);
        let coefficients = fit::solve(samples, &maxima, settings, envelope)?;
        let epsilon_ns =
            certificates::verify_feasible(samples, &maxima, &coefficients, settings.max_wave_ns)?;
        let frozen = FitCertificate {
            input_digest: digest,
            column_maxima: maxima,
            coefficient_words: coefficients
                .iter()
                .map(|&q| [q as u64, (q >> 64) as u64])
                .collect(),
            epsilon_ns,
            geometry_rank: rank,
            signed_basis,
        };
        Self::build(samples, settings, envelope, frozen, coefficients)
    }

    /// Replays only exact feasibility and input geometry. The floating point
    /// optimizer is intentionally not rerun to recover its platform rounding.
    pub(super) fn from_certificate(
        samples: &[FitSample<'_>],
        settings: &StructuredSettingsV2,
        envelope: EnvelopeSettings,
        frozen: FitCertificate,
    ) -> Result<Self> {
        if frozen.signed_basis.is_some() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        Self::from_certificate_inner(samples, settings, envelope, frozen)
    }

    pub(super) fn from_certificate_identified(
        samples: &[FitSample<'_>],
        settings: &StructuredSettingsV2,
        envelope: EnvelopeSettings,
        frozen: FitCertificate,
    ) -> Result<Self> {
        if frozen.signed_basis.is_none() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        Self::from_certificate_inner(samples, settings, envelope, frozen)
    }

    fn from_certificate_inner(
        samples: &[FitSample<'_>],
        settings: &StructuredSettingsV2,
        envelope: EnvelopeSettings,
        frozen: FitCertificate,
    ) -> Result<Self> {
        let (maxima, geometry, digest) = fit::validate_population(samples, settings, envelope)?;
        let rank = geometry.basis.len();
        drop(geometry);
        if frozen.input_digest != digest
            || frozen.column_maxima != maxima
            || frozen.geometry_rank != rank
            || frozen.coefficient_words.len() != maxima.len()
        {
            return Err(StructuredUnknown::WrongSource);
        }
        if frozen.signed_basis.is_some() {
            // The declared ceiling applies to the original constructor too;
            // replay cannot admit a cache that the original work budget forbids.
            signed_certificates::check_work(
                samples.len(),
                maxima.len(),
                rank,
                envelope.maximum_coordinate_visits,
            )?;
        }
        let coefficients: Vec<_> = frozen
            .coefficient_words
            .iter()
            .map(|q| u128::from(q[0]) | (u128::from(q[1]) << 64))
            .collect();
        if certificates::verify_feasible(samples, &maxima, &coefficients, settings.max_wave_ns)?
            != frozen.epsilon_ns
        {
            return Err(StructuredUnknown::WrongSource);
        }
        Self::build(samples, settings, envelope, frozen, coefficients)
    }

    fn build(
        samples: &[FitSample<'_>],
        settings: &StructuredSettingsV2,
        envelope: EnvelopeSettings,
        frozen: FitCertificate,
        coefficients: Vec<u128>,
    ) -> Result<Self> {
        let certificates = certificates::build(samples, frozen.epsilon_ns, envelope)?;
        let signed_basis = frozen
            .signed_basis
            .as_ref()
            .map(|certificate| {
                signed_certificates::SignedBasis::replay(
                    samples,
                    &frozen.column_maxima,
                    frozen.geometry_rank,
                    frozen.epsilon_ns,
                    certificate,
                )
            })
            .transpose()?;
        Ok(Self {
            frozen,
            coefficients,
            certificates,
            maximum_wave_ns: settings.max_wave_ns,
            signed_basis,
        })
    }

    pub(super) fn certificate(&self) -> &FitCertificate {
        &self.frozen
    }

    /// Bounded offline explanation; never used by a prediction or a signature.
    #[cfg(test)]
    pub(super) fn diagnose_identified_bound(&self, input: &[u64]) -> serde_json::Value {
        use serde_json::json;
        let signed = match (&self.signed_basis, &self.frozen.signed_basis) {
            (Some(cache), Some(frozen)) => cache.diagnose_bound(
                input,
                &self.frozen.column_maxima,
                self.frozen.epsilon_ns,
                frozen,
            ),
            _ => json!({"upper_ns":null,"failure":"signed_basis_absent"}),
        };
        let signed_upper = signed["upper_ns"].as_u64();
        let mut positive: Option<(usize, u64)> = None;
        let mut valid_positive = 0;
        let mut unavailable_positive = 0;
        let mut overflow_positive = 0;
        for (index, certificate) in self.certificates.iter().enumerate() {
            match certificates::upper_bound(certificate, input) {
                Ok(Some(bound)) => {
                    valid_positive += 1;
                    if positive.is_none_or(|(_, prior)| bound < prior) {
                        positive = Some((index, bound));
                    }
                }
                Ok(None) => unavailable_positive += 1,
                Err(_) => overflow_positive += 1,
            }
        }
        let (winner, selected) = match (signed_upper, positive) {
            (Some(s), Some((_, p))) if s < p => ("signed", Some(s)),
            (Some(s), Some((_, p))) if s == p => ("tie", Some(s)),
            (_, Some((_, p))) => ("positive", Some(p)),
            (Some(s), None) => ("signed", Some(s)),
            (None, None) => ("unavailable", None),
        };
        let mut maximum_support: Option<(usize, u64, u64)> = None;
        let mut unseen = 0;
        let mut above_maximum = 0;
        for (axis, (&query, &maximum)) in input.iter().zip(&self.frozen.column_maxima).enumerate() {
            if maximum == 0 {
                unseen += usize::from(query != 0);
                continue;
            }
            above_maximum += usize::from(query > maximum);
            if maximum_support.is_none_or(|(_, q, m)| {
                u128::from(query) * u128::from(m) > u128::from(q) * u128::from(maximum)
            }) {
                maximum_support = Some((axis, query, maximum));
            }
        }
        json!({
            "rank": self.frozen.geometry_rank,
            "axes": self.frozen.column_maxima.len(),
            "epsilon_ns": self.frozen.epsilon_ns,
            "numerical_support_error": self.check_observed_axes_detailed(input).err().map(|e|format!("{e:?}")),
            "maximum_wave_ns": self.maximum_wave_ns,
            "positive_certificate_count": self.certificates.len(),
            "valid_positive_candidates": valid_positive,
            "unavailable_positive_candidates": unavailable_positive,
            "failed_positive_candidates": overflow_positive,
            "positive_winner": positive.map(|(index, upper)|json!({"certificate_index":index,"upper_ns":upper})),
            "winner": winner,
            "selected_upper_ns": selected,
            "signed": signed,
            "axes_above_fit_maximum": above_maximum,
            "unseen_positive_axes": unseen,
            "maximum_normalized_support": maximum_support.map(|(axis,q,m)|json!({
                "axis":axis,"input_value":q,"fit_maximum":m,"ratio":q as f64 / m as f64
            })),
        })
    }

    /// Cold diagnostic only. Reuses the actual certificate bound calculation;
    /// failure to explain it must never change the original query result.
    pub(super) fn diagnose_bound(&self, input: &[u64]) -> Option<PredictionBoundDiagnostic<'_>> {
        self.check_observed_axes(input).ok()?;
        let mut selected: Option<(usize, u64)> = None;
        for (index, certificate) in self.certificates.iter().enumerate() {
            if let Some(bound) = certificates::upper_bound(certificate, input).ok()? {
                if selected.is_none_or(|(_, previous)| bound < previous) {
                    selected = Some((index, bound));
                }
            }
        }
        let (certificate_index, upper_ns) = selected?;
        let certificate = &self.certificates[certificate_index];
        let (mut p, mut q, mut limiting_axis) = (0_u128, 1_u128, 0_usize);
        for (axis, (&x, &a)) in input.iter().zip(&certificate.axes).enumerate() {
            if a != 0 && u128::from(x).checked_mul(q)? > p.checked_mul(a)? {
                (p, q, limiting_axis) = (u128::from(x), a, axis);
            }
        }
        Some(PredictionBoundDiagnostic {
            certificate_index,
            upper_ns,
            limiting_axis,
            input_value: input[limiting_axis],
            fit_maximum: self.frozen.column_maxima[limiting_axis],
            certificate_axis: certificate.axes[limiting_axis],
            certificate_bound_ns: certificate.bound_ns,
            certificate_axes: &certificate.axes,
        })
    }

    /// Numerical support only. Physical membership must be proven separately.
    /// Neither measured walls, epsilon, nor prediction caps select membership.
    pub(super) fn check_observed_axes(&self, input: &[u64]) -> Result<()> {
        self.check_observed_axes_detailed(input)
            .map_err(StructuredQueryFailureV2::reason)
    }
    fn check_observed_axes_detailed(&self, input: &[u64]) -> QueryResult<()> {
        if input.len() != self.frozen.column_maxima.len()
            || input.first() != Some(&1)
            || input.iter().any(|&x| x > EXACT_INTEGER_LIMIT)
        {
            return Err(StructuredUnknown::InvalidInput.into());
        }
        if input
            .iter()
            .zip(&self.frozen.column_maxima)
            .any(|(&x, &scale)| x != 0 && scale == 0)
        {
            return Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::UnidentifiedDirection,
            ));
        }
        Ok(())
    }

    /// No allocation, factorization, source scan, or fit on the query path.
    /// Empirical point used only by the explicitly declared residual strategy.
    pub(super) fn predict_fitted(&self, input: &[u64]) -> Result<u64> {
        self.predict_fitted_detailed(input)
            .map_err(StructuredQueryFailureV2::reason)
    }
    pub(super) fn predict_fitted_detailed(&self, input: &[u64]) -> QueryResult<u64> {
        self.check_observed_axes_detailed(input)?;
        let point = self.fitted_point_unchecked(input)?;
        if point == 0 || point > self.maximum_wave_ns {
            return Err(StructuredQueryFailureV2::OutsidePredictionRange(
                StructuredUnknown::Capacity,
            ));
        }
        Ok(point)
    }

    fn fitted_point_unchecked(&self, input: &[u64]) -> QueryResult<u64> {
        let (_, scaled_upper) =
            certificates::scaled_dot(input, &self.frozen.column_maxima, &self.coefficients)?;
        u64::try_from(certificates::ceil_div(scaled_upper, DENOMINATOR)?).map_err(|_| {
            StructuredQueryFailureV2::OutsidePredictionRange(StructuredUnknown::Numerical)
        })
    }

    /// No allocation, factorization, source scan, or fit on the query path.
    pub(super) fn predict(&self, input: &[u64]) -> Result<CostEnvelope> {
        self.predict_detailed(input)
            .map_err(StructuredQueryFailureV2::reason)
    }
    pub(super) fn predict_detailed(&self, input: &[u64]) -> QueryResult<CostEnvelope> {
        self.check_observed_axes_detailed(input)?;
        let mut upper = None;
        for certificate in &self.certificates {
            if let Some(bound) = certificates::upper_bound(certificate, input)? {
                upper = Some(upper.map_or(bound, |prior: u64| prior.min(bound)));
            }
        }
        let upper_ns = upper.ok_or(StructuredQueryFailureV2::OutsidePredictionRange(
            StructuredUnknown::Capacity,
        ))?;
        if upper_ns == 0 || upper_ns > self.maximum_wave_ns {
            return Err(StructuredQueryFailureV2::OutsidePredictionRange(
                StructuredUnknown::Capacity,
            ));
        }
        let typical_ns = self.fitted_point_unchecked(input)?;
        if typical_ns > upper_ns.saturating_add(1) {
            return Err(StructuredUnknown::Numerical.into());
        }
        // Outward rounding of a feasible point may differ by a nanosecond.
        // It is a display estimate; cap to the independently certified upper.
        Ok(CostEnvelope {
            typical_ns: typical_ns.min(upper_ns),
            lower_ns: 0,
            upper_ns,
        })
    }

    /// Each optional candidate is verified by exact integer dual arithmetic.
    /// Neither the floating suggestion nor an approximate span test authorizes
    /// a cost. An unusable candidate leaves the original positive bounds intact.
    pub(super) fn predict_identified_envelope_detailed(
        &self,
        input: &[u64],
    ) -> QueryResult<CostEnvelope> {
        self.check_observed_axes_detailed(input)?;
        let (Some(cache), Some(frozen)) = (&self.signed_basis, &self.frozen.signed_basis) else {
            return Err(StructuredUnknown::WrongProtocol.into());
        };
        let mut upper = cache.upper_bound(
            input,
            &self.frozen.column_maxima,
            self.frozen.epsilon_ns,
            frozen,
        );
        for certificate in &self.certificates {
            // An overflowing *candidate* is not a corrupted source. Another
            // independently verified certificate can still authorize the input.
            if let Ok(Some(bound)) = certificates::upper_bound(certificate, input) {
                upper = Some(upper.map_or(bound, |prior| prior.min(bound)));
            }
        }
        let upper_ns = upper.ok_or(StructuredQueryFailureV2::OutsidePredictionRange(
            StructuredUnknown::Capacity,
        ))?;
        if upper_ns == 0 || upper_ns > self.maximum_wave_ns {
            return Err(StructuredQueryFailureV2::OutsidePredictionRange(
                StructuredUnknown::Capacity,
            ));
        }
        let typical_ns = self.fitted_point_unchecked(input)?;
        if typical_ns
            .checked_sub(upper_ns)
            .is_some_and(|delta| delta > 1)
        {
            return Err(StructuredUnknown::Numerical.into());
        }
        Ok(CostEnvelope {
            typical_ns: typical_ns.min(upper_ns),
            lower_ns: 0,
            upper_ns,
        })
    }

    pub(super) fn retained_heap_bytes(&self) -> Option<usize> {
        let mut bytes = self
            .frozen
            .column_maxima
            .capacity()
            .checked_mul(std::mem::size_of::<u64>())?
            .checked_add(
                self.frozen
                    .coefficient_words
                    .capacity()
                    .checked_mul(std::mem::size_of::<[u64; 2]>())?,
            )?
            .checked_add(
                self.coefficients
                    .capacity()
                    .checked_mul(std::mem::size_of::<u128>())?,
            )?
            .checked_add(
                self.certificates
                    .capacity()
                    .checked_mul(std::mem::size_of::<UpperCertificate>())?,
            )?;
        for certificate in &self.certificates {
            bytes = bytes.checked_add(
                certificate
                    .axes
                    .capacity()
                    .checked_mul(std::mem::size_of::<u128>())?,
            )?;
        }
        if let Some(certificate) = &self.frozen.signed_basis {
            bytes = bytes.checked_add(certificate.retained_heap_bytes()?)?;
        }
        if let Some(cache) = &self.signed_basis {
            bytes = bytes.checked_add(cache.retained_heap_bytes()?)?;
        }
        Some(bytes)
    }

    pub(super) fn bind_parameters(&self, digest: &mut sha2::Sha256) {
        use sha2::Digest;
        digest.update(REVISION);
        digest.update(self.frozen.input_digest);
        digest.update(self.frozen.epsilon_ns.to_le_bytes());
        digest.update((self.frozen.geometry_rank as u64).to_le_bytes());
        digest.update((self.frozen.column_maxima.len() as u64).to_le_bytes());
        for (&scale, &coefficient) in self.frozen.column_maxima.iter().zip(&self.coefficients) {
            digest.update(scale.to_le_bytes());
            digest.update(coefficient.to_le_bytes());
        }
        if let Some(certificate) = &self.frozen.signed_basis {
            certificate.bind_parameters(digest);
        }
    }
}
