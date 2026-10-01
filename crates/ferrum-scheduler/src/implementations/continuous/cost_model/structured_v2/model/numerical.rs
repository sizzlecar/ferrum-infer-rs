use super::super::nonnegative::FitCertificate;
use super::physical_envelope::PhysicalEnvelope;
use super::*;

pub(super) enum NumericalModel {
    RowSpace {
        fit: RowSpaceFit,
        support: JointSupport,
        generators: PendingGenerators,
    },
    PhysicalEnvelope(Box<PhysicalEnvelope>),
}
impl NumericalModel {
    pub fn fit(
        samples: &[StructuredNumericObservationV2],
        settings: &StructuredSettingsV2,
        exemplar: &StructuredInputV2,
        contract: Option<&NonNegativeEnvelopeContractV1>,
        certificate: Option<FitCertificate>,
    ) -> Result<Self> {
        if let Some(contract) = contract {
            return Ok(Self::PhysicalEnvelope(Box::new(PhysicalEnvelope::fit(
                contract.clone(),
                samples,
                settings,
                certificate,
            )?)));
        }
        if certificate.is_some() || !exemplar.cost_template_policy().is_ordered() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        let rows: Vec<_> = samples
            .iter()
            .map(|s| FitRow {
                basis: &s.input.basis,
                wall_ns: s.wall_ns,
            })
            .collect();
        let fit = RowSpaceFit::fit(&rows, settings)?;
        let support = JointSupport::new(samples.iter().map(|s| s.input.support.as_slice()))?;
        let generators = PendingGenerators::new(&fit, exemplar)?;
        Ok(Self::RowSpace {
            fit,
            support,
            generators,
        })
    }
    pub fn physical(&self) -> Option<&PhysicalEnvelope> {
        match self {
            Self::PhysicalEnvelope(value) => Some(value),
            _ => None,
        }
    }
    pub fn physical_mut(&mut self) -> Option<&mut PhysicalEnvelope> {
        match self {
            Self::PhysicalEnvelope(value) => Some(value),
            _ => None,
        }
    }
    pub fn legacy(&self) -> Result<(&RowSpaceFit, &JointSupport, &PendingGenerators)> {
        match self {
            Self::RowSpace {
                fit,
                support,
                generators,
            } => Ok((fit, support, generators)),
            _ => Err(StructuredUnknown::WrongProtocol),
        }
    }
    pub fn actual_upper(&self, input: &StructuredInputV2) -> Result<u64> {
        match self {
            Self::RowSpace { fit, support, .. } => {
                if !support.contains(&input.support) {
                    return Err(StructuredUnknown::JointSupport);
                }
                fit.predict(&input.basis)
            }
            Self::PhysicalEnvelope(value) => {
                value.membership(input)?;
                value.actual_upper(input)
            }
        }
    }
    pub fn bind_fit(&self, digest: &mut Sha256) {
        match self {
            Self::RowSpace { fit, support, .. } => {
                fit.bind_parameters(digest);
                support.bind_parameters(digest);
            }
            Self::PhysicalEnvelope(value) => value.bind_fit(digest),
        }
    }
    pub fn fit_error_floor_ns(&self) -> u64 {
        match self {
            Self::RowSpace { fit, .. } => fit.fit_error_floor_ns(),
            Self::PhysicalEnvelope(value) => value.fit_error_floor_ns(),
        }
    }
    pub fn rank(&self) -> usize {
        match self {
            Self::RowSpace { fit, .. } => fit.rank(),
            Self::PhysicalEnvelope(value) => value.rank(),
        }
    }
    pub fn retained_heap_bytes(&self) -> Option<usize> {
        match self {
            Self::RowSpace {
                fit,
                support,
                generators,
            } => fit
                .retained_heap_bytes()?
                .checked_add(support.retained_heap_bytes()?)?
                .checked_add(generators.retained_heap_bytes()?),
            Self::PhysicalEnvelope(value) => {
                std::mem::size_of::<PhysicalEnvelope>().checked_add(value.retained_heap_bytes()?)
            }
        }
    }
}
