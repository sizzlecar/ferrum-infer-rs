//! Bounded Rust base predictors. Calibration, temporal eligibility and publication
//! belong to the learning owner. These fits alone are not client latency bounds.

mod boosted;
mod features;
mod physical;
#[cfg(test)]
mod tests;

pub use features::{
    FeatureDomain, Phase, ReuseKnowledge, WorkFeatures, FEATURE_COUNT, FEATURE_SCHEMA,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    #[error("invalid model configuration: {0}")]
    Configuration(&'static str),
    #[error("invalid training/model data: {0}")]
    InvalidData(&'static str),
    #[error("unsupported model domain")]
    UnsupportedDomain,
    #[error("model resource budget exhausted: {0}")]
    ResourceLimit(&'static str),
    #[error("model persistence: {0}")]
    Persistence(String),
}

/// Owner-local rejection facts. Feature coordinates and support bounds never
/// cross the neutral scheduler query interface.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PredictionRejection {
    ExecutionIdentityMismatch,
    FeatureDomainMismatch,
    OutsideObservedSupport,
    InvalidWorkFeatures(ModelError),
    InvalidBasePrediction(ModelError),
}

impl PredictionRejection {
    fn into_model_error(self) -> ModelError {
        match self {
            Self::ExecutionIdentityMismatch
            | Self::FeatureDomainMismatch
            | Self::OutsideObservedSupport => ModelError::UnsupportedDomain,
            Self::InvalidWorkFeatures(error) | Self::InvalidBasePrediction(error) => error,
        }
    }
}

/// Complete canonical assembly identity, not a hash or a model-name heuristic.
/// The owner must bind model/source, device, precision and executor configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct ExecutionIdentity(String);

impl TryFrom<String> for ExecutionIdentity {
    type Error = ModelError;

    fn try_from(identity: String) -> Result<Self, Self::Error> {
        Self::new(identity)
    }
}

impl ExecutionIdentity {
    pub fn new(identity: String) -> Result<Self, ModelError> {
        let result = Self(identity);
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> Result<(), ModelError> {
        let identity = &self.0;
        if identity.is_empty() || identity.len() > 64 * 1024 || identity.contains('\0') {
            return Err(ModelError::Configuration(
                "execution identity must be 1..=65536 bytes",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    Physical,
    Gbdt,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingSample {
    pub features: WorkFeatures,
    pub service_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasePrediction {
    pub central_seconds: f64,
    /// Calibration residual is actual - risk_base. Physical uses its center;
    /// GBDT uses its separate quantile head. Never add both as safety margins.
    pub risk_base_seconds: f64,
}

/// Explicit implementation resources, independent of the client's latency SLO.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingBudget {
    pub max_samples: usize,
    pub max_owned_bytes: usize,
    pub max_work_units: u64,
    pub max_trees: usize,
    pub max_depth: usize,
    pub split_candidates: usize,
    pub ridge_sweeps: usize,
    pub ridge_penalties: Vec<f64>,
    pub learning_rate: f64,
    pub risk_quantile: f64,
}

impl Default for TrainingBudget {
    fn default() -> Self {
        Self {
            max_samples: 4096,
            max_owned_bytes: 32 * 1024 * 1024,
            max_work_units: 50_000_000,
            max_trees: 32,
            max_depth: 3,
            split_candidates: 32,
            ridge_sweeps: 256,
            ridge_penalties: vec![0.0, 0.0001, 0.01, 1.0],
            learning_rate: 0.15,
            risk_quantile: 0.95,
        }
    }
}

impl TrainingBudget {
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.max_samples == 0
            || self.max_samples > 4096
            || self.max_owned_bytes == 0
            || self.max_owned_bytes > 32 * 1024 * 1024
            || self.max_work_units == 0
            || self.max_work_units > 500_000_000
            || self.max_trees == 0
            || self.max_trees > 32
            || self.max_depth > 3
            || self.split_candidates == 0
            || self.split_candidates > 32
            || self.ridge_sweeps == 0
            || self.ridge_sweeps > 256
            || self.ridge_penalties.is_empty()
            || self.ridge_penalties.len() > 4
            || self
                .ridge_penalties
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
            || !self.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate)
            || self.learning_rate == 0.0
            || !self.risk_quantile.is_finite()
            || self.risk_quantile <= 0.0
            || self.risk_quantile >= 1.0
        {
            return Err(ModelError::Configuration(
                "invalid training resource or objective bounds",
            ));
        }
        Ok(())
    }

    fn nodes_per_tree(&self) -> usize {
        (1 << (self.max_depth + 1)) - 1
    }

    /// Conservative bound on fit-owned Rust vector payloads plus vector headers.
    /// Borrowed input, allocator metadata, stack, retained snapshots and RSS are
    /// separate measurements; this is not a process-memory guarantee.
    pub fn owned_bytes_bound(&self, samples: usize) -> Result<usize, ModelError> {
        self.validate()?;
        let per_sample = std::mem::size_of::<ScaledSample>()
            + 4 * std::mem::size_of::<f64>()
            + (self.max_depth + 2)
                * (std::mem::size_of::<usize>() + 2 * std::mem::size_of::<f64>());
        let nodes = 2 * self.max_trees * self.nodes_per_tree();
        samples
            .checked_mul(per_sample)
            .and_then(|x| x.checked_add(nodes * std::mem::size_of::<boosted::Node>()))
            .and_then(|x| {
                x.checked_add(
                    2 * 64 * 1024
                        + 4096
                        + 2 * self.max_trees * std::mem::size_of::<Vec<boosted::Node>>(),
                )
            })
            .ok_or(ModelError::ResourceLimit("allocation extent"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Support {
    domain: FeatureDomain,
    minimum: [f64; FEATURE_COUNT],
    maximum: [f64; FEATURE_COUNT],
}

impl Support {
    fn include(&mut self, features: &WorkFeatures) {
        for (i, value) in features.values().iter().enumerate() {
            self.minimum[i] = self.minimum[i].min(*value);
            self.maximum[i] = self.maximum[i].max(*value);
        }
    }

    fn contains(&self, features: &WorkFeatures) -> bool {
        features.domain() == self.domain
            && features
                .values()
                .iter()
                .enumerate()
                .all(|(i, x)| x.is_finite() && *x >= self.minimum[i] && *x <= self.maximum[i])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Predictor {
    Physical(physical::Physical),
    Gbdt(boosted::Boosted),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "StoredModel")]
pub struct FittedModel {
    schema: u32,
    identity: ExecutionIdentity,
    support: Support,
    scales: [f64; FEATURE_COUNT],
    predictor: Predictor,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredModel {
    schema: u32,
    identity: ExecutionIdentity,
    support: Support,
    scales: [f64; FEATURE_COUNT],
    predictor: Predictor,
}

impl TryFrom<StoredModel> for FittedModel {
    type Error = ModelError;

    fn try_from(value: StoredModel) -> Result<Self, Self::Error> {
        let model = Self {
            schema: value.schema,
            identity: value.identity,
            support: value.support,
            scales: value.scales,
            predictor: value.predictor,
        };
        model.validate(&TrainingBudget::default())?;
        Ok(model)
    }
}

#[derive(Clone, Copy)]
struct ScaledSample {
    x: [f64; FEATURE_COUNT],
    y: f64,
}

struct WorkCounter {
    remaining: u64,
}

impl WorkCounter {
    fn charge(&mut self, units: usize) -> Result<(), ModelError> {
        self.remaining = self
            .remaining
            .checked_sub(units as u64)
            .ok_or(ModelError::ResourceLimit("training work"))?;
        Ok(())
    }
}

impl FittedModel {
    /// The owner supplies disjoint, time-ordered train/validation/calibration
    /// partitions. Calibration observations must not appear in these slices.
    pub fn train(
        family: ModelFamily,
        identity: ExecutionIdentity,
        training: &[TrainingSample],
        validation: &[TrainingSample],
        budget: &TrainingBudget,
    ) -> Result<Self, ModelError> {
        budget.validate()?;
        identity.validate()?;
        if training.len() < 2 || validation.is_empty() {
            return Err(ModelError::InvalidData(
                "training and validation must be populated",
            ));
        }
        let count = training
            .len()
            .checked_add(validation.len())
            .ok_or(ModelError::ResourceLimit("sample extent"))?;
        if count > budget.max_samples {
            return Err(ModelError::ResourceLimit("training samples"));
        }
        if budget.owned_bytes_bound(count)? > budget.max_owned_bytes {
            return Err(ModelError::ResourceLimit("training allocation"));
        }
        let domain = training[0].features.domain();
        for sample in training.iter().chain(validation) {
            sample.features.validate()?;
            if sample.features.domain() != domain {
                return Err(ModelError::UnsupportedDomain);
            }
            if !sample.service_seconds.is_finite()
                || sample.service_seconds < 0.0
                || sample.service_seconds > 86400.0
            {
                return Err(ModelError::InvalidData(
                    "service label must be finite seconds in 0..=86400",
                ));
            }
        }
        let mut support = Support {
            domain,
            minimum: [f64::INFINITY; FEATURE_COUNT],
            maximum: [0.0; FEATURE_COUNT],
        };
        for sample in training {
            support.include(&sample.features);
        }
        // Max scaling is stable even for highly collinear or widely varying work.
        let scales = support.maximum.map(|x| x.max(1.0));
        // Validation evaluates the frozen training scaling even outside the
        // fitting envelope. Its measured features may qualify the declared
        // support; unseen future positions remain outside that envelope.
        for sample in validation {
            support.include(&sample.features);
        }
        let scaled: Vec<_> = training
            .iter()
            .chain(validation)
            .map(|sample| ScaledSample {
                x: std::array::from_fn(|i| sample.features.values()[i] / scales[i]),
                y: sample.service_seconds,
            })
            .collect();
        let (training, validation) = scaled.split_at(training.len());
        let mut work = WorkCounter {
            remaining: budget.max_work_units,
        };
        let predictor = match family {
            ModelFamily::Physical => {
                Predictor::Physical(physical::train(training, validation, budget, &mut work)?)
            }
            ModelFamily::Gbdt => {
                Predictor::Gbdt(boosted::train(training, validation, budget, &mut work)?)
            }
        };
        let result = Self {
            schema: FEATURE_SCHEMA,
            identity,
            support,
            scales,
            predictor,
        };
        result.validate(budget)?;
        Ok(result)
    }

    pub fn domain(&self) -> FeatureDomain {
        self.support.domain
    }

    pub fn support_bounds(&self) -> (&[f64; FEATURE_COUNT], &[f64; FEATURE_COUNT]) {
        (&self.support.minimum, &self.support.maximum)
    }

    pub(crate) fn feature_schema(&self) -> u32 {
        self.schema
    }

    pub fn family(&self) -> ModelFamily {
        match self.predictor {
            Predictor::Physical(_) => ModelFamily::Physical,
            Predictor::Gbdt(_) => ModelFamily::Gbdt,
        }
    }

    pub fn risk_quantile(&self) -> Option<f64> {
        match &self.predictor {
            Predictor::Physical(_) => None,
            Predictor::Gbdt(model) => Some(model.quantile),
        }
    }

    pub fn predict(
        &self,
        identity: &ExecutionIdentity,
        features: &WorkFeatures,
    ) -> Result<BasePrediction, ModelError> {
        self.predict_detailed(identity, features)
            .map_err(PredictionRejection::into_model_error)
    }

    pub(crate) fn predict_detailed(
        &self,
        identity: &ExecutionIdentity,
        features: &WorkFeatures,
    ) -> Result<BasePrediction, PredictionRejection> {
        if identity != &self.identity {
            return Err(PredictionRejection::ExecutionIdentityMismatch);
        }
        if features.domain() != self.support.domain {
            return Err(PredictionRejection::FeatureDomainMismatch);
        }
        if features.values().iter().any(|x| !x.is_finite()) {
            // Previously rejected by Support::contains before model evaluation.
            return Err(PredictionRejection::InvalidWorkFeatures(
                ModelError::UnsupportedDomain,
            ));
        }
        if !self.support.contains(features) {
            return Err(PredictionRejection::OutsideObservedSupport);
        }
        self.predict_features_detailed(features)
    }

    /// Qualified out-of-fit observations may extend only the measured feature
    /// envelope. Their labels cannot alter fitting, hyperparameters or scaling.
    /// This envelope is an empirical support declaration, not proof that every
    /// unseen combination inside its box has calibrated conditional coverage.
    pub(crate) fn qualify_observed_features(
        &mut self,
        identity: &ExecutionIdentity,
        features: &WorkFeatures,
    ) -> Result<BasePrediction, ModelError> {
        let base = self.predict_observed_features(identity, features)?;
        self.support.include(features);
        Ok(base)
    }

    /// Copy only after a real observation has a valid fixed-model prediction.
    /// Existing published snapshots keep their original support declaration.
    pub(crate) fn qualify_observed_copy(
        &self,
        identity: &ExecutionIdentity,
        features: &WorkFeatures,
    ) -> Result<(Self, BasePrediction), ModelError> {
        let base = self.predict_observed_features(identity, features)?;
        let mut qualified = self.clone();
        qualified.support.include(features);
        Ok((qualified, base))
    }

    fn predict_observed_features(
        &self,
        identity: &ExecutionIdentity,
        features: &WorkFeatures,
    ) -> Result<BasePrediction, ModelError> {
        if identity != &self.identity || features.domain() != self.support.domain {
            return Err(ModelError::UnsupportedDomain);
        }
        self.predict_features(features)
    }

    fn predict_features(&self, features: &WorkFeatures) -> Result<BasePrediction, ModelError> {
        self.predict_features_detailed(features)
            .map_err(PredictionRejection::into_model_error)
    }

    fn predict_features_detailed(
        &self,
        features: &WorkFeatures,
    ) -> Result<BasePrediction, PredictionRejection> {
        features
            .validate()
            .map_err(PredictionRejection::InvalidWorkFeatures)?;
        let x = std::array::from_fn(|i| features.values()[i] / self.scales[i]);
        let result = match &self.predictor {
            Predictor::Physical(model) => {
                let center = model.predict(&x);
                BasePrediction {
                    central_seconds: center,
                    risk_base_seconds: center,
                }
            }
            Predictor::Gbdt(model) => model.predict(&x),
        };
        if !result.central_seconds.is_finite()
            || !result.risk_base_seconds.is_finite()
            || result.central_seconds < 0.0
            || result.risk_base_seconds < 0.0
        {
            return Err(PredictionRejection::InvalidBasePrediction(
                ModelError::InvalidData("invalid model prediction"),
            ));
        }
        Ok(result)
    }

    pub fn to_json(&self, budget: &TrainingBudget) -> Result<Vec<u8>, ModelError> {
        self.validate(budget)?;
        let bytes = serde_json::to_vec(self).map_err(|e| ModelError::Persistence(e.to_string()))?;
        if bytes.len() > budget.max_owned_bytes.min(1024 * 1024) {
            return Err(ModelError::ResourceLimit("persisted model bytes"));
        }
        Ok(bytes)
    }

    pub fn from_json(
        bytes: &[u8],
        expected: &ExecutionIdentity,
        budget: &TrainingBudget,
    ) -> Result<Self, ModelError> {
        budget.validate()?;
        // Reserve enough declared budget for the maximum permitted decoded tree
        // payload before serde allocates; string/input memory is separately bounded.
        if budget.owned_bytes_bound(0)? > budget.max_owned_bytes {
            return Err(ModelError::ResourceLimit("persisted model allocation"));
        }
        if bytes.len() > budget.max_owned_bytes.min(1024 * 1024) {
            return Err(ModelError::ResourceLimit("persisted model bytes"));
        }
        let result: Self =
            serde_json::from_slice(bytes).map_err(|e| ModelError::Persistence(e.to_string()))?;
        if &result.identity != expected {
            return Err(ModelError::UnsupportedDomain);
        }
        result.validate(budget)?;
        Ok(result)
    }

    fn validate(&self, budget: &TrainingBudget) -> Result<(), ModelError> {
        budget.validate()?;
        self.identity.validate()?;
        if self.schema != FEATURE_SCHEMA {
            return Err(ModelError::InvalidData("unsupported feature schema"));
        }
        for i in 0..FEATURE_COUNT {
            if !self.support.minimum[i].is_finite()
                || self.support.minimum[i] < 0.0
                || !self.support.maximum[i].is_finite()
                || self.support.maximum[i] < self.support.minimum[i]
                || !self.scales[i].is_finite()
                || self.scales[i] < 1.0
                || self.scales[i] > self.support.maximum[i].max(1.0)
            {
                return Err(ModelError::InvalidData(
                    "invalid persisted feature support/scales",
                ));
            }
        }
        match &self.predictor {
            Predictor::Physical(model) => model.validate(),
            Predictor::Gbdt(model) => model.validate(budget),
        }
    }
}
