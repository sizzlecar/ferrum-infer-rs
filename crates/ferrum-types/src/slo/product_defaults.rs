//! Mode-dependent product defaults at the shared typed configuration boundary.
//! Presence is retained only during parsing. Runtime configuration is explicit;
//! no missing option can later turn a manual policy into automatic calibration.
use super::*;
use serde::{
    de::DeserializeOwned, de::MapAccess, de::SeqAccess, de::Visitor, Deserializer, Serializer,
};
use serde_json::{Map, Value};
use std::{collections::BTreeSet, fmt, marker::PhantomData};

/// The inner typed configuration remains the sole field/type validator. This
/// cold wire adapter records which options the user supplied without copying
/// the resource-limit schema or weakening its deny_unknown_fields contract.
struct Supplied<T> {
    value: T,
    fields: BTreeSet<String>,
}

/// Unlike plain Value, this temporary wire value preserves the original typed
/// parser's rejection of duplicate fields at every nested JSON table.
struct UniqueValue(Value);
impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = UniqueValue;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a configuration value without duplicate fields")
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Bool(value)))
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::from(value)))
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::from(value)))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|n| UniqueValue(Value::Number(n)))
                    .ok_or_else(|| E::custom("non-finite configuration number"))
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value.to_owned())))
            }
            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value)))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                self.visit_unit()
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
                UniqueValue::deserialize(d)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(UniqueValue(value)) = access.next_element()? {
                    values.push(value);
                }
                Ok(UniqueValue(Value::Array(values)))
            }
            fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
                let mut values = Map::new();
                while let Some((key, UniqueValue(value))) =
                    access.next_entry::<String, UniqueValue>()?
                {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!("duplicate field `{key}`")));
                    }
                }
                Ok(UniqueValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(ValueVisitor)
    }
}

impl<T: Default> Default for Supplied<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            fields: BTreeSet::new(),
        }
    }
}
impl<'de, T: DeserializeOwned> Deserialize<'de> for Supplied<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputVisitor<T>(PhantomData<T>);
        impl<'de, T: DeserializeOwned> Visitor<'de> for InputVisitor<T> {
            type Value = Supplied<T>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a typed configuration table")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
                let mut fields = BTreeSet::new();
                let mut values = Map::new();
                while let Some((key, UniqueValue(value))) =
                    access.next_entry::<String, UniqueValue>()?
                {
                    if !fields.insert(key.clone()) {
                        return Err(serde::de::Error::custom(format!("duplicate field `{key}`")));
                    }
                    values.insert(key, value);
                }
                let value = serde_json::from_value(Value::Object(values))
                    .map_err(serde::de::Error::custom)?;
                Ok(Supplied { value, fields })
            }
        }
        deserializer.deserialize_map(InputVisitor(PhantomData))
    }
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Input {
    mode: SloMode,
    experiment_stage: Option<SloExperimentStageV1>,
    default_service_class: Option<String>,
    services: Vec<ServiceSloConfig>,
    cost_profile: Option<PathBuf>,
    prefill_reference: Option<SloPrefillReferenceConfig>,
    cost_observation: Supplied<SloCostObservationConfig>,
    required_query_observation: SloRequiredQueryObservationConfig,
    planner: SloPlannerConfig,
    admission: SloAdmissionConfig,
    output: Supplied<SloOutputConfig>,
}

impl From<Input> for SloConfig {
    fn from(input: Input) -> Self {
        let Input {
            mode,
            experiment_stage,
            default_service_class,
            services,
            cost_profile,
            prefill_reference,
            cost_observation,
            required_query_observation,
            planner,
            admission,
            output,
        } = input;
        let Supplied {
            value: mut cost,
            fields: cost_fields,
        } = cost_observation;
        let explicit_automatic = matches!(
            cost.live_structured_calibration,
            SloLiveStructuredCalibration::AutomaticV1 { .. }
        );
        // Explicit artifacts, legacy/disabled selections and controlled
        // experiments keep their declared behavior. Existing RequireSlo uses
        // a strict imported-model contract and is not silently reinterpreted.
        let default_automatic = mode == SloMode::Enforce
            && experiment_stage.is_none()
            && admission.time_policy == SloTimeAdmissionPolicy::CompleteRequests
            && cost_profile.is_none()
            && prefill_reference.is_none()
            && cost.profile_export.is_none()
            && cost.model == SloCostModelConfig::default()
            && !cost_fields.contains("live_structured_calibration")
            && (!cost_fields.contains("predictor")
                || cost.predictor == SloCostPredictor::StructuredWholeWaveV2)
            && (!cost_fields.contains("structured_capture")
                || cost.structured_capture == SloStructuredCostCapture::HostSettledV1);
        if default_automatic {
            // The product preset uses the existing validated numerical and
            // population protocols. All resource, age and qualification bounds
            // remain the ordinary automatic defaults. Explicit AutomaticV1
            // settings retain their own historical/default interpretation.
            cost.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 {
                settings: SloAutomaticCalibrationSettingsV1 {
                    population_schedule:
                        SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2,
                    numerical_strategy:
                        SloAutomaticCalibrationNumericalStrategyV1::IdentifiedFitGlobalResidualV1,
                    ..Default::default()
                },
            };
        }
        if default_automatic || explicit_automatic {
            if !cost_fields.contains("predictor") {
                cost.predictor = SloCostPredictor::StructuredWholeWaveV2;
            }
            if !cost_fields.contains("structured_capture") {
                cost.structured_capture = SloStructuredCostCapture::HostSettledV1;
            }
            if !cost_fields.contains("structured_actual_capture") {
                cost.structured_actual_capture = SloStructuredActualCapturePolicy::ConsumerDrivenV1;
            }
        }
        let Supplied {
            value: mut output,
            fields: output_fields,
        } = output;
        if mode == SloMode::Enforce
            && experiment_stage.is_none()
            && !output_fields.contains("transport")
        {
            output.transport = SloOutputTransport::Credited;
        }
        Self {
            mode,
            experiment_stage,
            default_service_class,
            services,
            cost_profile,
            prefill_reference,
            cost_observation: cost,
            required_query_observation,
            planner,
            admission,
            output,
        }
    }
}

/// The standalone cost config preserves its historical compact wire format.
/// An effective SLO policy must spell out these mode-dependent choices so a
/// serialize/reload cycle cannot turn explicit Disabled/Legacy into omission.
pub(super) fn serialize_cost<S: Serializer>(
    cost: &SloCostObservationConfig,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // Keep native typed serialization (including TOML's Option handling).
    // Only append choices that the standalone cost serializer omits.
    #[derive(Serialize)]
    struct Effective<'a> {
        #[serde(flatten)]
        value: &'a SloCostObservationConfig,
        #[serde(skip_serializing_if = "Option::is_none")]
        predictor: Option<SloCostPredictor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        structured_capture: Option<SloStructuredCostCapture>,
        #[serde(skip_serializing_if = "Option::is_none")]
        structured_actual_capture: Option<SloStructuredActualCapturePolicy>,
        #[serde(skip_serializing_if = "Option::is_none")]
        live_structured_calibration: Option<&'a SloLiveStructuredCalibration>,
    }
    Effective {
        value: cost,
        predictor: (cost.predictor == SloCostPredictor::LegacyFeatureModel)
            .then_some(cost.predictor),
        structured_capture: cost
            .structured_capture
            .is_disabled()
            .then_some(cost.structured_capture),
        structured_actual_capture: cost
            .structured_actual_capture
            .is_legacy()
            .then_some(cost.structured_actual_capture),
        live_structured_calibration: cost
            .live_structured_calibration
            .is_disabled()
            .then_some(&cost.live_structured_calibration),
    }
    .serialize(serializer)
}

#[cfg(test)]
mod tests;
