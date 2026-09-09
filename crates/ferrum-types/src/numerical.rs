//! User-requested numerical execution policy. Device capabilities and model
//! family contracts resolve this request before any model execution begins.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

/// Stable profile identity, interpreted within a model family's declared
/// catalog. The profile's numerical ABI has a separate contract version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct NumericalProfileId(String);

impl NumericalProfileId {
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 160
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
            })
        {
            return Err("numerical profile identity needs 1..=160 portable ASCII bytes".to_owned());
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NumericalProfileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<String> for NumericalProfileId {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl FromStr for NumericalProfileId {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl From<NumericalProfileId> for String {
    fn from(value: NumericalProfileId) -> Self {
        value.0
    }
}

/// A request, never proof that a numerical profile can execute on a device.
/// `Require` must fail if its complete declared program cannot be supported.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalExecutionPolicy {
    #[default]
    Auto,
    Require(NumericalProfileId),
}

impl FromStr for NumericalExecutionPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            _ => Ok(Self::Require(value.parse()?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_policy_preserves_identity_through_config_and_cli_parsing() {
        let explicit: NumericalExecutionPolicy = "fixture.f32-master".parse().unwrap();
        assert_eq!(
            explicit,
            NumericalExecutionPolicy::Require(
                NumericalProfileId::new("fixture.f32-master").unwrap()
            )
        );
        let json = serde_json::to_vec(&explicit).unwrap();
        assert_eq!(
            serde_json::from_slice::<NumericalExecutionPolicy>(&json).unwrap(),
            explicit
        );
        assert_eq!(
            "auto".parse::<NumericalExecutionPolicy>().unwrap(),
            NumericalExecutionPolicy::Auto
        );
    }

    #[test]
    fn invalid_profile_ids_are_rejected_at_the_deserialization_boundary() {
        for value in ["", "fixture f16", "fixture\nf16", "配置.f16"] {
            assert!(value.parse::<NumericalExecutionPolicy>().is_err());
            let json = serde_json::json!({ "require": value });
            assert!(serde_json::from_value::<NumericalExecutionPolicy>(json).is_err());
        }
    }
}
