//! Standard request controls and model-declared reasoning effort support.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

/// Standard reasoning effort vocabulary. A request's omitted or null effort is
/// represented by `Option::None`, separately from the explicit `"none"` value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ReasoningEffort {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            _ => Err(format!(
                "unsupported reasoning effort {value:?}; expected none, minimal, low, medium, high, xhigh, or max"
            )),
        }
    }
}

impl<'de> Deserialize<'de> for ReasoningEffort {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// An explicit model contract, independent of its output parser or whether a
/// template happens to interpolate an effort variable. Unknown support must not
/// be treated as a declaration that a requested effort is unsupported.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ReasoningEffortSupport {
    #[default]
    Unknown,
    /// Exhaustive supported values. An empty set explicitly supports no effort
    /// controls; omission of a declaration is represented by `Unknown` instead.
    Declared(BTreeSet<ReasoningEffort>),
}

impl ReasoningEffortSupport {
    pub fn supports(&self, effort: ReasoningEffort) -> Option<bool> {
        self.declared_efforts()
            .map(|efforts| efforts.contains(&effort))
    }

    pub fn declared_efforts(&self) -> Option<&BTreeSet<ReasoningEffort>> {
        match self {
            Self::Unknown => None,
            Self::Declared(efforts) => Some(efforts),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_wire_values_agree_with_text_parsing_and_display() {
        for (wire, effort) in [
            ("none", ReasoningEffort::None),
            ("minimal", ReasoningEffort::Minimal),
            ("low", ReasoningEffort::Low),
            ("medium", ReasoningEffort::Medium),
            ("high", ReasoningEffort::High),
            ("xhigh", ReasoningEffort::XHigh),
            ("max", ReasoningEffort::Max),
        ] {
            assert_eq!(wire.parse::<ReasoningEffort>().unwrap(), effort);
            assert_eq!(effort.to_string(), wire);
            let value = serde_json::Value::String(wire.to_owned());
            assert_eq!(serde_json::to_value(effort).unwrap(), value);
            assert_eq!(
                serde_json::from_value::<ReasoningEffort>(value).unwrap(),
                effort
            );
        }
    }

    #[test]
    fn effort_rejects_nonstandard_values_and_nonstring_wire_types() {
        for value in ["", "auto", "LOW", " high ", "unlimited"] {
            assert!(value.parse::<ReasoningEffort>().is_err());
            assert!(serde_json::from_value::<ReasoningEffort>(serde_json::json!(value)).is_err());
        }
        for value in [
            serde_json::json!(false),
            serde_json::json!(1),
            serde_json::json!({}),
            serde_json::json!({"high": null}),
        ] {
            assert!(serde_json::from_value::<ReasoningEffort>(value).is_err());
        }
    }

    #[test]
    fn optional_null_is_distinct_from_explicit_none() {
        assert_eq!(
            serde_json::from_value::<Option<ReasoningEffort>>(serde_json::Value::Null).unwrap(),
            None
        );
        assert_eq!(
            serde_json::from_value::<Option<ReasoningEffort>>(serde_json::json!("none")).unwrap(),
            Some(ReasoningEffort::None)
        );
    }

    #[test]
    fn unknown_support_is_distinct_from_an_exhaustive_declaration() {
        let unknown = ReasoningEffortSupport::default();
        assert_eq!(unknown.supports(ReasoningEffort::None), None);
        assert_eq!(unknown.supports(ReasoningEffort::High), None);
        assert_eq!(unknown.declared_efforts(), None);

        let declared = ReasoningEffortSupport::Declared(BTreeSet::from([ReasoningEffort::High]));
        assert_eq!(declared.supports(ReasoningEffort::High), Some(true));
        assert_eq!(declared.supports(ReasoningEffort::None), Some(false));
        assert_eq!(
            declared.declared_efforts(),
            Some(&BTreeSet::from([ReasoningEffort::High]))
        );
        let unsupported = ReasoningEffortSupport::Declared(BTreeSet::new());
        assert_eq!(unsupported.supports(ReasoningEffort::High), Some(false));
    }
}
