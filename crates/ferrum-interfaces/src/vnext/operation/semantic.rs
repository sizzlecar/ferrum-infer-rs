use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

use super::super::VNextError;

/// Stable semantic attribute identity. Attribute names are data, not ad-hoc
/// strings interpreted by an individual provider.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AttributeId(String);

impl AttributeId {
    pub fn new(value: impl Into<String>) -> Result<Self, VNextError> {
        let value = value.into();
        if value.is_empty() || value.len() > 160 {
            return Err(VNextError::InvalidIdentity {
                kind: "operation attribute",
                value,
                reason: "identity must contain between 1 and 160 bytes",
            });
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
        }) {
            return Err(VNextError::InvalidIdentity {
                kind: "operation attribute",
                value,
                reason: "identity contains a non-portable character",
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AttributeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<String> for AttributeId {
    type Error = VNextError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<AttributeId> for String {
    fn from(value: AttributeId) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeValueKind {
    Bool,
    Integer,
    Unsigned,
    Rational,
    Text,
    Integers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct CanonicalRational {
    numerator: i64,
    denominator: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalRationalWire {
    numerator: i64,
    denominator: u64,
}

impl<'de> Deserialize<'de> for CanonicalRational {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CanonicalRationalWire::deserialize(deserializer)?;
        Self::new(wire.numerator, wire.denominator).map_err(serde::de::Error::custom)
    }
}

impl CanonicalRational {
    pub fn new(numerator: i64, denominator: u64) -> Result<Self, VNextError> {
        if denominator == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "rational denominator must be non-zero".to_owned(),
            });
        }
        let divisor = gcd_u64(numerator.unsigned_abs(), denominator);
        let denominator = denominator / divisor;
        let reduced = i128::from(numerator) / i128::from(divisor);
        let numerator = i64::try_from(reduced).map_err(|_| VNextError::InvalidExecutionPlan {
            reason: "canonical rational numerator overflows i64".to_owned(),
        })?;
        Ok(Self {
            numerator,
            denominator,
        })
    }

    /// Parses a finite base-10 decimal or scientific-notation value without
    /// routing through binary floating point.
    pub fn from_decimal_str(raw: &str) -> Result<Self, VNextError> {
        let normalized = raw.to_ascii_lowercase();
        let (mantissa, exponent) = match normalized.split_once('e') {
            Some((mantissa, exponent)) => (
                mantissa,
                exponent.parse::<i32>().map_err(|error| {
                    invalid_decimal_rational(format!("invalid decimal exponent: {error}"))
                })?,
            ),
            None => (normalized.as_str(), 0),
        };
        let (negative, mantissa) = if let Some(unsigned) = mantissa.strip_prefix('-') {
            (true, unsigned)
        } else {
            (false, mantissa.strip_prefix('+').unwrap_or(mantissa))
        };
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if whole.is_empty()
            || !whole.bytes().all(|byte| byte.is_ascii_digit())
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(invalid_decimal_rational(format!(
                "invalid decimal rational {raw:?}"
            )));
        }

        let digits = format!("{whole}{fraction}");
        let mut magnitude = digits.parse::<u128>().map_err(|error| {
            invalid_decimal_rational(format!("decimal numerator overflows: {error}"))
        })?;
        let fractional_digits = i32::try_from(fraction.len()).map_err(|_| {
            invalid_decimal_rational("decimal rational has too many fractional digits")
        })?;
        let scale = fractional_digits
            .checked_sub(exponent)
            .ok_or_else(|| invalid_decimal_rational("decimal rational exponent overflows"))?;
        let denominator = if scale >= 0 {
            10_u128
                .checked_pow(scale as u32)
                .ok_or_else(|| invalid_decimal_rational("decimal rational denominator overflows"))?
        } else {
            magnitude = magnitude
                .checked_mul(10_u128.checked_pow(scale.unsigned_abs()).ok_or_else(|| {
                    invalid_decimal_rational("decimal rational numerator scale overflows")
                })?)
                .ok_or_else(|| invalid_decimal_rational("decimal rational numerator overflows"))?;
            1
        };
        let signed = if negative {
            -(i128::try_from(magnitude)
                .map_err(|_| invalid_decimal_rational("decimal rational numerator exceeds i128"))?)
        } else {
            i128::try_from(magnitude)
                .map_err(|_| invalid_decimal_rational("decimal rational numerator exceeds i128"))?
        };
        let numerator = i64::try_from(signed)
            .map_err(|_| invalid_decimal_rational("decimal rational numerator exceeds i64"))?;
        let denominator = u64::try_from(denominator)
            .map_err(|_| invalid_decimal_rational("decimal rational denominator exceeds u64"))?;
        Self::new(numerator, denominator)
    }

    pub const fn numerator(self) -> i64 {
        self.numerator
    }

    pub const fn denominator(self) -> u64 {
        self.denominator
    }
}

fn invalid_decimal_rational(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticValue {
    Bool(bool),
    Integer(i64),
    Unsigned(u64),
    Rational(CanonicalRational),
    Text(String),
    Integers(Vec<i64>),
}

impl SemanticValue {
    pub const fn kind(&self) -> AttributeValueKind {
        match self {
            Self::Bool(_) => AttributeValueKind::Bool,
            Self::Integer(_) => AttributeValueKind::Integer,
            Self::Unsigned(_) => AttributeValueKind::Unsigned,
            Self::Rational(_) => AttributeValueKind::Rational,
            Self::Text(_) => AttributeValueKind::Text,
            Self::Integers(_) => AttributeValueKind::Integers,
        }
    }

    pub fn validate(&self, context: &str) -> Result<(), VNextError> {
        match self {
            Self::Text(value) if value.is_empty() => Err(VNextError::InvalidExecutionPlan {
                reason: format!("{context} contains an empty text attribute"),
            }),
            Self::Integers(values) if values.is_empty() => Err(VNextError::InvalidExecutionPlan {
                reason: format!("{context} contains an empty integer-list attribute"),
            }),
            _ => Ok(()),
        }
    }
}
