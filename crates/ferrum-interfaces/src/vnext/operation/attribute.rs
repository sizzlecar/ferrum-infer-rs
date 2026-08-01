use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize};

use super::super::VNextError;
use super::foundation::invalid_operation;
use super::{AttributeId, AttributeValueKind, CanonicalRational, SemanticValue};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributeSpec {
    pub value_kind: AttributeValueKind,
    pub required: bool,
    pub constraint: AttributeConstraint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeConstraint {
    None,
    BoolEquals(bool),
    IntegerRange {
        minimum: i64,
        maximum: i64,
    },
    UnsignedRange {
        minimum: u64,
        maximum: u64,
    },
    RationalRange {
        minimum: CanonicalRational,
        maximum: CanonicalRational,
    },
    TextChoices {
        values: BTreeSet<String>,
    },
    IntegerListLength {
        minimum: u32,
        maximum: u32,
    },
}

/// Closed attribute vocabulary for one operation contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AttributeSchema {
    entries: BTreeMap<AttributeId, AttributeSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AttributeSchemaWire {
    entries: BTreeMap<AttributeId, AttributeSpec>,
}

impl<'de> Deserialize<'de> for AttributeSchema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AttributeSchemaWire::deserialize(deserializer)?;
        Self::new(wire.entries).map_err(serde::de::Error::custom)
    }
}

impl AttributeSchema {
    pub fn new(entries: BTreeMap<AttributeId, AttributeSpec>) -> Result<Self, VNextError> {
        for (attribute_id, spec) in &entries {
            spec.validate(attribute_id)?;
        }
        Ok(Self { entries })
    }

    pub fn empty() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn entries(&self) -> &BTreeMap<AttributeId, AttributeSpec> {
        &self.entries
    }

    pub fn validate_values(
        &self,
        values: &BTreeMap<AttributeId, SemanticValue>,
        context: &str,
    ) -> Result<(), VNextError> {
        for (attribute_id, value) in values {
            let spec = self.entries.get(attribute_id).ok_or_else(|| {
                invalid_operation(format!(
                    "{context} contains unknown attribute `{attribute_id}`"
                ))
            })?;
            value.validate(context)?;
            if value.kind() != spec.value_kind {
                return Err(invalid_operation(format!(
                    "{context} attribute `{attribute_id}` has the wrong value kind"
                )));
            }
            spec.validate_value(attribute_id, value)?;
        }
        if let Some(attribute_id) = self.entries.iter().find_map(|(attribute_id, spec)| {
            (spec.required && !values.contains_key(attribute_id)).then_some(attribute_id)
        }) {
            return Err(invalid_operation(format!(
                "{context} is missing required attribute `{attribute_id}`"
            )));
        }
        Ok(())
    }
}

impl AttributeSpec {
    fn validate(&self, attribute_id: &AttributeId) -> Result<(), VNextError> {
        let compatible = match (&self.value_kind, &self.constraint) {
            (_, AttributeConstraint::None) => true,
            (AttributeValueKind::Bool, AttributeConstraint::BoolEquals(_)) => true,
            (
                AttributeValueKind::Integer,
                AttributeConstraint::IntegerRange { minimum, maximum },
            ) => minimum <= maximum,
            (
                AttributeValueKind::Unsigned,
                AttributeConstraint::UnsignedRange { minimum, maximum },
            ) => minimum <= maximum,
            (AttributeValueKind::Text, AttributeConstraint::TextChoices { values }) => {
                !values.is_empty() && values.iter().all(|value| !value.is_empty())
            }
            (
                AttributeValueKind::Integers,
                AttributeConstraint::IntegerListLength { minimum, maximum },
            ) => minimum <= maximum,
            (
                AttributeValueKind::Rational,
                AttributeConstraint::RationalRange { minimum, maximum },
            ) => {
                (minimum.numerator() as i128) * (maximum.denominator() as i128)
                    <= (maximum.numerator() as i128) * (minimum.denominator() as i128)
            }
            _ => false,
        };
        if !compatible {
            return Err(invalid_operation(format!(
                "attribute `{attribute_id}` has an incompatible or invalid constraint"
            )));
        }
        Ok(())
    }

    fn validate_value(
        &self,
        attribute_id: &AttributeId,
        value: &SemanticValue,
    ) -> Result<(), VNextError> {
        let accepted = match (&self.constraint, value) {
            (AttributeConstraint::None, _) => true,
            (AttributeConstraint::BoolEquals(expected), SemanticValue::Bool(actual)) => {
                expected == actual
            }
            (
                AttributeConstraint::IntegerRange { minimum, maximum },
                SemanticValue::Integer(actual),
            ) => minimum <= actual && actual <= maximum,
            (
                AttributeConstraint::UnsignedRange { minimum, maximum },
                SemanticValue::Unsigned(actual),
            ) => minimum <= actual && actual <= maximum,
            (
                AttributeConstraint::RationalRange { minimum, maximum },
                SemanticValue::Rational(actual),
            ) => {
                (actual.numerator() as i128) * (minimum.denominator() as i128)
                    >= (minimum.numerator() as i128) * (actual.denominator() as i128)
                    && (actual.numerator() as i128) * (maximum.denominator() as i128)
                        <= (maximum.numerator() as i128) * (actual.denominator() as i128)
            }
            (AttributeConstraint::TextChoices { values }, SemanticValue::Text(actual)) => {
                values.contains(actual)
            }
            (
                AttributeConstraint::IntegerListLength { minimum, maximum },
                SemanticValue::Integers(actual),
            ) => (*minimum as usize) <= actual.len() && actual.len() <= (*maximum as usize),
            _ => false,
        };
        if !accepted {
            return Err(invalid_operation(format!(
                "attribute `{attribute_id}` violates its declared constraint"
            )));
        }
        Ok(())
    }
}
