use serde::{Deserialize, Serialize};
use std::fmt;

use super::VNextError;

fn validate_identity(kind: &'static str, value: &str) -> Result<(), VNextError> {
    if value.is_empty() {
        return Err(VNextError::InvalidIdentity {
            kind,
            value: value.to_owned(),
            reason: "identity must not be empty",
        });
    }
    if value.len() > 160 {
        return Err(VNextError::InvalidIdentity {
            kind,
            value: value.to_owned(),
            reason: "identity exceeds 160 bytes",
        });
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
    }) {
        return Err(VNextError::InvalidIdentity {
            kind,
            value: value.to_owned(),
            reason: "identity contains a non-portable character",
        });
    }
    Ok(())
}

macro_rules! stable_identity {
    ($name:ident, $kind:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, VNextError> {
                let value = value.into();
                validate_identity($kind, &value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl TryFrom<String> for $name {
            type Error = VNextError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

stable_identity!(CapabilityId, "capability");
stable_identity!(DeviceId, "device");
stable_identity!(ExternalModelMetadataId, "external model metadata");
stable_identity!(ModelFamilyId, "model family");
stable_identity!(NodeId, "node");
stable_identity!(OperationId, "operation");
stable_identity!(PlanId, "plan");
stable_identity!(ProgramValueId, "program value");
stable_identity!(ProviderId, "provider");
stable_identity!(QuantizationFormatId, "quantization format");
stable_identity!(RequestIdentity, "request");
stable_identity!(ResourceId, "resource");
stable_identity!(RunId, "run");
stable_identity!(SpanId, "span");
stable_identity!(StateId, "state");
stable_identity!(TensorId, "tensor");
stable_identity!(TokenizerId, "tokenizer");
stable_identity!(TransactionId, "transaction");
stable_identity!(WeightId, "weight");
stable_identity!(WeightFormatId, "weight format");
stable_identity!(WeightLayoutId, "weight layout");

/// A versioned contract uses major for breaking and minor for additive changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContractVersion {
    pub major: u16,
    pub minor: u16,
}

impl ContractVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    pub const fn satisfies(self, required: Self) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl fmt::Display for ContractVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}
