//! Private result of the shared checked provider encoder. A replay request must
//! return bindings before constructing resident compute; it cannot silently
//! fall back to building and discarding a compute command.
use ferrum_interfaces::vnext::{EncodedDeviceOperation, EncodedReusableExecutionBindings};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum EncodingTarget {
    Full,
    BindingsOnly,
}

pub(super) enum Encoding<C> {
    Full(EncodedDeviceOperation<C>),
    Bindings(EncodedReusableExecutionBindings<C>),
}

impl<C> Encoding<C> {
    pub(super) fn full(self) -> Result<EncodedDeviceOperation<C>, String> {
        match self {
            Self::Full(operation) => Ok(operation),
            Self::Bindings(_) => Err("full provider encoding returned only bindings".into()),
        }
    }

    pub(super) fn bindings(self) -> Result<EncodedReusableExecutionBindings<C>, String> {
        match self {
            Self::Bindings(bindings) => Ok(bindings),
            Self::Full(_) => Err("binding-only provider rebuilt resident compute".into()),
        }
    }
}

impl EncodingTarget {
    pub(super) fn projection_preparation(
        self,
    ) -> super::super::native_blocks::upstream_linear::ProjectionPreparation {
        use super::super::native_blocks::upstream_linear::ProjectionPreparation;
        match self {
            Self::Full => ProjectionPreparation::Full,
            Self::BindingsOnly => ProjectionPreparation::BindingsOnly,
        }
    }
}
