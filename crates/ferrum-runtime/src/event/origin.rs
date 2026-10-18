/// Typed origin for an executor-owned request lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorRequestOrigin {
    Product,
    Startup,
    Diagnostic,
}

impl ExecutorRequestOrigin {
    pub const fn namespace(self) -> &'static str {
        match self {
            Self::Product => "product",
            Self::Startup => "startup",
            Self::Diagnostic => "diagnostic",
        }
    }

    pub fn from_namespaced_request_identity(identity: &str) -> Option<Self> {
        let suffix = identity.strip_prefix("request.")?;
        let (namespace, request_id) = suffix.split_once('.')?;
        if request_id.is_empty() {
            return None;
        }
        match namespace {
            "product" => Some(Self::Product),
            "startup" => Some(Self::Startup),
            "diagnostic" => Some(Self::Diagnostic),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExecutorRequestOrigin;

    #[test]
    fn request_origin_namespaces_are_validated() {
        assert_eq!(ExecutorRequestOrigin::Product.namespace(), "product");
        assert_eq!(ExecutorRequestOrigin::Startup.namespace(), "startup");
        assert_eq!(ExecutorRequestOrigin::Diagnostic.namespace(), "diagnostic");
        assert_eq!(
            ExecutorRequestOrigin::from_namespaced_request_identity("request.product.123"),
            Some(ExecutorRequestOrigin::Product)
        );
        assert_eq!(
            ExecutorRequestOrigin::from_namespaced_request_identity("request.startup.123"),
            Some(ExecutorRequestOrigin::Startup)
        );
        assert_eq!(
            ExecutorRequestOrigin::from_namespaced_request_identity("request.diagnostic.123"),
            Some(ExecutorRequestOrigin::Diagnostic)
        );
        assert_eq!(
            ExecutorRequestOrigin::from_namespaced_request_identity("request.product."),
            None
        );
        assert_eq!(
            ExecutorRequestOrigin::from_namespaced_request_identity("request/external"),
            None
        );
    }
}
