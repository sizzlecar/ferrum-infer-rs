//! Opaque image request correlation. These identifiers carry no request content
//! and do not authorize execution, select a model, or establish timing evidence.
use uuid::Uuid;

pub const IMAGE_ATTEMPT_ID_HEADER: &str = "x-ferrum-image-attempt-id";
pub const IMAGE_REQUEST_ID_HEADER: &str = "x-ferrum-image-request-id";
pub const IMAGE_TRACE_SESSION_HEADER: &str = "x-ferrum-image-trace-session";

pub use crate::iterative::RequestCorrelation as ImageRequestCorrelation;

/// The only caller-supplied identity accepted by the image observation path.
/// Require the fixed hyphenated UUID wire spelling; arbitrary strings, URNs,
/// whitespace and content-derived identifiers are not accepted as metadata.
pub fn parse_image_attempt_id(value: Option<&str>) -> Result<Option<Uuid>, &'static str> {
    value
        .map(|value| {
            if value.len() != 36
                || value.bytes().enumerate().any(|(i, c)| {
                    if [8, 13, 18, 23].contains(&i) {
                        c != b'-'
                    } else {
                        !c.is_ascii_hexdigit()
                    }
                })
            {
                return Err("image attempt id must be a hyphenated UUID");
            }
            Uuid::parse_str(value).map_err(|_| "image attempt id must be a hyphenated UUID")
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_bounded_attempt_identity_is_parsed_and_service_mints_its_own_request() {
        assert_eq!(parse_image_attempt_id(None).unwrap(), None);
        let attempt = Uuid::new_v4();
        assert_eq!(
            parse_image_attempt_id(Some(&attempt.to_string())).unwrap(),
            Some(attempt)
        );
        for invalid in [
            "",
            "a prompt",
            "00000000000000000000000000000000",
            "{00000000-0000-0000-0000-000000000000}",
            "00000000-0000-0000-0000-00000000000g",
        ] {
            assert!(parse_image_attempt_id(Some(invalid)).is_err());
        }
        let a = ImageRequestCorrelation::new(Some(attempt));
        let b = ImageRequestCorrelation::new(Some(attempt));
        assert_eq!(a.client_attempt_id, b.client_attempt_id);
        assert_ne!(a.request_id, b.request_id);
        assert_ne!(a.request_id, attempt);
    }
}
