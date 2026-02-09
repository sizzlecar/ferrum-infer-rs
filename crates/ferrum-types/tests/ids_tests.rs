use ferrum_types::*;
use serde_json as json;

#[test]
fn request_and_batch_ids_are_unique_and_display() {
    let r1 = RequestId::new();
    let r2 = RequestId::new();
    assert_ne!(r1, r2);
    assert!(!r1.to_string().is_empty());

    let b1 = BatchId::new();
    let b2 = BatchId::new();
    assert_ne!(b1, b2);
    assert!(!b1.to_string().is_empty());
}

#[test]
fn model_and_client_id_conversions() {
    let m: ModelId = "llama".into();
    assert_eq!(m.as_str(), "llama");
    assert_eq!(m.to_string(), "llama");

    let c: ClientId = "acme".into();
    assert_eq!(c.as_str(), "acme");
    assert_eq!(c.to_string(), "acme");
}

#[test]
fn ids_serde_roundtrip() {
    let req = RequestId::new();
    let s = json::to_string(&req).unwrap();
    let back: RequestId = json::from_str(&s).unwrap();
    assert_eq!(req, back);
}
