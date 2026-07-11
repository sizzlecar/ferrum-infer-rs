use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

#[test]
fn legacy_backend_methods_are_mapped_82_of_82() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let map_path = manifest_dir
        .join("../../docs/goals/runtime-vnext-0.8.0-2026-07-10/G01A_LEGACY_CONTRACT_MAP.json");
    let map: Value = serde_json::from_slice(&fs::read(map_path).unwrap()).unwrap();
    assert_eq!(map["schema_version"], 1);
    assert_eq!(map["source"]["category"], "backend_trait_method");
    assert_eq!(map["source"]["expected_method_count"], 82);
    assert_eq!(map["summary"]["mapped"], 82);
    assert_eq!(map["summary"]["unmapped"], 0);
    assert_eq!(map["summary"]["missing_owner"], 0);
    assert_eq!(map["summary"]["special_case"], 0);

    let source =
        fs::read_to_string(manifest_dir.join("../ferrum-kernels/src/backend/traits.rs")).unwrap();
    let syntax = syn::parse_file(&source).unwrap();
    let backend_methods = syntax
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Trait(item_trait) if item_trait.ident == "Backend" => Some(
                item_trait
                    .items
                    .iter()
                    .filter_map(|item| match item {
                        syn::TraitItem::Fn(method) => Some(method.sig.ident.to_string()),
                        _ => None,
                    })
                    .collect::<BTreeSet<_>>(),
            ),
            _ => None,
        })
        .unwrap();
    let mappings = map["mappings"].as_array().unwrap();
    assert_eq!(mappings.len(), 82);
    let allowed_classifications = BTreeSet::from([
        "stable_device_primitive",
        "versioned_operation",
        "model_semantic",
        "dead_code",
    ]);
    let mut mapped = BTreeSet::new();
    for mapping in mappings {
        assert_eq!(mapping["legacy_trait"], "Backend");
        let method = mapping["legacy_method"].as_str().unwrap();
        assert!(
            backend_methods.contains(method),
            "missing legacy method `{method}`"
        );
        assert!(mapped.insert(method));
        assert!(allowed_classifications.contains(mapping["classification"].as_str().unwrap()));
        assert!(!mapping["owner"].as_str().unwrap().trim().is_empty());
        assert!(!mapping["disposition"].as_str().unwrap().trim().is_empty());
    }
    assert_eq!(mapped.len(), 82);
    println!("VNEXT LEGACY MAP PASS: 82/82");
}
