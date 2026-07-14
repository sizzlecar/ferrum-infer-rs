mod vnext_core_contract;

use vnext_core_contract::*;

fn vnext_source_files() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/vnext");
    let mut paths = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn generic_contracts_have_zero_architecture_names() {
    let names = [
        "qwen", "llama", "deepseek", "mistral", "mixtral", "gemma", "chatglm", "internlm",
        "baichuan",
    ];
    for path in vnext_source_files() {
        let source = fs::read_to_string(&path).unwrap().to_ascii_lowercase();
        for name in names {
            assert!(!source.contains(name), "{} contains {name}", path.display());
        }
    }
}

#[test]
fn silent_success_defaults_are_absent() {
    for path in vnext_source_files() {
        let source = fs::read_to_string(&path).unwrap();
        assert!(!source.contains("fn unsupported") || !source.contains("Ok(())"));
        assert!(!source.contains("std::env::var"));
        assert!(!source.contains("downcast_ref"));
    }
}

#[test]
fn failure_envelope_wire_limit_precedes_deserialization() {
    let at_limit = vec![b' '; MAX_FAILURE_ENVELOPE_WIRE_BYTES];
    match FailureEnvelope::decode_untrusted(&at_limit) {
        Err(VNextError::Serialization { context, message }) => {
            assert_eq!(context, "decode untrusted failure envelope");
            assert!(!message.contains("maximum is"));
        }
        other => panic!("equal-to-limit malformed payload hit wrong result: {other:?}"),
    }

    let over_limit = vec![b' '; MAX_FAILURE_ENVELOPE_WIRE_BYTES + 1];
    match FailureEnvelope::decode_untrusted(&over_limit) {
        Err(VNextError::Serialization { context, message }) => {
            assert_eq!(context, "decode untrusted failure envelope");
            assert!(message.contains("maximum is 8192"));
        }
        other => panic!("oversized payload hit wrong result: {other:?}"),
    }
}
