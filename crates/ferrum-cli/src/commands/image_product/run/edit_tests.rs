use super::*;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};
use std::fs;

fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
                output_bytes: 0,
            },
        )
        .unwrap(),
    )
}

#[test]
fn edit_cli_input_is_read_into_one_admitted_backing_and_retires() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.png");
    let wire = b"\x89PNG\r\n\x1a\ntransport-fixture";
    fs::write(&path, wire).unwrap();
    let (file, length) = open_image(&path).unwrap();
    let account = host(wire.len());
    let bytes = read_image(&account, file, length).unwrap();
    assert_eq!(bytes.bytes(), wire);
    assert!(account.owns(&bytes));
    assert_eq!(account.usage().unwrap().allocated_bytes(), wire.len());
    drop(bytes);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn edit_cli_rejects_invalid_extent_capacity_and_file_changes_without_leaking() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.png");
    assert!(open_image(dir.path()).is_err());
    fs::write(&path, b"").unwrap();
    assert!(open_image(&path).is_err());
    let wire = b"\x89PNG\r\n\x1a\n1234";
    fs::write(&path, wire).unwrap();
    let (file, length) = open_image(&path).unwrap();
    let account = host(wire.len() - 1);
    assert!(read_image(&account, file, length).is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let account = host(1024);
    for altered in [
        b"\x89PNG\r\n\x1a\n".as_slice(),
        b"\x89PNG\r\n\x1a\n12345".as_slice(),
        b"not a png...".as_slice(),
    ] {
        fs::write(&path, wire).unwrap();
        let (file, length) = open_image(&path).unwrap();
        fs::write(&path, altered).unwrap();
        assert!(read_image(&account, file, length).is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
