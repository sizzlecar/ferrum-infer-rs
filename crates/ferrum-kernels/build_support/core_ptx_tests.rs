use super::*;
use std::cell::Cell;

const PTX: &[u8] =
    b".version 8.7\n.target sm_86\n.address_size 64\n.visible .entry fresh() { ret; }\n";
const FILE: &str = "unit.ptx";
const OLD: &[u8] = b"previous PTX bytes";
const STAMP: &str = "previous signature";

fn fixture() -> Stage {
    let dir = Stage::new(&std::env::temp_dir()).unwrap();
    fs::write(dir.0.join(FILE), OLD).unwrap();
    fs::write(dir.0.join(format!("{FILE}.stamp")), STAMP).unwrap();
    dir
}

fn assert_previous(dir: &Stage) {
    assert_eq!(fs::read(dir.0.join(FILE)).unwrap(), OLD);
    assert_eq!(
        fs::read_to_string(dir.0.join(format!("{FILE}.stamp"))).unwrap(),
        STAMP
    );
    assert_eq!(
        fs::read_dir(&dir.0).unwrap().count(),
        2,
        "staging files must be removed"
    );
}

#[test]
fn compiler_success_without_new_output_cannot_certify_old_artifact() {
    let dir = fixture();
    let published = Cell::new(false);
    let error = compile_and_publish(
        &dir.0,
        FILE,
        "new",
        |path| {
            assert!(!path.exists());
            Ok(())
        },
        |_| {
            published.set(true);
            Ok(())
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
    assert!(!published.get());
    assert_previous(&dir);
}

#[test]
fn rejected_or_partial_compiler_outputs_preserve_artifact_and_stamp() {
    for bytes in [
        b"".as_slice(),
        b"compiler diagnostic, not PTX".as_slice(),
        PTX,
    ] {
        let dir = fixture();
        let published = Cell::new(false);
        let error = compile_and_publish(
            &dir.0,
            FILE,
            "new",
            |path| {
                fs::write(path, bytes)?;
                if bytes == PTX {
                    Err(io::Error::other("compiler failed after partial output"))
                } else {
                    Ok(())
                }
            },
            |_| {
                published.set(true);
                Ok(())
            },
        );
        assert!(error.is_err());
        assert!(!published.get());
        assert_previous(&dir);
    }
}

#[test]
fn cache_sees_only_verified_new_bytes_before_local_publication() {
    let dir = fixture();
    let signature = format!("output_contract={}\ninput=fixture", OUTPUT_CONTRACT);
    let identity = compile_and_publish(
        &dir.0,
        FILE,
        &signature,
        |path| fs::write(path, PTX),
        |path| {
            assert_eq!(fs::read(path)?, PTX);
            assert_ne!(path, dir.0.join(FILE));
            assert_eq!(fs::read(dir.0.join(FILE))?, OLD);
            assert_eq!(
                fs::read_to_string(dir.0.join(format!("{FILE}.stamp")))?,
                STAMP
            );
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(identity.size_bytes, PTX.len() as u64);
    assert_eq!(identity.sha256, format!("{:x}", Sha256::digest(PTX)));
    assert_eq!(fs::read(dir.0.join(FILE)).unwrap(), PTX);
    assert_eq!(
        fs::read_to_string(dir.0.join(format!("{FILE}.stamp"))).unwrap(),
        bound_stamp(&signature, &identity)
    );
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 2);
}

#[test]
fn cache_failure_preserves_previous_local_pair() {
    let dir = fixture();
    let error = compile_and_publish(
        &dir.0,
        FILE,
        "new",
        |path| fs::write(path, PTX),
        |_| Err(io::Error::other("cache publication rejected")),
    );
    assert!(error.is_err());
    assert_previous(&dir);
}

#[test]
fn stamp_replace_failure_rolls_back_the_artifact() {
    let dir = fixture();
    let stage = Stage::new(&dir.0).unwrap();
    let generated = stage.0.join(FILE);
    let staged_stamp = stage.0.join("input.stamp");
    let previous = stage.0.join("previous.ptx");
    let destination = dir.0.join(FILE);
    let stamp = dir.0.join(format!("{FILE}.stamp"));
    fs::write(&generated, PTX).unwrap();
    fs::write(&staged_stamp, "new").unwrap();
    fs::copy(&destination, &previous).unwrap();
    let error = replace_pair(
        &generated,
        &destination,
        &staged_stamp,
        &stamp,
        Some(&previous),
        |from, to| {
            if to == stamp {
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "injected stamp replace failure",
                ))
            } else {
                fs::rename(from, to)
            }
        },
    );
    assert_eq!(error.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
    drop(stage);
    assert_previous(&dir);
}

#[test]
fn fresh_destination_needs_no_existing_ptx_or_stamp() {
    let dir = Stage::new(&std::env::temp_dir()).unwrap();
    compile_and_publish(
        &dir.0,
        FILE,
        "fresh",
        |path| fs::write(path, PTX),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(fs::read(dir.0.join(FILE)).unwrap(), PTX);
    assert!(local_artifact_matches(&dir.0, FILE, "fresh").unwrap());
}

#[test]
fn local_stamp_rejects_changed_artifact_and_old_unbound_stamp() {
    let dir = fixture();
    compile_and_publish(&dir.0, FILE, "new", |path| fs::write(path, PTX), |_| Ok(())).unwrap();
    assert!(local_artifact_matches(&dir.0, FILE, "new").unwrap());
    assert!(!local_artifact_matches(&dir.0, FILE, "different input").unwrap());
    let changed = String::from_utf8(PTX.to_vec())
        .unwrap()
        .replace("fresh", "different");
    fs::write(dir.0.join(FILE), changed).unwrap();
    assert!(!local_artifact_matches(&dir.0, FILE, "new").unwrap());
    fs::write(dir.0.join(FILE), PTX).unwrap();
    fs::write(dir.0.join(format!("{FILE}.stamp")), "new").unwrap();
    assert!(!local_artifact_matches(&dir.0, FILE, "new").unwrap());
}

#[test]
fn shared_cache_restore_also_binds_local_output_bytes() {
    let dir = fixture();
    fs::write(dir.0.join(FILE), PTX).unwrap();
    fs::write(dir.0.join(format!("{FILE}.stamp")), "restored").unwrap();
    record_restored_artifact(&dir.0, FILE, "restored").unwrap();
    assert!(local_artifact_matches(&dir.0, FILE, "restored").unwrap());
    assert_eq!(fs::read(dir.0.join(FILE)).unwrap(), PTX);
}
