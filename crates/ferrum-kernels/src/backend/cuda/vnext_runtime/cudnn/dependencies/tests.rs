use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrum-cudnn-dependencies-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn explicit_dependency_selection_is_bounded_and_never_searches_for_relative_names() {
    assert!(validate_paths(&[]).is_ok());
    assert!(validate_paths(&[PathBuf::from("libnvrtc.so.13")]).is_err());
    let absolute = std::env::temp_dir().join("dependency.so.1");
    assert!(validate_paths(&[absolute.clone(), absolute]).is_err());
    let paths = (0..17)
        .map(|i| std::env::temp_dir().join(format!("dependency{i}.so.1")))
        .collect::<Vec<_>>();
    assert!(validate_paths(&paths).is_err());
}

#[test]
fn selected_dependency_bytes_and_load_order_are_fingerprinted_before_native_loading() {
    let files = Files::new();
    let first = files.write("first.so.1", b"native-library-a");
    let second = files.write("second.so.1", b"native-library-b");
    let paths = [first.clone(), second.clone()];
    let original = Selection::prepare(&paths).unwrap().fingerprint();
    assert_eq!(original, Selection::prepare(&paths).unwrap().fingerprint());
    assert_ne!(
        original,
        Selection::prepare(&[second, first.clone()])
            .unwrap()
            .fingerprint()
    );
    // Same file length: identity must include content, not only metadata.
    std::fs::write(&first, b"native-library-c").unwrap();
    assert_ne!(original, Selection::prepare(&paths).unwrap().fingerprint());
    assert_ne!(original, Selection::prepare(&[]).unwrap().fingerprint());
}

#[test]
fn missing_and_non_file_dependencies_fail_without_native_loading() {
    let files = Files::new();
    assert!(Selection::prepare(&[files.0.join("absent.so.1")]).is_err());
    assert!(Selection::prepare(&[files.0.clone()]).is_err());
    let empty = Selection::prepare(&[]).unwrap();
    assert!(empty.validate_loader_support().is_ok());
    assert!(unsafe { empty.load() }.unwrap().is_empty());
}

#[cfg(unix)]
#[test]
fn aliases_cannot_double_select_one_dependency() {
    let files = Files::new();
    let target = files.write("dependency.so.1.2", b"native-library");
    let alias = files.0.join("dependency.so.1");
    std::os::unix::fs::symlink(&target, &alias).unwrap();
    assert_eq!(
        Selection::prepare(&[target.clone()]).unwrap().fingerprint(),
        Selection::prepare(&[alias.clone()]).unwrap().fingerprint()
    );
    assert!(Selection::prepare(&[target, alias]).is_err());
}
