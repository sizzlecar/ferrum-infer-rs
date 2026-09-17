use super::*;

#[test]
#[cfg(unix)]
fn local_gguf_symlink_keeps_its_filename_in_run_and_serve_commands() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    fs::write(root.join("content-address"), b"fixture").unwrap();
    let selected = root.join("selected.gguf");
    std::os::unix::fs::symlink("content-address", &selected).unwrap();
    let mut args = Args::try_parse_from([
        "model-regression",
        "--ferrum-bin",
        "fixture",
        "--model",
        selected.to_str().unwrap(),
        "--backend",
        "metal",
        "--report-dir",
        "report",
    ])
    .unwrap();
    args.model = ferrum_bench_core::release_regression::model_sources::normalize_local_model_path(
        Path::new(&args.model),
    )
    .unwrap()
    .to_str()
    .unwrap()
    .to_owned();
    for entrypoint in ["run", "serve"] {
        let words = args.common_args(entrypoint);
        assert_eq!(words[1], selected.to_str().unwrap());
        assert_eq!(fs::read(&words[1]).unwrap(), b"fixture");
    }
}
