//! Native installer lifecycle and uninterrupted model-session upgrade verification.
#[path = "windows_installer_regression/evidence.rs"]
#[cfg_attr(not(windows), allow(dead_code))]
// Native lifecycle helpers remain testable on other hosts.
mod evidence;
#[path = "windows_installer_regression/layout.rs"]
#[cfg_attr(not(windows), allow(dead_code))]
mod layout;
#[path = "windows_installer_regression/model_io.rs"]
#[cfg_attr(not(windows), allow(dead_code))]
mod model_io;
#[cfg(windows)]
#[path = "windows_installer_regression/process_identity.rs"]
mod process_identity;
#[cfg(windows)]
#[path = "windows_installer_regression/session.rs"]
mod session;
#[cfg(windows)]
#[path = "windows_installer_regression/windows.rs"]
mod windows;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use evidence::digest;
#[cfg(windows)]
use evidence::{inventory, payload, verify_files};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[cfg(windows)]
use std::{
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(
    about = "Verify native Windows installation, optional version upgrade, reinstall and uninstall"
)]
struct Args {
    #[arg(long)]
    setup: PathBuf,
    #[arg(long)]
    setup_sha256: String,
    /// Previously accepted portable extraction, including ferrum-portable.json.
    #[arg(long)]
    payload_dir: PathBuf,
    /// Stable launcher packaged by both setups; distinct from the model executable.
    #[arg(long)]
    launcher: PathBuf,
    #[arg(long)]
    launcher_sha256: String,
    /// Install this older setup first, then upgrade using --setup.
    #[arg(long, requires_all = ["previous_setup_sha256", "previous_payload_dir", "model"])]
    previous_setup: Option<PathBuf>,
    #[arg(long, requires = "previous_setup")]
    previous_setup_sha256: Option<String>,
    #[arg(long, requires = "previous_setup")]
    previous_payload_dir: Option<PathBuf>,
    /// Existing local model for live upgrade; must be covered by --preserve-dir.
    #[arg(long, requires = "previous_setup")]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 128)]
    max_tokens: u32,
    #[arg(long, default_value_t = 2048)]
    context_tokens: u32,
    #[arg(long, default_value_t = 600)]
    startup_timeout_secs: u64,
    /// Existing external model/data directory whose complete bytes must survive.
    #[arg(long, required = true)]
    preserve_dir: Vec<PathBuf>,
    /// Must not exist. All evidence survives failure.
    #[arg(long)]
    report_dir: PathBuf,
    #[arg(long, default_value_t = 600)]
    timeout_secs: u64,
}

fn save(dir: &Path, name: &str, value: &impl serde::Serialize) -> Result<()> {
    fs::write(dir.join(name), serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

#[cfg(windows)]
fn process(args: &Args, stage: &str, exe: &Path, argv: &[String]) -> Result<()> {
    let mut child = Command::new(exe)
        .args(argv)
        .stdin(Stdio::null())
        .stdout(fs::File::create(
            args.report_dir.join(format!("{stage}.stdout.log")),
        )?)
        .stderr(fs::File::create(
            args.report_dir.join(format!("{stage}.stderr.log")),
        )?)
        .spawn()
        .with_context(|| format!("spawn {stage}"))?;
    let start = Instant::now();
    let mut event = json!({"program":exe,"args":argv,"pid":child.id(),"status":"running","started_at":chrono::Utc::now().to_rfc3339()});
    save(&args.report_dir, &format!("{stage}.json"), &event)?;
    loop {
        if let Some(status) = child.try_wait()? {
            event["status"] = json!("completed");
            event["exit_code"] = json!(status.code());
            event["elapsed_ms"] = json!(start.elapsed().as_millis());
            event["finished_at"] = json!(chrono::Utc::now().to_rfc3339());
            save(&args.report_dir, &format!("{stage}.json"), &event)?;
            ensure!(status.success(), "{stage} exited {status}");
            return Ok(());
        }
        if start.elapsed() >= Duration::from_secs(args.timeout_secs) {
            event["status"] = json!("timed_out_process_may_still_run");
            save(&args.report_dir, &format!("{stage}.json"), &event)?;
            anyhow::bail!(
                "{stage} timed out; PID {} may still run; no automatic cleanup",
                child.id()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(windows)]
fn lifecycle(args: &Args, report: &mut Value) -> Result<()> {
    let install = windows::local_app_data()?.join("Programs").join("Ferrum");
    report["install_dir"] = json!(install);
    let before = windows::path()?;
    report["original_user_path"] = json!(before);
    ensure!(
        evidence::absent(&install)?,
        "existing installation directory: {}",
        install.display()
    );
    ensure!(
        !windows::uninstall_exists()? && !windows::ownership_exists()?,
        "existing Ferrum uninstall/ownership registry key"
    );
    let app = install
        .to_str()
        .context("installation path is not Unicode")?;
    ensure!(
        !app.contains(';'),
        "installation directory contains PATH separator"
    );
    ensure!(
        windows::entry_count(before.as_ref(), app)? == 0,
        "user PATH already contains the installation directory"
    );
    let records = payload(&args.payload_dir)?;
    verify_files(&args.payload_dir, &records)?;
    let launcher = digest(&args.launcher)?;
    ensure!(
        launcher.sha256 == args.launcher_sha256,
        "launcher bytes changed"
    );
    let version = evidence::payload_version(&args.payload_dir)?;
    report["version"] = json!(version);
    let previous = if let Some(setup) = &args.previous_setup {
        let directory = args
            .previous_payload_dir
            .as_ref()
            .context("previous payload is required")?;
        let hash = args
            .previous_setup_sha256
            .as_ref()
            .context("previous setup SHA is required")?;
        let previous_version = evidence::payload_version(directory)?;
        evidence::verify_upgrade_versions(&previous_version, &version)?;
        let identity = digest(setup)?;
        ensure!(&identity.sha256 == hash, "previous setup SHA-256 differs");
        let previous_records = payload(directory)?;
        verify_files(directory, &previous_records)?;
        ensure!(
            layout::core_record(&previous_records)?.sha256 != layout::core_record(&records)?.sha256,
            "upgrade must use distinct real core executables"
        );
        report["scope"] = json!("first_install_live_upgrade_same_setup_reinstall_uninstall");
        report["previous_setup"] =
            json!({"path":setup,"identity":identity,"version":previous_version});
        report["previous_payload"] = json!(previous_records);
        Some((setup, hash, previous_version, previous_records))
    } else {
        None
    };
    let mut preserved = Vec::new();
    for path in &args.preserve_dir {
        let path = path.canonicalize()?;
        ensure!(
            !evidence::paths_overlap(&path, &install)?
                && !evidence::paths_overlap(&path, &args.report_dir)?,
            "preserved directory overlaps installation or evidence output"
        );
        preserved.push((path.clone(), inventory(&path)?));
    }
    report["preserved_before"] = json!(preserved);
    if let Some(model) = &args.model {
        ensure!(
            model.is_dir() && preserved.iter().any(|(path, _)| model.starts_with(path)),
            "local model must be inside a --preserve-dir directory"
        );
    }
    report["payload"] = json!(records);
    save(&args.report_dir, "report.json", report)?;

    // Hashing external models can take time. Refuse a concurrent install or PATH edit.
    ensure!(
        evidence::absent(&install)?
            && !windows::uninstall_exists()?
            && !windows::ownership_exists()?
            && windows::path()? == before,
        "user installation state changed during preflight"
    );

    let expected_path = evidence::appended_path(before.as_ref(), app)?;
    let mut sentinel = None;
    let mut installed_files = std::collections::BTreeSet::new();
    let mut old_session: Option<session::Session> = None;
    let mut old_selection: Option<layout::Selection> = None;
    let mut stages = Vec::new();
    if let Some((setup, hash, previous_version, previous_records)) = &previous {
        stages.push((
            "install_previous",
            *setup,
            hash.as_str(),
            previous_records.as_slice(),
            previous_version.as_str(),
            args.previous_payload_dir.as_ref().unwrap(),
        ));
    }
    stages.push((
        if previous.is_some() {
            "upgrade"
        } else {
            "install"
        },
        &args.setup,
        args.setup_sha256.as_str(),
        records.as_slice(),
        version.as_str(),
        &args.payload_dir,
    ));
    stages.push((
        "same_setup_reinstall",
        &args.setup,
        args.setup_sha256.as_str(),
        records.as_slice(),
        version.as_str(),
        &args.payload_dir,
    ));
    for (stage, setup, setup_hash, expected_records, expected_version, source) in stages {
        ensure!(digest(setup)?.sha256 == setup_hash, "setup bytes changed");
        let install_stage = || {
            process(
                args,
                stage,
                setup,
                &[
                    "/VERYSILENT".into(),
                    "/SUPPRESSMSGBOXES".into(),
                    "/NORESTART".into(),
                    "/SP-".into(),
                    format!(
                        "/LOG={}",
                        args.report_dir.join(format!("{stage}.inno.log")).display()
                    ),
                ],
            )
        };
        if stage == "upgrade" {
            let live = old_session
                .as_mut()
                .context("missing owned old model session")?;
            report["live_upgrade"]["during"] = live.while_upgrading(install_stage)?;
        } else {
            install_stage()?;
        }
        let selected = layout::inspect(&install, source, &launcher)?;
        report[format!("{stage}_selected_payload")] = json!(selected);
        if let (Some(old), Some((_, _, _, old_records))) = (&old_selection, &previous) {
            verify_files(&old.directory, old_records)?;
            ensure!(
                payload(&old.directory)? == *old_records,
                "upgrade changed retained old payload"
            );
        }
        let version_stage = format!("{stage}_version");
        process(
            args,
            &version_stage,
            &install.join("ferrum.exe"),
            &["--version".into()],
        )?;
        let actual_version =
            fs::read_to_string(args.report_dir.join(format!("{version_stage}.stdout.log")))?;
        ensure!(
            actual_version.trim() == format!("ferrum {expected_version}"),
            "installed executable version differs from its payload"
        );
        report[format!("{stage}_version")] = json!(actual_version.trim());
        if stage == "install_previous" {
            let mut live = session::Session::start(args, &selected, expected_version)?;
            report["live_upgrade"]["before"] = live.proof("before-upgrade")?;
            old_selection = Some(selected.clone());
            old_session = Some(live);
            save(&args.report_dir, "report.json", report)?;
        } else if stage == "upgrade" {
            let live = old_session
                .as_mut()
                .context("missing old session after upgrade")?;
            report["live_upgrade"]["new_entry_version_while_old_alive"] =
                json!(actual_version.trim());
            report["live_upgrade"]["after"] = live.proof("after-upgrade")?;
            save(&args.report_dir, "report.json", report)?;
            // Only the QA-owned launcher and its job are stopped. The installer never stops it.
            report["live_upgrade"]["cleanup"] = old_session.take().unwrap().stop()?;
            save(&args.report_dir, "report.json", report)?;
            report["new_run"] = session::new_run(args, &selected)?;
        }
        let current = windows::path()?;
        report[format!("{stage}_user_path")] = json!(current);
        ensure!(
            current.as_ref() == Some(&expected_path),
            "{stage} changed PATH prefix/type or did not add the exact suffix"
        );
        ensure!(
            windows::entry_count(current.as_ref(), app)? == 1,
            "{stage} PATH entry count differs from one"
        );
        ensure!(
            windows::uninstall_exists()? && windows::ownership_exists()?,
            "{stage} registry registration is missing"
        );
        if let Some((path, bytes)) = &sentinel {
            ensure!(
                fs::read(path)? == *bytes,
                "reinstall changed an unrelated user file"
            );
        } else {
            let path = install.join("installer-regression-user-file.txt");
            let bytes = b"User data fixture: installer must preserve this file.\r\n".to_vec();
            use std::io::Write;
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?
                .write_all(&bytes)?;
            sentinel = Some((path, bytes));
        }
        // Include files from both versions in the uninstall check, excluding our user file.
        installed_files.extend(
            inventory(&install)?
                .files
                .into_keys()
                .filter(|path| path != "installer-regression-user-file.txt"),
        );
        // Explicit version inventory also binds uninstall ownership to the accepted manifest.
        installed_files.extend(layout::records_for_version(
            expected_records,
            &selected.version_dir,
        ));
        report["installer_created_files"] = json!(installed_files);
        save(&args.report_dir, "report.json", report)?;
    }
    ensure!(
        old_session.is_none(),
        "owned session must be stopped before uninstall"
    );
    let uninstaller = install.join("unins000.exe");
    let registered = windows::uninstall_command()?
        .context("missing UninstallString")?
        .text()?;
    ensure!(
        registered.trim_matches('"') == uninstaller.to_string_lossy(),
        "uninstall command does not name the owned executable"
    );
    report["uninstaller"] = json!({"path":uninstaller,"identity":digest(&uninstaller)?});
    save(&args.report_dir, "report.json", report)?;
    process(
        args,
        "uninstall",
        &uninstaller,
        &[
            "/VERYSILENT".into(),
            "/SUPPRESSMSGBOXES".into(),
            "/NORESTART".into(),
            format!(
                "/LOG={}",
                args.report_dir.join("uninstall.inno.log").display()
            ),
        ],
    )?;
    // Inno's original unins000.exe can exit before its temporary copy finishes
    // deleting installed files. Wait for the declared postconditions, not just
    // the bootstrap process exit, before inspecting PATH and retained user data.
    let uninstall_wait = Instant::now();
    loop {
        let registered = windows::uninstall_exists()? || windows::ownership_exists()?;
        let mut files_remain = !evidence::absent(&uninstaller)?;
        for path in &installed_files {
            files_remain |= !evidence::absent(&install.join(path))?;
        }
        if !registered && !files_remain {
            report["uninstall_completion"] = json!({
                "observed_at": chrono::Utc::now().to_rfc3339(),
                "postcondition_wait_ms": uninstall_wait.elapsed().as_millis(),
                "installer_files_and_registration_removed": true,
            });
            break;
        }
        ensure!(
            uninstall_wait.elapsed() < Duration::from_secs(args.timeout_secs),
            "uninstaller bootstrap exited, but installed files or registration remain"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    let after = windows::path()?;
    report["final_user_path"] = json!(after);
    ensure!(
        after == before,
        "uninstall did not restore original PATH bytes/type/existence"
    );
    ensure!(
        !windows::uninstall_exists()? && !windows::ownership_exists()?,
        "uninstall left Ferrum registration/ownership"
    );
    for path in &installed_files {
        ensure!(
            evidence::absent(&install.join(path))?,
            "uninstall retained installer-created file {path}"
        );
    }
    ensure!(
        evidence::absent(&uninstaller)?,
        "uninstall executable remains"
    );
    for (path, expected) in &preserved {
        ensure!(
            &inventory(path)? == expected,
            "external directory changed: {}",
            path.display()
        );
    }
    let (sentinel_path, bytes) = sentinel.context("missing owned sentinel")?;
    ensure!(
        fs::read(&sentinel_path)? == bytes,
        "uninstall removed/changed the unrelated file fixture"
    );
    report["preserved_user_file"] =
        json!({"path":sentinel_path,"identity":digest(&sentinel_path)?});
    // Keep the retained user-file fixture as evidence. Never remove the directory or rewrite PATH.
    Ok(())
}

#[cfg(not(windows))]
fn lifecycle(_args: &Args, _report: &mut Value) -> Result<()> {
    anyhow::bail!("real installer lifecycle requires native Windows")
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    ensure!(
        args.timeout_secs > 0 && args.startup_timeout_secs > 0,
        "timeouts must be positive"
    );
    ensure!(
        args.max_tokens > 0 && args.context_tokens > args.max_tokens,
        "invalid inference token capacity"
    );
    fs::create_dir(&args.report_dir).context("report directory must be new")?;
    args.report_dir = args.report_dir.canonicalize()?;
    let mut report = json!({"schema_version":1,"status":"running","scope":"first_install_same_setup_reinstall_uninstall","automatic_failure_cleanup":false});
    let result = (|| {
        args.setup = args.setup.canonicalize()?;
        args.payload_dir = args.payload_dir.canonicalize()?;
        args.launcher = args.launcher.canonicalize()?;
        let launcher = digest(&args.launcher)?;
        ensure!(
            launcher.sha256 == args.launcher_sha256,
            "launcher SHA-256 does not match explicit input"
        );
        report["launcher"] = json!({"path":args.launcher,"identity":launcher});
        args.model = args.model.as_ref().map(|p| p.canonicalize()).transpose()?;
        args.previous_setup = args
            .previous_setup
            .as_ref()
            .map(|p| p.canonicalize())
            .transpose()?;
        args.previous_payload_dir = args
            .previous_payload_dir
            .as_ref()
            .map(|p| p.canonicalize())
            .transpose()?;
        let identity = digest(&args.setup)?;
        ensure!(
            identity.sha256 == args.setup_sha256,
            "setup SHA-256 does not match explicit input"
        );
        report["setup"] = json!({"path":args.setup,"identity":identity});
        lifecycle(&args, &mut report)
    })();
    report["status"] = json!(if result.is_ok() { "passed" } else { "failed" });
    report["error"] = json!(result.as_ref().err().map(|e| format!("{e:#}")));
    save(&args.report_dir, "report.json", &report)?;
    result
}
