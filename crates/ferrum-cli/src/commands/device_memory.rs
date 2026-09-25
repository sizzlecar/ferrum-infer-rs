//! Shared validation for the explicit run/serve device allocation sampler.

use ferrum_types::{
    Device, DeviceMemorySamplingConfig, ExecutionResourceAuthority, FerrumError, Result,
};
use std::path::Path;

pub(super) const HELP: &str = "Write native Metal device-allocation samples every 250 ms before weight loading through shutdown. Reports MTLDevice.currentAllocatedSize, not RSS or total physical VRAM. Sampled peaks exclude device initialization and may miss shorter peaks. Requires a new file separate from other diagnostics.";

pub(super) fn validate_output_paths<'a>(
    path: Option<&Path>,
    other_outputs: impl IntoIterator<Item = (&'a str, Option<&'a Path>)>,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    DeviceMemorySamplingConfig {
        jsonl_path: path.to_owned(),
    }
    .validate_output_paths(
        other_outputs
            .into_iter()
            .filter_map(|(label, path)| path.map(|path| (label, path))),
    )
    .map_err(FerrumError::config)
}

pub(super) fn resolve(
    path: Option<&Path>,
    device: &Device,
) -> Result<Option<DeviceMemorySamplingConfig>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let config = DeviceMemorySamplingConfig {
        jsonl_path: path.to_owned(),
    };
    config.validate().map_err(FerrumError::config)?;
    let supported = match device {
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        Device::Metal => true,
        _ => false,
    };
    if !supported {
        return Err(FerrumError::unsupported(format!(
            "--device-memory-jsonl requires a supported native Metal backend; selected device is {device}"
        )));
    }
    Ok(Some(config))
}

pub(super) fn validate_authority(
    config: Option<&DeviceMemorySamplingConfig>,
    authority: ExecutionResourceAuthority,
) -> Result<()> {
    if config.is_some() && authority != ExecutionResourceAuthority::PlanRuntime {
        return Err(FerrumError::unsupported(
            "--device-memory-jsonl requires a native plan runtime; legacy execution does not expose device allocation sampling",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Parser, Subcommand};
    use std::path::PathBuf;

    #[derive(Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: TestCommand,
    }

    #[derive(Subcommand)]
    enum TestCommand {
        Run(crate::commands::run::RunCommand),
        Serve(crate::commands::serve::ServeCommand),
    }

    fn parsed_path(command: &str, extra: &[&str]) -> Option<PathBuf> {
        let mut args = vec!["ferrum", command, "test-model"];
        args.extend_from_slice(extra);
        let parsed = TestCli::try_parse_from(args).unwrap();
        match parsed.command {
            TestCommand::Run(command) => {
                assert!(command.memory_profile_jsonl.is_none());
                assert!(command.profile_jsonl.is_none());
                command.device_memory_jsonl
            }
            TestCommand::Serve(command) => {
                assert!(command.memory_profile_jsonl.is_none());
                assert!(command.profile_jsonl.is_none());
                command.device_memory_jsonl
            }
        }
    }

    #[test]
    fn device_memory_sampling_run_and_serve_default_off() {
        for command in ["run", "serve"] {
            let path = parsed_path(command, &[]);
            assert!(path.is_none());
            assert!(resolve(path.as_deref(), &Device::CPU).unwrap().is_none());
        }
    }

    #[test]
    fn device_memory_sampling_run_and_serve_reject_unsupported_backends() {
        for command in ["run", "serve"] {
            let path = parsed_path(command, &["--device-memory-jsonl", "samples.jsonl"]);
            let mut devices = vec![Device::CPU, Device::ROCm(0), Device::CUDA(0)];
            if !cfg!(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios")
            )) {
                devices.push(Device::Metal);
            }
            for device in devices {
                let error = resolve(path.as_deref(), &device).unwrap_err();
                assert!(error.to_string().contains("requires a supported native"));
            }
        }
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn device_memory_sampling_run_and_serve_resolve_the_same_typed_config() {
        for command in ["run", "serve"] {
            let path = parsed_path(
                command,
                &["--device-memory-jsonl", "evidence/samples.jsonl"],
            );
            let config = resolve(path.as_deref(), &Device::Metal).unwrap().unwrap();
            assert_eq!(config.jsonl_path, PathBuf::from("evidence/samples.jsonl"));
            config.validate().unwrap();
        }
    }

    #[test]
    fn device_memory_sampling_rejects_empty_path() {
        assert!(resolve(Some(Path::new("")), &Device::CPU)
            .unwrap_err()
            .to_string()
            .contains("must not be empty"));
    }

    #[test]
    fn device_memory_sampling_rejects_legacy_without_changing_default_execution() {
        let config = DeviceMemorySamplingConfig {
            jsonl_path: PathBuf::from("samples.jsonl"),
        };
        validate_authority(None, ExecutionResourceAuthority::LegacyEngine).unwrap();
        validate_authority(Some(&config), ExecutionResourceAuthority::PlanRuntime).unwrap();
        assert!(
            validate_authority(Some(&config), ExecutionResourceAuthority::LegacyEngine)
                .unwrap_err()
                .to_string()
                .contains("legacy execution")
        );
    }

    #[tokio::test]
    async fn device_memory_sampling_entrypoints_reject_cpu_before_model_resolution() {
        for command in ["run", "serve"] {
            let parsed = TestCli::try_parse_from([
                "ferrum",
                command,
                "/device-memory-unresolved-model.gguf",
                "--backend",
                "cpu",
                "--device-memory-jsonl",
                "samples.jsonl",
            ])
            .unwrap();
            let config = crate::config::CliConfig::default();
            let result = match parsed.command {
                TestCommand::Run(command) => crate::commands::run::execute(command, config).await,
                TestCommand::Serve(command) => {
                    crate::commands::serve::execute(command, config).await
                }
            };
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("requires a supported native Metal backend"));
        }
    }

    #[tokio::test]
    async fn device_memory_sampling_entrypoints_reject_shared_diagnostic_outputs() {
        for command in ["run", "serve"] {
            for flag in [
                "--profile-jsonl",
                "--memory-profile-jsonl",
                "--scheduler-trace-jsonl",
                "--effective-config-json",
                "--decision-trace-jsonl",
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "/device-memory-unresolved-model.gguf",
                    "--backend",
                    "cpu",
                    "--device-memory-jsonl",
                    "samples.jsonl",
                    flag,
                    "./samples.jsonl",
                ])
                .unwrap();
                let config = crate::config::CliConfig::default();
                let result = match parsed.command {
                    TestCommand::Run(command) => {
                        crate::commands::run::execute(command, config).await
                    }
                    TestCommand::Serve(command) => {
                        crate::commands::serve::execute(command, config).await
                    }
                };
                assert!(
                    result.unwrap_err().to_string().contains("conflicts with"),
                    "{command} {flag}"
                );
            }
        }
        validate_output_paths(None, [("profile", Some(Path::new("samples.jsonl")))]).unwrap();
        validate_output_paths(
            Some(Path::new("samples.jsonl")),
            [("profile", Some(Path::new("profile.jsonl")))],
        )
        .unwrap();
    }

    #[test]
    fn device_memory_sampling_cannot_be_silently_ignored_by_synthetic_slice() {
        for command in ["run", "serve"] {
            let error = TestCli::try_parse_from([
                "ferrum",
                command,
                "--observability-vertical-slice-out",
                "synthetic-evidence",
                "--device-memory-jsonl",
                "samples.jsonl",
            ])
            .err()
            .expect("synthetic slices do not create a device runtime");
            assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
        }
    }
}
