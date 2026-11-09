//! Explicit process-lifetime representation reuse. This never selects another
//! precision or skips the model's native prompt-expansion protocol.
use super::*;
use ferrum_engine::image_generation::p_h3::PH3ImageAssets;
use ferrum_quantization::ProcessMappedDenseUploadPolicy;

#[derive(Debug, Default)]
pub enum PH3ImagePromptExpansionWeightCacheConfig {
    #[default]
    Disabled,
    /// The first admitted expansion writes an anonymous private safetensors
    /// file; later requests reuse its immutable mapping. The file is released
    /// with the process. OS page-cache residency is not a Host-ledger promise.
    ProcessMapped {
        directory: PathBuf,
        maximum_file_bytes: u64,
        maximum_metadata_bytes: usize,
        upload_policy: ProcessMappedDenseUploadPolicy,
    },
}

impl<'de> Deserialize<'de> for PH3ImagePromptExpansionWeightCacheConfig {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        // Explicit newtype content prevents an internally tagged unit variant
        // from discarding fields that would silently disable a requested cache.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Empty {}
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Disabled(Empty),
            ProcessMapped {
                directory: PathBuf,
                maximum_file_bytes: u64,
                maximum_metadata_bytes: usize,
                #[serde(default)]
                upload_policy: ProcessMappedDenseUploadPolicy,
            },
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Disabled(_) => Self::Disabled,
            Wire::ProcessMapped {
                directory,
                maximum_file_bytes,
                maximum_metadata_bytes,
                upload_policy,
            } => Self::ProcessMapped {
                directory,
                maximum_file_bytes,
                maximum_metadata_bytes,
                upload_policy,
            },
        })
    }
}

impl PH3ImagePromptExpansionWeightCacheConfig {
    pub(super) fn validate(&self) -> Result<()> {
        if let Self::ProcessMapped {
            directory,
            maximum_file_bytes,
            maximum_metadata_bytes,
            ..
        } = self
        {
            ensure!(
                !directory.as_os_str().is_empty(),
                "process-mapped weights need an explicit directory"
            );
            ferrum_quantization::ProcessMappedDenseLimits {
                maximum_file_bytes: *maximum_file_bytes,
                maximum_metadata_bytes: *maximum_metadata_bytes,
            }
            .validate()
            .map_err(anyhow::Error::msg)?;
        }
        Ok(())
    }

    pub(super) fn resolve_paths(&mut self, base: &Path) {
        if let Self::ProcessMapped { directory, .. } = self {
            if directory.is_relative() && !directory.as_os_str().is_empty() {
                *directory = base.join(&*directory);
            }
        }
    }

    pub(super) fn apply(&self, assets: PH3ImageAssets) -> Result<PH3ImageAssets> {
        self.validate()?;
        match self {
            Self::Disabled => Ok(assets),
            Self::ProcessMapped {
                directory,
                maximum_file_bytes,
                maximum_metadata_bytes,
                upload_policy,
            } => assets.with_prompt_expansion_process_mapped_weights_and_upload_policy(
                directory.clone(),
                ferrum_quantization::ProcessMappedDenseLimits {
                    maximum_file_bytes: *maximum_file_bytes,
                    maximum_metadata_bytes: *maximum_metadata_bytes,
                },
                *upload_policy,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_mapped_configuration_is_explicit_bounded_and_resolves_relative_directory() {
        let mut config: PH3ImagePromptExpansionWeightCacheConfig =
            serde_json::from_value(serde_json::json!({
                "kind":"process_mapped", "directory":"weights-cache",
                "maximum_file_bytes":4096, "maximum_metadata_bytes":1024,
            }))
            .unwrap();
        config.validate().unwrap();
        config.resolve_paths(Path::new("/configuration"));
        let PH3ImagePromptExpansionWeightCacheConfig::ProcessMapped {
            directory,
            upload_policy,
            ..
        } = config
        else {
            panic!("explicit mapped policy")
        };
        assert_eq!(directory, Path::new("/configuration/weights-cache"));
        assert_eq!(
            upload_policy,
            ProcessMappedDenseUploadPolicy::ReadOnlyMapping
        );
        for (field, value) in [
            ("directory", serde_json::json!("")),
            ("maximum_file_bytes", serde_json::json!(0)),
            ("maximum_metadata_bytes", serde_json::json!(0)),
        ] {
            let mut value_config = serde_json::json!({"kind":"process_mapped","directory":"weights-cache","maximum_file_bytes":4096,"maximum_metadata_bytes":1024});
            value_config[field] = value;
            assert!(
                serde_json::from_value::<PH3ImagePromptExpansionWeightCacheConfig>(value_config)
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
        assert!(
            serde_json::from_value::<PH3ImagePromptExpansionWeightCacheConfig>(
                serde_json::json!({"kind":"process_mapped","directory":"weights-cache"})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<PH3ImagePromptExpansionWeightCacheConfig>(
                serde_json::json!({"kind":"disabled","directory":"weights-cache"})
            )
            .is_err()
        );
    }

    #[test]
    fn process_mapped_upload_policy_is_explicit_and_rejects_unknown_values() {
        let mut value = serde_json::json!({
            "kind":"process_mapped", "directory":"weights-cache",
            "maximum_file_bytes":4096, "maximum_metadata_bytes":1024,
            "upload_policy":"staged",
        });
        let config: PH3ImagePromptExpansionWeightCacheConfig =
            serde_json::from_value(value.clone()).unwrap();
        config.validate().unwrap();
        assert!(matches!(
            config,
            PH3ImagePromptExpansionWeightCacheConfig::ProcessMapped {
                upload_policy: ProcessMappedDenseUploadPolicy::Staged,
                ..
            }
        ));
        value["upload_policy"] = serde_json::json!("automatic");
        assert!(serde_json::from_value::<PH3ImagePromptExpansionWeightCacheConfig>(value).is_err());
        assert!(
            serde_json::from_value::<PH3ImagePromptExpansionWeightCacheConfig>(
                serde_json::json!({"kind":"disabled", "upload_policy":"staged"})
            )
            .is_err()
        );
    }
}
