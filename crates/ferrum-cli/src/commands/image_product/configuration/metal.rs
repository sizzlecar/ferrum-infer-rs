use super::*;
use ferrum_engine::image_generation::{
    ImagePrecisionPolicy, ImageStreamLimit, ImageStreamRequirements,
};
use std::num::NonZeroU64;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MetalMemoryConfig {
    ExplicitTotal {
        bytes: NonZeroU64,
    },
    AutoRecommended {
        utilization: f64,
        maximum_bytes: Option<NonZeroU64>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetalConfig {
    pub memory: MetalMemoryConfig,
    #[serde(default)]
    pub precision_policy: ImagePrecisionPolicy,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    #[serde(default)]
    pub maximum_streams: ImageStreamLimit,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    pub host: HostConfig,
    pub pipeline: PipelineConfig,
}
impl MetalConfig {
    /// Wire/config validation is available even in binaries without Metal.
    /// Native capacity/command ABI validation follows before device creation.
    pub fn validate(&self) -> Result<()> {
        let limits = self.pipeline.resolved_limits()?;
        let host = self.host.limits()?;
        ensure!(
            !self.precision_policy.uses_fp8()
                || host.class_bytes(ferrum_runtime::HostMemoryClass::WeightBacking) > 0,
            "Metal FP8 conversion requires a nonzero host weight backing allowance"
        );
        ensure!(
            host.class_bytes(ferrum_runtime::HostMemoryClass::InputPreprocess) >= 16
                && host.class_bytes(ferrum_runtime::HostMemoryClass::Output) > 0,
            "Metal image host input and output allowances must be positive"
        );
        let ceiling = match self.memory {
            MetalMemoryConfig::ExplicitTotal { bytes } => bytes.get(),
            MetalMemoryConfig::AutoRecommended {
                utilization,
                maximum_bytes,
            } => {
                ensure!(
                    utilization.is_finite() && utilization > 0.0 && utilization <= 1.0,
                    "Metal working-set utilization must be in (0,1]"
                );
                maximum_bytes.map_or(u64::MAX, NonZeroU64::get)
            }
        };
        let streams = self
            .maximum_streams
            .resolve(ImageStreamRequirements::for_serial_pipeline(
                limits.static_initialization,
                limits.executor,
            ))?
            .resolved_streams();
        let opaque = self
            .stream_memory_bytes
            .checked_mul(u64::try_from(streams)?)
            .and_then(|n| n.checked_add(self.bootstrap_memory_bytes))
            .ok_or_else(|| anyhow::anyhow!("Metal opaque allowance overflows"))?;
        ensure!(
            self.bootstrap_memory_bytes > 0 && self.stream_memory_bytes > 0 && opaque < ceiling,
            "Metal opaque allowances leave no tensor capacity"
        );
        ensure!(
            self.maximum_in_flight_per_stream > 0
                && self.maximum_commands_per_submission > 0
                && self.maximum_commands_per_submission <= u32::MAX as usize
                && limits.static_initialization.maximum_commands_per_batch()
                    <= self.maximum_commands_per_submission,
            "Metal command and in-flight bounds are invalid"
        );
        ensure!(
            !self.opaque_memory_profile.is_empty()
                && self.opaque_memory_profile.len() <= 256
                && self.opaque_memory_profile.is_ascii()
                && !self
                    .opaque_memory_profile
                    .bytes()
                    .any(|b| b.is_ascii_control()),
            "Metal requires a bounded ASCII opaque memory profile"
        );
        Ok(())
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn resolved(
        &self,
    ) -> Result<ferrum_engine::image_generation::metal::QwenImage21MetalConfig> {
        use ferrum_engine::image_generation::metal::{
            ImageMetalMemoryPolicy, QwenImage21MetalConfig,
        };
        self.validate()?;
        let config = QwenImage21MetalConfig {
            memory: match self.memory {
                MetalMemoryConfig::ExplicitTotal { bytes } => {
                    ImageMetalMemoryPolicy::ExplicitTotal(bytes)
                }
                MetalMemoryConfig::AutoRecommended {
                    utilization,
                    maximum_bytes,
                } => ImageMetalMemoryPolicy::AutoRecommended {
                    utilization,
                    maximum_bytes,
                },
            },
            precision_policy: self.precision_policy,
            bootstrap_memory_bytes: self.bootstrap_memory_bytes,
            stream_memory_bytes: self.stream_memory_bytes,
            maximum_streams: self.maximum_streams,
            maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
            maximum_commands_per_submission: self.maximum_commands_per_submission,
            opaque_memory_profile: self.opaque_memory_profile.clone(),
            host: self.host.limits()?,
            pipeline: self.pipeline.resolved_limits()?,
        };
        config.validate()?;
        Ok(config)
    }
}
