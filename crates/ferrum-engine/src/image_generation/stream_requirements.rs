//! Managed stream capacity derived from the serial image execution schedule.
//! Static initialization completes before execution, and the service worker
//! finishes a batch before starting another. Failed work stops the pipeline.
//! The runtime's bootstrap allocation stream and opaque library internals are
//! separate owners, not additional execution-lane slots.

use anyhow::{ensure, Result};
use ferrum_execution::TensorExecutorConfig;
use ferrum_runtime::{BlockingStreamRequirement, StaticInitializationPolicy};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageStreamRequirements {
    static_initialization: BlockingStreamRequirement,
    tensor_execution: BlockingStreamRequirement,
    request_state: BlockingStreamRequirement,
}

impl ImageStreamRequirements {
    /// Applies only to the blocking, serial component schedule used by the
    /// image pipeline and its single service worker. Parallel execution needs
    /// an explicit schedule that accounts for overlapping requirements.
    pub fn for_serial_pipeline(
        initialization: StaticInitializationPolicy,
        executor: TensorExecutorConfig,
    ) -> Self {
        Self {
            static_initialization: initialization.stream_requirement(),
            tensor_execution: executor.execution_stream_requirement(),
            request_state: executor.request_state_stream_requirement(),
        }
    }

    pub fn required_streams(self) -> usize {
        self.static_initialization
            .maximum_live_streams()
            .max(self.tensor_execution.maximum_live_streams())
            .max(self.request_state.maximum_live_streams())
    }
}

/// Omission or `"auto"` uses the schedule requirement. A positive integer
/// preserves an explicit caller ceiling without inventing extra parallel work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageStreamLimit {
    #[default]
    Auto,
    Explicit(usize),
}

impl Serialize for ImageStreamLimit {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Explicit(limit) => limit.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ImageStreamLimit {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Number(usize),
            Name(String),
        }
        match Wire::deserialize(deserializer)? {
            Wire::Number(limit) if limit > 0 => Ok(Self::Explicit(limit)),
            Wire::Name(name) if name == "auto" => Ok(Self::Auto),
            _ => Err(serde::de::Error::custom(
                "maximum_streams must be auto or a positive integer",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStreamLimitSource {
    Auto,
    Explicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageStreamPlan {
    source: ImageStreamLimitSource,
    required_streams: usize,
    resolved_streams: usize,
    requirements: ImageStreamRequirements,
}

impl ImageStreamPlan {
    pub const fn source(self) -> ImageStreamLimitSource {
        self.source
    }
    pub const fn required_streams(self) -> usize {
        self.required_streams
    }
    pub const fn resolved_streams(self) -> usize {
        self.resolved_streams
    }
}

impl ImageStreamLimit {
    pub fn resolve(self, requirements: ImageStreamRequirements) -> Result<ImageStreamPlan> {
        let required_streams = requirements.required_streams();
        let (source, resolved_streams) = match self {
            Self::Auto => (ImageStreamLimitSource::Auto, required_streams),
            Self::Explicit(limit) => (ImageStreamLimitSource::Explicit, limit),
        };
        ensure!(
            resolved_streams > 0 && resolved_streams >= required_streams,
            "CUDA stream limit {resolved_streams} is below the serial schedule requirement {required_streams}"
        );
        Ok(ImageStreamPlan {
            source,
            required_streams,
            resolved_streams,
            requirements,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_schedule_uses_executor_declarations_and_does_not_sum_disjoint_phases() {
        let requirements = ImageStreamRequirements::for_serial_pipeline(
            StaticInitializationPolicy::new(1024, 8).unwrap(),
            TensorExecutorConfig::default(),
        );
        let plan = ImageStreamLimit::Auto.resolve(requirements).unwrap();
        assert_eq!(plan.required_streams(), 1);
        assert_eq!(plan.resolved_streams(), 1);
        assert_eq!(plan.source(), ImageStreamLimitSource::Auto);
        assert_eq!(requirements.request_state.maximum_live_streams(), 0);
        let wider = ImageStreamRequirements {
            static_initialization: BlockingStreamRequirement::new(3),
            tensor_execution: BlockingStreamRequirement::new(2),
            request_state: BlockingStreamRequirement::new(0),
        };
        assert_eq!(
            ImageStreamLimit::Auto
                .resolve(wider)
                .unwrap()
                .resolved_streams(),
            3
        );
        assert!(ImageStreamLimit::Explicit(2).resolve(wider).is_err());
        let explicit = ImageStreamLimit::Explicit(5).resolve(wider).unwrap();
        assert_eq!(explicit.required_streams(), 3);
        assert_eq!(explicit.resolved_streams(), 5);
        assert_eq!(explicit.source(), ImageStreamLimitSource::Explicit);
        assert!(ImageStreamLimit::Explicit(0).resolve(requirements).is_err());
        let receipt = serde_json::to_value(plan).unwrap();
        assert_eq!(receipt["source"], "auto");
        assert_eq!(receipt["required_streams"], 1);
        assert_eq!(receipt["resolved_streams"], 1);
    }

    #[test]
    fn stream_limit_wire_preserves_positive_integers_and_rejects_invalid_values() {
        for policy in [ImageStreamLimit::Auto, ImageStreamLimit::Explicit(4)] {
            assert_eq!(
                serde_json::from_value::<ImageStreamLimit>(serde_json::to_value(policy).unwrap())
                    .unwrap(),
                policy
            );
        }
        for wire in ["0", "-1", "1.5", "null", "{}", "\"automatic\"", "\"2\""] {
            assert!(serde_json::from_str::<ImageStreamLimit>(wire).is_err());
        }
    }
}
