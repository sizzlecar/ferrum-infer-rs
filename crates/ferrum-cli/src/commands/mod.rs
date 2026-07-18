//! CLI Commands - Ollama-style interface

use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SequenceFitPolicyArg {
    FullInputMustFit,
    ImmediateOnly,
}

impl SequenceFitPolicyArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::FullInputMustFit => "full-input-must-fit",
            Self::ImmediateOnly => "immediate-only",
        }
    }
}

pub mod bench;
pub mod bench_serve;
pub mod embed;
pub mod list;
pub mod pull;
pub mod replay_bundle;
pub mod run;
pub mod serve;
pub mod stop;
pub mod transcribe;
pub mod tts;
