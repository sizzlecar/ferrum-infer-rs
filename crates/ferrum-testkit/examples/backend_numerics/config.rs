use ferrum_testkit::op_diff::{
    gemm::GemmOp,
    metal_context::{MetalContextOp, SUBMISSION_PHASES},
    required::{run_required, RequiredBackend, RequiredReport},
    rms_norm::RmsNormOp,
    silu_mul::SiluMulOp,
    NMSE_FP16_TOL, NMSE_FP32_TOL,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub(super) const USAGE: &str = "backend_numerics --require-backend metal|cuda --op OP --report PATH \
    [--seed N] [--max-nmse VALUE]\n\
    Operators and defaults:\n\
      rms-norm [--tokens 4] [--dim 128] [--eps 1e-6]\n\
      gemm [--m 64] [--n 32] [--k 32]    C[m,n] = A[m,k] * B[n,k]^T\n\
      silu-mul [--tokens 4] [--intermediate 256]\n\
      metal-context [--tokens 3] [--intermediate 33] [--k 35] (Metal only)\n\
    Seed defaults to 42; max-nmse=1e-7 for Metal F32, 1e-6 for CUDA F16.\n\
    Operator-specific options cannot be used with a different operator.\n\
    The report must be a new file in an existing directory. NotRun and Failed exit nonzero.\n\
    This checks the selected Backend trait fixture, not all model/runtime/precision paths or performance.";

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "kebab-case")]
pub(super) enum Operation {
    RmsNorm {
        tokens: usize,
        dim: usize,
        eps: f32,
    },
    Gemm {
        m: usize,
        n: usize,
        k: usize,
    },
    SiluMul {
        tokens: usize,
        intermediate: usize,
    },
    MetalContext {
        tokens: usize,
        intermediate: usize,
        k: usize,
    },
}

impl Operation {
    pub(super) fn output_shape(&self) -> [usize; 2] {
        match *self {
            Self::MetalContext {
                tokens,
                intermediate,
                ..
            } => [SUBMISSION_PHASES.len(), tokens.saturating_mul(intermediate)],
            Self::RmsNorm { tokens, dim, .. } => [tokens, dim],
            Self::Gemm { m, n, .. } => [m, n],
            Self::SiluMul {
                tokens,
                intermediate,
            } => [tokens, intermediate],
        }
    }

    pub(super) fn execution_path(&self) -> &'static str {
        match self {
            Self::MetalContext { .. } => {
                "MetalContext::submit_and_wait (Backend F32 compute/blit/compute)"
            }
            Self::RmsNorm { .. } => "Backend::rms_norm",
            Self::Gemm { .. } => "Backend::gemm",
            Self::SiluMul { .. } => "Backend::fused_silu_mul_split",
        }
    }

    fn expected_output_len(&self) -> Result<usize, String> {
        match *self {
            Self::MetalContext {
                tokens,
                intermediate,
                k,
            } => MetalContextOp {
                tokens,
                intermediate,
                k,
            }
            .expected_output_len(),
            Self::RmsNorm { tokens, dim, eps } => {
                RmsNormOp { tokens, dim, eps }.expected_output_len()
            }
            Self::Gemm { m, n, k } => GemmOp { m, n, k }.expected_output_len(),
            Self::SiluMul {
                tokens,
                intermediate,
            } => SiluMulOp {
                tokens,
                intermediate,
            }
            .expected_output_len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct Config {
    pub(super) require_backend: RequiredBackend,
    // Flattening preserves the original RMSNorm report's op/tokens/dim/eps.
    #[serde(flatten)]
    pub(super) op: Operation,
    pub(super) seed: u64,
    pub(super) max_nmse: f64,
}

impl Config {
    pub(super) fn validate(&self) -> Result<usize, String> {
        if !self.max_nmse.is_finite() || self.max_nmse <= 0.0 {
            return Err("--max-nmse must be finite and strictly positive".into());
        }
        if matches!(self.op, Operation::MetalContext { .. })
            && self.require_backend != RequiredBackend::Metal
        {
            return Err("metal-context requires the Metal backend".into());
        }
        self.op.expected_output_len()
    }

    pub(super) fn execute(&self) -> RequiredReport {
        match self.op {
            Operation::MetalContext {
                tokens,
                intermediate,
                k,
            } => run_required(
                &MetalContextOp {
                    tokens,
                    intermediate,
                    k,
                },
                self.require_backend,
                self.seed,
                self.max_nmse,
            ),
            Operation::RmsNorm { tokens, dim, eps } => run_required(
                &RmsNormOp { tokens, dim, eps },
                self.require_backend,
                self.seed,
                self.max_nmse,
            ),
            Operation::Gemm { m, n, k } => run_required(
                &GemmOp { m, n, k },
                self.require_backend,
                self.seed,
                self.max_nmse,
            ),
            Operation::SiluMul {
                tokens,
                intermediate,
            } => run_required(
                &SiluMulOp {
                    tokens,
                    intermediate,
                },
                self.require_backend,
                self.seed,
                self.max_nmse,
            ),
        }
    }

    pub(super) fn precision(&self) -> Precision {
        let storage = match self.require_backend {
            RequiredBackend::Metal => StoragePrecision::F32,
            RequiredBackend::Cuda => StoragePrecision::F16,
        };
        let kernel_entrypoint = match (&self.op, self.require_backend) {
            (Operation::MetalContext { tokens: 1, .. }, _) => {
                "gemv_f32 -> blit -> silu_mul_split_f32 (legacy MetalContext)"
            }
            (Operation::MetalContext { .. }, _) => {
                "gemm_f32_v2 -> blit -> silu_mul_split_f32 (legacy MetalContext)"
            }
            (Operation::RmsNorm { .. }, RequiredBackend::Metal) => "rms_norm_f32",
            (Operation::RmsNorm { .. }, RequiredBackend::Cuda) => "rms_norm_f16",
            (Operation::Gemm { m: 1, .. }, RequiredBackend::Metal) => "gemv_f32",
            (Operation::Gemm { .. }, RequiredBackend::Metal) => "gemm_f32_v2",
            (Operation::Gemm { .. }, RequiredBackend::Cuda) => {
                "cublasGemmEx(CUBLAS_COMPUTE_32F_FAST_16F)"
            }
            (Operation::SiluMul { .. }, RequiredBackend::Metal) => "silu_mul_split_f32",
            (Operation::SiluMul { .. }, RequiredBackend::Cuda) => "fused_silu_mul_interleaved_f16",
        };
        Precision {
            reference_storage: StoragePrecision::F32,
            backend_input_storage: storage,
            backend_output_storage: storage,
            kernel_entrypoint,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum StoragePrecision {
    F32,
    F16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Precision {
    pub(super) reference_storage: StoragePrecision,
    pub(super) backend_input_storage: StoragePrecision,
    pub(super) backend_output_storage: StoragePrecision,
    pub(super) kernel_entrypoint: &'static str,
}

#[derive(Debug, PartialEq)]
pub(super) struct Args {
    pub(super) config: Config,
    pub(super) report: PathBuf,
}

pub(super) fn parse_args(
    arguments: impl IntoIterator<Item = String>,
) -> Result<Option<Args>, String> {
    let arguments: Vec<_> = arguments.into_iter().collect();
    if arguments == ["--help"] || arguments == ["-h"] {
        return Ok(None);
    }
    let mut fields = BTreeMap::new();
    let mut arguments = arguments.into_iter();
    while let Some(key) = arguments.next() {
        if !matches!(
            key.as_str(),
            "--require-backend"
                | "--op"
                | "--report"
                | "--tokens"
                | "--dim"
                | "--eps"
                | "--m"
                | "--n"
                | "--k"
                | "--intermediate"
                | "--seed"
                | "--max-nmse"
        ) {
            return Err(format!("unknown option {key:?}\n{USAGE}"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if value.is_empty() || value.starts_with("--") {
            return Err(format!("missing value for {key}"));
        }
        if fields.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option {key}"));
        }
    }
    let backend = match fields.remove("--require-backend").as_deref() {
        Some("metal") => RequiredBackend::Metal,
        Some("cuda") => RequiredBackend::Cuda,
        Some(other) => return Err(format!("unsupported required backend {other:?}")),
        None => {
            return Err(
                "--require-backend is required; CPU cannot substitute for an accelerator".into(),
            )
        }
    };
    let op = match fields.remove("--op").as_deref() {
        Some("rms-norm") => Operation::RmsNorm {
            tokens: number(&mut fields, "--tokens", "4")?,
            dim: number(&mut fields, "--dim", "128")?,
            eps: number(&mut fields, "--eps", "1e-6")?,
        },
        Some("gemm") => Operation::Gemm {
            m: number(&mut fields, "--m", "64")?,
            n: number(&mut fields, "--n", "32")?,
            k: number(&mut fields, "--k", "32")?,
        },
        Some("metal-context") => Operation::MetalContext {
            tokens: number(&mut fields, "--tokens", "3")?,
            intermediate: number(&mut fields, "--intermediate", "33")?,
            k: number(&mut fields, "--k", "35")?,
        },
        Some("silu-mul") => Operation::SiluMul {
            tokens: number(&mut fields, "--tokens", "4")?,
            intermediate: number(&mut fields, "--intermediate", "256")?,
        },
        Some(other) => return Err(format!("unsupported operator {other:?}")),
        None => return Err("--op is required".into()),
    };
    let report = fields.remove("--report").ok_or("--report is required")?;
    let max_nmse = match fields.remove("--max-nmse") {
        Some(value) => value.parse().map_err(|_| "--max-nmse must be a float")?,
        None => match backend {
            RequiredBackend::Metal => NMSE_FP32_TOL,
            RequiredBackend::Cuda => NMSE_FP16_TOL,
        },
    };
    let config = Config {
        require_backend: backend,
        op,
        seed: number(&mut fields, "--seed", "42")?,
        max_nmse,
    };
    if let Some(option) = fields.keys().next() {
        return Err(format!(
            "option {option} does not apply to the selected operator"
        ));
    }
    config.validate()?;
    Ok(Some(Args {
        config,
        report: report.into(),
    }))
}

fn number<T: std::str::FromStr>(
    fields: &mut BTreeMap<String, String>,
    key: &str,
    default: &str,
) -> Result<T, String> {
    fields
        .remove(key)
        .unwrap_or_else(|| default.into())
        .parse()
        .map_err(|_| format!("invalid numeric value for {key}"))
}
