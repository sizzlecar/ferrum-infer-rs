//! Closed admission policy for sharing a bounded tensor runtime with the
//! dense F16 causal providers. This is a declaration, not measured qualification.
use ferrum_runtime::{BufferUsage, CapabilityId, DeviceCommandPhase};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CudaTensorExecutionProfile {
    #[default]
    TensorOnly,
    TensorAndDenseF16CausalV1,
    TensorAndDenseF16VisualCausalV1,
    TensorAndDenseBf16CausalV1,
    TensorAndDenseBf16VisualCausalV1,
}

/// Additional allowance for the exact dense-F16 portable causal provider set.
/// Modules are admitted separately from the tensor provider's bootstrap slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDenseCausalProfile {
    module_memory_bytes: u64,
    opaque_memory_profile: String,
    visual_prompt: bool,
    decode_replay: Option<CudaCausalDecodeReplay>,
    bf16: bool,
}

/// Compatibility name; `new` retains the original F16 execution profile.
pub type CudaDenseF16CausalProfile = CudaDenseCausalProfile;

/// Bounded graph inventory on each causal lane, charged to its original account.
/// Opaque bytes are a declared allowance, not a measured CUDA memory profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaCausalDecodeReplay {
    pub maximum_executables: usize,
    pub opaque_memory_bytes: u64,
}

impl CudaCausalDecodeReplay {
    pub const BLAS_WORKSPACE_BYTES: u64 = 32 << 20;

    pub fn memory_bytes_per_stream(self) -> u64 {
        self.opaque_memory_bytes + Self::BLAS_WORKSPACE_BYTES
    }
}

/// An N2 continuation requires the exact work declared by Core's admitted
/// reusable program. An untagged prefill cannot acquire replay from its size.
pub(crate) fn permits_command_replay(
    enabled: bool,
    phase: DeviceCommandPhase,
    participants: u32,
    tokens: u64,
    admitted_shape: Option<(u32, u64)>,
) -> bool {
    if !enabled {
        return false;
    }
    if phase != DeviceCommandPhase::Compute {
        return true;
    }
    if participants == 0 {
        return false;
    }
    let n1 = u64::from(participants);
    match admitted_shape {
        None => tokens == n1,
        Some(shape) => shape == (participants, tokens) && (tokens == n1 || tokens == 2 * n1),
    }
}

impl CudaDenseCausalProfile {
    pub fn new(
        module_memory_bytes: u64,
        opaque_memory_profile: impl Into<String>,
    ) -> Result<Self, String> {
        let opaque_memory_profile = opaque_memory_profile.into();
        if module_memory_bytes == 0
            || opaque_memory_profile.is_empty()
            || opaque_memory_profile.len() > 256
            || !opaque_memory_profile.is_ascii()
            || opaque_memory_profile.bytes().any(|b| b.is_ascii_control())
        {
            return Err("CUDA causal modules require a positive allowance and a bounded ASCII profile identity".into());
        }
        Ok(Self {
            module_memory_bytes,
            opaque_memory_profile,
            visual_prompt: false,
            decode_replay: None,
            bf16: false,
        })
    }
    pub fn new_bf16(
        module_memory_bytes: u64,
        opaque_memory_profile: impl Into<String>,
    ) -> Result<Self, String> {
        let mut profile = Self::new(module_memory_bytes, opaque_memory_profile)?;
        profile.bf16 = true;
        Ok(profile)
    }
    pub const fn is_bf16(&self) -> bool {
        self.bf16
    }
    pub const fn module_memory_bytes(&self) -> u64 {
        self.module_memory_bytes
    }
    pub fn opaque_memory_profile(&self) -> &str {
        &self.opaque_memory_profile
    }
    pub fn with_visual_prompt_v1(mut self) -> Self {
        self.visual_prompt = true;
        self
    }
    pub const fn is_visual_prompt_v1(&self) -> bool {
        self.visual_prompt
    }
    pub fn with_decode_replay(
        mut self,
        maximum_executables: usize,
        opaque_memory_bytes: u64,
    ) -> Result<Self, String> {
        if maximum_executables == 0
            || maximum_executables > u32::MAX as usize
            || opaque_memory_bytes == 0
            || opaque_memory_bytes
                .checked_add(CudaCausalDecodeReplay::BLAS_WORKSPACE_BYTES)
                .is_none_or(|bytes| bytes > isize::MAX as u64)
        {
            return Err(
                "causal decode replay requires bounded positive graph slots and opaque bytes"
                    .into(),
            );
        }
        self.decode_replay = Some(CudaCausalDecodeReplay {
            maximum_executables,
            opaque_memory_bytes,
        });
        Ok(self)
    }
    pub fn decode_replay_limits(&self) -> Option<(usize, u64)> {
        self.decode_replay
            .map(|p| (p.maximum_executables, p.opaque_memory_bytes))
    }
    /// Graph allowance plus the separately owned 32 MiB cuBLAS workspace.
    /// Eager-only profiles retain the vendor workspace policy and add no bytes.
    pub fn decode_replay_memory_bytes_per_stream(&self) -> u64 {
        self.decode_replay
            .map_or(0, CudaCausalDecodeReplay::memory_bytes_per_stream)
    }
    pub(crate) fn decode_replay(&self) -> Option<CudaCausalDecodeReplay> {
        self.decode_replay
    }
    pub(crate) const fn execution_profile(&self) -> CudaTensorExecutionProfile {
        if self.bf16 && self.visual_prompt {
            CudaTensorExecutionProfile::TensorAndDenseBf16VisualCausalV1
        } else if self.bf16 {
            CudaTensorExecutionProfile::TensorAndDenseBf16CausalV1
        } else if self.visual_prompt {
            CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1
        } else {
            CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
        }
    }
}

pub(crate) const DENSE_F16_CAUSAL_CAPABILITIES: [&str; 12] = [
    ferrum_runtime::TOKEN_EMBEDDING_F16_CAPABILITY_ID,
    ferrum_runtime::LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
    ferrum_runtime::LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID,
    ferrum_runtime::RMS_NORM_F16_CAPABILITY_ID,
    ferrum_runtime::DENSE_SWIGLU_F16_CAPABILITY_ID,
    ferrum_runtime::RESIDUAL_ADD_F16_CAPABILITY_ID,
    ferrum_runtime::GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
    ferrum_runtime::GATED_DELTA_CANDIDATE_SCAN_F16_CAPABILITY_ID,
    ferrum_runtime::CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
    ferrum_runtime::TAIL_TWO_GREEDY_F16_CAPABILITY_ID,
    ferrum_runtime::CONCAT_COLUMNS_F16_CAPABILITY_ID,
    ferrum_runtime::DENSE_LINEAR_F16_CAPABILITY_ID,
];
pub(crate) const DENSE_F16_VISUAL_CAUSAL_CAPABILITIES: [&str; 14] = [
    DENSE_F16_CAUSAL_CAPABILITIES[0],
    DENSE_F16_CAUSAL_CAPABILITIES[1],
    DENSE_F16_CAUSAL_CAPABILITIES[2],
    DENSE_F16_CAUSAL_CAPABILITIES[3],
    DENSE_F16_CAUSAL_CAPABILITIES[4],
    DENSE_F16_CAUSAL_CAPABILITIES[5],
    DENSE_F16_CAUSAL_CAPABILITIES[6],
    DENSE_F16_CAUSAL_CAPABILITIES[7],
    DENSE_F16_CAUSAL_CAPABILITIES[8],
    DENSE_F16_CAUSAL_CAPABILITIES[9],
    DENSE_F16_CAUSAL_CAPABILITIES[10],
    DENSE_F16_CAUSAL_CAPABILITIES[11],
    ferrum_runtime::ROW_SELECT_F16_CAPABILITY_ID,
    ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_CAPABILITY_ID,
];

pub(crate) const DENSE_BF16_CAUSAL_CAPABILITIES: [&str; 8] = [
    ferrum_runtime::TOKEN_EMBEDDING_BF16_CAPABILITY_ID,
    ferrum_runtime::LAST_TOKEN_DENSE_LINEAR_BF16_CAPABILITY_ID,
    ferrum_runtime::LAST_TOKEN_MASKED_ARGMAX_BF16_CAPABILITY_ID,
    ferrum_runtime::RMS_NORM_BF16_CAPABILITY_ID,
    ferrum_runtime::DENSE_SWIGLU_BF16_CAPABILITY_ID,
    ferrum_runtime::RESIDUAL_ADD_BF16_CAPABILITY_ID,
    ferrum_runtime::GATED_DELTA_RECURRENT_ATTENTION_BF16_CAPABILITY_ID,
    ferrum_runtime::CAUSAL_PAGED_ATTENTION_BF16_CAPABILITY_ID,
];

/// Native BF16 projection GEMMs require Ampere or newer. Check the actual
/// context before publishing these capabilities or installing any weights.
pub(crate) fn validate_causal_compute_capability(
    capabilities: &BTreeSet<CapabilityId>,
    major: i32,
) -> Result<(), String> {
    if major < 8
        && capabilities
            .iter()
            .any(|capability| DENSE_BF16_CAUSAL_CAPABILITIES.contains(&capability.as_str()))
    {
        return Err(
            "native BF16 causal execution requires CUDA compute capability 8.0 or newer".into(),
        );
    }
    Ok(())
}
pub(crate) const DENSE_BF16_VISUAL_CAUSAL_CAPABILITIES: [&str; 10] = [
    DENSE_BF16_CAUSAL_CAPABILITIES[0],
    DENSE_BF16_CAUSAL_CAPABILITIES[1],
    DENSE_BF16_CAUSAL_CAPABILITIES[2],
    DENSE_BF16_CAUSAL_CAPABILITIES[3],
    DENSE_BF16_CAUSAL_CAPABILITIES[4],
    DENSE_BF16_CAUSAL_CAPABILITIES[5],
    DENSE_BF16_CAUSAL_CAPABILITIES[6],
    DENSE_BF16_CAUSAL_CAPABILITIES[7],
    ferrum_runtime::ROW_SELECT_BF16_CAPABILITY_ID,
    ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_BF16_CAPABILITY_ID,
];

pub(crate) fn validate_capabilities(
    capabilities: &BTreeSet<CapabilityId>,
    profile: CudaTensorExecutionProfile,
) -> Result<(), String> {
    if profile == CudaTensorExecutionProfile::TensorOnly {
        return crate::backend::cuda_tensor_capability_policy::validate(capabilities);
    }
    let required: &[&str] = match profile {
        CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1 => {
            &DENSE_F16_VISUAL_CAUSAL_CAPABILITIES
        }
        CudaTensorExecutionProfile::TensorAndDenseBf16CausalV1 => &DENSE_BF16_CAUSAL_CAPABILITIES,
        CudaTensorExecutionProfile::TensorAndDenseBf16VisualCausalV1 => {
            &DENSE_BF16_VISUAL_CAUSAL_CAPABILITIES
        }
        _ => &DENSE_F16_CAUSAL_CAPABILITIES,
    };
    let tensor = capabilities
        .iter()
        .filter(|c| !required.contains(&c.as_str()))
        .cloned()
        .collect();
    crate::backend::cuda_tensor_capability_policy::validate(&tensor)?;
    for capability in required {
        if !capabilities.iter().any(|c| c.as_str() == *capability) {
            return Err(format!(
                "CUDA tensor+causal profile lacks required capability: {capability}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_profile_capabilities(
    capabilities: &BTreeSet<CapabilityId>,
    profile: Option<&CudaDenseF16CausalProfile>,
) -> Result<(), String> {
    let execution = profile.map_or(
        CudaTensorExecutionProfile::TensorOnly,
        CudaDenseF16CausalProfile::execution_profile,
    );
    let Some(_) = profile.and_then(CudaDenseF16CausalProfile::decode_replay) else {
        return validate_capabilities(capabilities, execution);
    };
    let required = [
        ferrum_runtime::DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
        ferrum_runtime::DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID,
    ];
    for id in required {
        if !capabilities.iter().any(|c| c.as_str() == id) {
            return Err(format!("causal decode replay lacks capability: {id}"));
        }
    }
    let remaining = capabilities
        .iter()
        .filter(|c| !required.contains(&c.as_str()))
        .cloned()
        .collect();
    validate_capabilities(&remaining, execution)
}

pub(crate) const fn permits_zero(profile: CudaTensorExecutionProfile, usage: BufferUsage) -> bool {
    matches!(usage, BufferUsage::Weights | BufferUsage::Persistent)
        || (matches!(
            profile,
            CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
                | CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1
                | CudaTensorExecutionProfile::TensorAndDenseBf16CausalV1
                | CudaTensorExecutionProfile::TensorAndDenseBf16VisualCausalV1
        ) && matches!(usage, BufferUsage::State))
}

pub(crate) fn add_module_allowance(
    opaque_bytes: u64,
    profile: Option<&CudaDenseF16CausalProfile>,
) -> Result<u64, String> {
    opaque_bytes
        .checked_add(profile.map_or(0, CudaDenseF16CausalProfile::module_memory_bytes))
        .ok_or_else(|| "CUDA tensor+causal opaque budget overflows".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_causal_replay_requires_exact_admission_for_two_row_work() {
        let compute = DeviceCommandPhase::Compute;
        assert!(permits_command_replay(true, compute, 2, 2, None));
        assert!(permits_command_replay(true, compute, 1, 2, Some((1, 2))));
        assert!(permits_command_replay(true, compute, 2, 4, Some((2, 4))));
        // Replay disabled and ordinary untagged multi-token prefill stay eager.
        assert!(!permits_command_replay(false, compute, 2, 4, Some((2, 4))));
        assert!(!permits_command_replay(true, compute, 2, 4, None));
        assert!(!permits_command_replay(true, compute, 2, 6, None));
        // A real capture identity does not authorize different or empty work.
        assert!(!permits_command_replay(true, compute, 1, 2, Some((2, 4))));
        assert!(!permits_command_replay(true, compute, 2, 3, Some((2, 4))));
        assert!(!permits_command_replay(true, compute, 2, 2, Some((2, 4))));
        assert!(!permits_command_replay(true, compute, 2, 6, Some((2, 6))));
        assert!(!permits_command_replay(true, compute, 0, 0, Some((0, 0))));
    }

    fn capabilities() -> BTreeSet<CapabilityId> {
        std::iter::once(ferrum_runtime::SILU_BF16_CAPABILITY_ID)
            .chain(DENSE_F16_CAUSAL_CAPABILITIES)
            .map(|c| CapabilityId::new(c).unwrap())
            .collect()
    }
    #[test]
    fn bounded_decode_replay_is_explicit_and_preserves_closed_capabilities() {
        let eager = CudaDenseF16CausalProfile::new(64, "test-causal-v1").unwrap();
        assert_eq!(eager.decode_replay_limits(), None);
        assert_eq!(eager.decode_replay_memory_bytes_per_stream(), 0);
        let mut caps = capabilities();
        validate_profile_capabilities(&caps, Some(&eager)).unwrap();
        let replay = eager.clone().with_decode_replay(8, 128).unwrap();
        assert_eq!(replay.decode_replay_limits(), Some((8, 128)));
        assert_eq!(
            replay.decode_replay_memory_bytes_per_stream(),
            128 + CudaCausalDecodeReplay::BLAS_WORKSPACE_BYTES
        );
        let maximum_graph_bytes = isize::MAX as u64 - CudaCausalDecodeReplay::BLAS_WORKSPACE_BYTES;
        assert!(eager
            .clone()
            .with_decode_replay(1, maximum_graph_bytes)
            .is_ok());
        assert!(eager
            .clone()
            .with_decode_replay(1, maximum_graph_bytes + 1)
            .is_err());
        assert!(validate_profile_capabilities(&caps, Some(&replay)).is_err());
        for id in [
            ferrum_runtime::DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
            ferrum_runtime::DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID,
        ] {
            caps.insert(CapabilityId::new(id).unwrap());
        }
        validate_profile_capabilities(&caps, Some(&replay)).unwrap();
        assert!(validate_profile_capabilities(&caps, Some(&eager)).is_err());
        assert!(validate_profile_capabilities(&caps, None).is_err());
        caps.insert(CapabilityId::new("capability.operation.unknown").unwrap());
        assert!(validate_profile_capabilities(&caps, Some(&replay)).is_err());
        for (slots, bytes) in [(0, 128), (8, 0), (usize::MAX, 128), (8, u64::MAX)] {
            assert!(eager.clone().with_decode_replay(slots, bytes).is_err());
        }
    }

    #[test]
    fn typed_causal_capabilities_are_complete_and_tensor_only_stays_closed() {
        let caps = capabilities();
        assert!(validate_capabilities(&caps, CudaTensorExecutionProfile::TensorOnly).is_err());
        validate_capabilities(&caps, CudaTensorExecutionProfile::TensorAndDenseF16CausalV1)
            .unwrap();
        for missing in DENSE_F16_CAUSAL_CAPABILITIES {
            let mut incomplete = caps.clone();
            incomplete.remove(&CapabilityId::new(missing).unwrap());
            assert!(validate_capabilities(
                &incomplete,
                CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
            )
            .is_err());
        }
        for foreign in [
            ferrum_runtime::TOKEN_EMBEDDING_F32_MASTER_CAPABILITY_ID,
            ferrum_runtime::DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
            "capability.operation.unknown",
        ] {
            let mut unknown = caps.clone();
            unknown.insert(CapabilityId::new(foreign).unwrap());
            assert!(validate_capabilities(
                &unknown,
                CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
            )
            .is_err());
        }
    }
    #[test]
    fn recurrent_zero_is_explicit_and_never_authorizes_activation_or_scratch() {
        for profile in [
            CudaTensorExecutionProfile::TensorOnly,
            CudaTensorExecutionProfile::TensorAndDenseF16CausalV1,
            CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1,
            CudaTensorExecutionProfile::TensorAndDenseBf16CausalV1,
            CudaTensorExecutionProfile::TensorAndDenseBf16VisualCausalV1,
        ] {
            assert!(permits_zero(profile, BufferUsage::Weights));
            assert!(permits_zero(profile, BufferUsage::Persistent));
            assert!(!permits_zero(profile, BufferUsage::Activations));
            assert!(!permits_zero(profile, BufferUsage::Scratch));
            assert_eq!(
                permits_zero(profile, BufferUsage::State),
                profile != CudaTensorExecutionProfile::TensorOnly
            );
        }
    }
    #[test]
    fn visual_causal_mode_is_a_closed_extension_and_never_leaks_into_text_or_tensor_defaults() {
        let old = capabilities();
        let mut visual = old.clone();
        visual.extend(
            [
                ferrum_runtime::ROW_SELECT_F16_CAPABILITY_ID,
                ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_CAPABILITY_ID,
            ]
            .into_iter()
            .map(|c| CapabilityId::new(c).unwrap()),
        );
        assert!(validate_capabilities(&visual, CudaTensorExecutionProfile::TensorOnly).is_err());
        assert!(validate_capabilities(
            &visual,
            CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
        )
        .is_err());
        validate_capabilities(
            &visual,
            CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1,
        )
        .unwrap();
        assert!(validate_capabilities(
            &old,
            CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1
        )
        .is_err());
        visual.insert(CapabilityId::new("capability.unowned.visual").unwrap());
        assert!(validate_capabilities(
            &visual,
            CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1
        )
        .is_err());
        let text = CudaDenseF16CausalProfile::new(4096, "native-causal-v1").unwrap();
        assert_eq!(
            text.execution_profile(),
            CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
        );
        let visual = text.with_visual_prompt_v1();
        assert_eq!(
            visual.execution_profile(),
            CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1
        );
        assert_eq!(visual.module_memory_bytes(), 4096);
    }
    #[test]
    fn causal_module_declaration_is_checked_without_loading_a_driver() {
        let p = CudaDenseF16CausalProfile::new(1, "experimental-unqualified-v1").unwrap();
        assert_eq!(p.module_memory_bytes(), 1);
        assert_eq!(p.opaque_memory_profile(), "experimental-unqualified-v1");
        for (bytes, name) in [
            (0, "v1".to_owned()),
            (1, String::new()),
            (1, "x".repeat(257)),
            (1, "bad\nprofile".to_owned()),
            (1, "非ASCII".to_owned()),
        ] {
            assert!(CudaDenseF16CausalProfile::new(bytes, name).is_err());
        }
        assert_eq!(add_module_allowance(40, None).unwrap(), 40);
        assert_eq!(add_module_allowance(40, Some(&p)).unwrap(), 41);
        assert!(add_module_allowance(u64::MAX, Some(&p)).is_err());
    }

    #[test]
    fn native_bf16_rejects_pre_ampere_without_changing_f16_qualification() {
        let f16 = capabilities();
        validate_causal_compute_capability(&f16, 7).unwrap();
        for capability in DENSE_BF16_CAUSAL_CAPABILITIES {
            let bf16 = BTreeSet::from([CapabilityId::new(capability).unwrap()]);
            assert!(validate_causal_compute_capability(&bf16, 7).is_err());
            validate_causal_compute_capability(&bf16, 8).unwrap();
        }
    }
}
