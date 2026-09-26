//! CUDA FP16 causal KV is stored in 16-token vLLM blocks, not token-major.
//! Read complete blocks, then gather every written K/V element in token order.
//! Unwritten block padding is not semantic model state; it is never inspected
//! for finiteness or used to choose which semantic elements enter the check.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct VllmKvLayout {
    heads: usize,
    dimension: usize,
}

impl VllmKvLayout {
    const TOKENS: usize = 16;
    const K_PACK: usize = 8;

    pub(super) fn from_tensor(tensor: &ProgramTensorSpec) -> Self {
        let [2, heads, dimension] = tensor.dimensions.as_slice() else {
            panic!("causal KV fixture requires [K/V, heads, dimension]")
        };
        assert_eq!(tensor.element_type, ElementType::F16);
        assert_eq!(tensor.layout, ResolvedTensorLayout::Contiguous);
        let layout = Self {
            heads: usize::try_from(*heads).unwrap(),
            dimension: usize::try_from(*dimension).unwrap(),
        };
        let block = layout.bytes_per_token() * Self::TOKENS as u64;
        // These are the actual CUDA selector's VllmBlocks16 conditions. A
        // different layout needs its own readback contract, not this gather.
        assert!(layout.heads > 0 && layout.dimension > 0);
        assert_eq!(layout.dimension % Self::K_PACK, 0);
        assert!(block <= 64 * 1024 && (64 * 1024) % block == 0);
        layout
    }

    pub(super) fn bytes_per_token(self) -> u64 {
        (2 * self.heads * self.dimension * 2) as u64
    }

    pub(super) fn readback_bytes(self, tokens: usize) -> u64 {
        assert!(tokens > 0);
        (tokens.div_ceil(Self::TOKENS) * Self::TOKENS) as u64 * self.bytes_per_token()
    }

    pub(super) fn gather(self, physical: &[u8], tokens: usize) -> Vec<u8> {
        assert_eq!(physical.len() as u64, self.readback_bytes(tokens));
        let half_block = self.heads * self.dimension * Self::TOKENS;
        let mut logical = Vec::with_capacity(tokens * self.bytes_per_token() as usize);
        for token in 0..tokens {
            let block = token / Self::TOKENS;
            let within = token % Self::TOKENS;
            for kind in 0..2 {
                for head in 0..self.heads {
                    for dimension in 0..self.dimension {
                        let within_block = if kind == 0 {
                            head * self.dimension * Self::TOKENS
                                + (dimension / Self::K_PACK) * Self::TOKENS * Self::K_PACK
                                + within * Self::K_PACK
                                + dimension % Self::K_PACK
                        } else {
                            half_block
                                + head * self.dimension * Self::TOKENS
                                + dimension * Self::TOKENS
                                + within
                        };
                        let offset = (block * 2 * half_block + within_block) * 2;
                        logical.extend_from_slice(&physical[offset..offset + 2]);
                    }
                }
            }
        }
        assert_eq!(logical.len(), tokens * self.bytes_per_token() as usize);
        logical
    }
}

#[test]
fn causal_vllm_readback_gathers_all_live_elements_across_blocks_and_pages() {
    let tensor = ProgramTensorSpec {
        dimensions: vec![2, 2, 128],
        element_type: ElementType::F16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let layout = VllmKvLayout::from_tensor(&tensor);
    // Independent physical-order writer. Poison all unwritten token slots;
    // expected output includes both K and V for every head/dimension/token.
    for tokens in [1_usize, 4, 15, 16, 17, 64, 65, 129] {
        let value = |token: usize, kind: usize, head: usize, dimension: usize| {
            half::f16::from_f32(
                (1 + token * 7 + kind * 11 + head * 13 + dimension * 3) as f32 / 1024.0,
            )
            .to_le_bytes()
        };
        let mut physical = Vec::new();
        for block in 0..tokens.div_ceil(16) {
            for kind in 0..2 {
                for head in 0..2 {
                    let (outer, inner) = if kind == 0 { (16, 8) } else { (128, 1) };
                    for group in 0..outer {
                        for within in 0..16 {
                            for lane in 0..inner {
                                let token = block * 16 + within;
                                physical.extend_from_slice(&if token < tokens {
                                    value(token, kind, head, group * inner + lane)
                                } else {
                                    half::f16::NAN.to_le_bytes()
                                });
                            }
                        }
                    }
                }
            }
        }
        let mut expected = Vec::new();
        for token in 0..tokens {
            for kind in 0..2 {
                for head in 0..2 {
                    for dimension in 0..128 {
                        expected.extend_from_slice(&value(token, kind, head, dimension));
                    }
                }
            }
        }
        let gathered = layout.gather(&physical, tokens);
        assert_eq!(gathered, expected, "tokens={tokens}");
        // A nonfinite *written* element must survive gathering and fail the
        // existing state validity assertion, rather than being filtered out.
        physical[..2].copy_from_slice(&half::f16::NAN.to_le_bytes());
        assert!(half::f16::from_le_bytes(
            layout.gather(&physical, tokens)[..2].try_into().unwrap()
        )
        .is_nan());
    }
}
