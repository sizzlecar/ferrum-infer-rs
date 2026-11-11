//! Batch independent explicit-position decode CTAs without changing their math.
use super::*;

pub(super) struct PreparedExplicitDecode(PackedFallbackLaunch);

impl PreparedExplicitDecode {
    pub(super) fn for_launches(
        launches: &[CausalAttentionLaunch],
        binding: BindingLayout,
        independent_sequence_kv: bool,
    ) -> Result<Option<Self>, String> {
        // Preparing all rows before attention is valid only when their writes
        // target independently owned KV. Each row retains its own RoPE tables.
        if !independent_sequence_kv
            || launches.len() > u16::MAX as usize
            || binding.slot_bytes < BINDING_CONTROL_BYTES
            || launches.iter().enumerate().any(|(index, launch)| {
                launch.tokens != 1
                    || launch.rotary.is_none()
                    || launch.packed_token_start != index as u64
                    || Some(launch.binding_offset) != (index as u64).checked_mul(binding.slot_bytes)
            })
        {
            return Ok(None);
        }
        packed_fallback_geometry(launches, binding).map(|geometry| geometry.map(Self))
    }

    pub(super) fn dispatch_count(&self, output_gate: bool, post_attention_norm: bool) -> u64 {
        // Shared norm/QKV/projection/residual, local preparation, one attention
        // grid, then the unchanged per-request output gates.
        7 + u64::from(post_attention_norm)
            + u64::from(self.0.participant_grid) * (1 + u64::from(output_gate))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue(
        &self,
        stream: &CudaStream,
        functions: &CausalAttentionFunctions,
        launches: &[CausalAttentionLaunch],
        binding: u64,
        shape: CudaCausalAttentionShape,
        layout: ScratchLayout,
        scratch: u64,
        query_norm: u64,
        key_norm: u64,
        regions: &[CudaBufferRegion],
        output_gate: bool,
    ) -> Result<(), CudaDeviceRuntimeError> {
        for launch in launches {
            let control = scratch_pointer(binding, launch.binding_offset)?;
            let [cos, sin] = launch.rotary.ok_or_else(|| {
                CudaDeviceRuntimeError::contract("explicit decode lost its rotary tables")
            })?;
            launch_prepare_with_rope(
                stream,
                &functions.prepare,
                scratch_pointer(scratch, launch.packed_query_raw)?,
                scratch_pointer(scratch, launch.packed_key_raw)?,
                scratch_pointer(scratch, launch.packed_value_raw)?,
                query_norm,
                key_norm,
                scratch_pointer(scratch, launch.packed_query)?,
                control,
                scratch_pointer(control, BINDING_CONTROL_BYTES)?,
                *launch,
                shape,
                i32::from(launch.path.uses_vllm_layout()),
                None,
                Some([regions[cos].device_ptr(), regions[sin].device_ptr()]),
            )?;
        }
        // The original kernel's z dimension selects a request's control/table
        // slot and packed row. Its per-head CTA and F32 reduction stay intact;
        // explicit rotary launches do not select the grouped-query algorithm.
        launch_fallback_attention(
            stream,
            functions,
            scratch_pointer(scratch, layout.query)?,
            scratch_pointer(scratch, layout.query_raw)?,
            binding,
            scratch_pointer(binding, BINDING_CONTROL_BYTES)?,
            scratch_pointer(scratch, layout.context)?,
            launches[0],
            shape,
            i32::from(self.0.path.uses_vllm_layout()),
            Some(self.0),
        )?;
        if output_gate {
            for launch in launches {
                launch_attention_gate(
                    stream,
                    &functions.attention_gate,
                    scratch_pointer(scratch, launch.packed_context)?,
                    scratch_pointer(scratch, launch.packed_query_raw)?,
                    launch.tokens,
                    shape,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch(index: u64) -> CausalAttentionLaunch {
        CausalAttentionLaunch {
            input_region: 0,
            output_region: 1,
            binding_offset: index * 256,
            packed_token_start: index,
            packed_query_raw: index * 1024,
            packed_key_raw: index * 256,
            packed_value_raw: index * 256,
            packed_query: index * 512,
            packed_context: index * 512,
            tokens: 1,
            tokens_i32: 1,
            sequence_tokens: 1024 + index,
            sequence_tokens_i32: 1024 + index as i32,
            table_entries_i32: 65,
            replay_topology: CausalAttentionReplayTopology::PartitionStable(
                CausalAttentionReplayEnvelope {
                    sequence_capacity_tokens: 2048,
                    table_capacity_entries: 128,
                },
            ),
            path: CausalAttentionKernelPath::VllmAddressedFallback,
            rotary: Some([2 + index as usize * 2, 3 + index as usize * 2]),
        }
    }

    #[test]
    fn explicit_decode_requires_independent_single_rows_and_canonical_binding_slots() {
        let layout = BindingLayout {
            required_bytes: 512,
            slot_bytes: 256,
        };
        let rows = [launch(0), launch(1)];
        let prepared = PreparedExplicitDecode::for_launches(&rows, layout, true)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.0.participant_grid, 2);
        assert_eq!(prepared.0.binding_slot_bytes, 256);
        assert_eq!(prepared.dispatch_count(true, false), 11);
        assert!(PreparedExplicitDecode::for_launches(&rows, layout, false)
            .unwrap()
            .is_none());
        assert!(
            PreparedExplicitDecode::for_launches(&rows[..1], layout, true)
                .unwrap()
                .is_none()
        );
        let invalidations: [fn(&mut CausalAttentionLaunch); 5] = [
            |row: &mut CausalAttentionLaunch| row.tokens = 2,
            |row: &mut CausalAttentionLaunch| row.rotary = None,
            |row: &mut CausalAttentionLaunch| row.binding_offset = 0,
            |row: &mut CausalAttentionLaunch| row.packed_token_start = 0,
            |row: &mut CausalAttentionLaunch| {
                row.path = CausalAttentionKernelPath::TokenMajorFallback
            },
        ];
        for change in invalidations {
            let mut invalid = rows;
            change(&mut invalid[1]);
            assert!(PreparedExplicitDecode::for_launches(&invalid, layout, true)
                .unwrap()
                .is_none());
        }
    }

    #[test]
    fn explicit_decode_never_exceeds_cuda_grid_z_capacity() {
        let rows = (0..=u16::MAX)
            .map(|index| launch(u64::from(index)))
            .collect::<Vec<_>>();
        let layout = BindingLayout {
            required_bytes: rows.len() as u64 * 256,
            slot_bytes: 256,
        };
        assert!(PreparedExplicitDecode::for_launches(&rows, layout, true)
            .unwrap()
            .is_none());
        assert!(
            PreparedExplicitDecode::for_launches(&rows[..u16::MAX as usize], layout, true)
                .unwrap()
                .is_some()
        );
    }
}
