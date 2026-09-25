//! Last-token projection's actual ordered gather/compute/scatter evidence.
use super::*;
use ferrum_interfaces::execution_cost::{
    SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1,
    StatisticalTransferKindV1,
};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

fn blit(builder: &mut SelectedCommandCostBuilderV1, bytes: u64, gather: bool) -> Option<()> {
    static SOURCE: OnceLock<[u8; 32]> = OnceLock::new();
    let signature = *SOURCE.get_or_init(|| Sha256::digest(FINGERPRINT_SOURCE.as_bytes()).into());
    let class = SelectedAlgorithmClassV1::new(
        if gather {
            "MTLBlit.last_token.gather.shared_contiguous"
        } else {
            "MTLBlit.last_token.scatter.shared_contiguous"
        },
        1,
        signature,
        Sha256::digest(b"aligned4.buffer-copy.byte-exact.v1").into(),
    )
    .ok()?;
    builder
        .transfer(class, StatisticalTransferKindV1::DeviceToDevice, bytes)
        .ok()
}
pub(super) fn evidence(
    pipelines: &MetalLinearPipelines,
    launches: &[LinearLaunch],
    tokens: u64,
    packed: Option<(LastTokenPackedScratchLayout, u32, bool)>,
) -> Option<SelectedCommandCostEvidenceV1> {
    if launches.is_empty() || launches.iter().any(|l| l.transform.is_some()) {
        return None;
    }
    if packed.is_none() && launches.iter().any(|launch| launch.params.rows != 1) {
        return None;
    }
    let mut builder = SelectedCommandCostBuilderV1::new(tokens);
    let scratch = packed.map_or(0, |(layout, _, _)| layout.required_bytes);
    if let Some((layout, count, shared_input)) = packed {
        if launches.len() != 1 || count == 0 || launches[0].params.rows != count {
            return None;
        }
        if !shared_input {
            for _ in 0..count {
                blit(&mut builder, layout.input_row_bytes, true)?;
            }
        }
    }
    for &launch in launches {
        selected::projection(&mut builder, pipelines, launch, None, scratch)?;
    }
    if let Some((layout, count, _)) = packed {
        for _ in 0..count {
            blit(&mut builder, layout.output_row_bytes, false)?;
        }
    }
    builder.finish().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_head_accounts_actual_packed_gather_compute_and_scatter() {
        let device = Device::system_default().expect("actual Metal PSO catalog");
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let part = PreparedLinearPart {
            region: 0,
            format: LinearPhysicalFormat::Q6K,
            out_features: 64,
            output_offset: 0,
            transform: None,
        };
        for dtype in [ElementType::F16, ElementType::F32] {
            for count in [2, 5, 8, 17, 33] {
                let layout = LastTokenPackedScratchLayout::new(count, 256, 64, dtype).unwrap();
                let launch = linear_launch_typed(
                    part,
                    0,
                    0,
                    count,
                    256,
                    64,
                    0,
                    layout.output_offset_bytes,
                    dtype,
                )
                .unwrap();
                let shared = evidence(
                    &pipelines,
                    &[launch],
                    count + 3,
                    Some((layout, count as u32, true)),
                )
                .unwrap();
                let gathered = evidence(
                    &pipelines,
                    &[launch],
                    count + 3,
                    Some((layout, count as u32, false)),
                )
                .unwrap();
                shared
                    .validate_command(count + 3, launch.dispatch_count(), count)
                    .unwrap();
                gathered
                    .validate_command(count + 3, launch.dispatch_count(), count * 2)
                    .unwrap();
                assert_eq!(
                    shared.work().device_to_device_bytes,
                    count * layout.output_row_bytes
                );
                assert_eq!(
                    gathered.work().device_to_device_bytes,
                    count * (layout.input_row_bytes + layout.output_row_bytes)
                );
                assert_eq!(shared.work().inner_work_units, count * 256 * 64);
                assert_eq!(shared.work().peak_scratch_bytes, layout.required_bytes);
                assert_ne!(shared.family_signature(), gathered.family_signature());
                assert!(evidence(&pipelines, &[launch], count, None).is_none());
                assert!(evidence(
                    &pipelines,
                    &[launch],
                    count,
                    Some((layout, count as u32 + 1, true))
                )
                .is_none());
            }
        }
    }

    #[test]
    fn selected_head_scalar_and_participant_loop_have_no_invented_transfers() {
        let device = Device::system_default().expect("actual Metal PSO catalog");
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let part = PreparedLinearPart {
            region: 0,
            format: LinearPhysicalFormat::Q6K,
            out_features: 64,
            output_offset: 0,
            transform: None,
        };
        let launch = linear_launch_typed(part, 0, 0, 1, 256, 64, 0, 0, ElementType::F32).unwrap();
        for count in [1, 3] {
            let cost = evidence(&pipelines, &vec![launch; count], count as u64 + 7, None).unwrap();
            cost.validate_command(count as u64 + 7, count as u64, 0)
                .unwrap();
            assert_eq!(cost.work().device_to_device_bytes, 0);
            assert_eq!(cost.work().peak_scratch_bytes, 0);
            assert_eq!(cost.work().inner_work_units, count as u64 * 256 * 64);
        }
        assert!(evidence(&pipelines, &[], 1, None).is_none());
        let mut unsupported = launch;
        unsupported.format = LinearPhysicalFormat::Native(GgufBlockFormat::Iq4Xs);
        assert!(evidence(&pipelines, &[unsupported], 1, None).is_none());
    }
}
