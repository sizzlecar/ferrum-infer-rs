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
    projection: &LastTokenProjection,
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
    let mut builder = crate::backend::metal::vnext_runtime::selected_cost_builder(
        projection.structured_capture(),
        tokens,
    );
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
        match projection {
            LastTokenProjection::Strict(p) => {
                selected::projection(&mut builder, p, launch, None, scratch)?
            }
            LastTokenProjection::Half(p) => p.append_statistical(&mut builder, launch, scratch)?,
        }
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
    fn selected_head_tracks_real_small_split_tiled_and_gather_scatter_work() {
        check_selected_head(ferrum_types::SloStructuredCostCapture::Disabled);
    }

    #[test]
    fn selected_head_algorithm_capture_covers_strict_half_and_real_transfer_geometry() {
        check_selected_head(ferrum_types::SloStructuredCostCapture::HostSettledV1);
    }

    fn check_selected_head(capture: ferrum_types::SloStructuredCostCapture) {
        let device = Device::system_default().expect("real Metal head pipeline catalog");
        let strict = LastTokenProjection::Strict(Arc::new(
            MetalLinearPipelines::new(&device)
                .unwrap()
                .with_structured_capture(capture),
        ));
        let half = LastTokenProjection::Half(Arc::new(
            half_head::HalfHeadPipelines::new(&device)
                .unwrap()
                .with_structured_capture(capture),
        ));
        for projection in [&strict, &half] {
            let part = PreparedLinearPart {
                region: 0,
                format: LinearPhysicalFormat::Q6K,
                out_features: 64,
                output_offset: 0,
                transform: None,
            };
            for count in [2, 5, 7, 8, 16] {
                let layout =
                    LastTokenPackedScratchLayout::new(count, 256, 64, ElementType::F32).unwrap();
                let launch = linear_launch_typed(
                    part,
                    0,
                    0,
                    count,
                    256,
                    64,
                    0,
                    layout.output_offset_bytes,
                    ElementType::F32,
                )
                .unwrap();
                let tokens = count + 3;
                let shared = evidence(
                    projection,
                    &[launch],
                    tokens,
                    Some((layout, count as u32, true)),
                )
                .unwrap();
                let gathered = evidence(
                    projection,
                    &[launch],
                    tokens,
                    Some((layout, count as u32, false)),
                )
                .unwrap();
                shared
                    .validate_command(tokens, projection.dispatch_count(launch), count)
                    .unwrap();
                gathered
                    .validate_command(tokens, projection.dispatch_count(launch), count * 2)
                    .unwrap();
                for command in [&shared, &gathered] {
                    if capture.is_disabled() {
                        assert!(command.algorithm_work().is_none());
                    } else {
                        let work = command.algorithm_work().unwrap().unwrap();
                        work.validate_command(command).unwrap();
                        assert!(work.entries().iter().any(|entry| entry.kind()
                            == ferrum_interfaces::execution_cost::AlgorithmWorkKindV1::Kernel));
                        assert!(work.entries().iter().any(|entry| entry.kind() == ferrum_interfaces::execution_cost::AlgorithmWorkKindV1::DeviceToDevice));
                    }
                }
                assert_eq!(shared.work().device_to_device_bytes, count * 64 * 4);
                assert_eq!(
                    gathered.work().device_to_device_bytes,
                    count * (256 + 64) * 4
                );
                assert_ne!(shared.family_signature(), gathered.family_signature());
                assert_eq!(shared.work().peak_scratch_bytes, layout.required_bytes);
                assert_eq!(shared.work().inner_work_units, count * 64 * 256);
            }
        }
        let a = LinearParams {
            rows: 7,
            in_features: 256,
            out_features: 64,
            output_stride: 64,
            output_column_offset: 0,
        };
        let mut launch = linear_launch_typed(
            PreparedLinearPart {
                region: 0,
                format: LinearPhysicalFormat::Q6K,
                out_features: 64,
                output_offset: 0,
                transform: None,
            },
            0,
            0,
            7,
            256,
            64,
            0,
            0,
            ElementType::F32,
        )
        .unwrap();
        let seven = evidence(
            &half,
            &[launch],
            7,
            Some((
                LastTokenPackedScratchLayout::new(7, 256, 64, ElementType::F32).unwrap(),
                7,
                true,
            )),
        )
        .unwrap();
        launch.params = LinearParams { rows: 8, ..a };
        let eight = evidence(
            &half,
            &[launch],
            8,
            Some((
                LastTokenPackedScratchLayout::new(8, 256, 64, ElementType::F32).unwrap(),
                8,
                true,
            )),
        )
        .unwrap();
        seven.validate_command(7, 2, 7).unwrap();
        eight.validate_command(8, 1, 8).unwrap();
        assert_ne!(seven.family_signature(), eight.family_signature());
        assert_eq!(
            eight.work().padded_units,
            32 * 64,
            "actual half-head M32 tile remains unchanged"
        );
        launch.activation_type = ElementType::F16;
        assert!(
            evidence(
                &half,
                &[launch],
                8,
                Some((
                    LastTokenPackedScratchLayout::new(8, 256, 64, ElementType::F32).unwrap(),
                    8,
                    true
                ))
            )
            .is_none(),
            "no strict profile substitution for unsupported half-head ABI"
        );
    }
}

#[cfg(test)]
mod strict_regression {
    use super::*;

    #[test]
    fn selected_head_accounts_actual_packed_gather_compute_and_scatter() {
        let device = Device::system_default().expect("actual Metal PSO catalog");
        let pipelines =
            LastTokenProjection::Strict(Arc::new(MetalLinearPipelines::new(&device).unwrap()));
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
        let pipelines =
            LastTokenProjection::Strict(Arc::new(MetalLinearPipelines::new(&device).unwrap()));
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
