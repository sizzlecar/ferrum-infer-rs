//! Project the already selected invocation; this code never chooses a route.
use super::*;
use ferrum_runtime::{
    DeviceNativeOperationId, DeviceProjectionRole, PackedProjectionGeometry,
    ParticipantRowSelector, ParticipantTokenSelector, ProjectionArithmetic, ProjectionDimensions,
    ProjectionInputTransform, ProjectionInvocationTemplate, ProjectionLaunchGeometry,
    ProjectionLayout,
};

fn geometry(config: cudarc::driver::LaunchConfig) -> ProjectionLaunchGeometry {
    ProjectionLaunchGeometry {
        grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
        block: [config.block_dim.0, config.block_dim.1, config.block_dim.2],
        shared_bytes: config.shared_mem_bytes,
    }
}
fn implementation(name: &'static str) -> DeviceNativeOperationId {
    DeviceNativeOperationId::new(name).expect("installed static operation identity")
}

pub(super) fn project(
    parts: &[weights::MatrixPart],
    launches: &[(usize, usize, u32)],
    selections: &[Vec<ProjectionPlan>],
    participants: u32,
    packed: bool,
    output_stride: u32,
) -> Result<Arc<[ProjectionInvocationTemplate]>, String> {
    let mut facts = Vec::new();
    for (launch_index, (&(_, _, m), selected)) in launches.iter().zip(selections).enumerate() {
        for (part, selection) in parts.iter().zip(selected) {
            let (arithmetic, input_transforms) = match selection {
                ProjectionPlan::Mmq { plan, .. } => {
                    let g = plan.geometry();
                    // One invocation comprises copy/pad, quantize, dot,
                    // optional fixup and publish. It is not a single kernel.
                    (
                        ProjectionArithmetic::PackedQuantized {
                            implementation: implementation(plan.operator()),
                            geometry: PackedProjectionGeometry {
                                padded_k: g.padded_inputs,
                                padded_n: g.padded_outputs,
                                row_tile: g.j,
                                column_tile: g.i,
                                blocks: g.blocks,
                                threads: g.nthreads,
                                shared_bytes: g.shared_bytes,
                                fixup: g.fixup != 0,
                                guard_blocks: g.guard_blocks,
                                pack_abi: g.pack_abi,
                                packed_bytes: g.packed_bytes,
                                fixup_bytes: g.fixup_bytes,
                                workspace_bytes: plan.workspace_bytes(),
                            },
                        },
                        Box::default(),
                    )
                }
                ProjectionPlan::Strict(selection) => {
                    let transforms =
                        match (selection.transform, part.transform.as_ref()) {
                            (Some(transform), Some(spec)) => vec![ProjectionInputTransform {
                                spec: spec.clone(),
                                implementation: implementation(transform.implementation()),
                                launch: geometry(transform.launch()),
                            }]
                            .into_boxed_slice(),
                            (None, None) => Box::default(),
                            _ => return Err(
                                "selected projection transform differs from its original matrix"
                                    .into(),
                            ),
                        };
                    (
                        ProjectionArithmetic::Direct {
                            implementation: implementation(selection.linear.implementation()),
                            row_tile: selection.linear.row_tile(),
                            launch: geometry(selection.linear.launch()),
                        },
                        transforms,
                    )
                }
            };
            let rows = if packed {
                (0..participants)
                    .map(|participant_index| ParticipantRowSelector {
                        participant_index,
                        source: ParticipantTokenSelector::LastSourceToken,
                        physical_rows: participant_index..participant_index + 1,
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![ParticipantRowSelector {
                    participant_index: u32::try_from(launch_index)
                        .map_err(|_| "projection participant ordinal overflows")?,
                    source: ParticipantTokenSelector::LastSourceToken,
                    physical_rows: 0..1,
                }]
            };
            facts.push(ProjectionInvocationTemplate {
                ordinal: u32::try_from(facts.len())
                    .map_err(|_| "projection invocation ordinal overflows")?,
                role: DeviceProjectionRole::OutputHead,
                component: part.component_id.clone(),
                dimensions: ProjectionDimensions {
                    m,
                    k: part.columns,
                    n: part.rows,
                },
                layout: ProjectionLayout {
                    input_stride: part.columns,
                    output_stride,
                    output_offset: part.output_offset,
                    input_type: ElementType::F32,
                    output_type: ElementType::F32,
                },
                arithmetic,
                input_transforms,
                rows: rows.into_boxed_slice(),
            });
        }
    }
    Ok(facts.into())
}
