use super::*;
use ferrum_runtime::{
    Nvfp4MatrixLayout, PhysicalWeightComponentView, PhysicalWeightLayout, WeightTensorSpec,
};

/// Metadata for the two Stored operands of one packed GEMM. The selected
/// package remains their source; a model program must install these records
/// through its ordinary Stored schema and runtime admission.
#[derive(Clone, Debug)]
pub struct FastH3Nvfp4WeightOperands {
    pub geometry: Nvfp4MatrixLayout,
    pub packed_values: WeightTensorSpec,
    pub scales: WeightTensorSpec,
    /// Exact original source declarations, including flattened scale shape.
    pub components: [WeightComponentSpec; 2],
}

impl FastH3TransformerPackage {
    /// Select complete B and scale_B components without unpacking or uploading
    /// them. Caller-owned weight IDs let the common graph reuse one operand.
    /// The final invocation alpha is deliberately absent: its producer must
    /// compose the stored F32 multiplier with the activation global scale.
    pub fn nvfp4_weight_operands(
        &self,
        canonical_name: &str,
        packed_weight_id: WeightId,
        scale_weight_id: WeightId,
    ) -> Result<FastH3Nvfp4WeightOperands> {
        let Some(FastH3Weight {
            storage: FastH3WeightStorage::Nvfp4 { matrix, components },
            ..
        }) = self.weight(canonical_name)
        else {
            anyhow::bail!("FastH3 parameter {canonical_name} is not an NVFP4 matrix");
        };
        FastH3Nvfp4WeightOperands::from_components(
            matrix,
            components,
            packed_weight_id,
            scale_weight_id,
        )
    }
}

impl FastH3Nvfp4WeightOperands {
    pub(in crate::fast_h3) fn from_components(
        matrix: &Nvfp4MatrixSpec,
        components: &[WeightComponentSpec],
        packed_weight_id: WeightId,
        scale_weight_id: WeightId,
    ) -> Result<Self> {
        ensure!(
            packed_weight_id != scale_weight_id,
            "NVFP4 operand IDs must differ"
        );
        let select = |name: &str| -> Result<WeightComponentSpec> {
            components
                .iter()
                .find(|c| c.external_names == [name])
                .cloned()
                .context("checked NVFP4 source component is absent")
        };
        let components = [select(&matrix.names.packed)?, select(&matrix.names.scales)?];
        let geometry = Nvfp4MatrixLayout::new(
            matrix.geometry.rows().try_into()?,
            matrix.geometry.columns().try_into()?,
        )?;
        let view = |id, component: &WeightComponentSpec, dimensions: [u64; 2]| WeightTensorSpec {
            id,
            dimensions: dimensions.to_vec(),
            logical_element_type: component.physical_element_type(),
            physical_layout: PhysicalWeightLayout::PhysicalComponent(
                PhysicalWeightComponentView::new(component.id.clone()),
            ),
            required: true,
        };
        Ok(FastH3Nvfp4WeightOperands {
            geometry,
            packed_values: view(packed_weight_id, &components[0], geometry.packed_shape()),
            scales: view(scale_weight_id, &components[1], geometry.scale_shape()),
            components,
        })
    }
}
