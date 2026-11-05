use super::*;
use anyhow::{ensure, Result};
use ferrum_quantization::SafetensorsArchive;
pub(crate) struct PH3ImageVaeEncoderVerticalCheckpoint<'a> {
    config: &'a PH3ImageVaeEncoderVerticalConfig,
    source: &'a SafetensorsArchive,
    host: HostMemoryAccount,
}
impl<'a> PH3ImageVaeEncoderVerticalCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageVaeEncoderVerticalConfig,
        source: &'a SafetensorsArchive,
        host: HostMemoryAccount,
        limits: PH3ImageVaeEncoderVerticalLimits,
    ) -> Result<Self> {
        PH3ImageVaeEncoderVerticalFamily::new(limits)?.check(config)?;
        for entry in weights::entries(config) {
            let t = source.tensor(&entry.name)?;
            ensure!(
                t.element_type() == Some(ElementType::F32) && t.shape() == entry.original,
                "encoder source {} must retain its exact original F32 shape",
                entry.name
            );
            ensure!(
                t.bytes()
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                "encoder source {} contains nonfinite values",
                entry.name
            );
        }
        Ok(Self {
            config,
            source,
            host,
        })
    }
    pub fn source(&self) -> &'a SafetensorsArchive {
        self.source
    }
}
impl WeightComponentSource for PH3ImageVaeEncoderVerticalCheckpoint<'_> {
    fn component<'s>(
        &'s self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        let [name] = c.external_names.as_slice() else {
            return Err(invalid("one exact encoder source required"));
        };
        let e = weights::entries(self.config)
            .into_iter()
            .find(|e| &e.name == name)
            .ok_or_else(|| invalid("unknown encoder source component"))?;
        if c.dimensions != e.physical
            || c.encoding
                != (WeightEncoding::Dense {
                    element_type: ElementType::F32,
                })
        {
            return Err(invalid(
                "encoder component differs from declared physical last-tap mapping",
            ));
        }
        let t = self
            .source
            .tensor(name)
            .map_err(|e| invalid(e.to_string()))?;
        if t.element_type() != Some(ElementType::F32) || t.shape() != e.original {
            return Err(invalid("encoder original F32 source shape changed"));
        }
        let payload = |bytes, retained| -> Result<WeightComponentPayload<'s>, VNextError> {
            WeightComponentPayload::new(
                c,
                name.clone(),
                t.source_file(),
                c.dimensions.clone(),
                ElementType::F32,
                bytes,
            )?
            .with_retained_host_memory(retained)
        };
        let [outputs, inputs, time, height, width] = e.original.as_slice() else {
            return self.source.component(c);
        };
        if *time == 1 {
            return payload(t.bytes(), t.retained_host_memory().clone());
        }
        // Official causal_zero is weight[:,:, -input_T:, :, :], hence the
        // last tap for T=1. Each O/I pair has its own temporal stride; taking
        // one contiguous file tail or its center tap would be incorrect.
        let bytes = usize::try_from(c.physical_bytes()?)
            .map_err(|_| invalid("last-tap backing exceeds address space"))?;
        let mut backing = self
            .host
            .allocate_zeroed(HostMemoryClass::WeightBacking, bytes)
            .map_err(|e| invalid(e.to_string()))?;
        let plane = usize::try_from(
            height
                .checked_mul(*width)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| invalid("kernel plane bytes overflow"))?,
        )
        .map_err(|_| invalid("kernel plane exceeds address space"))?;
        for pair in 0..usize::try_from(outputs * inputs)
            .map_err(|_| invalid("kernel channel pairs exceed address space"))?
        {
            let start = (pair * (*time as usize) + *time as usize - 1) * plane;
            backing.as_mut()[pair * plane..(pair + 1) * plane]
                .copy_from_slice(&t.bytes()[start..start + plane]);
        }
        WeightComponentPayload::from_managed_host(
            c,
            vec![name.clone()],
            vec![t.source_file().to_owned()],
            c.dimensions.clone(),
            ElementType::F32,
            backing.freeze(),
        )
    }
}
