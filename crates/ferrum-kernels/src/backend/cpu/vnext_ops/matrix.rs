use ferrum_interfaces::vnext::WeightEncoding;
use rayon::prelude::*;

use super::super::vnext_runtime::CpuRuntimeError;
use super::scalar::CpuFloat;
use crate::gguf_blocks::GgufBlockFormat;

#[derive(Debug, Clone, Copy)]
pub(super) enum CpuMatrixFormat {
    Dense(CpuFloat),
    Native(GgufBlockFormat),
}

impl CpuMatrixFormat {
    pub(super) fn from_encoding(encoding: &WeightEncoding) -> Result<Self, CpuRuntimeError> {
        match encoding {
            WeightEncoding::Dense { element_type } => Ok(Self::Dense((*element_type).try_into()?)),
            WeightEncoding::BlockQuantized(spec) => Ok(Self::Native(
                GgufBlockFormat::from_spec(spec).map_err(CpuRuntimeError::new)?,
            )),
            _ => Err(CpuRuntimeError::new(
                "CPU matrix encoding has no installed kernel",
            )),
        }
    }

    fn row_bytes(self, columns: usize) -> Result<usize, CpuRuntimeError> {
        match self {
            Self::Dense(dtype) => dtype.byte_len(1, columns),
            Self::Native(format)
                if columns != 0 && columns.is_multiple_of(format.block_values()) =>
            {
                (columns / format.block_values())
                    .checked_mul(format.block_bytes())
                    .ok_or_else(|| CpuRuntimeError::new("CPU matrix row byte size overflows"))
            }
            _ => Err(CpuRuntimeError::new(
                "CPU matrix requires complete native blocks per row",
            )),
        }
    }
}

/// A checked borrowed matrix; native blocks remain compressed for the entire
/// model lifetime. Each dot product decodes individual values in its registers.
pub(super) struct CpuMatrix<'a> {
    bytes: &'a [u8],
    format: CpuMatrixFormat,
    rows: usize,
    columns: usize,
    row_bytes: usize,
}

impl<'a> CpuMatrix<'a> {
    pub(super) fn new(
        bytes: &'a [u8],
        format: CpuMatrixFormat,
        rows: usize,
        columns: usize,
    ) -> Result<Self, CpuRuntimeError> {
        let row_bytes = format.row_bytes(columns)?;
        if rows == 0 || rows.checked_mul(row_bytes) != Some(bytes.len()) {
            return Err(CpuRuntimeError::new(
                "CPU matrix bytes differ from the declared physical shape",
            ));
        }
        Ok(Self {
            bytes,
            format,
            rows,
            columns,
            row_bytes,
        })
    }

    #[inline]
    fn value(&self, row: &[u8], column: usize) -> f32 {
        match self.format {
            CpuMatrixFormat::Dense(dtype) => dtype.read(row, column),
            CpuMatrixFormat::Native(format) => {
                let start = column / format.block_values() * format.block_bytes();
                format.decode_value(
                    &row[start..start + format.block_bytes()],
                    column % format.block_values(),
                )
            }
        }
    }

    /// Accumulation stays F32 for both declared activation profiles. Work is
    /// parallelized between independent output elements; scheduling cannot
    /// change a dot product's reduction order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn linear(
        &self,
        input: &[u8],
        input_type: CpuFloat,
        output: &mut [u8],
        output_type: CpuFloat,
        rows: usize,
        output_stride: usize,
        output_column_offset: usize,
    ) -> Result<(), CpuRuntimeError> {
        if input.len() != input_type.byte_len(rows, self.columns)?
            || output.len() != output_type.byte_len(rows, output_stride)?
            || output_column_offset
                .checked_add(self.rows)
                .is_none_or(|n| n > output_stride)
        {
            return Err(CpuRuntimeError::new(
                "CPU linear activation shape or output stride differs",
            ));
        }
        let input_row_bytes = input_type.byte_len(1, self.columns)?;
        output
            .par_chunks_exact_mut(output_type.bytes())
            .enumerate()
            .for_each(|(index, destination)| {
                let column = index % output_stride;
                let Some(weight_row) = column
                    .checked_sub(output_column_offset)
                    .filter(|&row| row < self.rows)
                else {
                    return;
                };
                let input_start = index / output_stride * input_row_bytes;
                let input = &input[input_start..input_start + input_row_bytes];
                let weight =
                    &self.bytes[weight_row * self.row_bytes..(weight_row + 1) * self.row_bytes];
                let mut sum = 0.0_f32;
                for column in 0..self.columns {
                    sum += input_type.read(input, column) * self.value(weight, column);
                }
                output_type.write(destination, 0, sum);
            });
        Ok(())
    }

    pub(super) fn embedding(
        &self,
        token_ids: &[u8],
        output: &mut [u8],
        output_type: CpuFloat,
    ) -> Result<(), CpuRuntimeError> {
        if token_ids.is_empty()
            || !token_ids.len().is_multiple_of(4)
            || output.len() != output_type.byte_len(token_ids.len() / 4, self.columns)?
        {
            return Err(CpuRuntimeError::new(
                "CPU embedding token or output shape differs",
            ));
        }
        let token = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap()) as usize;
        if token_ids
            .chunks_exact(4)
            .any(|bytes| token(bytes) >= self.rows)
        {
            return Err(CpuRuntimeError::new(
                "CPU embedding token exceeds the declared vocabulary",
            ));
        }
        let row_bytes = output_type.byte_len(1, self.columns)?;
        output
            .par_chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(|(index, destination)| {
                let id = token(&token_ids[index * 4..index * 4 + 4]);
                let source = &self.bytes[id * self.row_bytes..(id + 1) * self.row_bytes];
                for column in 0..self.columns {
                    output_type.write(destination, column, self.value(source, column));
                }
            });
        Ok(())
    }
}
