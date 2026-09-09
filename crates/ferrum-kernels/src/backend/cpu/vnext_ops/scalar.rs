use ferrum_interfaces::vnext::ElementType;
use half::f16;

use super::super::vnext_runtime::CpuRuntimeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CpuFloat {
    F16,
    F32,
}

impl TryFrom<ElementType> for CpuFloat {
    type Error = CpuRuntimeError;
    fn try_from(value: ElementType) -> Result<Self, Self::Error> {
        match value {
            ElementType::F16 => Ok(Self::F16),
            ElementType::F32 => Ok(Self::F32),
            _ => Err(CpuRuntimeError::new(
                "CPU operator requires declared F16 or F32 storage",
            )),
        }
    }
}

impl CpuFloat {
    pub(super) const fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    #[inline]
    pub(super) fn read(self, bytes: &[u8], index: usize) -> f32 {
        match self {
            Self::F16 => {
                f16::from_le_bytes(bytes[2 * index..2 * index + 2].try_into().unwrap()).to_f32()
            }
            Self::F32 => f32::from_le_bytes(bytes[4 * index..4 * index + 4].try_into().unwrap()),
        }
    }

    #[inline]
    pub(super) fn write(self, bytes: &mut [u8], index: usize, value: f32) {
        match self {
            Self::F16 => {
                bytes[2 * index..2 * index + 2].copy_from_slice(&f16::from_f32(value).to_le_bytes())
            }
            Self::F32 => bytes[4 * index..4 * index + 4].copy_from_slice(&value.to_le_bytes()),
        }
    }

    pub(super) fn byte_len(self, rows: usize, columns: usize) -> Result<usize, CpuRuntimeError> {
        rows.checked_mul(columns)
            .and_then(|n| n.checked_mul(self.bytes()))
            .filter(|&n| n != 0)
            .ok_or_else(|| {
                CpuRuntimeError::new("CPU tensor shape is empty or exceeds addressable memory")
            })
    }
}
