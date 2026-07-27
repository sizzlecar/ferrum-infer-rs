//! Shared decoding for native launch boundaries that return a stage and the
//! underlying platform status in one non-zero integer.

const STATUS_STAGE_SHIFT: u32 = 16;
const NATIVE_STATUS_MASK: u32 = 0xffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StagedNativeStatus {
    stage: u16,
    native_status: u16,
}

impl StagedNativeStatus {
    pub(crate) fn decode(code: i32) -> Option<Self> {
        (code != 0).then(|| {
            let code = code as u32;
            Self {
                stage: (code >> STATUS_STAGE_SHIFT) as u16,
                native_status: (code & NATIVE_STATUS_MASK) as u16,
            }
        })
    }

    pub(crate) const fn stage(self) -> u16 {
        self.stage
    }

    pub(crate) const fn native_status(self) -> u16 {
        self.native_status
    }
}

#[cfg(test)]
mod tests {
    use super::StagedNativeStatus;

    #[test]
    fn staged_native_status_preserves_stage_and_platform_code() {
        assert_eq!(StagedNativeStatus::decode(0), None);

        let status = StagedNativeStatus::decode((5 << 16) | 701).unwrap();
        assert_eq!(status.stage(), 5);
        assert_eq!(status.native_status(), 701);

        let legacy = StagedNativeStatus::decode(17).unwrap();
        assert_eq!(legacy.stage(), 0);
        assert_eq!(legacy.native_status(), 17);
    }
}
