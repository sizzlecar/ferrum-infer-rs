//! Explicit population coordinate. Reserved membership never replaces the
//! original accepted FIFO ordinal or attests a source without its live ledger.
use super::*;

pub const POPULATION_REVISION: &str = "preexecuted_single_scope_members_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredPopulationV1 {
    DenseFifo,
    ReservedMembers { rule_signature: [u8; 32] },
}
impl StructuredPopulationV1 {
    pub(super) fn validate(self) -> Result<()> {
        if matches!(self, Self::ReservedMembers { rule_signature } if rule_signature == [0; 32]) {
            return Err(StructuredUnknown::WrongProtocol);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredMemberBindingV1 {
    rule_signature: [u8; 32],
    offered_ordinal: u64,
    member_ordinal: u64,
}
impl StructuredMemberBindingV1 {
    pub fn new(
        rule_signature: [u8; 32],
        offered_ordinal: u64,
        member_ordinal: u64,
    ) -> Result<Self> {
        if rule_signature == [0; 32]
            || offered_ordinal == 0
            || member_ordinal == 0
            || member_ordinal > offered_ordinal
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        Ok(Self {
            rule_signature,
            offered_ordinal,
            member_ordinal,
        })
    }
    pub fn rule_signature(self) -> [u8; 32] {
        self.rule_signature
    }
    pub fn offered_ordinal(self) -> u64 {
        self.offered_ordinal
    }
    pub fn member_ordinal(self) -> u64 {
        self.member_ordinal
    }
}
impl StructuredNumericObservationV1 {
    pub(super) fn population_ordinal(&self, population: StructuredPopulationV1) -> Result<u64> {
        match (population, self.membership) {
            (StructuredPopulationV1::DenseFifo, None) => Ok(self.ordinal),
            (StructuredPopulationV1::ReservedMembers { rule_signature }, Some(member))
                if member.rule_signature == rule_signature =>
            {
                Ok(member.member_ordinal)
            }
            _ => Err(StructuredUnknown::WrongProtocol),
        }
    }
}
