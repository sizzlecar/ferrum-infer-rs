//! Metadata-only helpers for existing family fixtures. These deliberately do
//! not claim device compatibility; compiler/runtime tests supply real catalogs.

use ferrum_interfaces::vnext::{
    ModelFamilyRegistration, NumericalExecutionPolicy, PreparedModelFamily, VNextError,
};

use super::{DefinedProductionModel, PreparedProductionModel};

pub(super) trait PrepareFamilyFixture: ModelFamilyRegistration {
    fn prepare_fixture(&self, raw: &serde_json::Value) -> Result<PreparedModelFamily, VNextError> {
        let definition = self.define(raw)?;
        let candidates = definition
            .numerical_profiles()
            .candidates(&NumericalExecutionPolicy::Auto)?;
        self.prepare(&definition, &candidates[0].id)
    }
}
impl<R: ModelFamilyRegistration> PrepareFamilyFixture for R {}

pub(super) fn prepare_product_fixture(
    defined: DefinedProductionModel,
) -> ferrum_types::Result<PreparedProductionModel> {
    let candidates = defined
        .definition()
        .numerical_profiles()
        .candidates(&NumericalExecutionPolicy::Auto)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    defined.prepare(&candidates[0].id)
}
