//! Device placement and tensor-sharding utilities.
//!
//! Layer-split execution consumes the placement types in this module and
//! constructs real model stages through the engine's layer-split composition.
//! A placement or sharding description alone does not provide distributed
//! execution or collective communication.

pub mod config;
pub mod device;
pub mod tensor_parallel;

pub use config::{LayerDistribution, LayerRange, ParallelConfig, ParallelismType};
pub use device::{global_device_manager, DeviceCapability, DeviceInfo, DeviceManager};
pub use tensor_parallel::{
    LayerParallelType, TensorParallelConfig, TensorParallelGroup, TransformerParallelMapping,
    WeightShard,
};
