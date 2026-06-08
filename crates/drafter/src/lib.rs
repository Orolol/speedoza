pub mod dflash;
#[cfg(feature = "cuda")]
pub mod gpu;

pub use dflash::{
    DFlashConfig, DFlashDrafter, DFlashLayerWeights, DFlashManifest, DFlashWeightRef,
    LayerAttentionKind,
};
#[cfg(feature = "cuda")]
pub use gpu::{DFlashDrafterDevice, DFlashLayerDevice, DrafterVramReport};
