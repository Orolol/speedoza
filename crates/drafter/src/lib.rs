pub mod dflash;
#[cfg(feature = "cuda")]
pub mod forward;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod handoff;
#[cfg(feature = "cuda")]
pub mod propose;

pub use dflash::{
    DFlashConfig, DFlashDrafter, DFlashLayerWeights, DFlashManifest, DFlashWeightRef,
    LayerAttentionKind,
};
#[cfg(feature = "cuda")]
pub use forward::{DrafterForward, DrafterForwardWorkspace, DrafterWorkspaceReport};
#[cfg(feature = "cuda")]
pub use gpu::{DFlashDrafterDevice, DFlashLayerDevice, DrafterVramReport};
#[cfg(feature = "cuda")]
pub use handoff::{TargetHiddenCapture, TargetHiddenCaptureSlot};
#[cfg(feature = "cuda")]
pub use propose::{DFlashProposeWorkspace, propose_block};
