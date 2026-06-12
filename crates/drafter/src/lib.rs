pub mod dflash;
pub mod eagle3;
#[cfg(feature = "cuda")]
pub mod eagle3_forward;
#[cfg(feature = "cuda")]
pub mod eagle3_gpu;
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
pub use eagle3::{
    Eagle3Config, Eagle3Drafter, Eagle3LayerWeights, Eagle3Manifest, Eagle3WeightRef,
};
#[cfg(feature = "cuda")]
pub use eagle3_forward::{
    Eagle3DraftChain, Eagle3DraftToken, Eagle3Forward, Eagle3ForwardWorkspace,
    Eagle3WorkspaceReport,
};
#[cfg(feature = "cuda")]
pub use eagle3_gpu::{Eagle3DrafterDevice, Eagle3LayerDevice, Eagle3VramReport};
#[cfg(feature = "cuda")]
pub use forward::{DrafterForward, DrafterForwardWorkspace, DrafterWorkspaceReport};
#[cfg(feature = "cuda")]
pub use gpu::{DFlashDrafterDevice, DFlashLayerDevice, DrafterVramReport};
#[cfg(feature = "cuda")]
pub use handoff::{TargetHiddenCapture, TargetHiddenCaptureSlot};
#[cfg(feature = "cuda")]
pub use propose::{DFlashProposeWorkspace, propose_block};
