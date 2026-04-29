use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid Qwen3.6 topology: {0}")]
    InvalidTopology(String),

    #[error("invalid tensor metadata for {name}: {reason}")]
    InvalidTensor { name: String, reason: String },

    #[error("unsupported operation without CUDA backend: {0}")]
    UnsupportedNoCuda(&'static str),

    #[error("kernel {kernel} returned error code {code}")]
    KernelLaunch { kernel: &'static str, code: i32 },

    #[error("runtime error: {0}")]
    Runtime(String),
}
