//! Safe wrappers around the CUDA stream / graph FFI exposed by the kernel
//! library. The engine uses these to capture the decode forward into a CUDA
//! graph and replay it across decode iterations, dropping ~600 host kernel
//! launches per token down to a single graph launch.

use qwen36_fp4_core::{CoreError, Result};

use crate::backend::DevicePtr;

/// Pointer-sized CUDA stream handle. NULL maps to the legacy default stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStream(pub *mut core::ffi::c_void);

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub const NULL: Self = Self(core::ptr::null_mut());

    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Allocate a fresh non-blocking CUDA stream owned by an [`OwnedCudaStream`].
    pub fn create() -> Result<OwnedCudaStream> {
        let mut raw: *mut core::ffi::c_void = core::ptr::null_mut();
        let status = unsafe { ffi::qwen36_cuda_stream_create(&mut raw) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "qwen36_cuda_stream_create failed with status {status}"
            )));
        }
        Ok(OwnedCudaStream(CudaStream(raw)))
    }

    pub fn synchronize(&self) -> Result<()> {
        let status = unsafe { ffi::qwen36_cuda_stream_synchronize(self.0) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "qwen36_cuda_stream_synchronize failed with status {status}"
            )));
        }
        Ok(())
    }
}

/// Owns a CUDA stream and destroys it on drop.
pub struct OwnedCudaStream(CudaStream);

impl OwnedCudaStream {
    pub fn handle(&self) -> CudaStream {
        self.0
    }
}

impl Drop for OwnedCudaStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = ffi::qwen36_cuda_stream_destroy(self.0.0);
            }
        }
    }
}

/// Sets the ambient CUDA stream every kernel launch will use. NULL restores
/// the legacy default stream. The setter is single-stream by design — the
/// engine flips streams once during graph capture, never concurrently.
pub fn set_active_stream(stream: CudaStream) {
    unsafe { ffi::qwen36_set_active_stream(stream.0) }
}

pub fn get_active_stream() -> CudaStream {
    CudaStream(unsafe { ffi::qwen36_get_active_stream() })
}

/// Atomically advance a device-side `int32_t` by 1. Captured into the decode
/// graph so each replay steps the position counter without host involvement.
pub fn increment_i32(target: DevicePtr) -> Result<()> {
    let status = unsafe { ffi::qwen36_increment_i32(target) };
    if status != 0 {
        return Err(CoreError::Runtime(format!(
            "qwen36_increment_i32 failed with status {status}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct CudaGraph(*mut core::ffi::c_void);

unsafe impl Send for CudaGraph {}

#[derive(Debug, Clone, Copy)]
pub struct CudaGraphExec(*mut core::ffi::c_void);

unsafe impl Send for CudaGraphExec {}

/// Begin recording all kernel launches on `stream` into a CUDA graph. Pair
/// with [`end_capture`] to retrieve the recorded graph.
pub fn begin_capture(stream: CudaStream) -> Result<()> {
    let status = unsafe { ffi::qwen36_cuda_stream_begin_capture(stream.0) };
    if status != 0 {
        return Err(CoreError::Runtime(format!(
            "qwen36_cuda_stream_begin_capture failed with status {status}"
        )));
    }
    Ok(())
}

pub fn end_capture(stream: CudaStream) -> Result<CudaGraph> {
    let mut graph: *mut core::ffi::c_void = core::ptr::null_mut();
    let status = unsafe { ffi::qwen36_cuda_stream_end_capture(stream.0, &mut graph) };
    if status != 0 {
        return Err(CoreError::Runtime(format!(
            "qwen36_cuda_stream_end_capture failed with status {status}"
        )));
    }
    Ok(CudaGraph(graph))
}

pub fn instantiate(graph: CudaGraph) -> Result<CudaGraphExec> {
    let mut exec: *mut core::ffi::c_void = core::ptr::null_mut();
    let status = unsafe { ffi::qwen36_cuda_graph_instantiate(graph.0, &mut exec) };
    if status != 0 {
        return Err(CoreError::Runtime(format!(
            "qwen36_cuda_graph_instantiate failed with status {status}"
        )));
    }
    Ok(CudaGraphExec(exec))
}

pub fn destroy_graph(graph: CudaGraph) -> Result<()> {
    let status = unsafe { ffi::qwen36_cuda_graph_destroy(graph.0) };
    if status != 0 {
        return Err(CoreError::Runtime(format!(
            "qwen36_cuda_graph_destroy failed with status {status}"
        )));
    }
    Ok(())
}

pub fn destroy_graph_exec(exec: CudaGraphExec) -> Result<()> {
    let status = unsafe { ffi::qwen36_cuda_graph_exec_destroy(exec.0) };
    if status != 0 {
        return Err(CoreError::Runtime(format!(
            "qwen36_cuda_graph_exec_destroy failed with status {status}"
        )));
    }
    Ok(())
}

pub fn launch(exec: CudaGraphExec, stream: CudaStream) -> Result<()> {
    let status = unsafe { ffi::qwen36_cuda_graph_launch(exec.0, stream.0) };
    if status != 0 {
        return Err(CoreError::Runtime(format!(
            "qwen36_cuda_graph_launch failed with status {status}"
        )));
    }
    Ok(())
}

mod ffi {
    use super::*;
    type Stream = *mut core::ffi::c_void;
    type Graph = *mut core::ffi::c_void;
    type GraphExec = *mut core::ffi::c_void;

    #[link(name = "qwen36_fp4_kernels")]
    unsafe extern "C" {
        pub fn qwen36_get_active_stream() -> Stream;
        pub fn qwen36_set_active_stream(stream: Stream);
        pub fn qwen36_cuda_stream_create(out: *mut Stream) -> i32;
        pub fn qwen36_cuda_stream_destroy(stream: Stream) -> i32;
        pub fn qwen36_cuda_stream_synchronize(stream: Stream) -> i32;
        pub fn qwen36_cuda_stream_begin_capture(stream: Stream) -> i32;
        pub fn qwen36_cuda_stream_end_capture(stream: Stream, out: *mut Graph) -> i32;
        pub fn qwen36_cuda_graph_instantiate(graph: Graph, out: *mut GraphExec) -> i32;
        pub fn qwen36_cuda_graph_destroy(graph: Graph) -> i32;
        pub fn qwen36_cuda_graph_exec_destroy(exec: GraphExec) -> i32;
        pub fn qwen36_cuda_graph_launch(exec: GraphExec, stream: Stream) -> i32;
        pub fn qwen36_increment_i32(target: DevicePtr) -> i32;
    }
}
