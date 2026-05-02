#[cfg(feature = "cuda")]
use qwen36_fp4_core::{CoreError, Result};
#[cfg(feature = "cuda")]
use serde::Serialize;

#[cfg(feature = "cuda")]
use crate::backend::DevicePtr;

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaDeviceBuffer {
    ptr: DevicePtr,
    bytes: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize)]
pub struct CudaDiagnostics {
    pub driver_version: i32,
    pub runtime_version: i32,
    pub device_count: i32,
    pub active_device: i32,
    pub sm_major: i32,
    pub sm_minor: i32,
    pub multiprocessor_count: i32,
    pub total_global_mem: usize,
    pub device_name: String,
    pub libcuda_path: String,
    pub cudart_path: String,
    pub last_cuda_error: i32,
    pub last_cuda_error_name: String,
    pub last_cuda_error_string: String,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CudaCounters {
    pub malloc_calls: u64,
    pub free_calls: u64,
    pub h2d_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_calls: u64,
    pub d2h_bytes: u64,
    pub d2d_calls: u64,
    pub d2d_bytes: u64,
    pub d2d_async_calls: u64,
    pub d2d_async_bytes: u64,
    pub memset_calls: u64,
    pub memset_bytes: u64,
    pub synchronize_calls: u64,
    pub stream_synchronize_calls: u64,
    pub graph_launch_calls: u64,
}

#[cfg(feature = "cuda")]
impl CudaDeviceBuffer {
    pub fn alloc(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(CoreError::Runtime(
                "cannot allocate a zero-byte CUDA buffer".to_owned(),
            ));
        }
        let mut allocation = ffi::DeviceAllocation {
            ptr: DevicePtr::NULL,
            bytes: 0,
        };
        let code = unsafe { ffi::qwen36_cuda_malloc(&mut allocation, bytes) };
        if code != 0 {
            let detail = cuda_diagnostics()
                .map(|diag| {
                    format!(
                        "; driver={} runtime={} gpu='{}' libcuda='{}' cudart='{}' last_cuda_error={} {} ({})",
                        diag.driver_version,
                        diag.runtime_version,
                        diag.device_name,
                        diag.libcuda_path,
                        diag.cudart_path,
                        diag.last_cuda_error,
                        diag.last_cuda_error_name,
                        diag.last_cuda_error_string
                    )
                })
                .unwrap_or_else(|err| format!("; diagnostics unavailable: {err}"));
            return Err(CoreError::Runtime(format!(
                "qwen36_cuda_malloc({bytes} bytes) failed with code {code}{detail}"
            )));
        }
        Ok(Self {
            ptr: allocation.ptr,
            bytes: allocation.bytes,
        })
    }

    pub fn zeroed(bytes: usize) -> Result<Self> {
        let buffer = Self::alloc(bytes)?;
        buffer.memset(0)?;
        Ok(buffer)
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn ptr_at(&self, offset_bytes: usize) -> Result<DevicePtr> {
        if offset_bytes > self.bytes {
            return Err(CoreError::Runtime(format!(
                "CUDA pointer offset {offset_bytes} exceeds buffer of {} bytes",
                self.bytes
            )));
        }
        self.ptr
            .offset_bytes(offset_bytes)
            .ok_or_else(|| CoreError::Runtime("CUDA pointer offset overflow".to_owned()))
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn copy_from_host(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.bytes {
            return Err(CoreError::Runtime(format!(
                "host copy of {} bytes exceeds CUDA buffer of {} bytes",
                bytes.len(),
                self.bytes
            )));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        check("qwen36_cuda_memcpy_h2d", unsafe {
            ffi::qwen36_cuda_memcpy_h2d(self.ptr, bytes.as_ptr().cast(), bytes.len())
        })
    }

    pub fn copy_to_host(&self, bytes: &mut [u8]) -> Result<()> {
        self.copy_to_host_at(0, bytes)
    }

    pub fn copy_to_host_at(&self, source_offset_bytes: usize, bytes: &mut [u8]) -> Result<()> {
        let end = source_offset_bytes
            .checked_add(bytes.len())
            .ok_or_else(|| {
                CoreError::Runtime("device-to-host copy source offset overflow".to_owned())
            })?;
        if end > self.bytes {
            return Err(CoreError::Runtime(format!(
                "device copy from byte range {source_offset_bytes}..{end} exceeds CUDA buffer of {} bytes",
                self.bytes
            )));
        }
        if bytes.len() > self.bytes {
            return Err(CoreError::Runtime(format!(
                "device copy of {} bytes exceeds CUDA buffer of {} bytes",
                bytes.len(),
                self.bytes
            )));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let source = self.ptr.offset_bytes(source_offset_bytes).ok_or_else(|| {
            CoreError::Runtime("device-to-host copy source pointer overflow".to_owned())
        })?;
        check("qwen36_cuda_memcpy_d2h", unsafe {
            ffi::qwen36_cuda_memcpy_d2h(bytes.as_mut_ptr().cast(), source, bytes.len())
        })
    }

    pub fn copy_from_device(&self, source: &Self, bytes: usize) -> Result<()> {
        if bytes > source.bytes {
            return Err(CoreError::Runtime(format!(
                "device-to-device copy of {bytes} bytes exceeds source buffer of {} bytes",
                source.bytes
            )));
        }
        self.copy_from_device_ptr(source.ptr, bytes)
    }

    pub fn copy_from_device_ptr(&self, source: DevicePtr, bytes: usize) -> Result<()> {
        self.copy_from_device_ptr_at(0, source, bytes)
    }

    pub fn copy_from_device_ptr_at(
        &self,
        destination_offset_bytes: usize,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        let end = destination_offset_bytes.checked_add(bytes).ok_or_else(|| {
            CoreError::Runtime("device-to-device copy destination offset overflow".to_owned())
        })?;
        if end > self.bytes {
            return Err(CoreError::Runtime(format!(
                "device-to-device copy to byte range {destination_offset_bytes}..{end} exceeds destination buffer of {} bytes",
                self.bytes
            )));
        }
        if bytes > self.bytes {
            return Err(CoreError::Runtime(format!(
                "device-to-device copy of {bytes} bytes exceeds destination buffer of {} bytes",
                self.bytes
            )));
        }
        if source == DevicePtr::NULL {
            return Err(CoreError::Runtime(
                "cannot copy from a null CUDA source pointer".to_owned(),
            ));
        }
        if bytes == 0 {
            return Ok(());
        }
        let destination = self
            .ptr
            .offset_bytes(destination_offset_bytes)
            .ok_or_else(|| {
                CoreError::Runtime("device-to-device copy destination pointer overflow".to_owned())
            })?;
        check("qwen36_cuda_memcpy_d2d_async", unsafe {
            ffi::qwen36_cuda_memcpy_d2d_async(destination, source, bytes)
        })
    }

    pub fn memset(&self, value: u8) -> Result<()> {
        check("qwen36_cuda_memset", unsafe {
            ffi::qwen36_cuda_memset(self.ptr, i32::from(value), self.bytes)
        })
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != DevicePtr::NULL {
            unsafe {
                let _ = ffi::qwen36_cuda_free(self.ptr);
            }
            self.ptr = DevicePtr::NULL;
        }
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_synchronize() -> Result<()> {
    check("qwen36_cuda_synchronize", unsafe {
        ffi::qwen36_cuda_synchronize()
    })
}

#[cfg(feature = "cuda")]
pub fn cuda_diagnostics() -> Result<CudaDiagnostics> {
    let mut raw = ffi::CudaDiagnosticsRaw {
        driver_version: 0,
        runtime_version: 0,
        device_count: 0,
        active_device: 0,
        sm_major: 0,
        sm_minor: 0,
        multiprocessor_count: 0,
        total_global_mem: 0,
        device_name: [0; 128],
        libcuda_path: [0; 512],
        cudart_path: [0; 512],
        last_cuda_error: 0,
        last_cuda_error_name: [0; 64],
        last_cuda_error_string: [0; 256],
    };
    check("qwen36_cuda_get_diagnostics", unsafe {
        ffi::qwen36_cuda_get_diagnostics(&mut raw)
    })?;
    Ok(CudaDiagnostics {
        driver_version: raw.driver_version,
        runtime_version: raw.runtime_version,
        device_count: raw.device_count,
        active_device: raw.active_device,
        sm_major: raw.sm_major,
        sm_minor: raw.sm_minor,
        multiprocessor_count: raw.multiprocessor_count,
        total_global_mem: raw.total_global_mem,
        device_name: c_array_to_string(&raw.device_name),
        libcuda_path: c_array_to_string(&raw.libcuda_path),
        cudart_path: c_array_to_string(&raw.cudart_path),
        last_cuda_error: raw.last_cuda_error,
        last_cuda_error_name: c_array_to_string(&raw.last_cuda_error_name),
        last_cuda_error_string: c_array_to_string(&raw.last_cuda_error_string),
    })
}

#[cfg(feature = "cuda")]
pub fn cuda_counters_reset() -> Result<()> {
    check("qwen36_cuda_counters_reset", unsafe {
        ffi::qwen36_cuda_counters_reset()
    })
}

#[cfg(feature = "cuda")]
pub fn cuda_counters_read() -> Result<CudaCounters> {
    let mut raw = ffi::CudaCountersRaw {
        malloc_calls: 0,
        free_calls: 0,
        h2d_calls: 0,
        h2d_bytes: 0,
        d2h_calls: 0,
        d2h_bytes: 0,
        d2d_calls: 0,
        d2d_bytes: 0,
        d2d_async_calls: 0,
        d2d_async_bytes: 0,
        memset_calls: 0,
        memset_bytes: 0,
        synchronize_calls: 0,
        stream_synchronize_calls: 0,
        graph_launch_calls: 0,
    };
    check("qwen36_cuda_counters_read", unsafe {
        ffi::qwen36_cuda_counters_read(&mut raw)
    })?;
    Ok(CudaCounters {
        malloc_calls: raw.malloc_calls,
        free_calls: raw.free_calls,
        h2d_calls: raw.h2d_calls,
        h2d_bytes: raw.h2d_bytes,
        d2h_calls: raw.d2h_calls,
        d2h_bytes: raw.d2h_bytes,
        d2d_calls: raw.d2d_calls,
        d2d_bytes: raw.d2d_bytes,
        d2d_async_calls: raw.d2d_async_calls,
        d2d_async_bytes: raw.d2d_async_bytes,
        memset_calls: raw.memset_calls,
        memset_bytes: raw.memset_bytes,
        synchronize_calls: raw.synchronize_calls,
        stream_synchronize_calls: raw.stream_synchronize_calls,
        graph_launch_calls: raw.graph_launch_calls,
    })
}

#[cfg(feature = "cuda")]
fn c_array_to_string(bytes: &[u8]) -> String {
    let nul = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..nul]).into_owned()
}

/// Pin a memory window to the L2 cache via the active stream's access policy.
/// `hit_ratio` is the fraction of accesses that should be cached, in [0, 1].
/// Best effort under L2 pressure. No-op on the legacy default stream.
#[cfg(feature = "cuda")]
pub fn cuda_set_l2_access_window(
    base: crate::backend::DevicePtr,
    bytes: usize,
    hit_ratio: f32,
) -> Result<()> {
    check("qwen36_cuda_set_l2_access_window", unsafe {
        ffi::qwen36_cuda_set_l2_access_window(base, bytes, hit_ratio)
    })
}

#[cfg(feature = "cuda")]
pub fn cuda_clear_l2_access_window() -> Result<()> {
    check("qwen36_cuda_clear_l2_access_window", unsafe {
        ffi::qwen36_cuda_clear_l2_access_window()
    })
}

#[cfg(feature = "cuda")]
fn check(kernel: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CoreError::KernelLaunch { kernel, code })
    }
}

#[cfg(feature = "cuda")]
mod ffi {
    use crate::backend::DevicePtr;

    #[repr(C)]
    pub struct DeviceAllocation {
        pub ptr: DevicePtr,
        pub bytes: usize,
    }

    #[repr(C)]
    pub struct CudaDiagnosticsRaw {
        pub driver_version: i32,
        pub runtime_version: i32,
        pub device_count: i32,
        pub active_device: i32,
        pub sm_major: i32,
        pub sm_minor: i32,
        pub multiprocessor_count: i32,
        pub total_global_mem: usize,
        pub device_name: [u8; 128],
        pub libcuda_path: [u8; 512],
        pub cudart_path: [u8; 512],
        pub last_cuda_error: i32,
        pub last_cuda_error_name: [u8; 64],
        pub last_cuda_error_string: [u8; 256],
    }

    #[repr(C)]
    pub struct CudaCountersRaw {
        pub malloc_calls: u64,
        pub free_calls: u64,
        pub h2d_calls: u64,
        pub h2d_bytes: u64,
        pub d2h_calls: u64,
        pub d2h_bytes: u64,
        pub d2d_calls: u64,
        pub d2d_bytes: u64,
        pub d2d_async_calls: u64,
        pub d2d_async_bytes: u64,
        pub memset_calls: u64,
        pub memset_bytes: u64,
        pub synchronize_calls: u64,
        pub stream_synchronize_calls: u64,
        pub graph_launch_calls: u64,
    }

    #[link(name = "qwen36_fp4_kernels")]
    unsafe extern "C" {
        pub fn qwen36_cuda_malloc(out: *mut DeviceAllocation, bytes: usize) -> i32;
        pub fn qwen36_cuda_free(ptr: DevicePtr) -> i32;
        pub fn qwen36_cuda_memcpy_h2d(
            dst: DevicePtr,
            src: *const std::ffi::c_void,
            bytes: usize,
        ) -> i32;
        pub fn qwen36_cuda_memcpy_d2h(
            dst: *mut std::ffi::c_void,
            src: DevicePtr,
            bytes: usize,
        ) -> i32;
        pub fn qwen36_cuda_memcpy_d2d_async(dst: DevicePtr, src: DevicePtr, bytes: usize) -> i32;
        pub fn qwen36_cuda_memset(dst: DevicePtr, value: i32, bytes: usize) -> i32;
        pub fn qwen36_cuda_synchronize() -> i32;
        pub fn qwen36_cuda_get_diagnostics(out: *mut CudaDiagnosticsRaw) -> i32;
        pub fn qwen36_cuda_counters_reset() -> i32;
        pub fn qwen36_cuda_counters_read(out: *mut CudaCountersRaw) -> i32;
        pub fn qwen36_cuda_set_l2_access_window(
            base: DevicePtr,
            bytes: usize,
            hit_ratio: f32,
        ) -> i32;
        pub fn qwen36_cuda_clear_l2_access_window() -> i32;
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn round_trips_host_bytes_through_cuda_buffer() {
        let input = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let buffer = CudaDeviceBuffer::alloc(input.len()).unwrap();
        buffer.copy_from_host(&input).unwrap();
        cuda_synchronize().unwrap();

        let mut observed = [0_u8; 8];
        buffer.copy_to_host(&mut observed).unwrap();
        assert_eq!(observed, input);
    }
}
