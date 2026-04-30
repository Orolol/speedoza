#[cfg(feature = "cuda")]
use qwen36_fp4_core::{CoreError, Result};

#[cfg(feature = "cuda")]
use crate::backend::DevicePtr;

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaDeviceBuffer {
    ptr: DevicePtr,
    bytes: usize,
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
        check("qwen36_cuda_malloc", unsafe {
            ffi::qwen36_cuda_malloc(&mut allocation, bytes)
        })?;
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
        check("qwen36_cuda_memcpy_d2d", unsafe {
            ffi::qwen36_cuda_memcpy_d2d(destination, source, bytes)
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
        pub fn qwen36_cuda_memcpy_d2d(dst: DevicePtr, src: DevicePtr, bytes: usize) -> i32;
        pub fn qwen36_cuda_memset(dst: DevicePtr, value: i32, bytes: usize) -> i32;
        pub fn qwen36_cuda_synchronize() -> i32;
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
