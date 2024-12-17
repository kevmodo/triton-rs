//! Triton inference response handling.
use crate::{check_err, Error};
use libc::c_void;
use std::ffi::CString;
use std::ptr;
use std::slice;
#[cfg(feature = "tracing")]
use tracing::error;

/// Represents a response to a Triton inference request.
///
/// This struct wraps the raw Triton response pointer and provides safe methods
/// to create and send responses back to clients.
pub struct Response {
    ptr: *mut triton_sys::TRITONBACKEND_Response,
    sent: bool,
}

impl Response {
    /// Creates a new response associated with the given request.
    pub fn new(request: &super::Request) -> Result<Self, Error> {
        let mut response: *mut triton_sys::TRITONBACKEND_Response = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseNew(&mut response, request.as_ptr())
        })?;
        Ok(Self {
            ptr: response,
            sent: false,
        })
    }

    /// Creates a new output tensor in the response.
    ///
    /// # Arguments
    /// * `name` - Name of the output tensor
    /// * `datatype` - The data type of the tensor (as a Triton type code)
    /// * `shape` - The shape of the output tensor
    pub fn output(&mut self, name: &str, datatype: u32, shape: &[i64]) -> Result<Output, Error> {
        let mut output: *mut triton_sys::TRITONBACKEND_Output = ptr::null_mut();
        let name = CString::new(name).expect("CString::new failed");

        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseOutput(
                self.ptr,
                &mut output,
                name.as_ptr(),
                datatype,
                shape.as_ptr(),
                shape.len() as u32,
            )
        })?;

        Ok(Output { ptr: output })
    }

    /// Sends the response back to the client.
    ///
    /// This consumes the response object and finalizes the inference response.
    pub fn send(mut self) -> Result<(), Error> {
        let send_flags =
            triton_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL;
        self.sent = true;
        let err = ptr::null_mut();
        check_err(unsafe { triton_sys::TRITONBACKEND_ResponseSend(self.ptr, send_flags, err) })
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        if !self.sent {
            let _result = check_err(unsafe { triton_sys::TRITONBACKEND_ResponseDelete(self.ptr) });
            #[cfg(feature = "tracing")]
            if let Err(error) = _result {
                error!(error, "Failed to delete response");
            }
        }
    }
}

/// Represents an output tensor in a Triton response.
///
/// This struct provides methods to allocate and manipulate the output
/// tensor's buffer.
pub struct Output {
    ptr: *mut triton_sys::TRITONBACKEND_Output,
}

impl Output {
    /// Allocates and returns a mutable buffer for the output tensor.
    ///
    /// `size` - The size in bytes to allocate for the buffer
    pub fn buffer(&mut self, size: usize) -> Result<&mut [u8], Error> {
        let mut buffer: *mut c_void = ptr::null_mut();
        let mut memory_type: triton_sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id = 0;

        check_err(unsafe {
            triton_sys::TRITONBACKEND_OutputBuffer(
                self.ptr,
                &mut buffer,
                size as u64,
                &mut memory_type,
                &mut memory_type_id,
            )
        })?;

        Ok(unsafe { slice::from_raw_parts_mut(buffer as *mut u8, size) })
    }
}
