use crate::{check_err, decode_string, Error};
use libc::c_void;
use std::borrow::Cow;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use std::slice;

pub struct Request {
    ptr: *mut triton_sys::TRITONBACKEND_Request,
}

impl Request {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Request) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONBACKEND_Request {
        self.ptr
    }

    pub fn get_input(&self, name: &str) -> Result<Input, Error> {
        let name = CString::new(name).expect("CString::new failed");

        let mut input: *mut triton_sys::TRITONBACKEND_Input = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_RequestInput(self.ptr, name.as_ptr(), &mut input)
        })?;

        Ok(Input::from_ptr(input))
    }
}

pub struct Input {
    ptr: *mut triton_sys::TRITONBACKEND_Input,
}
impl Input {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Input) -> Self {
        Self { ptr }
    }

    /// Gets a reference to the buffer associated with the input. Note the buffer index must
    /// be less than the buffer count (
    fn raw_buffer(&self, buffer_index: u32) -> Result<&[u8], Error> {
        let mut buffer: *const c_void = ptr::null_mut();
        let mut memory_type = triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
        let mut memory_type_id = 0;
        let mut buffer_byte_size = 0;
        check_err(unsafe {
            triton_sys::TRITONBACKEND_InputBuffer(
                self.ptr,
                buffer_index,
                &mut buffer,
                &mut buffer_byte_size,
                &mut memory_type,
                &mut memory_type_id,
            )
        })?;

        Ok(unsafe { slice::from_raw_parts(buffer as *const u8, buffer_byte_size as usize) })
    }

    pub fn buffer(&self) -> Result<Cow<[u8]>, Error> {
        let properties = self.properties()?;
        match properties.buffer_count {
            1 => Ok(Cow::Borrowed(self.raw_buffer(0)?)),
            _ => {
                let mut retval = Vec::with_capacity(properties.byte_size as usize);
                for buf_idx in 0..properties.buffer_count {
                    let buffer = self.raw_buffer(buf_idx)?;
                    retval.extend_from_slice(buffer);
                }
                Ok(Cow::Owned(retval))
            }
        }
    }

    pub fn as_string(&self) -> Result<String, Error> {
        let buffer = self.buffer()?;

        let strings = decode_string(&buffer)?;
        Ok(strings.first().unwrap().clone())
    }

    pub fn as_u64(&self) -> Result<u64, Error> {
        let buffer = self.buffer()?;

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buffer);

        Ok(u64::from_le_bytes(bytes))
    }

    pub fn properties(&self) -> Result<InputProperties, Error> {
        let mut name = ptr::null();
        let mut datatype = 0u32;
        let shape = ptr::null_mut();
        let mut dims_count = 0u32;
        let mut byte_size = 0u64;
        let mut buffer_count = 0u32;

        check_err(unsafe {
            triton_sys::TRITONBACKEND_InputProperties(
                self.ptr,
                &mut name,
                &mut datatype,
                shape,
                &mut dims_count,
                &mut byte_size,
                &mut buffer_count,
            )
        })?;

        let name = unsafe { CStr::from_ptr(name) };

        Ok(InputProperties {
            name,
            datatype,
            // shape,
            dims_count,
            byte_size,
            buffer_count,
        })
    }
}

#[derive(Debug)]
pub struct InputProperties<'a> {
    pub name: &'a CStr,
    pub datatype: u32,
    // shape: Vec<i64>,
    pub dims_count: u32,
    pub byte_size: u64,
    pub buffer_count: u32,
}
