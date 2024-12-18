use crate::{check_err, decode_string, Error};
use libc::c_void;
use std::borrow::Cow;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use std::slice;
use std::str::FromStr;
use triton_sys::TRITONSERVER_DataType;

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
        match memory_type {
            triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU
            | triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED => {
                debug_assert!(buffer.is_aligned());
                debug_assert!(!buffer.is_null());
                let buffer = buffer as *const u8;
                Ok(unsafe { slice::from_raw_parts(buffer, buffer_byte_size as usize) })
            }
            _ => Err(Error::from("GPU memory is unsupported")),
        }
    }

    pub fn buffer(&self) -> Result<Cow<[u8]>, Error> {
        let properties = self.properties()?;
        match properties.buffer_count {
            1 => {
                let retval = self.raw_buffer(0)?;
                Ok(Cow::Borrowed(retval))
            }
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
        let mut dims_count = 0u32;
        let mut byte_size = 0u64;
        let mut buffer_count = 0u32;

        check_err(unsafe {
            triton_sys::TRITONBACKEND_InputProperties(
                self.ptr,
                &mut name,
                &mut datatype,
                ptr::null_mut(),
                &mut dims_count,
                &mut byte_size,
                &mut buffer_count,
            )
        })?;

        let name = unsafe { CStr::from_ptr(name) };

        Ok(InputProperties {
            name,
            datatype,
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
    // pub shape: &'a [i64],
    pub dims_count: u32,
    pub byte_size: u64,
    pub buffer_count: u32,
}

/// Represents an input or output data type for a model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Boolean
    Bool,
    /// Unsigned 8-bit integer
    UInt8,
    /// Unsigned 16-bit integer
    UInt16,
    /// Unsigned 32-bit integer
    UInt32,
    /// Unsigned 64-bit integer
    UInt64,
    /// Signed 8-bit integer
    Int8,
    /// Signed 16-bit integer
    Int16,
    /// Signed 32-bit integer
    Int32,
    /// Signed 64-bit integer
    Int64,
    /// 16-bit floating point
    Fp16,
    /// 32-bit floating point
    Fp32,
    /// 64-bit floating point
    Fp64,
    /// Arbitrary bytes
    Bytes,
    /// 16-bit floating point (bfloat16)
    Bf16,
}

impl From<DataType> for TRITONSERVER_DataType {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Bool => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL,
            DataType::UInt8 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8,
            DataType::UInt16 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16,
            DataType::UInt32 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32,
            DataType::UInt64 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64,
            DataType::Int8 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8,
            DataType::Int16 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16,
            DataType::Int32 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32,
            DataType::Int64 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64,
            DataType::Fp16 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16,
            DataType::Fp32 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
            DataType::Fp64 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64,
            DataType::Bytes => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES,
            DataType::Bf16 => triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16,
        }
    }
}

impl TryFrom<u32> for DataType {
    type Error = Error;
    fn try_from(data_type: u32) -> Result<Self, Error> {
        match data_type {
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL => Ok(DataType::Bool),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8 => Ok(DataType::UInt8),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16 => Ok(DataType::UInt16),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32 => Ok(DataType::UInt32),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64 => Ok(DataType::UInt64),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8 => Ok(DataType::Int8),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16 => Ok(DataType::Int16),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32 => Ok(DataType::Int32),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64 => Ok(DataType::Int64),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16 => Ok(DataType::Fp16),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32 => Ok(DataType::Fp32),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64 => Ok(DataType::Fp64),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES => Ok(DataType::Bytes),
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16 => Ok(DataType::Bf16),
            _ => Err(Error::from("Unknown TRITONSERVER_DataType")),
        }
    }
}

impl FromStr for DataType {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "TYPE_BOOL" => Ok(DataType::Bool),
            "TYPE_UINT8" => Ok(DataType::UInt8),
            "TYPE_UINT16" => Ok(DataType::UInt16),
            "TYPE_UINT32" => Ok(DataType::UInt32),
            "TYPE_UINT64" => Ok(DataType::UInt64),
            "TYPE_INT8" => Ok(DataType::Int8),
            "TYPE_INT16" => Ok(DataType::Int16),
            "TYPE_INT32" => Ok(DataType::Int32),
            "TYPE_INT64" => Ok(DataType::Int64),
            "TYPE_FP16" => Ok(DataType::Fp16),
            "TYPE_FP32" => Ok(DataType::Fp32),
            "TYPE_FP64" => Ok(DataType::Fp64),
            "TYPE_BYTES" => Ok(DataType::Bytes),
            "TYPE_BF16" => Ok(DataType::Bf16),
            _ => Err(Error::from("Unknown data type")),
        }
    }
}

impl From<DataType> for &'static str {
    fn from(data_type: DataType) -> &'static str {
        match data_type {
            DataType::Bool => "TYPE_BOOL",
            DataType::UInt8 => "TYPE_UINT8",
            DataType::UInt16 => "TYPE_UINT16",
            DataType::UInt32 => "TYPE_UINT32",
            DataType::UInt64 => "TYPE_UINT64",
            DataType::Int8 => "TYPE_INT8",
            DataType::Int16 => "TYPE_INT16",
            DataType::Int32 => "TYPE_INT32",
            DataType::Int64 => "TYPE_INT64",
            DataType::Fp16 => "TYPE_FP16",
            DataType::Fp32 => "TYPE_FP32",
            DataType::Fp64 => "TYPE_FP64",
            DataType::Bytes => "TYPE_BYTES",
            DataType::Bf16 => "TYPE_BF16",
        }
    }
}
