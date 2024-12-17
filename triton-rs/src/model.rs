use crate::{check_err, Error};
use libc::c_char;
#[cfg(feature = "json")]
use serde_json::Value;
use std::any::{Any, TypeId};
use std::ffi::CStr;
use std::fs::File;
use std::io::prelude::*;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::ptr;
#[cfg(feature = "tracing")]
use tracing::error;

pub struct Model {
    ptr: *mut triton_sys::TRITONBACKEND_Model,
}

impl Model {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Model) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, Error> {
        let mut model_name: *const c_char = ptr::null_mut();
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelName(self.ptr, &mut model_name) })?;

        let c_str = unsafe { CStr::from_ptr(model_name) };
        Ok(c_str.to_string_lossy().to_string())
    }

    pub fn set_data<D>(&self, data: D) -> Result<(), Error>
    where
        D: Any + Send + 'static,
    {
        self.drop_data()?;
        let data = Box::new(data);
        let data_ptr = Box::into_raw(data);
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelSetState(self.ptr, data_ptr as *mut std::ffi::c_void)
        })?;
        Ok(())
    }

    pub fn drop_data(&self) -> Result<Option<Box<dyn Any>>, Error> {
        let mut state_ptr: *mut std::ffi::c_void = ptr::null_mut();
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelState(self.ptr, &mut state_ptr) })?;
        if !state_ptr.is_null() {
            return unsafe {
                let data = Box::from_raw(state_ptr as *mut Box<dyn Any>);
                triton_sys::TRITONBACKEND_ModelSetState(self.ptr, ptr::null_mut());
                Ok(Some(data))
            };
        }
        Ok(None)
    }

    pub fn data<D>(&self) -> Result<Option<&D>, Error>
    where
        D: Any,
    {
        let mut state_ptr = std::ptr::null_mut();
        unsafe {
            check_err(triton_sys::TRITONBACKEND_ModelState(
                self.ptr,
                &mut state_ptr,
            ))?;
        }
        if state_ptr.is_null() {
            Ok(None)
        } else {
            let data = unsafe { &*(state_ptr as *const D) };
            if TypeId::of::<D>() == (*data).type_id() {
                Ok(Some(data))
            } else {
                Err(Error::from("Model state type mismatch"))
            }
        }
    }

    pub fn version(&self) -> Result<u64, Error> {
        let mut version = 0u64;
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelVersion(self.ptr, &mut version) })?;
        Ok(version)
    }

    pub fn location(&self) -> Result<String, Error> {
        let mut artifact_type: triton_sys::TRITONBACKEND_ArtifactType = 0u32;
        let mut location: *const c_char = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelRepository(self.ptr, &mut artifact_type, &mut location)
        })?;

        let c_str = unsafe { CStr::from_ptr(location) };
        Ok(c_str.to_string_lossy().to_string())
    }

    pub fn path(&self, filename: &str) -> Result<PathBuf, Error> {
        Ok(PathBuf::from(format!(
            "{}/{}/{}",
            self.location()?,
            self.version()?,
            filename
        )))
    }

    pub fn load_file(&self, filename: &str) -> Result<Vec<u8>, Error> {
        let path = self.path(filename)?;
        let mut f = File::open(path)?;

        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;

        Ok(buffer)
    }

    pub fn config(&self) -> Result<ModelConfig, Error> {
        ModelConfig::from_model(self)
    }
}

pub struct ModelConfig {
    ptr: *mut triton_sys::TRITONSERVER_Message,
}

impl ModelConfig {
    fn from_model(model: &Model) -> Result<Self, Error> {
        let mut ptr = MaybeUninit::uninit();
        unsafe {
            check_err(triton_sys::TRITONBACKEND_ModelConfig(
                model.ptr,
                1,
                ptr.as_mut_ptr(),
            ))?;
            Ok(Self {
                ptr: ptr.assume_init(),
            })
        }
    }

    /// Get the model configuration as a JSON-formatted string
    pub fn as_str(&self) -> Result<&str, Error> {
        let mut base = MaybeUninit::uninit();
        let mut byte_size = MaybeUninit::uninit();
        unsafe {
            check_err(triton_sys::TRITONSERVER_MessageSerializeToJson(
                self.ptr,
                base.as_mut_ptr(),
                byte_size.as_mut_ptr(),
            ))?;
            let base = base.assume_init();
            let byte_size = byte_size.assume_init();
            let json_slice = std::slice::from_raw_parts(base as *const u8, byte_size);
            Ok(std::str::from_utf8(json_slice)?)
        }
    }

    #[cfg(feature = "json")]
    /// Get the model configuration as a [`serde_json::Value`]
    pub fn as_json(&self) -> Result<Value, Error> {
        let mut base = MaybeUninit::uninit();
        let mut byte_size = MaybeUninit::uninit();
        unsafe {
            check_err(triton_sys::TRITONSERVER_MessageSerializeToJson(
                self.ptr,
                base.as_mut_ptr(),
                byte_size.as_mut_ptr(),
            ))?;
            let base = base.assume_init();
            let byte_size = byte_size.assume_init();
            let json_slice = std::slice::from_raw_parts(base as *const u8, byte_size);
            Ok(serde_json::from_slice(json_slice)?)
        }
    }
}

impl Drop for ModelConfig {
    fn drop(&mut self) {
        unsafe {
            let _result = check_err(triton_sys::TRITONSERVER_MessageDelete(self.ptr));
            #[cfg(feature = "tracing")]
            if let Err(error) = _result {
                error!(error, "Error deleting model config");
            }
        }
    }
}
