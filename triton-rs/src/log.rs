//! Tracing support for backends

use std::ffi::CString;

use tracing::{Event, Subscriber};
use tracing_subscriber::{
    fmt::format::{PrettyVisitor, Writer},
    layer::Context,
    Layer,
};

/// A [`tracing::Subscriber`] that logs to the Triton log system.
///
/// Example:
/// ```rust
/// use tracing_subscriber::prelude::*;
/// use triton_rs::TritonLogger;
///
/// tracing_subscriber::registry()
///     .with(TritonLogger)
///    .init();
///
/// tracing::error!("This is an error message");
/// tracing::warn!("This is a warning message");
/// tracing::info!("This is an info message");
pub struct TritonLogger;

/// Triton log levels
enum TritonLogLevel {
    /// TRITONSERVER_LOG_VERBOSE
    Verbose,
    /// TRITONSERVER_LOG_INFO
    Info,
    /// TRITONSERVER_LOG_WARN
    Warning,
    /// TRITONSERVER_LOG_ERROR
    Error,
}

impl From<tracing::Level> for TritonLogLevel {
    fn from(level: tracing::Level) -> Self {
        match level {
            tracing::Level::ERROR => TritonLogLevel::Error,
            tracing::Level::WARN => TritonLogLevel::Warning,
            tracing::Level::DEBUG => TritonLogLevel::Verbose,
            tracing::Level::INFO => TritonLogLevel::Info,
            tracing::Level::TRACE => TritonLogLevel::Verbose,
        }
    }
}

impl From<TritonLogLevel> for triton_sys::TRITONSERVER_LogLevel {
    fn from(level: TritonLogLevel) -> Self {
        match level {
            TritonLogLevel::Error => triton_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_ERROR,
            TritonLogLevel::Warning => triton_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_WARN,
            TritonLogLevel::Verbose => {
                triton_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_VERBOSE
            }
            TritonLogLevel::Info => triton_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_INFO,
        }
    }
}

impl<S: Subscriber> Layer<S> for TritonLogger {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let level: TritonLogLevel = (*event.metadata().level()).into();
        let triton_level: triton_sys::TRITONSERVER_LogLevel = level.into();
        // Safety: The level is converted to a valid Triton log level
        let enabled = unsafe { triton_sys::TRITONSERVER_LogIsEnabled(triton_level) };
        if enabled {
            let target = event.metadata().target();
            let filename = event.metadata().file().unwrap_or(target);
            let line: i32 = event.metadata().line().unwrap_or(0) as i32;
            let filename_c = CString::new(filename).unwrap();

            let mut message = String::new();
            let writer = Writer::new(&mut message);
            let mut visitor = PrettyVisitor::new(writer, false);
            event.record(&mut visitor);
            let message_c = CString::new(message).unwrap();

            unsafe {
                triton_sys::TRITONSERVER_LogMessage(
                    triton_level,
                    filename_c.as_ptr(),
                    line,
                    message_c.as_ptr(),
                )
            };
        }
    }
}
