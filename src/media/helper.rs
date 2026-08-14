use super::helper_protocol::{DecodeRequest, DecodeResult, PROTOCOL_VERSION};
use super::{MediaError, MediaErrorKind};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

const MAX_HELPER_STDOUT: usize = 1024 * 1024;
const MAX_HELPER_STDERR: usize = 64 * 1024;

pub async fn decode(request: &DecodeRequest) -> Result<DecodeResult, MediaError> {
    validate_request_paths(request)?;
    let helper = sibling_helper()?;
    let mut child = Command::new(helper)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| {
            MediaError::new(
                MediaErrorKind::ComponentUnavailable,
                "media helper unavailable",
            )
        })?;
    let job = assign_job_limits(&child, request.limits.ram_bytes, request.limits.cpu_time_ms)?;
    let body = serde_json::to_vec(request)
        .map_err(|_| MediaError::new(MediaErrorKind::Internal, "cannot encode helper request"))?;
    let mut stdin = child.stdin.take().ok_or_else(|| {
        MediaError::new(MediaErrorKind::Internal, "media helper stdin unavailable")
    })?;
    stdin.write_all(&body).await.map_err(|_| {
        MediaError::new(
            MediaErrorKind::ComponentUnavailable,
            "media helper write failed",
        )
    })?;
    drop(stdin);
    let timeout = std::time::Duration::from_millis(request.limits.wall_time_ms.max(1));
    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| MediaError::new(MediaErrorKind::Timeout, "media decoder timed out"))?
        .map_err(|_| {
            MediaError::new(MediaErrorKind::ComponentUnavailable, "media helper failed")
        })?;
    drop(job);
    if output.stdout.len() > MAX_HELPER_STDOUT || output.stderr.len() > MAX_HELPER_STDERR {
        return Err(MediaError::new(
            MediaErrorKind::DecodedTooLarge,
            "media helper output exceeded limit",
        ));
    }
    if !output.status.success() {
        return Err(MediaError::invalid("media decoding failed"));
    }
    let result: DecodeResult = serde_json::from_slice(&output.stdout)
        .map_err(|_| MediaError::invalid("invalid media helper result"))?;
    validate_result(request, &result)?;
    Ok(result)
}

fn validate_request_paths(request: &DecodeRequest) -> Result<(), MediaError> {
    if request.protocol_version != PROTOCOL_VERSION
        || !request.input_path.is_absolute()
        || !request.output_dir.is_absolute()
        || request.limits.max_output_bytes == 0
    {
        return Err(MediaError::invalid("invalid media helper request"));
    }
    Ok(())
}

fn validate_result(request: &DecodeRequest, result: &DecodeResult) -> Result<(), MediaError> {
    if result.protocol_version != PROTOCOL_VERSION
        || result.kind != request.expected_kind
        || result.audio_processed
        || result.frames.len() > request.limits.max_frames
        || result.width > request.limits.max_width
        || result.height > request.limits.max_height
        || result.source_fps.is_some_and(|value| !value.is_finite())
    {
        return Err(MediaError::invalid("media helper result violates contract"));
    }
    let output_root = request
        .output_dir
        .canonicalize()
        .map_err(|_| MediaError::invalid("media helper output directory missing"))?;
    let mut total = 0u64;
    for frame in &result.frames {
        let path = frame
            .path
            .canonicalize()
            .map_err(|_| MediaError::invalid("media helper frame missing"))?;
        if !path.starts_with(&output_root) {
            return Err(MediaError::invalid(
                "media helper frame escapes output directory",
            ));
        }
        let expected = u64::from(frame.width)
            .checked_mul(u64::from(frame.height))
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                MediaError::new(MediaErrorKind::DecodedTooLarge, "decoded frame too large")
            })?;
        let bytes = std::fs::read(&path)
            .map_err(|_| MediaError::invalid("cannot read media helper frame"))?;
        if frame.bytes != expected || bytes.len() as u64 != expected {
            return Err(MediaError::invalid("media helper frame size mismatch"));
        }
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(&frame.sha256) {
            return Err(MediaError::invalid("media helper frame hash mismatch"));
        }
        total = total.checked_add(expected).ok_or_else(|| {
            MediaError::new(MediaErrorKind::DecodedTooLarge, "decoded media too large")
        })?;
    }
    if total > request.limits.max_output_bytes {
        return Err(MediaError::new(
            MediaErrorKind::DecodedTooLarge,
            "decoded media output exceeds limit",
        ));
    }
    Ok(())
}

fn sibling_helper() -> Result<PathBuf, MediaError> {
    if let Some(path) = std::env::var_os("QWEN36_MEDIA_HELPER") {
        return Ok(PathBuf::from(path));
    }
    let executable = std::env::current_exe().map_err(|_| {
        MediaError::new(
            MediaErrorKind::ComponentUnavailable,
            "runtime path unavailable",
        )
    })?;
    let name = if cfg!(windows) {
        "qwen36-media-helper.exe"
    } else {
        "qwen36-media-helper"
    };
    Ok(executable.with_file_name(name))
}

#[cfg(windows)]
fn assign_job_limits(
    child: &tokio::process::Child,
    memory_bytes: u64,
    cpu_time_ms: u64,
) -> Result<JobHandle, MediaError> {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_JOB_TIME,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    let process = child.raw_handle().ok_or_else(|| {
        MediaError::new(MediaErrorKind::Internal, "media helper handle unavailable")
    })?;
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return Err(MediaError::new(
                MediaErrorKind::Internal,
                "cannot create media job",
            ));
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_JOB_MEMORY
            | JOB_OBJECT_LIMIT_JOB_TIME;
        info.BasicLimitInformation.ActiveProcessLimit = 2;
        info.BasicLimitInformation.PerJobUserTimeLimit =
            i64::try_from(cpu_time_ms.saturating_mul(10_000)).unwrap_or(i64::MAX);
        info.JobMemoryLimit = usize::try_from(memory_bytes).unwrap_or(usize::MAX);
        let configured = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        ) != 0;
        let assigned = configured && AssignProcessToJobObject(job, process) != 0;
        if !assigned {
            CloseHandle(job);
            return Err(MediaError::new(
                MediaErrorKind::Internal,
                "cannot isolate media helper",
            ));
        }
        Ok(JobHandle(job))
    }
}

#[cfg(windows)]
struct JobHandle(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl Drop for JobHandle {
    fn drop(&mut self) {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
    }
}

#[cfg(not(windows))]
struct JobHandle;

#[cfg(not(windows))]
fn assign_job_limits(
    _child: &tokio::process::Child,
    _memory_bytes: u64,
    _cpu_time_ms: u64,
) -> Result<JobHandle, MediaError> {
    Ok(JobHandle)
}
