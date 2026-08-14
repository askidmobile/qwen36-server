use super::{sniff_kind, MediaConfig, MediaError, MediaErrorKind, MediaKind, OwnerDigest};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct StoredMedia {
    pub id: String,
    pub kind: MediaKind,
    pub declared_mime: String,
    pub encoded_bytes: u64,
    pub path: PathBuf,
}

#[derive(Debug)]
struct Entry {
    owner_digest: [u8; 32],
    media: StoredMedia,
    expires_at: Instant,
    state: EntryState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryState {
    Uploaded,
    Claimed,
}

#[derive(Debug, Default)]
struct StoreState {
    entries: HashMap<String, Entry>,
    in_flight_uploads: usize,
    temp_reserved_bytes: u64,
    decoded_reserved_bytes: u64,
}

#[derive(Debug)]
pub struct MediaStore {
    root: PathBuf,
    config: MediaConfig,
    state: Mutex<StoreState>,
}

impl Drop for MediaStore {
    fn drop(&mut self) {
        let paths: Vec<PathBuf> = self
            .state
            .get_mut()
            .expect("media store lock")
            .entries
            .drain()
            .map(|(_, entry)| entry.media.path)
            .collect();
        for path in paths {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl MediaStore {
    pub fn new(config: MediaConfig) -> Result<Self, MediaError> {
        std::fs::create_dir_all(&config.temp_root)
            .map_err(|_| MediaError::new(MediaErrorKind::Internal, "cannot create media store"))?;
        let root = config.temp_root.canonicalize().map_err(|_| {
            MediaError::new(MediaErrorKind::Internal, "cannot canonicalize media store")
        })?;
        Ok(Self {
            root,
            config,
            state: Mutex::new(StoreState::default()),
        })
    }

    pub fn spawn_reaper(store: std::sync::Arc<Self>, interval: std::time::Duration) {
        tokio::spawn(async move {
            let mut timer = tokio::time::interval(interval);
            loop {
                timer.tick().await;
                store.reap_expired();
            }
        });
    }

    pub fn upload(
        &self,
        owner: OwnerDigest,
        declared_mime: &str,
        bytes: &[u8],
    ) -> Result<StoredMedia, MediaError> {
        let kind = sniff_kind(bytes, declared_mime)?;
        let limit = match kind {
            MediaKind::Image => self.config.max_image_bytes,
            MediaKind::Video => self.config.max_video_bytes,
        };
        let encoded_bytes = u64::try_from(bytes.len())
            .map_err(|_| MediaError::new(MediaErrorKind::EncodedTooLarge, "media too large"))?;
        if encoded_bytes == 0 || encoded_bytes > limit {
            return Err(MediaError::new(
                MediaErrorKind::EncodedTooLarge,
                "media exceeds encoded size limit",
            ));
        }
        self.reap_expired();
        {
            let mut state = self.state.lock().expect("media store lock");
            if state.entries.len() + state.in_flight_uploads >= self.config.max_pending_uploads {
                return Err(MediaError::new(
                    MediaErrorKind::UploadLimit,
                    "pending upload limit reached",
                ));
            }
            if state
                .temp_reserved_bytes
                .checked_add(encoded_bytes)
                .filter(|total| *total <= self.config.max_temp_bytes)
                .is_none()
            {
                return Err(MediaError::new(
                    MediaErrorKind::StorageExhausted,
                    "temporary media storage exhausted",
                ));
            }
            state.in_flight_uploads += 1;
            state.temp_reserved_bytes += encoded_bytes;
        }
        let reservation = UploadReservation {
            store: self,
            bytes: encoded_bytes,
            committed: false,
        };

        let id = uuid::Uuid::new_v4().simple().to_string();
        let final_path = self.root.join(format!("{id}.media"));
        let part_path = self.root.join(format!("{id}.part"));
        let mut partial = PartialFile::create(&part_path)?;
        partial.write_all(bytes)?;
        partial.sync_all()?;
        std::fs::rename(&part_path, &final_path).map_err(|_| {
            MediaError::new(MediaErrorKind::Internal, "cannot finalize media upload")
        })?;
        partial.keep = true;

        let media = StoredMedia {
            id: id.clone(),
            kind,
            declared_mime: declared_mime.to_string(),
            encoded_bytes,
            path: final_path,
        };
        let mut state = self.state.lock().expect("media store lock");
        state.in_flight_uploads = state.in_flight_uploads.saturating_sub(1);
        state.entries.insert(
            id,
            Entry {
                owner_digest: owner.bytes(),
                media: media.clone(),
                expires_at: Instant::now() + self.config.upload_ttl,
                state: EntryState::Uploaded,
            },
        );
        drop(state);
        reservation.commit();
        Ok(media)
    }

    pub fn claim_batch(
        self: &std::sync::Arc<Self>,
        owner: OwnerDigest,
        ids: &[String],
    ) -> Result<ClaimedMedia, MediaError> {
        if ids.is_empty() {
            return Ok(ClaimedMedia {
                store: self.clone(),
                media: Vec::new(),
            });
        }
        let unique: HashSet<&str> = ids.iter().map(String::as_str).collect();
        if unique.len() != ids.len() {
            return Err(MediaError::invalid("duplicate media ID in request"));
        }
        self.reap_expired();
        let digest = owner.bytes();
        let mut state = self.state.lock().expect("media store lock");
        for id in ids {
            let entry = state.entries.get(id).ok_or_else(|| {
                MediaError::new(MediaErrorKind::Invalid, "media ID is unavailable")
            })?;
            if entry.owner_digest != digest {
                return Err(MediaError::new(
                    MediaErrorKind::Invalid,
                    "media ID is unavailable",
                ));
            }
            if entry.state != EntryState::Uploaded {
                return Err(MediaError::new(
                    MediaErrorKind::Conflict,
                    "media ID is already claimed",
                ));
            }
        }
        let mut media = Vec::with_capacity(ids.len());
        for id in ids {
            let entry = state.entries.get_mut(id).expect("validated media ID");
            entry.state = EntryState::Claimed;
            media.push(entry.media.clone());
        }
        Ok(ClaimedMedia {
            store: self.clone(),
            media,
        })
    }

    pub fn reserve_decoded(
        self: &std::sync::Arc<Self>,
        bytes: u64,
    ) -> Result<DecodedReservation, MediaError> {
        let mut state = self.state.lock().expect("media store lock");
        if state
            .decoded_reserved_bytes
            .checked_add(bytes)
            .filter(|total| *total <= self.config.max_decoded_bytes)
            .is_none()
        {
            return Err(MediaError::new(
                MediaErrorKind::DecodedTooLarge,
                "decoded media budget exceeded",
            ));
        }
        state.decoded_reserved_bytes += bytes;
        Ok(DecodedReservation {
            store: self.clone(),
            bytes,
        })
    }

    pub fn reap_expired(&self) -> usize {
        self.reap_at(Instant::now())
    }

    pub fn purge_all(&self) -> usize {
        self.reap_matching(|_| true)
    }

    pub fn cleanup_orphans(&self) -> usize {
        let known: HashSet<PathBuf> = self
            .state
            .lock()
            .expect("media store lock")
            .entries
            .values()
            .map(|entry| entry.media.path.clone())
            .collect();
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return 0;
        };
        let mut removed = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let removable = path.extension().and_then(|value| value.to_str()) == Some("part")
                || (path.extension().and_then(|value| value.to_str()) == Some("media")
                    && !known.contains(&path));
            if removable && std::fs::remove_file(path).is_ok() {
                removed += 1;
            }
        }
        removed
    }

    fn reap_at(&self, now: Instant) -> usize {
        self.reap_matching(|entry| entry.expires_at <= now && entry.state == EntryState::Uploaded)
    }

    fn reap_matching(&self, predicate: impl Fn(&Entry) -> bool) -> usize {
        let removed = {
            let mut state = self.state.lock().expect("media store lock");
            let ids: Vec<String> = state
                .entries
                .iter()
                .filter(|(_, entry)| predicate(entry))
                .map(|(id, _)| id.clone())
                .collect();
            let mut paths = Vec::with_capacity(ids.len());
            for id in ids {
                if let Some(entry) = state.entries.remove(&id) {
                    state.temp_reserved_bytes = state
                        .temp_reserved_bytes
                        .saturating_sub(entry.media.encoded_bytes);
                    paths.push(entry.media.path);
                }
            }
            paths
        };
        let count = removed.len();
        for path in removed {
            let _ = std::fs::remove_file(path);
        }
        count
    }

    fn delete_claimed(&self, media: &[StoredMedia]) {
        let paths = {
            let mut state = self.state.lock().expect("media store lock");
            let mut paths = Vec::new();
            for item in media {
                if let Some(entry) = state.entries.remove(&item.id) {
                    state.temp_reserved_bytes = state
                        .temp_reserved_bytes
                        .saturating_sub(entry.media.encoded_bytes);
                    paths.push(entry.media.path);
                }
            }
            paths
        };
        for path in paths {
            let _ = std::fs::remove_file(path);
        }
    }

    #[doc(hidden)]
    pub fn force_expire_for_test(&self, id: &str, ago: std::time::Duration) {
        if let Some(entry) = self
            .state
            .lock()
            .expect("media store lock")
            .entries
            .get_mut(id)
        {
            entry.expires_at = Instant::now() - ago;
        }
    }
}

#[derive(Debug)]
pub struct ClaimedMedia {
    store: std::sync::Arc<MediaStore>,
    pub media: Vec<StoredMedia>,
}

impl Drop for ClaimedMedia {
    fn drop(&mut self) {
        self.store.delete_claimed(&self.media);
    }
}

#[derive(Debug)]
pub struct DecodedReservation {
    store: std::sync::Arc<MediaStore>,
    bytes: u64,
}

impl Drop for DecodedReservation {
    fn drop(&mut self) {
        let mut state = self.store.state.lock().expect("media store lock");
        state.decoded_reserved_bytes = state.decoded_reserved_bytes.saturating_sub(self.bytes);
    }
}

struct UploadReservation<'a> {
    store: &'a MediaStore,
    bytes: u64,
    committed: bool,
}

impl UploadReservation<'_> {
    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for UploadReservation<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let mut state = self.store.state.lock().expect("media store lock");
            state.in_flight_uploads = state.in_flight_uploads.saturating_sub(1);
            state.temp_reserved_bytes = state.temp_reserved_bytes.saturating_sub(self.bytes);
        }
    }
}

struct PartialFile {
    path: PathBuf,
    file: std::fs::File,
    keep: bool,
}

impl PartialFile {
    fn create(path: &Path) -> Result<Self, MediaError> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|_| MediaError::new(MediaErrorKind::Internal, "cannot create media upload"))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            keep: false,
        })
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), MediaError> {
        self.file
            .write_all(bytes)
            .map_err(|_| MediaError::new(MediaErrorKind::Internal, "cannot write media upload"))
    }

    fn sync_all(&self) -> Result<(), MediaError> {
        self.file
            .sync_all()
            .map_err(|_| MediaError::new(MediaErrorKind::Internal, "cannot sync media upload"))
    }
}

impl Drop for PartialFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
