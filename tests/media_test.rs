use qwen36_server::media::fetch::is_public_ip;
use qwen36_server::media::{MediaConfig, MediaErrorKind, MediaKind, MediaStore, OwnerDigest};
use std::sync::{Arc, Barrier};
use std::time::Duration;

struct Fixture {
    root: std::path::PathBuf,
    store: Arc<MediaStore>,
}

impl Fixture {
    fn new(mut config: MediaConfig) -> Self {
        let root = std::env::temp_dir().join(format!("qwen36-media-test-{}", uuid::Uuid::new_v4()));
        config.temp_root = root.clone();
        let store = Arc::new(MediaStore::new(config).unwrap());
        Self { root, store }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.store.purge_all();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn png() -> Vec<u8> {
    b"\x89PNG\r\n\x1a\nfixture".to_vec()
}

fn owner(value: &str) -> OwnerDigest {
    OwnerDigest::from_key(value)
}

#[test]
fn upload_claim_and_drop_delete_bytes() {
    let fixture = Fixture::new(MediaConfig::default());
    let uploaded = fixture
        .store
        .upload(owner("a"), "image/png", &png())
        .unwrap();
    assert_eq!(uploaded.kind, MediaKind::Image);
    assert!(uploaded.path.is_file());
    let claim = fixture
        .store
        .clone()
        .claim_batch(owner("a"), std::slice::from_ref(&uploaded.id))
        .unwrap();
    assert_eq!(claim.media.len(), 1);
    assert!(fixture
        .store
        .clone()
        .claim_batch(owner("a"), std::slice::from_ref(&uploaded.id))
        .is_err());
    drop(claim);
    assert!(!uploaded.path.exists());
    assert!(fixture
        .store
        .clone()
        .claim_batch(owner("a"), &[uploaded.id])
        .is_err());
}

#[test]
fn owner_binding_and_duplicate_batch_fail_without_partial_claim() {
    let fixture = Fixture::new(MediaConfig::default());
    let first = fixture
        .store
        .upload(owner("a"), "image/png", &png())
        .unwrap();
    let second = fixture
        .store
        .upload(owner("a"), "image/png", &png())
        .unwrap();
    let foreign = fixture
        .store
        .clone()
        .claim_batch(owner("b"), std::slice::from_ref(&first.id))
        .unwrap_err();
    assert_eq!(foreign.kind, MediaErrorKind::Invalid);
    let duplicate = fixture
        .store
        .clone()
        .claim_batch(owner("a"), &[first.id.clone(), first.id.clone()])
        .unwrap_err();
    assert_eq!(duplicate.kind, MediaErrorKind::Invalid);
    let missing = fixture
        .store
        .clone()
        .claim_batch(owner("a"), &[first.id.clone(), "missing".into()])
        .unwrap_err();
    assert_eq!(missing.kind, MediaErrorKind::Invalid);
    let claim = fixture
        .store
        .clone()
        .claim_batch(owner("a"), &[first.id, second.id])
        .unwrap();
    assert_eq!(claim.media.len(), 2);
}

#[test]
fn concurrent_claim_has_single_winner() {
    let fixture = Fixture::new(MediaConfig::default());
    let uploaded = fixture
        .store
        .upload(owner("a"), "image/png", &png())
        .unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let mut threads = Vec::new();
    for _ in 0..2 {
        let store = fixture.store.clone();
        let id = uploaded.id.clone();
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            store.claim_batch(owner("a"), &[id])
        }));
    }
    barrier.wait();
    let results: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter_map(|result| result.as_ref().err())
            .filter(|error| error.kind == MediaErrorKind::Conflict)
            .count(),
        1
    );
}

#[test]
fn ttl_and_quotas_cleanup() {
    let mut config = MediaConfig::default();
    config.max_pending_uploads = 1;
    config.max_temp_bytes = png().len() as u64;
    config.max_decoded_bytes = 4;
    let fixture = Fixture::new(config);
    let uploaded = fixture
        .store
        .upload(owner("a"), "image/png", &png())
        .unwrap();
    let pending = fixture
        .store
        .upload(owner("a"), "image/png", &png())
        .unwrap_err();
    assert_eq!(pending.kind, MediaErrorKind::UploadLimit);
    let reserve = fixture.store.clone().reserve_decoded(4).unwrap();
    assert_eq!(
        fixture.store.clone().reserve_decoded(1).unwrap_err().kind,
        MediaErrorKind::DecodedTooLarge
    );
    drop(reserve);
    assert!(fixture.store.clone().reserve_decoded(4).is_ok());
    fixture
        .store
        .force_expire_for_test(&uploaded.id, Duration::from_secs(1));
    assert_eq!(fixture.store.reap_expired(), 1);
    assert!(!uploaded.path.exists());
}

#[test]
fn mime_spoof_and_size_fail_before_file() {
    let mut config = MediaConfig::default();
    config.max_image_bytes = 8;
    let fixture = Fixture::new(config);
    assert_eq!(
        fixture
            .store
            .upload(owner("a"), "video/mp4", &png())
            .unwrap_err()
            .kind,
        MediaErrorKind::Invalid
    );
    assert_eq!(
        fixture
            .store
            .upload(owner("a"), "image/png", &png())
            .unwrap_err()
            .kind,
        MediaErrorKind::EncodedTooLarge
    );
    assert_eq!(std::fs::read_dir(&fixture.root).unwrap().count(), 0);
}

#[test]
fn ssrf_address_classes_are_fail_closed() {
    for address in [
        "127.0.0.1",
        "10.0.0.1",
        "100.64.0.1",
        "169.254.169.254",
        "192.0.2.1",
        "198.18.0.1",
        "203.0.113.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "2001:db8::1",
        "::ffff:127.0.0.1",
    ] {
        assert!(!is_public_ip(address.parse().unwrap()), "{address}");
    }
}
