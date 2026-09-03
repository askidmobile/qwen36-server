use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use qwen36_server::component_manager::{ComponentKind, ComponentManager, ComponentState};

#[test]
fn component_lifecycle_and_lease() {
    let cm = ComponentManager::new(
        Some(PathBuf::from("vision.gguf")),
        Some(PathBuf::from("mtp.gguf")),
        Duration::from_millis(50),
    );

    assert_eq!(cm.state(ComponentKind::Vision), ComponentState::Unloaded);
    assert_eq!(cm.state(ComponentKind::Mtp), ComponentState::Unloaded);

    let lease_v = cm.acquire_lease(ComponentKind::Vision).unwrap();
    assert!(matches!(
        cm.state(ComponentKind::Vision),
        ComponentState::Warm { .. }
    ));

    // Eviction while active must do nothing
    assert!(cm.evict_for_vram_pressure().is_none());

    drop(lease_v);
    // After dropping lease, state remains warm
    assert!(matches!(
        cm.state(ComponentKind::Vision),
        ComponentState::Warm { .. }
    ));

    // MTP lease
    let lease_m = cm.acquire_lease(ComponentKind::Mtp).unwrap();
    assert!(matches!(
        cm.state(ComponentKind::Mtp),
        ComponentState::Warm { .. }
    ));
    drop(lease_m);

    // Eviction prioritizes MTP
    assert_eq!(cm.evict_for_vram_pressure(), Some(ComponentKind::Mtp));
    assert_eq!(cm.state(ComponentKind::Mtp), ComponentState::Unloaded);

    // TTL expiry
    std::thread::sleep(Duration::from_millis(60));
    let unloads = cm.check_ttl();
    assert_eq!(unloads, vec![ComponentKind::Vision]);
    assert_eq!(cm.state(ComponentKind::Vision), ComponentState::Unloaded);
}

#[test]
fn component_unconfigured_error() {
    let cm = ComponentManager::new(None, None, Duration::from_secs(60));
    assert!(cm.acquire_lease(ComponentKind::Vision).is_err());
    assert!(cm.acquire_lease(ComponentKind::Mtp).is_err());
}
