use axum::{extract::State, Json};

use crate::{
    api::AppState,
    metrics::{RuntimeStatus, StatusModel, StatusSnapshot},
};

pub async fn status(State(state): State<AppState>) -> Json<StatusSnapshot> {
    state.host_status.spawn();
    let info = state.engine.model_info();
    let mut totals = state.metrics.totals();
    let recent_requests = state.metrics.recent();
    let host = state.host_status.snapshot();
    let engine_load = state.engine.in_flight();
    let engine_in_flight = engine_load.unwrap_or(totals.active_requests);
    let queued = engine_in_flight.saturating_sub(info.slots);
    if engine_load.is_some() {
        totals.active_requests = engine_in_flight.saturating_sub(queued);
    }
    let context_used = recent_requests
        .first()
        .map(|request| {
            request
                .prompt_tokens
                .saturating_add(request.completion_tokens) as usize
        })
        .unwrap_or(0);
    let engine_state = if state.engine.load_error().is_some() {
        "error"
    } else if state.engine.ready() {
        "ready"
    } else {
        "loading"
    };

    Json(StatusSnapshot {
        model: StatusModel {
            id: info.id,
            quant: info.quant,
            context_length: info.context_length,
            slots: info.slots,
        },
        state: engine_state.into(),
        uptime_s: state.metrics.uptime().as_secs(),
        runtime: RuntimeStatus::current(),
        totals,
        context_used,
        in_flight: engine_in_flight,
        queued,
        recent_requests,
        gpu: host.gpu,
        system: host.system,
    })
}
