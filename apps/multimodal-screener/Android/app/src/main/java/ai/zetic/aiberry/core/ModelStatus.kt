package ai.zetic.aiberry.core

/** Lifecycle of an on-device model, surfaced to the UI. */
sealed interface ModelStatus {
    data object Idle : ModelStatus
    /** 0..1, first run only (model is cached afterwards). */
    data class Downloading(val progress: Float) : ModelStatus
    data object Loading : ModelStatus
    data object Running : ModelStatus
    data object Ready : ModelStatus
    data class Failed(val message: String) : ModelStatus

    val isBusy: Boolean
        get() = this is Downloading || this is Loading || this is Running

    val isFailure: Boolean
        get() = this is Failed
}
