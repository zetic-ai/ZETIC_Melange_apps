package ai.zetic.skinclassifier.core

/**
 * Lifecycle of the on-device skin model, surfaced to the UI. Mirrors iOS `LoadPhase`.
 */
sealed interface ModelStatus {
    data object Idle : ModelStatus
    /** Real file download, 0..1, first run only (model is cached afterwards). */
    data class Downloading(val progress: Float) : ModelStatus
    /** Compile / init on the device NPU — no determinate progress. */
    data object Preparing : ModelStatus
    data object Ready : ModelStatus
    data class Failed(val message: String) : ModelStatus

    val isReady: Boolean get() = this is Ready
    val isBusy: Boolean get() = this is Downloading || this is Preparing
    val errorMessage: String? get() = (this as? Failed)?.message
}
