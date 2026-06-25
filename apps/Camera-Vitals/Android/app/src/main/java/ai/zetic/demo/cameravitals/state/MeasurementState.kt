package ai.zetic.demo.cameravitals.state

/** Top-level finite-state machine that drives which screen is shown. */
sealed interface MeasurementState {
    data class LoadingModel(val progress: Float) : MeasurementState
    data object PermissionDenied : MeasurementState
    data class Warmup(val framesFilled: Int) : MeasurementState
    data object Live : MeasurementState
    data class ErrorState(val message: String) : MeasurementState
}

/** Result of a guided 30-second measurement. */
data class MeasurementReport(
    val avgBPM: Int,
    val minBPM: Int,
    val maxBPM: Int,
    val avgQuality: Double,
    val series: List<Double>
)
