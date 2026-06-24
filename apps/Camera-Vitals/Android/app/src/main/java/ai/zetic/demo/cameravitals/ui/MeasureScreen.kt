package ai.zetic.demo.cameravitals.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.demo.cameravitals.AppConfig
import ai.zetic.demo.cameravitals.state.MeasurementState
import ai.zetic.demo.cameravitals.state.VitalsViewModel

/** The live measurement screen: camera + face lock, heart rate, waveform, and a guided scan. */
@Composable
fun MeasureScreen(vm: VitalsViewModel) {
    val isWarmup = vm.state is MeasurementState.Warmup
    val canMeasure = vm.state is MeasurementState.Live && vm.faceFound
    val displayBpm = if (vm.state is MeasurementState.Warmup) null else vm.bpm

    Column(
        Modifier.fillMaxSize().padding(horizontal = 18.dp).padding(top = 8.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Camera Vitals", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Theme.textPrimary)
                Text("Powered by ZETIC Melange", fontSize = 12.sp, fontWeight = FontWeight.Medium, color = Theme.textSecondary)
            }
            OnDeviceHud(latencyMs = vm.latencyMs)
        }

        // Camera card — flexible height so the whole screen always fits (no fixed iOS size).
        Box(
            Modifier
                .fillMaxWidth()
                .weight(1f)
                .heightIn(min = 200.dp)
                .clip(RoundedCornerShape(Theme.cardRadius))
                .background(Color.Black)
        ) {
            CameraPreview(vm, Modifier.fillMaxSize())
            FaceLockOverlay(
                faceBox = vm.faceBox,
                bufferSize = vm.bufferSize,
                quality = vm.quality,
                faceFound = vm.faceFound,
                modifier = Modifier.fillMaxSize()
            )
            Column(Modifier.fillMaxSize().padding(12.dp)) {
                if (isWarmup) {
                    WarmupBar(fraction = vm.warmupProgress)
                }
                Spacer(Modifier.weight(1f))
                Row {
                    SignalQualityBadge(vm.quality, vm.faceFound, vm.lowLight)
                    Spacer(Modifier.weight(1f))
                }
            }
        }

        // BPM card
        Card(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 22.dp, vertical = 18.dp)) {
                BpmReadout(bpm = displayBpm, quality = vm.quality)
                Spacer(Modifier.weight(1f))
            }
        }

        // Waveform card
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.fillMaxWidth().padding(horizontal = 22.dp, vertical = 16.dp)) {
                Text("Pulse waveform", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Theme.textSecondary)
                Spacer(Modifier.height(8.dp))
                WaveformChart(samples = vm.waveform, modifier = Modifier.fillMaxWidth().height(64.dp))
            }
        }

        // Measure control
        if (vm.isMeasuring) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                ProgressBar(fraction = vm.measureProgress, track = Theme.accentSoft, fill = Theme.accent, height = 8.dp)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Measuring… hold still", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = Theme.textSecondary)
                    Spacer(Modifier.weight(1f))
                    Text(
                        "${((1 - vm.measureProgress) * AppConfig.MEASURE_DURATION_SEC).toInt()}s",
                        fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Theme.accent
                    )
                    Spacer(Modifier.size(12.dp))
                    Text(
                        "Cancel", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = Theme.poor,
                        modifier = Modifier.clickable { vm.cancelMeasurement() }
                    )
                }
            }
        } else {
            val bg = if (canMeasure) Theme.accent else Theme.textSecondary.copy(alpha = 0.4f)
            Box(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(bg)
                    .clickable(enabled = canMeasure) { vm.startMeasurement() }
                    .padding(vertical = 16.dp),
                contentAlignment = Alignment.Center
            ) {
                Text("Measure 30s", fontSize = 17.sp, fontWeight = FontWeight.Bold, color = Color.White)
            }
        }
    }
}

@Composable
private fun WarmupBar(fraction: Double) {
    Column {
        ProgressBar(fraction = fraction, track = Color.White.copy(alpha = 0.35f), fill = Theme.accent, height = 5.dp)
        Spacer(Modifier.height(4.dp))
        Text("Stabilizing… keep still", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
    }
}

@Composable
private fun ProgressBar(fraction: Double, track: Color, fill: Color, height: androidx.compose.ui.unit.Dp) {
    Box(
        Modifier
            .fillMaxWidth()
            .height(height)
            .clip(CircleShape)
            .background(track)
    ) {
        Box(
            Modifier
                .fillMaxWidth(fraction.coerceIn(0.0, 1.0).toFloat())
                .fillMaxHeight()
                .clip(CircleShape)
                .background(fill)
        )
    }
}
