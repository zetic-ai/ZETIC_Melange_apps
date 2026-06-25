package ai.zetic.demo.cameravitals.ui

import android.graphics.Rect
import android.util.Size
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.max
import kotlin.math.roundToInt

/** Large heart-rate readout with a heart pulsing at the measured rate. */
@Composable
fun BpmReadout(bpm: Double?, quality: Double) {
    val text = if (bpm != null && bpm > 0) bpm.roundToInt().toString() else "––"
    val intervalMs = if (bpm != null && bpm > 30) (60000.0 / bpm).toInt() else 1000
    val transition = rememberInfiniteTransition(label = "beat")
    val scale by transition.animateFloat(
        initialValue = 0.92f,
        targetValue = 1.16f,
        animationSpec = infiniteRepeatable(tween(intervalMs / 2), RepeatMode.Reverse),
        label = "scale"
    )
    Row(verticalAlignment = Alignment.Bottom) {
        Icon(
            Icons.Filled.Favorite, contentDescription = null, tint = Theme.accent,
            modifier = Modifier
                .padding(bottom = 14.dp)
                .size(30.dp)
                .graphicsLayer {
                    scaleX = scale; scaleY = scale
                    alpha = if (bpm == null) 0.35f else 1f
                }
        )
        Spacer(Modifier.width(14.dp))
        Text(text, fontSize = 76.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, color = Theme.textPrimary)
        Spacer(Modifier.width(8.dp))
        Text("BPM", fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = Theme.textSecondary, modifier = Modifier.padding(bottom = 14.dp))
    }
}

/** Scrolling PPG waveform, auto-scaled to its min/max. */
@Composable
fun WaveformChart(samples: FloatArray, modifier: Modifier = Modifier, color: Color = Theme.accent) {
    Canvas(modifier) {
        if (samples.size < 2) return@Canvas
        var lo = samples[0]; var hi = samples[0]
        for (v in samples) { if (v < lo) lo = v; if (v > hi) hi = v }
        val range = max(hi - lo, 1e-4f)
        val stepX = size.width / (samples.size - 1)
        val path = Path()
        for (i in samples.indices) {
            val x = i * stepX
            val norm = (samples[i] - lo) / range
            val y = size.height * (1 - norm) * 0.8f + size.height * 0.1f
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, color, style = Stroke(width = 6f, cap = StrokeCap.Round, join = StrokeJoin.Round))
    }
}

/** Draws the locked face ROI over the camera preview, colored by signal quality. */
@Composable
fun FaceLockOverlay(faceBox: Rect?, bufferSize: Size, quality: Double, faceFound: Boolean, modifier: Modifier = Modifier) {
    Canvas(modifier) {
        if (!faceFound || faceBox == null || bufferSize.width == 0 || bufferSize.height == 0) return@Canvas
        val bw = bufferSize.width.toFloat()
        val bh = bufferSize.height.toFloat()
        val scale = max(size.width / bw, size.height / bh)
        val dx = (size.width - bw * scale) / 2f
        val dy = (size.height - bh * scale) / 2f
        // Mirror X to match the front-camera preview.
        val left = size.width - (faceBox.right * scale + dx)
        val top = faceBox.top * scale + dy
        val w = faceBox.width() * scale
        val h = faceBox.height() * scale
        drawRoundRect(
            color = Theme.quality(quality),
            topLeft = Offset(left, top),
            size = androidx.compose.ui.geometry.Size(w, h),
            cornerRadius = CornerRadius(36f, 36f),
            style = Stroke(width = 6f)
        )
    }
}

/** Compact pill conveying the current guidance / signal quality. */
@Composable
fun SignalQualityBadge(quality: Double, faceFound: Boolean, lowLight: Boolean) {
    val label = when {
        !faceFound -> "Center your face"
        lowLight -> "More light"
        else -> Theme.qualityLabel(quality)
    }
    val color = if (!faceFound || lowLight) Theme.poor else Theme.quality(quality)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .clip(CircleShape)
            .background(Color.White.copy(alpha = 0.88f))
            .padding(horizontal = 14.dp, vertical = 9.dp)
    ) {
        Box(Modifier.size(9.dp).clip(CircleShape).background(color))
        Spacer(Modifier.width(8.dp))
        Text(label, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = Theme.textPrimary)
    }
}

/** On-device / NPU + live latency chip. */
@Composable
fun OnDeviceHud(latencyMs: Double) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .clip(CircleShape)
            .background(Theme.accentSoft)
            .padding(horizontal = 12.dp, vertical = 7.dp)
    ) {
        Icon(Icons.Filled.Memory, contentDescription = null, tint = Theme.accent, modifier = Modifier.size(14.dp))
        Spacer(Modifier.width(6.dp))
        Text("On-device · NPU", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Theme.accent)
        if (latencyMs > 0) {
            Spacer(Modifier.width(6.dp))
            Text("· ${latencyMs.roundToInt()} ms", fontSize = 12.sp, color = Theme.textSecondary)
        }
    }
}
