package ai.zetic.aiberry.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * Semicircular well-being gauge echoing Aiberry's results screen, drawn in the
 * app's palette. [value] is 0…1 (higher = brighter affect); the arc runs red
 * (low) on the left to green (bright) on the right with a marker at the value.
 * 1:1 with iOS-Aiberry's `ScoreGauge`.
 */
@Composable
fun ScoreGauge(value: Float, score: Int, band: String) {
    val lineWidth = 18.dp
    val height = 160.dp

    val animated by animateFloatAsState(
        targetValue = value.coerceIn(0f, 1f),
        animationSpec = spring(dampingRatio = 0.8f, stiffness = 150f),
        label = "gaugeValue",
    )

    // red (low) -> orange -> yellow -> green (bright)
    val colors = listOf(
        Color(0.83f, 0.36f, 0.33f),
        Color(0.90f, 0.62f, 0.26f),
        Color(0.86f, 0.78f, 0.30f),
        Color(0.40f, 0.62f, 0.36f),
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(height),
        contentAlignment = Alignment.TopCenter,
    ) {
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            val stroke = lineWidth.toPx()
            val w = size.width
            val d = min(w, (height.toPx() - stroke) * 2f)   // diameter that fits
            val r = (d - stroke) / 2f
            val centerX = w / 2f
            val centerY = height.toPx() - stroke / 2f
            val center = Offset(centerX, centerY)

            // Top semicircle: sweepGradient starts at 0° (3 o'clock) clockwise. Align stops
            // so the gradient spans only the visible 180°..360° top arc (left -> right).
            val brush = Brush.sweepGradient(
                colorStops = arrayOf(
                    0.50f to colors[0],
                    0.667f to colors[1],
                    0.833f to colors[2],
                    1.0f to colors[3],
                ),
                center = center,
            )
            drawArc(
                brush = brush,
                startAngle = 180f,
                sweepAngle = 180f,
                useCenter = false,
                topLeft = Offset(centerX - r, centerY - r),
                size = Size(r * 2f, r * 2f),
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )

            // Value marker at angle π(1+value).
            val theta = PI * (1f + animated.coerceIn(0f, 1f))
            val mx = centerX + r * cos(theta).toFloat()
            val my = centerY + r * sin(theta).toFloat()
            val markerR = stroke / 2f + 3.dp.toPx()
            drawCircle(color = Color.White, radius = markerR, center = Offset(mx, my))
            drawCircle(
                color = Theme.Accent,
                radius = markerR,
                center = Offset(mx, my),
                style = Stroke(width = 4.dp.toPx()),
            )
        }

        // Centered score + band, nudged up toward the arc center (matches iOS offset).
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.offset(y = (height.value * 0.34f).dp),
        ) {
            Text(
                text = "$score",
                fontFamily = Theme.Serif,
                fontSize = 46.sp,
                fontWeight = FontWeight.SemiBold,
                color = Theme.Ink,
            )
            Text(
                text = band,
                fontFamily = Theme.Serif,
                fontSize = 22.sp,
                color = Theme.InkSoft,
            )
        }
    }
}
