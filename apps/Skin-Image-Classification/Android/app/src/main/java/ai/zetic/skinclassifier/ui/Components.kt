package ai.zetic.skinclassifier.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.skinclassifier.model.ClassScore
import ai.zetic.skinclassifier.model.SkinClass
import kotlin.math.roundToInt
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.text.TextStyle
import androidx.compose.foundation.text.BasicText

/** Tap handler without the default Material ripple, for the custom button surfaces. */
fun Modifier.clickableNoRipple(enabled: Boolean, onClick: () -> Unit): Modifier =
    this.clickable(enabled = enabled, onClick = onClick)

/** Filled brand-gradient primary action button. */
@Composable
fun PrimaryButton(
    title: String,
    icon: ImageVector,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(if (enabled) Theme.brandGradient else Brush.horizontalGradient(listOf(Theme.InkFaint, Theme.InkFaint)))
            .clickableNoRipple(enabled, onClick)
            .padding(vertical = 16.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = Color(0.03f, 0.05f, 0.08f), modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(8.dp))
        Text(title, color = Color(0.03f, 0.05f, 0.08f), fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
    }
}

/** Outlined translucent secondary action button. */
@Composable
fun SecondaryButton(
    title: String,
    icon: ImageVector,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(Color.White.copy(alpha = 0.06f))
            .border(1.dp, Theme.Hairline, RoundedCornerShape(18.dp))
            .clickableNoRipple(true, onClick)
            .padding(vertical = 15.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = Theme.Ink, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(8.dp))
        Text(title, color = Theme.Ink, fontSize = 15.sp, fontWeight = FontWeight.Medium)
    }
}

/** Non-dismissible safety banner shown on the capture and results screens. */
@Composable
fun DisclaimerBanner(modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(Color.White.copy(alpha = 0.05f))
            .border(1.dp, Theme.Amber.copy(alpha = 0.25f), RoundedCornerShape(14.dp))
            .padding(horizontal = 14.dp, vertical = 11.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Filled.Info, contentDescription = null, tint = Theme.Amber, modifier = Modifier.size(16.dp))
        Spacer(Modifier.width(10.dp))
        Text(
            "Demo only — not a medical device. This is not a diagnosis. Consult a healthcare professional.",
            color = Theme.InkSoft,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
        )
    }
}

/** Severity badge: tinted dot + label. */
@Composable
fun SeverityBadge(skinClass: SkinClass) {
    val tint = skinClass.tint
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(50))
            .background(tint.copy(alpha = 0.12f))
            .border(1.dp, tint.copy(alpha = 0.30f), RoundedCornerShape(50))
            .padding(horizontal = 10.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(tint))
        Spacer(Modifier.width(7.dp))
        Text(
            skinClass.severityLabel.uppercase(),
            color = tint,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold,
            letterSpacing = 0.8.sp,
        )
    }
}

/** Animated circular confidence gauge with a centered percentage. */
@Composable
fun ConfidenceRing(value: Float, tint: Color, size: androidx.compose.ui.unit.Dp = 116.dp) {
    val animated by animateFloatAsState(targetValue = value, animationSpec = tween(900), label = "ring")
    Box(modifier = Modifier.size(size), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.size(size)) {
            val stroke = 10.dp.toPx()
            val inset = stroke / 2
            val arcSize = Size(this.size.width - stroke, this.size.height - stroke)
            drawArc(
                color = Color.White.copy(alpha = 0.08f),
                startAngle = 0f,
                sweepAngle = 360f,
                useCenter = false,
                topLeft = androidx.compose.ui.geometry.Offset(inset, inset),
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )
            drawArc(
                brush = Brush.sweepGradient(listOf(tint.copy(alpha = 0.7f), tint)),
                startAngle = -90f,
                sweepAngle = 360f * animated.coerceIn(0f, 1f),
                useCenter = false,
                topLeft = androidx.compose.ui.geometry.Offset(inset, inset),
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Row(verticalAlignment = Alignment.Bottom) {
                Text(
                    "${(animated * 100).roundToInt()}",
                    color = Theme.Ink,
                    fontSize = 34.sp,
                    fontWeight = FontWeight.SemiBold,
                    fontFamily = FontFamily.Monospace,
                )
                Text("%", color = Theme.InkSoft, fontSize = 18.sp, fontFamily = FontFamily.Monospace)
            }
            Text(
                "CONFIDENCE",
                color = Theme.InkFaint,
                fontSize = 10.sp,
                letterSpacing = 1.2.sp,
            )
        }
    }
}

/** "Full distribution" — a labeled probability bar per class. */
@Composable
fun ClassDistributionBars(ranked: List<ClassScore>, maxRows: Int = 5) {
    Column(modifier = Modifier.glassCard(corner = 22.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(
            "FULL DISTRIBUTION",
            color = Theme.InkFaint,
            fontSize = 12.sp,
            fontWeight = FontWeight.SemiBold,
            letterSpacing = 1.4.sp,
        )
        ranked.take(maxRows).forEachIndexed { index, score ->
            val isTop = index == 0
            val tint = score.skinClass.tint
            Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        score.skinClass.title,
                        color = if (isTop) Theme.Ink else Theme.InkSoft,
                        fontSize = 13.sp,
                        fontWeight = if (isTop) FontWeight.SemiBold else FontWeight.Normal,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        "${(score.probability * 100).roundToInt()}%",
                        color = if (isTop) tint else Theme.InkFaint,
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace,
                    )
                }
                val animated by animateFloatAsState(
                    targetValue = score.probability.coerceIn(0f, 1f),
                    animationSpec = tween(700),
                    label = "bar",
                )
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(7.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color.White.copy(alpha = 0.06f)),
                ) {
                    Box(
                        Modifier
                            .fillMaxWidth(animated)
                            .height(7.dp)
                            .clip(RoundedCornerShape(50))
                            .background(Brush.horizontalGradient(listOf(tint.copy(alpha = 0.6f), tint))),
                    )
                }
            }
        }
    }
}

/** Small mono caption (e.g. the on-device latency line). */
@Composable
fun MonoCaption(text: String, color: Color = Theme.InkFaint) {
    BasicText(
        text = text,
        style = TextStyle(color = color, fontSize = 11.sp, fontFamily = FontFamily.Monospace),
    )
}
