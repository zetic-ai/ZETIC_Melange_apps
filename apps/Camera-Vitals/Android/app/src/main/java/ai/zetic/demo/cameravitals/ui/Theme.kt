package ai.zetic.demo.cameravitals.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/** Light medical-clean design tokens (mirrors iOS Theme). */
object Theme {
    val background = Color(0xFFF5F7FC)
    val card = Color.White
    val accent = Color(0xFF2E73F2)
    val accentSoft = Color(0x1F2E73F2)
    val textPrimary = Color(0xFF1A2133)
    val textSecondary = Color(0xFF6B788C)
    val good = Color(0xFF33B87A)
    val fair = Color(0xFFF29E2E)
    val poor = Color(0xFFE64D4D)

    val cardRadius = 22.dp

    fun quality(q: Double): Color = if (q >= 0.6) good else if (q >= 0.3) fair else poor
    fun qualityLabel(q: Double): String =
        if (q >= 0.6) "Good signal" else if (q >= 0.3) "Fair signal" else "Stabilizing…"
}

@Composable
fun CameraVitalsTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Theme.accent,
            background = Theme.background,
            surface = Theme.card,
            onPrimary = Color.White,
            onBackground = Theme.textPrimary,
            onSurface = Theme.textPrimary
        ),
        content = content
    )
}

/** Reusable soft white card. */
@Composable
fun Card(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Box(
        modifier
            .shadow(8.dp, RoundedCornerShape(Theme.cardRadius), clip = false)
            .clip(RoundedCornerShape(Theme.cardRadius))
            .background(Theme.card)
    ) { content() }
}
