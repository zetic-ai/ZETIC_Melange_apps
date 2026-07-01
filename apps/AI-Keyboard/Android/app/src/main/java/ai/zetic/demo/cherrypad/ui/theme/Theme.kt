package ai.zetic.demo.cherrypad.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/** CherryPad palette — ported 1:1 from the iOS `Theme.swift`. */
object CherryColors {
    val Cherry = Color(0xFFD81E34)        // primary accent
    val CherryDark = Color(0xFFA8132A)    // pressed / emphasis
    val CherrySoft = Color(0xFFFBE3E6)    // tinted chip background
    val Background = Color(0xFFFFF7F4)    // warm off-white canvas
    val Surface = Color(0xFFFFFFFF)       // cards
    val SurfaceMuted = Color(0xFFF1ECEA)  // input wells / inactive chips
    val TextPrimary = Color(0xFF1C1A19)
    val TextSecondary = Color(0xFF837C78)
    val OnCherry = Color(0xFFFFFFFF)
    val CardShadow = Color(0x0F000000)    // black @ 6%
}

/** Shared corner-radius scale. */
object CherryDims {
    val cardRadius = 20.dp
    val chipRadius = 14.dp
}

private val LightScheme = lightColorScheme(
    primary = CherryColors.Cherry,
    onPrimary = CherryColors.OnCherry,
    background = CherryColors.Background,
    onBackground = CherryColors.TextPrimary,
    surface = CherryColors.Surface,
    onSurface = CherryColors.TextPrimary,
    surfaceVariant = CherryColors.SurfaceMuted,
    onSurfaceVariant = CherryColors.TextSecondary,
    error = CherryColors.CherryDark,
)

@Composable
fun CherryTheme(content: @Composable () -> Unit) {
    // The design is a fixed light aesthetic (matches iOS); ignore system dark mode.
    @Suppress("UNUSED_EXPRESSION") isSystemInDarkTheme()
    MaterialTheme(colorScheme = LightScheme, content = content)
}
