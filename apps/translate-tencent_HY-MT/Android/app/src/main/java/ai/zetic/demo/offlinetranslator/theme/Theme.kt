package ai.zetic.demo.offlinetranslator.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** Dark palette tuned to DeepL's iOS app — mirrors `Theme/DeepLTheme.swift` value-for-value. */
object Theme {
    val background = Color.Black
    val surface = Color(0xFF1B1B1D)        // the big translate card
    val surfaceRaised = Color(0xFF2C2C2E)  // pills, language buttons
    val surfaceRaisedHi = Color(0xFF3A3A3C)
    val accent = Color(0xFF1A8FE3)         // bright blue: Paste, profile
    val accentDeep = Color(0xFF0B63CE)     // selected segment
    val textPrimary = Color.White
    val textSecondary = Color(0xFF8E8E93)
    val textTertiary = Color(0xFF636366)
    val separator = Color(0xFF2E2E30)
    val online = Color(0xFF32D74B)
}

/** Forces a dark Material3 scheme so the whole app reads like the iOS reference regardless of system setting. */
@Composable
fun OfflineTranslatorTheme(content: @Composable () -> Unit) {
    @Suppress("UNUSED_EXPRESSION") isSystemInDarkTheme() // intentionally ignored — always dark
    val colors = darkColorScheme(
        primary = Theme.accent,
        background = Theme.background,
        surface = Theme.surface,
        onPrimary = Color.White,
        onBackground = Theme.textPrimary,
        onSurface = Theme.textPrimary,
    )
    MaterialTheme(colorScheme = colors, content = content)
}
