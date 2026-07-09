package ai.zetic.demo.posemotion.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color

/** Dark sports-analytics tokens — same palette as the iOS Theme.swift. */
object Theme {
    val Background = Color(0xFF0B0E13)
    val Card = Color(0xFF161B22)
    val Accent = Color(0xFF46E08C)          // court green
    val AccentSoft = Accent.copy(alpha = 0.14f)
    val Ball = Color(0xFFFF9E26)            // ball orange
    val LeftSide = Color(0xFF4DCCFF)        // left limbs
    val RightSide = Color(0xFFFF738C)       // right limbs
    val Torso = Color.White
    val TextPrimary = Color(0xFFEDF2F7)
    val TextSecondary = Color(0xFF8C99AB)
    val Good = Accent
    val Poor = Color(0xFFF25959)
}

@Composable
fun PoseMotionTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Theme.Accent,
            background = Theme.Background,
            surface = Theme.Card,
            onPrimary = Color.Black,
            onBackground = Theme.TextPrimary,
            onSurface = Theme.TextPrimary,
        )
    ) {
        Surface(modifier = Modifier.fillMaxSize(), color = Theme.Background, content = content)
    }
}
