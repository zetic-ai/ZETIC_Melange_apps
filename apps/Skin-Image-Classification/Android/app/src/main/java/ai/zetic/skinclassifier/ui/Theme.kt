package ai.zetic.skinclassifier.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Dark, on-device-AI palette ported from the iOS app's `Theme.swift`: near-black background,
 * cyan→periwinkle brand gradient, and severity tints (mint / amber / coral).
 */
object Theme {
    val Bg = Color(0.04f, 0.05f, 0.08f)

    val Accent = Color(0.36f, 0.92f, 0.93f)   // cyan glow
    val Accent2 = Color(0.55f, 0.62f, 1.00f)  // periwinkle

    val Mint = Color(0.40f, 0.92f, 0.70f)
    val Amber = Color(1.00f, 0.78f, 0.36f)
    val Coral = Color(1.00f, 0.45f, 0.48f)

    val Ink = Color.White
    val InkSoft = Color.White.copy(alpha = 0.62f)
    val InkFaint = Color.White.copy(alpha = 0.40f)

    val Card = Color.White.copy(alpha = 0.05f)
    val Hairline = Color.White.copy(alpha = 0.12f)

    val Corner: Dp = 22.dp

    val brandGradient = Brush.horizontalGradient(listOf(Accent, Accent2))
}

/** Standard frosted "glass" card container. */
fun Modifier.glassCard(padding: Dp = 18.dp, corner: Dp = Theme.Corner): Modifier = this
    .fillMaxWidth()
    .clip(RoundedCornerShape(corner))
    .background(Theme.Card)
    .border(1.dp, Theme.Hairline, RoundedCornerShape(corner))
    .padding(padding)
