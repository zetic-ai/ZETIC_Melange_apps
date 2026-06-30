package com.brew.ui.theme

import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp

/**
 * Forced-light Brew theme. Ignores [isSystemInDarkTheme] entirely (the iOS app
 * is locked to `.preferredColorScheme(.light)`).
 */
@Composable
fun BrewTheme(content: @Composable () -> Unit) {
    val scheme = lightColorScheme(
        primary = BrewColors.accent,
        background = BrewColors.canvas,
        surface = BrewColors.canvas,
        onPrimary = Color.White,
        onBackground = BrewColors.ink,
        onSurface = BrewColors.ink,
    )
    MaterialTheme(colorScheme = scheme, typography = BrewTypography, content = content)
}

/** Serif display face (New York on iOS → the platform serif here). */
val Serif: FontFamily = FontFamily.Serif

/** Reusable card background — mirrors iOS `cardBackground(fill:cornerRadius:)`. */
fun Modifier.cardBackground(fill: Color = BrewColors.card, cornerRadius: Int = 20): Modifier =
    this.clip(RoundedCornerShape(cornerRadius.dp)).background(fill)
