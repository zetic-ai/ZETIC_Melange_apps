package com.yeonseok.melangecounsel.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.unit.dp

enum class ThemeMode {
    SYSTEM,
    LIGHT,
    DARK
}

private val LightColors = lightColorScheme(
    primary = Sage40,
    onPrimary = Cream95,
    primaryContainer = Sage90,
    onPrimaryContainer = Sage20,
    secondary = Peach40,
    onSecondary = Cream95,
    secondaryContainer = Peach80.copy(alpha = 0.4f),
    onSecondaryContainer = WarmGray20,
    tertiary = Terracotta,
    surface = Linen,
    onSurface = WarmGray20,
    surfaceVariant = Cream90,
    onSurfaceVariant = WarmGray40,
    surfaceContainerLowest = Cream95,
    surfaceContainerLow = Cream90,
    surfaceContainer = Cream90.copy(alpha = 0.7f),
    outline = WarmGray80,
    outlineVariant = WarmGray80.copy(alpha = 0.5f)
)

private val DarkColors = darkColorScheme(
    primary = Sage60,
    onPrimary = Sage10,
    primaryContainer = Sage20,
    onPrimaryContainer = Sage80,
    secondary = Peach80,
    onSecondary = Sage10,
    secondaryContainer = Peach40.copy(alpha = 0.3f),
    onSecondaryContainer = Cream90,
    tertiary = Peach80,
    surface = Sage10,
    onSurface = Cream90,
    surfaceVariant = Sage20,
    onSurfaceVariant = WarmGray80,
    surfaceContainerLowest = Sage10,
    surfaceContainerLow = Sage20.copy(alpha = 0.6f),
    surfaceContainer = Sage20,
    outline = WarmGray40,
    outlineVariant = WarmGray40.copy(alpha = 0.5f)
)

private val AppShapes = Shapes(
    small = RoundedCornerShape(10.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(22.dp),
    extraLarge = RoundedCornerShape(28.dp)
)

@Composable
fun CounselTheme(themeMode: ThemeMode, content: @Composable () -> Unit) {
    val useDark = when (themeMode) {
        ThemeMode.SYSTEM -> isSystemInDarkTheme()
        ThemeMode.LIGHT -> false
        ThemeMode.DARK -> true
    }

    MaterialTheme(
        colorScheme = if (useDark) DarkColors else LightColors,
        typography = AppTypography,
        shapes = AppShapes,
        content = content
    )
}
