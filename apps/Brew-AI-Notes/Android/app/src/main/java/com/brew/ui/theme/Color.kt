package com.brew.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * Brew's warm, paper-like palette (ported 1:1 from the iOS `Theme.swift`).
 * The app is forced light, so these are absolute colors, not a Material scheme.
 */
object BrewColors {
    // Backgrounds
    val canvas = Color(0xFFF4F3EE)       // app background (cream)
    val card = Color(0xFFECEBE4)         // resting card fill
    val cardElevated = Color(0xFFFBFAF6) // sheets / floating bars / input fields
    val iconTile = Color(0xFFDCE3CB)     // green-tinted icon square
    val iconTileInk = Color(0xFF6E7B4E)  // icon glyph / "ready" accents

    // Text
    val ink = Color(0xFF1B1B19)
    val inkSecondary = Color(0xFF6F6E69)
    val inkTertiary = Color(0xFF9C9B95)

    // Accents
    val accent = Color(0xFF32342E)       // near-black pills, FAB, send buttons, user bubbles
    val recording = Color(0xFF7BB661)    // recording level green
    val profile = Color(0xFFD9CEF2)      // lavender profile circle
    val warning = Color(0xFFD9803F)      // recording-problem / unavailable amber
}
