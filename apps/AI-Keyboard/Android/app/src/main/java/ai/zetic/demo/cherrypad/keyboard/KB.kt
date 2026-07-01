package ai.zetic.demo.cherrypad.keyboard

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Keyboard design tokens — ported 1:1 from the iOS `KB` enum. The keyboard has a single
 * FIXED height; the processing/result panel takes over the QWERTY area rather than resizing.
 */
object KB {
    // Colors
    val cherry = Color(0xFFD81E34)
    val cherryDark = Color(0xFFA8132A)
    val cherrySoft = Color(0xFFFBE3E6)
    val background = Color(0xFFE5E5EA)   // iOS systemGray5 (light)
    val keyFill = Color(0xFFFFFFFF)      // letter keys / cards
    val specialFill = Color(0xFFC7C7CC)  // iOS systemGray3 — shift/delete/plane/globe/return
    val textPrimary = Color(0xFF000000)
    val textSecondary = Color(0xFF8E8E93)
    val keyShadow = Color(0x29000000)    // black @ 16%

    // Radii
    val radiusKey: Dp = 5.dp
    val radiusControl: Dp = 12.dp

    // Spacing
    val sideMargin: Dp = 4.dp
    val sectionGap: Dp = 8.dp
    val barGap: Dp = 6.dp
    val keyGapH: Dp = 5.dp
    val keyGapV: Dp = 9.dp

    // Heights / widths
    val keyHeight: Dp = 42.dp
    val actionButtonH: Dp = 52.dp
    val controlHeight: Dp = 48.dp
    val secondaryWidth: Dp = 52.dp
    val resultMaxHeight: Dp = 168.dp

    // Fixed key widths (bottom row + specials)
    val shiftDeleteWidth: Dp = 44.dp
    val globeWidth: Dp = 44.dp
    val inlinePlaneWidth: Dp = 56.dp     // "#+=" / "123" inside row 3
    val planeToggleWidth: Dp = 84.dp     // "123"/"ABC" bottom-left
    val returnWidth: Dp = 92.dp

    /** Fixed keyboard height: sectionGap + (actionBar + sectionGap + 4 rows) + bottom pad. */
    val keyboardHeight: Dp =
        sectionGap + (actionButtonH + sectionGap + (keyHeight * 4 + keyGapV * 3)) + 4.dp
}
