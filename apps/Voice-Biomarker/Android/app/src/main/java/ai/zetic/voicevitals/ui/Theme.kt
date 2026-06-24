package ai.zetic.voicevitals.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Warm, editorial "private notes" palette: cream background, serif headlines,
 * soft rounded cards, sage-green icon tiles, charcoal pill buttons.
 */
object Theme {
    val Bg = Color(0.953f, 0.945f, 0.918f)   // warm cream
    val Card = Color(0.912f, 0.903f, 0.872f) // soft warm-gray card
    val CardAlt = Color(0.972f, 0.966f, 0.945f) // lighter inner fill

    val Ink = Color(0.118f, 0.118f, 0.106f)  // near-black text
    val InkSoft = Color(0.435f, 0.427f, 0.400f) // muted secondary text

    val Tile = Color(0.792f, 0.851f, 0.733f) // sage icon tile
    val TileInk = Color(0.310f, 0.447f, 0.282f) // deep green glyph
    val Dark = Color(0.168f, 0.165f, 0.149f) // charcoal pill
    val Lavender = Color(0.792f, 0.749f, 0.925f) // avatar accent

    // Semantic aliases.
    val Accent = TileInk
    val Positive = TileInk
    val Warn = Color(0.85f, 0.62f, 0.26f)
    val Danger = Color(0.83f, 0.36f, 0.33f)

    val Corner: Dp = 22.dp
    val Serif = FontFamily.Serif
}

/** Standard soft "card" container. */
fun Modifier.card(padding: Dp = 18.dp): Modifier = this
    .fillMaxWidth()
    .clip(RoundedCornerShape(Theme.Corner))
    .background(Theme.Card)
    .padding(padding)
