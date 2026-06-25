package ai.zetic.aiberry.ui

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border

/**
 * A calm, professional "presence" indicator in the shared sage palette — the focal
 * element during the guided check-in. No cartoon face, no emoji: it mirrors the
 * RecordButton / IconTile look. A live input-level ring shows it's listening; a quiet
 * spinner shows it's working. 1:1 with iOS-Aiberry's `PresenceOrb`.
 */
@Composable
fun PresenceOrb(
    listening: Boolean = false,
    level: Float = 0f,
    thinking: Boolean = false,
    size: Dp = 168.dp,
    modifier: Modifier = Modifier,
) {
    val ringWidth by animateDpAsState(
        targetValue = (2f + level.coerceIn(0f, 1f) * 14f).dp,
        animationSpec = tween(durationMillis = 120),
        label = "orbRing",
    )
    val outer = size * 1.12f

    Box(
        modifier = modifier.size(outer),
        contentAlignment = Alignment.Center,
    ) {
        // Live input-level ring (only while listening).
        if (listening) {
            Canvas(modifier = Modifier.size(outer)) {
                val stroke = ringWidth.toPx()
                val radius = (this.size.minDimension - stroke) / 2f
                drawCircle(
                    color = ai.zetic.aiberry.ui.Theme.TileInk.copy(alpha = 0.28f),
                    radius = radius,
                    style = Stroke(width = stroke),
                )
            }
        }

        // Main orb.
        Box(
            modifier = Modifier
                .size(size)
                .clip(CircleShape)
                .background(Theme.Tile)
                .border(1.dp, Theme.TileInk.copy(alpha = 0.15f), CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            if (thinking) {
                CircularProgressIndicator(
                    color = Theme.TileInk,
                    modifier = Modifier.size(size * 0.32f),
                )
            } else {
                Icon(
                    imageVector = if (listening) Icons.Filled.GraphicEq else Icons.Filled.Person,
                    contentDescription = null,
                    tint = Theme.TileInk,
                    modifier = Modifier.size(size * 0.32f),
                )
            }
        }
    }
}
