package com.brew.ui.components

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.brew.ui.theme.BrewColors

/** Three animated dots (ported from iOS `TypingIndicator`). */
@Composable
fun TypingIndicator() {
    val transition = rememberInfiniteTransition(label = "typing")
    Row(horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically) {
        for (i in 0..2) {
            val a by transition.animateFloat(
                initialValue = 0.3f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(600, delayMillis = i * 200),
                    repeatMode = RepeatMode.Reverse,
                ),
                label = "dot$i",
            )
            androidx.compose.foundation.layout.Box(
                modifier = Modifier
                    .size(7.dp)
                    .alpha(a)
                    .clip(CircleShape)
                    .background(BrewColors.inkSecondary),
            )
        }
    }
}

/** Five animated green level capsules (ported from iOS `LevelBars`). */
@Composable
fun LevelBars(level: Float, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(3.dp),
        verticalAlignment = Alignment.Bottom,
    ) {
        val heights = listOf(8, 14, 20, 14, 8)
        heights.forEachIndexed { i, max ->
            val active = level >= (i + 1) / 6f * 0.6f || level > 0.05f && i <= (level * 5).toInt()
            androidx.compose.foundation.layout.Box(
                modifier = Modifier
                    .size(width = 3.dp, height = (if (active) max else 6).dp)
                    .clip(CircleShape)
                    .background(if (active) BrewColors.recording else BrewColors.recording.copy(alpha = 0.3f)),
            )
        }
    }
}

/** Renders an elapsed-seconds count as mm:ss. */
fun timeString(seconds: Int): String {
    val m = seconds / 60
    val s = seconds % 60
    return "%d:%02d".format(m, s)
}

@Composable
fun Dot(modifier: Modifier = Modifier) {
    Text("·", color = BrewColors.inkSecondary, modifier = modifier)
}
