package ai.zetic.aiberry.ui

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.keyframes
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * A conversation bubble — the app's question (leading, light card) or the user's
 * answer (trailing, deep-green), in the shared VoiceVitals palette. 1:1 with
 * iOS-Aiberry's `ChatBubble`.
 */
@Composable
fun ChatBubble(text: String, isUser: Boolean) {
    Row(modifier = Modifier.fillMaxWidth()) {
        if (isUser) Spacer(Modifier.width(40.dp))
        if (!isUser) {
            BubbleBody(text, isUser)
            Spacer(Modifier.width(40.dp))
        } else {
            Spacer(Modifier.weight(1f))
            BubbleBody(text, isUser)
        }
    }
}

@Composable
private fun BubbleBody(text: String, isUser: Boolean) {
    Text(
        text = text,
        fontSize = 16.sp,
        fontWeight = if (isUser) FontWeight.Medium else FontWeight.Normal,
        color = if (isUser) Color.White else Theme.Ink,
        modifier = Modifier
            .clip(RoundedCornerShape(18.dp))
            .background(if (isUser) Theme.TileInk else Theme.CardAlt)
            .padding(horizontal = 16.dp, vertical = 12.dp),
    )
}

/** Animated "listening…" placeholder shown in the user's bubble while recording. */
@Composable
fun ListeningBubble() {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(18.dp))
                .background(Theme.TileInk.copy(alpha = 0.9f))
                .padding(horizontal = 18.dp, vertical = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(5.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            val transition = rememberInfiniteTransition(label = "dots")
            repeat(3) { i ->
                val alpha by transition.animateFloat(
                    initialValue = 0.4f,
                    targetValue = 0.4f,
                    animationSpec = infiniteRepeatable(
                        animation = keyframes {
                            durationMillis = 1200
                            0.4f at 0 using LinearEasing
                            1f at (i * 400 + 200) using LinearEasing
                            0.4f at (i * 400 + 400) using LinearEasing
                        },
                        repeatMode = RepeatMode.Restart,
                    ),
                    label = "dot$i",
                )
                Box(
                    modifier = Modifier
                        .size(7.dp)
                        .alpha(alpha)
                        .clip(CircleShape)
                        .background(Color.White),
                )
            }
        }
    }
}
