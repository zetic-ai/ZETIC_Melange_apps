package ai.zetic.aiberry.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Autorenew
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Circle
import androidx.compose.material.icons.filled.Downloading
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.aiberry.core.ModelStatus
import kotlin.math.roundToInt

/** Big serif headline. */
@Composable
fun EditorialTitle(text: String, modifier: Modifier = Modifier) {
    Text(
        text = text,
        fontFamily = Theme.Serif,
        fontSize = 40.sp,
        color = Theme.Ink,
        maxLines = 2,
        modifier = modifier.fillMaxWidth(),
    )
}

/** Rounded sage tile holding a line-art glyph. */
@Composable
fun IconTile(icon: ImageVector, size: Int = 56) {
    Box(
        modifier = Modifier
            .size(size.dp)
            .clip(RoundedCornerShape(15.dp))
            .background(Theme.Tile),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = Theme.TileInk,
            modifier = Modifier.size((size * 0.42f).dp),
        )
    }
}

/** Section header inside a card: sage tile + title + caption. */
@Composable
fun CardHeader(icon: ImageVector, title: String, subtitle: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        IconTile(icon = icon, size = 52)
        Spacer(Modifier.width(14.dp))
        Column {
            Text(title, fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = Theme.Ink)
            Spacer(Modifier.height(3.dp))
            Text(subtitle, fontSize = 12.sp, color = Theme.InkSoft)
        }
    }
}

/** Large tap-to-record button with a live input-level ring. */
@Composable
fun RecordButton(
    isRecording: Boolean,
    level: Float,
    busy: Boolean,
    onClick: () -> Unit,
) {
    val ringWidth by animateDpAsState(
        targetValue = (2f + level * 12f).dp,
        animationSpec = tween(durationMillis = 80),
        label = "ring",
    )
    val ringColor by animateColorAsState(
        targetValue = if (isRecording) Theme.Danger else Theme.TileInk.copy(alpha = 0.35f),
        label = "ringColor",
    )
    Box(
        modifier = Modifier
            .size(128.dp)
            .clip(CircleShape)
            .background(if (isRecording) Theme.Dark else Theme.Tile)
            .border(ringWidth, ringColor, CircleShape)
            .clickable(enabled = !busy) { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        if (busy) {
            CircularProgressIndicator(color = Theme.TileInk, modifier = Modifier.size(44.dp))
        } else {
            Icon(
                imageVector = if (isRecording) Icons.Filled.Stop else Icons.Filled.Mic,
                contentDescription = if (isRecording) "Stop recording" else "Record",
                tint = if (isRecording) Color.White else Theme.TileInk,
                modifier = Modifier.size(44.dp),
            )
        }
    }
}

/** Horizontal probability bar with a label and percentage. */
@Composable
fun BarRow(
    label: String,
    value: Float,
    tint: Color = Theme.Accent,
    highlighted: Boolean = false,
) {
    val animated by animateFloatAsState(
        targetValue = value.coerceIn(0f, 1f),
        animationSpec = spring(dampingRatio = 0.8f, stiffness = 380f),
        label = "bar",
    )
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                label,
                fontSize = 15.sp,
                fontWeight = if (highlighted) FontWeight.SemiBold else FontWeight.Normal,
                color = Theme.Ink,
                modifier = Modifier.weight(1f),
            )
            Text(
                "${(value * 100).roundToInt()}%",
                fontSize = 15.sp,
                color = Theme.InkSoft,
            )
        }
        Spacer(Modifier.height(5.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(10.dp)
                .clip(CircleShape)
                .background(Theme.Bg),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(animated.coerceAtLeast(0.02f))
                    .clip(CircleShape)
                    .background(tint.copy(alpha = if (highlighted) 1f else 0.7f)),
            )
        }
    }
}

/** Status line describing what the on-device model is doing. */
@Composable
fun StatusLine(status: ModelStatus) {
    val (icon, text) = when (status) {
        is ModelStatus.Idle -> Icons.Filled.Circle to "Ready"
        is ModelStatus.Downloading ->
            Icons.Filled.Downloading to "Downloading model… ${(status.progress * 100).roundToInt()}%"
        is ModelStatus.Loading -> Icons.Filled.Memory to "Preparing on NPU…"
        is ModelStatus.Running -> Icons.Filled.GraphicEq to "Analyzing on-device…"
        is ModelStatus.Ready -> Icons.Filled.CheckCircle to "Done"
        is ModelStatus.Failed -> Icons.Filled.Warning to status.message
    }
    val color = if (status.isFailure) Theme.Danger else Theme.InkSoft
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(14.dp))
        Spacer(Modifier.width(6.dp))
        Text(text, fontSize = 12.sp, color = color, maxLines = 2)
    }
}

/** Sample play/stop button used in both screens. */
@Composable
fun SampleButton(
    label: String,
    isActive: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(Theme.Tile.copy(alpha = 0.5f))
            .clickable { onClick() }
            .padding(vertical = 12.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            imageVector = if (isActive) Icons.Filled.Stop else Icons.Filled.PlayArrow,
            contentDescription = null,
            tint = Theme.TileInk,
            modifier = Modifier.size(18.dp),
        )
        Spacer(Modifier.width(8.dp))
        Text(label, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Theme.TileInk)
    }
}
