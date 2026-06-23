package ai.zetic.demo.offlinetranslator.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.demo.offlinetranslator.theme.Theme
import ai.zetic.demo.offlinetranslator.ui.components.PasteButton

/**
 * Landing state: prompt text, Paste, and the voice/image source buttons. Tapping the body starts
 * editing. Mirrors iOS `IdleView`.
 */
@Composable
fun IdleView(
    modelReady: Boolean,
    onActivate: () -> Unit,
    onPaste: () -> Unit,
    onVoice: () -> Unit,
    onImage: () -> Unit,
) {
    Column(Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null,
                    onClick = onActivate,
                ),
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            Text(
                text = "Type, paste, talk, or snap a photo to translate",
                color = Theme.textSecondary,
                fontSize = 24.sp,
                fontWeight = FontWeight.Normal,
            )
            PasteButton(enabled = modelReady, onClick = { onPaste(); onActivate() })
        }

        Spacer(Modifier.weight(1f))

        // Offline input sources: voice (speech-to-text) and image (OCR).
        Row(verticalAlignment = Alignment.CenterVertically) {
            sourceButton(Icons.Filled.Mic, "Voice input", onVoice)
            Spacer(Modifier.width(16.dp))
            sourceButton(Icons.Filled.PhotoCamera, "Image input", onImage)
        }
    }
}

@Composable
private fun sourceButton(icon: ImageVector, contentDescription: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(53.dp)
            .clip(CircleShape)
            .background(Theme.surfaceRaised)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = contentDescription,
            tint = Theme.accent,
            modifier = Modifier.size(22.dp),
        )
    }
}
