package ai.zetic.demo.offlinetranslator.ui

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.BookmarkBorder
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.demo.offlinetranslator.theme.Theme

/**
 * Result state: source on top, streamed translation on the bottom, with an action footer.
 * Tapping the source returns to editing. Mirrors iOS `ResultView`.
 */
@Composable
fun ResultView(
    sourceText: String,
    translatedText: String,
    isTranslating: Boolean,
    onEditSource: () -> Unit,
    onSpeakSource: () -> Unit,
    onSpeakTranslation: () -> Unit,
    onShare: () -> Unit,
    onCopy: () -> Unit,
) {
    Column(Modifier.fillMaxSize()) {
        // Source (top half)
        Column(
            modifier = Modifier.fillMaxWidth().weight(1f).padding(bottom = 14.dp),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null,
                        onClick = onEditSource,
                    ),
            ) {
                Text(text = sourceText, color = Theme.textPrimary, fontSize = 20.sp)
            }
            Spacer(Modifier.height(12.dp))
            IconButton(Icons.AutoMirrored.Filled.VolumeUp, onSpeakSource)
        }

        Box(Modifier.fillMaxWidth().height(1.dp).background(Theme.separator))

        // Translation (bottom half)
        Column(
            modifier = Modifier.fillMaxWidth().weight(1f).padding(top = 16.dp),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .verticalScroll(rememberScrollState()),
            ) {
                TranslationText(translatedText, isTranslating)
            }
            Spacer(Modifier.height(12.dp))
            Footer(
                onSpeak = onSpeakTranslation,
                onShare = onShare,
                onCopy = onCopy,
                shareEnabled = translatedText.isNotEmpty(),
            )
        }
    }
}

@Composable
private fun TranslationText(translatedText: String, isTranslating: Boolean) {
    if (translatedText.isEmpty() && isTranslating) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(
                color = Theme.textSecondary,
                strokeWidth = 2.dp,
                modifier = Modifier.size(16.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text("Translating on device…", color = Theme.textSecondary, fontSize = 15.sp)
        }
    } else {
        val caretAlpha = if (isTranslating) blinkingAlpha() else 0f
        val text: AnnotatedString = buildAnnotatedString {
            append(translatedText)
            if (isTranslating) {
                withStyle(SpanStyle(color = Theme.accent.copy(alpha = caretAlpha))) {
                    append(" ▍")
                }
            }
        }
        Text(text = text, color = Theme.textPrimary, fontSize = 20.sp)
    }
}

@Composable
private fun blinkingAlpha(): Float {
    val transition = rememberInfiniteTransition(label = "caret")
    val alpha by transition.animateFloat(
        initialValue = 1f,
        targetValue = 0f,
        animationSpec = infiniteRepeatable(tween(600), RepeatMode.Reverse),
        label = "caretAlpha",
    )
    return alpha
}

@Composable
private fun Footer(
    onSpeak: () -> Unit,
    onShare: () -> Unit,
    onCopy: () -> Unit,
    shareEnabled: Boolean,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        IconButton(Icons.AutoMirrored.Filled.VolumeUp, onSpeak)
        Spacer(Modifier.width(20.dp))
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(percent = 50))
                .background(Theme.surfaceRaised)
                .padding(horizontal = 13.dp, vertical = 7.dp),
        ) {
            Text("Alternatives", color = Theme.textSecondary, fontSize = 13.sp, fontWeight = FontWeight.Medium)
        }
        Spacer(Modifier.weight(1f))
        IconButton(Icons.Filled.BookmarkBorder, onClick = { /* placeholder, inert like iOS */ })
        Spacer(Modifier.width(20.dp))
        IconButton(Icons.Filled.Share, onShare, enabled = shareEnabled)
        Spacer(Modifier.width(20.dp))
        IconButton(Icons.Filled.ContentCopy, onCopy)
    }
}

@Composable
private fun IconButton(icon: ImageVector, onClick: () -> Unit, enabled: Boolean = true) {
    val tint = if (enabled) Theme.textPrimary else Theme.textTertiary
    Box(
        modifier = Modifier
            .size(28.dp)
            .clickable(
                enabled = enabled,
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = onClick,
            ),
        contentAlignment = Alignment.Center,
    ) {
        Icon(imageVector = icon, contentDescription = null, tint = tint, modifier = Modifier.size(18.dp))
    }
}
