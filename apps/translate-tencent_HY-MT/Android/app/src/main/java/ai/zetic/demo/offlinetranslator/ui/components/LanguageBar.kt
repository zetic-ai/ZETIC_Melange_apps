package ai.zetic.demo.offlinetranslator.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.demo.offlinetranslator.model.Language
import ai.zetic.demo.offlinetranslator.theme.Theme

/** The persistent bottom language selector: source pill · swap · target pill. */
@Composable
fun LanguageBar(
    source: Language,
    target: Language,
    onTapSource: () -> Unit,
    onTapTarget: () -> Unit,
    onSwap: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        pill(source.englishName, onTapSource)
        Spacer(Modifier.width(10.dp))
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(RoundedCornerShape(percent = 50))
                .clickable(onClick = onSwap),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                imageVector = if (source.isDetect) Icons.AutoMirrored.Filled.ArrowForward else Icons.Filled.SwapHoriz,
                contentDescription = "Swap languages",
                tint = Theme.textPrimary,
                modifier = Modifier.size(20.dp),
            )
        }
        Spacer(Modifier.width(10.dp))
        pill(target.englishName, onTapTarget)
    }
}

@Composable
private fun RowScope.pill(title: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .weight(1f)
            .clip(RoundedCornerShape(13.dp))
            .background(Theme.surfaceRaised)
            .clickable(onClick = onClick)
            .padding(vertical = 14.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = title,
            color = Theme.textPrimary,
            fontSize = 16.sp,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}
