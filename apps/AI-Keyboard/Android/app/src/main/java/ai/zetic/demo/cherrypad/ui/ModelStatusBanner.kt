package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.llm.LLMService
import ai.zetic.demo.cherrypad.ui.theme.CherryColors
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Inline model status. Renders nothing when the model is Ready. */
@Composable
fun ModelStatusBanner(
    phase: LLMService.Phase,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier,
) {
    when (phase) {
        is LLMService.Phase.Ready -> Unit

        is LLMService.Phase.Failed -> Row(
            modifier = modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(CherryColors.CherrySoft)
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Icon(Icons.Filled.Warning, contentDescription = null, tint = CherryColors.CherryDark, modifier = Modifier.size(18.dp))
            Text(phase.message, color = CherryColors.TextPrimary, fontSize = 13.sp, modifier = Modifier.weight(1f))
            Text(
                "Retry",
                color = CherryColors.Cherry,
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.clickable { onRetry() },
            )
        }

        else -> {
            val text = when (phase) {
                is LLMService.Phase.Downloading -> "Downloading model… ${(phase.progress * 100).toInt()}%"
                else -> "Preparing on-device model…"
            }
            Row(
                modifier = modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(CherryColors.SurfaceMuted)
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp,
                    color = CherryColors.Cherry,
                )
                Text(text, color = CherryColors.TextSecondary, fontSize = 13.sp, fontWeight = FontWeight.Medium)
                Spacer(Modifier.weight(1f))
            }
        }
    }
}
