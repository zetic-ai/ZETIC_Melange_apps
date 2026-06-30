package com.brew.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.brew.engine.ModelPhase
import com.brew.ui.theme.BrewColors
import com.brew.ui.theme.cardBackground

/** Compact AI model status pill (ported from iOS `ModelStatusChip`). */
@Composable
fun ModelStatusChip(phase: ModelPhase, modifier: Modifier = Modifier, onRetry: () -> Unit = {}) {
    when (phase) {
        is ModelPhase.Ready -> Unit // nothing when ready
        else -> {
            val clickable = phase is ModelPhase.Failed
            Row(
                modifier = modifier
                    .cardBackground(BrewColors.cardElevated, 18)
                    .let { if (clickable) it.clickable { onRetry() } else it }
                    .padding(horizontal = 12.dp, vertical = 7.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                when (phase) {
                    is ModelPhase.Downloading -> {
                        CircularProgressIndicator(
                            progress = { phase.progress },
                            modifier = Modifier.size(13.dp),
                            strokeWidth = 2.dp,
                            color = BrewColors.inkSecondary,
                        )
                        Text(
                            "Downloading AI · ${(phase.progress * 100).toInt()}%",
                            fontSize = 13.sp,
                            fontFamily = FontFamily.Monospace,
                            color = BrewColors.inkSecondary,
                        )
                    }

                    is ModelPhase.Failed -> {
                        Icon(
                            Icons.Filled.Refresh,
                            contentDescription = null,
                            tint = BrewColors.warning,
                            modifier = Modifier.size(14.dp),
                        )
                        Text(
                            "AI unavailable — tap to retry",
                            fontSize = 13.sp,
                            color = BrewColors.warning,
                        )
                    }

                    else -> {
                        CircularProgressIndicator(
                            modifier = Modifier.size(13.dp),
                            strokeWidth = 2.dp,
                            color = BrewColors.inkSecondary,
                        )
                        Text("Preparing AI…", fontSize = 13.sp, color = BrewColors.inkSecondary)
                    }
                }
            }
        }
    }
}
