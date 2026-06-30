package com.brew.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.brew.ui.components.LevelBars
import com.brew.ui.components.timeString
import com.brew.ui.theme.BrewColors
import com.brew.ui.theme.Serif
import com.brew.vm.RecordingState

@Composable
fun RecordingSheet(
    state: RecordingState,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit,
    onCancel: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(BrewColors.canvas)
            .padding(horizontal = 24.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("English", fontSize = 14.sp, fontWeight = FontWeight.Medium, color = BrewColors.inkSecondary)
            Spacer(Modifier.weight(1f))
            Text(
                "Cancel",
                fontSize = 15.sp,
                color = BrewColors.inkSecondary,
                modifier = Modifier.clickable { onCancel() },
            )
        }

        Text("New Note", fontFamily = Serif, fontSize = 38.sp, color = BrewColors.inkSecondary)

        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                if (state.healthy) Icons.Filled.GraphicEq else Icons.Filled.Warning,
                contentDescription = null,
                tint = if (state.healthy) BrewColors.iconTileInk else BrewColors.warning,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text(
                when {
                    !state.healthy -> "Recording problem — audio may not be saving."
                    state.isPaused -> "Paused"
                    else -> "Recording… everything you say will be transcribed when you stop."
                },
                fontSize = 16.sp,
                color = if (state.healthy) BrewColors.ink else BrewColors.warning,
            )
        }

        Spacer(Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            PillButton(
                icon = if (state.isPaused) Icons.Filled.PlayArrow else Icons.Filled.Pause,
                onClick = { if (state.isPaused) onResume() else onPause() },
            )
            Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    timeString(state.elapsedSeconds),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    fontFamily = FontFamily.Monospace,
                    color = BrewColors.ink,
                )
                Spacer(Modifier.height(6.dp))
                LevelBars(if (state.isPaused) 0f else state.level)
            }
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(28.dp))
                    .background(BrewColors.cardElevated)
                    .clickable { onStop() }
                    .padding(horizontal = 26.dp, vertical = 16.dp),
            ) {
                Text("Stop", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = BrewColors.ink)
            }
        }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
private fun PillButton(icon: androidx.compose.ui.graphics.vector.ImageVector, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(28.dp))
            .background(BrewColors.cardElevated)
            .clickable { onClick() }
            .padding(horizontal = 24.dp, vertical = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = null, tint = BrewColors.ink, modifier = Modifier.size(20.dp))
    }
}
