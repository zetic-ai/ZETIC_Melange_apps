package com.brew.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.brew.ui.theme.BrewColors
import com.brew.ui.theme.cardBackground
import com.brew.vm.RecordingState

/** Pinned bar shown on the list while recording (iOS `RecordingMiniBar`). */
@Composable
fun RecordingMiniBar(state: RecordingState, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .cardBackground(BrewColors.accent, 28)
            .clickable { onClick() }
            .padding(horizontal = 20.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text("New note", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
        Spacer(Modifier.width(8.dp))
        Text("·", color = Color.White.copy(alpha = 0.6f))
        Spacer(Modifier.width(8.dp))
        Text(
            timeString(state.elapsedSeconds),
            fontSize = 16.sp,
            fontWeight = FontWeight.SemiBold,
            fontFamily = FontFamily.Monospace,
            color = BrewColors.recording,
        )
        Spacer(Modifier.weight(1f))
        LevelBars(if (state.isPaused) 0f else state.level)
    }
}
