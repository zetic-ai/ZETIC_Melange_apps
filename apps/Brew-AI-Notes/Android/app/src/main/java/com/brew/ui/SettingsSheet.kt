package com.brew.ui

import androidx.compose.foundation.background
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
import androidx.compose.foundation.clickable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.brew.engine.ModelPhase
import com.brew.ui.theme.BrewColors
import com.brew.vm.SettingsViewModel

@Composable
fun SettingsSheet(vm: SettingsViewModel = viewModel()) {
    val phase by vm.llmPhase.collectAsStateWithLifecycle()
    val stats by vm.stats.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(BrewColors.canvas)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        Text("Settings", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = BrewColors.ink, modifier = Modifier.align(Alignment.CenterHorizontally))

        Section("ON-DEVICE AI") {
            AiStatusRow(phase, vm::retryLlm)
        }

        Section("STORAGE") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Description, contentDescription = null, tint = BrewColors.inkSecondary, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(10.dp))
                Text("Notes", fontSize = 16.sp, color = BrewColors.ink)
                Spacer(Modifier.weight(1f))
                Text("${stats.notesCount}", fontSize = 16.sp, color = BrewColors.inkSecondary)
            }
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.GraphicEq, contentDescription = null, tint = BrewColors.inkSecondary, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(10.dp))
                Text("Recordings", fontSize = 16.sp, color = BrewColors.ink)
                Spacer(Modifier.weight(1f))
                Text(stats.recordingsLabel, fontSize = 16.sp, color = BrewColors.inkSecondary)
            }
        }

        Section("ABOUT") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Info, contentDescription = null, tint = BrewColors.inkSecondary, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(10.dp))
                Text("Version", fontSize = 16.sp, color = BrewColors.ink)
                Spacer(Modifier.weight(1f))
                Text(vm.versionLabel, fontSize = 16.sp, color = BrewColors.inkSecondary)
            }
        }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(title, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = BrewColors.inkSecondary)
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(18.dp))
                .background(BrewColors.cardElevated)
                .padding(16.dp),
        ) { content() }
    }
}

@Composable
private fun AiStatusRow(phase: ModelPhase, onRetry: () -> Unit) {
    when (phase) {
        is ModelPhase.Ready -> StatusLine(Icons.Filled.CheckCircle, BrewColors.iconTileInk, "Ready — runs privately on this device")
        is ModelPhase.Downloading -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(progress = { phase.progress }, modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = BrewColors.inkSecondary)
            Spacer(Modifier.width(10.dp))
            Text("Downloading model… ${(phase.progress * 100).toInt()}%", fontSize = 16.sp, color = BrewColors.ink)
        }
        is ModelPhase.Failed -> Column {
            StatusLine(Icons.Filled.Warning, BrewColors.warning, "Unavailable")
            Spacer(Modifier.height(4.dp))
            Text(phase.message, fontSize = 14.sp, color = BrewColors.inkSecondary)
            Spacer(Modifier.height(12.dp))
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(BrewColors.card)
                    .clickable { onRetry() }
                    .padding(horizontal = 18.dp, vertical = 10.dp),
            ) { Text("Retry download", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = BrewColors.ink) }
        }
        else -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = BrewColors.inkSecondary)
            Spacer(Modifier.width(10.dp))
            Text("Preparing…", fontSize = 16.sp, color = BrewColors.ink)
        }
    }
}

@Composable
private fun StatusLine(icon: androidx.compose.ui.graphics.vector.ImageVector, tint: androidx.compose.ui.graphics.Color, text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, contentDescription = null, tint = tint, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(10.dp))
        Text(text, fontSize = 16.sp, color = BrewColors.ink)
    }
}
