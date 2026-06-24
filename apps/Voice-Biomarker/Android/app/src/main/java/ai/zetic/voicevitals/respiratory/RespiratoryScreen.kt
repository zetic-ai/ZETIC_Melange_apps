package ai.zetic.voicevitals.respiratory

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Air
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import ai.zetic.voicevitals.R
import ai.zetic.voicevitals.core.AudioRecorder
import ai.zetic.voicevitals.core.ModelStatus
import ai.zetic.voicevitals.core.SamplePlayer
import ai.zetic.voicevitals.ui.BarRow
import ai.zetic.voicevitals.ui.CardHeader
import ai.zetic.voicevitals.ui.EditorialTitle
import ai.zetic.voicevitals.ui.LatencyBadge
import ai.zetic.voicevitals.ui.PrivacyBanner
import ai.zetic.voicevitals.ui.RecordButton
import ai.zetic.voicevitals.ui.SampleButton
import ai.zetic.voicevitals.ui.StatusLine
import ai.zetic.voicevitals.ui.Theme
import ai.zetic.voicevitals.ui.card
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RespiratoryScreen(model: YamnetModel) {
    val context = LocalContext.current
    val recorder = remember { AudioRecorder() }
    val player = remember { SamplePlayer() }
    var showResults by remember { mutableStateOf(false) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = false)

    fun startRecording() {
        recorder.record(autoStopSeconds = 3.0) { model.analyze(it) }
    }

    val micPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> if (granted) startRecording() }

    fun onRecordTap() {
        if (recorder.isRecording) return
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) startRecording() else micPermission.launch(Manifest.permission.RECORD_AUDIO)
    }

    LaunchedEffect(model.status) {
        if (model.status is ModelStatus.Ready && model.topRespiratory != null) showResults = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Theme.Bg)
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        EditorialTitle("Respiratory Sounds", modifier = Modifier.padding(top = 8.dp, bottom = 2.dp))

        PrivacyBanner()

        Box(Modifier.card()) {
            CardHeader(
                icon = Icons.Filled.Air,
                title = "Acoustic Events",
                subtitle = "YAMNet · cough / breath / wheeze",
            )
        }

        Column(
            modifier = Modifier.card(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            RecordButton(
                isRecording = recorder.isRecording,
                level = recorder.level,
                busy = model.status.isBusy,
                onClick = { onRecordTap() },
            )
            Text(
                if (recorder.isRecording) "Listening… cough or breathe" else "Tap, then cough or breathe",
                fontSize = 15.sp,
                color = Theme.InkSoft,
            )
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                StatusLine(model.status)
                Spacer(Modifier.weight(1f))
                LatencyBadge(model.latencyMs)
            }
        }

        Column(
            modifier = Modifier.card(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("Audio samples", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Theme.InkSoft)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SampleButton(
                    label = "Sample 1",
                    isActive = player.playing == "cough_sound",
                    onClick = { playSample(player, context, "cough_sound", R.raw.cough_sound) },
                    modifier = Modifier.weight(1f),
                )
                SampleButton(
                    label = "Sample 2",
                    isActive = player.playing == "sigh_sound",
                    onClick = { playSample(player, context, "sigh_sound", R.raw.sigh_sound) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }

    if (showResults) {
        ModalBottomSheet(
            onDismissRequest = { showResults = false },
            sheetState = sheetState,
            containerColor = Theme.Bg,
        ) {
            ResultsContent(model)
        }
    }
}

private fun playSample(player: SamplePlayer, context: Context, key: String, resId: Int) {
    if (player.playing == key) player.stop() else player.play(context, resId, key)
}

@Composable
private fun ResultsContent(model: YamnetModel) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        val top = model.topRespiratory
        if (top != null) {
            Column(
                modifier = Modifier.card(),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Top respiratory event", fontSize = 15.sp, color = Theme.InkSoft)
                    Spacer(Modifier.weight(1f))
                    Text(top.name, fontFamily = Theme.Serif, fontSize = 22.sp, color = Theme.Ink)
                }
                HorizontalDivider(color = Theme.Bg)
                model.respiratoryEvents.take(6).forEach { event ->
                    BarRow(
                        label = event.name,
                        value = event.score,
                        tint = Theme.TileInk,
                        highlighted = event.index == top.index,
                    )
                }
            }
        }

        if (model.topEvents.isNotEmpty()) {
            Column(
                modifier = Modifier.card(),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text("All sounds (top 5)", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Theme.InkSoft)
                model.topEvents.forEach { event ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(event.name, fontSize = 12.sp, color = Theme.Ink)
                        Spacer(Modifier.weight(1f))
                        Text("${(event.score * 100).roundToInt()}%", fontSize = 12.sp, color = Theme.InkSoft)
                    }
                }
            }
        }
        Spacer(Modifier.height(8.dp))
    }
}
