package ai.zetic.voicevitals.emotion

import android.Manifest
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
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
fun EmotionScreen(model: EmotionModel) {
    val context = LocalContext.current
    val recorder = remember { AudioRecorder() }
    val player = remember { SamplePlayer() }
    var showResults by remember { mutableStateOf(false) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = false)

    val micPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) recorder.record(autoStopSeconds = null) { model.analyze(it) }
    }

    fun toggle() {
        if (recorder.isRecording) {
            player.stop()
            recorder.stop()
        } else {
            val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
            if (granted) {
                recorder.record(autoStopSeconds = null) { model.analyze(it) }
            } else {
                micPermission.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    LaunchedEffect(model.status) {
        if (model.status is ModelStatus.Ready && model.top != null) showResults = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Theme.Bg)
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        EditorialTitle("Speech Emotions", modifier = Modifier.padding(top = 8.dp, bottom = 2.dp))

        PrivacyBanner()

        Box(Modifier.card()) {
            CardHeader(
                icon = Icons.Filled.GraphicEq,
                title = "Speech Emotion",
                subtitle = "wav2vec2-large-xlsr · 7 emotions",
            )
        }

        // Recording card
        Column(
            modifier = Modifier.card(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            RecordButton(
                isRecording = recorder.isRecording,
                level = recorder.level,
                busy = model.status.isBusy,
                onClick = { toggle() },
            )
            Text(
                if (recorder.isRecording) "Recording… tap to stop" else "Tap to record",
                fontSize = 15.sp,
                color = Theme.InkSoft,
            )
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                StatusLine(model.status)
                Spacer(Modifier.weight(1f))
                LatencyBadge(model.latencyMs)
            }
        }

        // Samples card
        Column(
            modifier = Modifier.card(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("Audio samples", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Theme.InkSoft)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SampleButton(
                    label = "Sample 1",
                    isActive = player.playing == "angry_voice",
                    onClick = { playSample(player, context, "angry_voice", R.raw.angry_voice) },
                    modifier = Modifier.weight(1f),
                )
                SampleButton(
                    label = "Sample 2",
                    isActive = player.playing == "happy_voice",
                    onClick = { playSample(player, context, "happy_voice", R.raw.happy_voice) },
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

private fun playSample(player: SamplePlayer, context: android.content.Context, key: String, resId: Int) {
    if (player.playing == key) player.stop() else player.play(context, resId, key)
}

@Composable
private fun ResultsContent(model: EmotionModel) {
    val top = model.top ?: return
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        // Hero card
        val tint = EmotionStyle.color(top.label)
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(Theme.Corner))
                .background(
                    Brush.verticalGradient(
                        listOf(tint.copy(alpha = 0.30f), tint.copy(alpha = 0.08f)),
                    ),
                )
                .padding(vertical = 26.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(EmotionStyle.emoji(top.label), fontSize = 60.sp)
            Text(top.label, fontFamily = Theme.Serif, fontSize = 40.sp, color = Theme.Ink)
            Text(
                "${(top.probability * 100).roundToInt()}% confidence",
                fontSize = 15.sp,
                color = Theme.InkSoft,
            )
        }

        // Breakdown card
        Column(
            modifier = Modifier.card(),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text("All emotions", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Theme.InkSoft)
            model.scores.forEach { score ->
                BarRow(
                    label = "${EmotionStyle.emoji(score.label)}  ${score.label}",
                    value = score.probability,
                    tint = EmotionStyle.color(score.label),
                    highlighted = score.label == top.label,
                )
            }
        }
        Spacer(Modifier.height(8.dp))
    }
}
