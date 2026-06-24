package ai.zetic.aiberry.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.QuestionAnswer
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.aiberry.session.CheckInViewModel

// MARK: - Landing

@Composable
fun LandingScreen(vm: CheckInViewModel, onStart: () -> Unit) {
    val scroll = rememberScrollState()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scroll)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Spacer(Modifier.height(8.dp))
        EditorialTitle("Conversational\nmultimodal screener")

        PrivacyBanner()

        Column(
            modifier = Modifier.card(),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            ModelRow("Facial expression", "On-device · Melange") { StatusLine(vm.face.status) }
            Box(Modifier.fillMaxWidth().height(1.dp).background(Theme.Bg))
            ModelRow("Voice tone", "On-device · Melange") { StatusLine(vm.voice.status) }
        }

        if (vm.modelsFailed) {
            vm.loadError?.let { msg ->
                Row(
                    modifier = Modifier.card(14.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Icon(
                        Icons.Filled.WarningAmber,
                        contentDescription = null,
                        tint = Theme.Danger,
                        modifier = Modifier.size(20.dp),
                    )
                    Text(msg, fontSize = 13.sp, color = Theme.Danger, lineHeight = 18.sp)
                }
            }
            PrimaryButton(
                text = "Retry",
                enabled = true,
                onClick = { vm.preloadAll() },
                icon = Icons.Filled.Refresh,
            )
        } else {
            PrimaryButton(
                text = if (vm.modelsReady) "Start check-in" else "Preparing models…",
                enabled = vm.modelsReady,
                onClick = onStart,
            )
        }
    }
}

@Composable
private fun ModelRow(title: String, subtitle: String, status: @Composable () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(modifier = Modifier.weight(1f)) {
            Text(title, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = Theme.Ink)
            Spacer(Modifier.height(2.dp))
            Text(subtitle, fontSize = 12.sp, color = Theme.InkSoft)
        }
        status()
    }
}

// MARK: - Intro / consent

@Composable
fun IntroScreen(vm: CheckInViewModel) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(22.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Spacer(Modifier.weight(1f))
        PresenceOrb(size = 150.dp)
        Text(
            "How it works",
            fontFamily = Theme.Serif,
            fontSize = 30.sp,
            fontWeight = FontWeight.SemiBold,
            color = Theme.Ink,
        )
        Column(
            modifier = Modifier.card(),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Point(Icons.Filled.QuestionAnswer, "Answer a few open questions out loud, naturally.")
            Point(Icons.Filled.Face, "Your expression and voice are analyzed on-device as you talk.")
            Point(Icons.Filled.Lock, "Nothing is recorded to the cloud. It works in Airplane Mode.")
        }
        DisclaimerNote()
        Spacer(Modifier.weight(1f))
        PrimaryButton(text = "Begin", enabled = true, onClick = { vm.begin() })
    }
}

@Composable
private fun Point(icon: ImageVector, text: String) {
    Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        IconTile(icon = icon, size = 40)
        Text(
            text,
            fontSize = 14.sp,
            color = Theme.Ink,
            modifier = Modifier.weight(1f),
        )
    }
}

// MARK: - Conversation

@Composable
fun ConversationScreen(vm: CheckInViewModel) {
    val context = LocalContext.current
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp),
    ) {
        // Header: brand mark + camera PiP.
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            verticalAlignment = Alignment.Top,
        ) {
            BrandMark()
            Spacer(Modifier.weight(1f))
            CameraPreview(
                controller = vm.camera,
                onFrame = { b, r -> vm.face.ingest(b, r) },
                modifier = Modifier
                    .size(width = 96.dp, height = 128.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .border(1.dp, Theme.TileInk.copy(alpha = 0.25f), RoundedCornerShape(16.dp)),
            )
        }

        Spacer(Modifier.weight(1f))

        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(28.dp),
        ) {
            PresenceOrb(listening = vm.isRecording, level = vm.micLevel, size = 168.dp)
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                ChatBubble(vm.spokenText, isUser = false)
                if (vm.isRecording) ListeningBubble()
            }
        }

        Spacer(Modifier.weight(1f))

        // Optional countdown progress.
        LinearProgressIndicator(
            progress = { vm.countdownProgress.toFloat() },
            modifier = Modifier.fillMaxWidth().height(3.dp),
            color = Theme.Accent,
            trackColor = Theme.Card,
        )
        Spacer(Modifier.height(12.dp))

        PrimaryButton(
            text = if (vm.questionIndex + 1 < vm.totalQuestions) "Next question" else "See results",
            enabled = vm.canAdvance,
            onClick = { vm.advance() },
            icon = Icons.AutoMirrored.Filled.ArrowForward,
        )
        Spacer(Modifier.height(8.dp))

        // Bottom bar: mic | question count | end.
        Row(
            modifier = Modifier.fillMaxWidth().padding(vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row(
                modifier = Modifier.weight(1f),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    if (vm.isRecording) Icons.Filled.Mic else Icons.Filled.MicOff,
                    contentDescription = null,
                    tint = if (vm.isRecording) Theme.Accent else Theme.InkSoft,
                    modifier = Modifier.size(14.dp),
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    "Mic",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = if (vm.isRecording) Theme.Accent else Theme.InkSoft,
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    "Question ${vm.questionIndex + 1} of ${vm.totalQuestions}",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = Theme.Ink,
                )
            }
            Row(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(8.dp))
                    .clickable {
                        Haptics.tap(context)
                        vm.cancel()
                    },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    Icons.Filled.Close,
                    contentDescription = null,
                    tint = Theme.Danger,
                    modifier = Modifier.size(14.dp),
                )
                Spacer(Modifier.width(6.dp))
                Text("End", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = Theme.Danger)
            }
        }
    }
}

/** A restrained header mark (sage square + serif label) — no brand colors. */
@Composable
private fun BrandMark(size: Int = 20) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Box(
            modifier = Modifier
                .size(size.dp)
                .clip(RoundedCornerShape(5.dp))
                .background(Theme.Tile),
        )
        Text("Screening", fontFamily = Theme.Serif, fontSize = size.sp, fontWeight = FontWeight.SemiBold, color = Theme.Ink)
    }
}

// MARK: - Analyzing

@Composable
fun AnalyzingScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(30.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        PresenceOrb(thinking = true, size = 150.dp)
        Text("Reading your check-in…", fontFamily = Theme.Serif, fontSize = 24.sp, color = Theme.Ink)
        Text("Combining face and voice, on-device", fontSize = 14.sp, color = Theme.InkSoft)
    }
}

// MARK: - Shared helpers

/** Primary filled action button in the deep-green accent. */
@Composable
fun PrimaryButton(
    text: String,
    enabled: Boolean,
    onClick: () -> Unit,
    icon: ImageVector? = null,
) {
    val context = LocalContext.current
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(if (enabled) Theme.Accent else Theme.Accent.copy(alpha = 0.4f))
            .clickable(enabled = enabled) {
                Haptics.tap(context)
                onClick()
            }
            .padding(vertical = 15.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(text, fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
        if (icon != null) {
            Spacer(Modifier.width(8.dp))
            Icon(icon, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
        }
    }
}

/** Reused non-diagnostic disclaimer. */
@Composable
fun DisclaimerNote() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Top,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Icon(Icons.Filled.Info, contentDescription = null, tint = Theme.InkSoft, modifier = Modifier.size(14.dp))
        Text(
            "This is a technology demo, not a medical or diagnostic assessment.",
            fontSize = 12.sp,
            color = Theme.InkSoft,
        )
    }
}
