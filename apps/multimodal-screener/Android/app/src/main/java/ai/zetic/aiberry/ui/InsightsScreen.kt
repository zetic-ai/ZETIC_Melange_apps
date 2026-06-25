package ai.zetic.aiberry.ui

import androidx.compose.foundation.background
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
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.QuestionAnswer
import androidx.compose.material.icons.filled.Verified
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.aiberry.emotion.EmotionScore
import ai.zetic.aiberry.emotion.EmotionStyle
import ai.zetic.aiberry.session.CheckInViewModel
import ai.zetic.aiberry.session.ScreeningReport
import kotlin.math.roundToInt

private enum class InsightsTab(val label: String) {
    Score("Score"),
    Insights("Screening Insights"),
    Transcript("Transcript"),
}

@Composable
fun InsightsScreen(vm: CheckInViewModel, report: ScreeningReport) {
    var tab by remember { mutableStateOf(InsightsTab.Score) }
    val scroll = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scroll)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth()) {
            Text("Results", fontFamily = Theme.Serif, fontSize = 30.sp, fontWeight = FontWeight.SemiBold, color = Theme.Ink)
        }

        // Tab bar.
        Row(modifier = Modifier.fillMaxWidth()) {
            InsightsTab.entries.forEach { t ->
                val active = t == tab
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clickable { tab = t },
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Text(
                        t.label,
                        fontSize = 12.sp,
                        fontWeight = if (active) FontWeight.Bold else FontWeight.Normal,
                        color = if (active) Theme.Accent else Theme.InkSoft,
                        maxLines = 1,
                    )
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(2.dp)
                            .background(if (active) Theme.Accent else Color.Transparent),
                    )
                }
            }
        }

        when (tab) {
            InsightsTab.Score -> ScoreTab(report)
            InsightsTab.Insights -> InsightsTabContent(report)
            InsightsTab.Transcript -> TranscriptTab(report)
        }

        DisclaimerNote()
        PrimaryButton(text = "New check-in", enabled = true, onClick = { vm.restart() })
    }
}

// MARK: Score

@Composable
private fun ScoreTab(report: ScreeningReport) {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Column(
            modifier = Modifier.card(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ScoreGauge(value = report.wellbeing / 100f, score = report.wellbeing, band = report.band)
            Text("Composite well-being", fontSize = 12.sp, color = Theme.InkSoft)
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            ModalityChip("Face", report.faceTop, Modifier.weight(1f))
            ModalityChip("Voice", report.voiceTop, Modifier.weight(1f))
        }

        Row(
            modifier = Modifier.card(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Filled.Verified, contentDescription = null, tint = Theme.InkSoft, modifier = Modifier.size(14.dp))
            Spacer(Modifier.width(6.dp))
            Text("Evidence", fontSize = 12.sp, color = Theme.InkSoft)
            Spacer(Modifier.weight(1f))
            Text(
                "${(report.confidence * 100).roundToInt()}% · ${report.faceFrames} face reads",
                fontSize = 12.sp,
                color = Theme.InkSoft,
            )
        }
    }
}

@Composable
private fun ModalityChip(title: String, score: EmotionScore?, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(Theme.CardAlt)
            .padding(vertical = 14.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(title, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = Theme.InkSoft)
        Text(score?.let { EmotionStyle.emoji(it.label) } ?: "—", fontSize = 34.sp)
        Text(score?.label ?: "n/a", fontSize = 14.sp, fontWeight = FontWeight.Medium, color = Theme.Ink)
    }
}

// MARK: Screening insights

@Composable
private fun InsightsTabContent(report: ScreeningReport) {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Column(modifier = Modifier.card(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Sub-dimensions", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = Theme.InkSoft)
            DimensionRow("Mood", report.mood, "Mood", report)
            DimensionRow("Energy", report.energy, "Energy", report)
            DimensionRow("Rate of speech", report.rateOfSpeech, null, report)
        }

        Column(modifier = Modifier.card(), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text("Emotion blend (face + voice)", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = Theme.InkSoft)
            report.fused.forEachIndexed { index, score ->
                BarRow(
                    label = "${EmotionStyle.emoji(score.label)}  ${score.label}",
                    value = score.probability,
                    tint = EmotionStyle.color(score.label),
                    highlighted = index == 0,
                )
            }
        }
    }
}

@Composable
private fun DimensionRow(name: String, value: Int, driverKey: String?, report: ScreeningReport) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        BarRow(label = name, value = value / 100f, tint = Theme.Accent)
        val drivers = driverKey?.let { report.drivers[it] }
        if (!drivers.isNullOrEmpty()) {
            Text(
                "Driven mostly by ${drivers.joinToString(", ")}",
                fontSize = 11.sp,
                color = Theme.InkSoft,
            )
        }
    }
}

// MARK: Transcript

@Composable
private fun TranscriptTab(report: ScreeningReport) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        if (report.transcript.isEmpty()) {
            // Defensive only — a finished session always has the asked questions.
            Column(
                modifier = Modifier.card(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(Icons.Filled.QuestionAnswer, contentDescription = null, tint = Theme.InkSoft, modifier = Modifier.size(28.dp))
                Text("No transcript available", fontSize = 14.sp, color = Theme.Ink)
            }
            return@Column
        }

        // Always show the questions that were asked; show the spoken answer when it
        // was transcribed, otherwise a per-answer note. (The questions are known even
        // when on-device speech recognition is unavailable on this device/locale.)
        val anyAnswer = report.transcript.any { it.answer.isNotBlank() }
        report.transcript.forEach { qa ->
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                ChatBubble(qa.question, isUser = false)
                if (qa.answer.isNotBlank()) {
                    ChatBubble(qa.answer, isUser = true)
                } else {
                    Text(
                        "Spoken answer not transcribed on this device.",
                        fontSize = 12.sp,
                        color = Theme.InkSoft,
                    )
                }
            }
        }
        if (anyAnswer) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Icon(Icons.Filled.Lock, contentDescription = null, tint = Theme.InkSoft, modifier = Modifier.size(12.dp))
                Text("Transcribed on-device", fontSize = 11.sp, color = Theme.InkSoft)
            }
        } else {
            Text(
                "On-device speech recognition may be unavailable on this device or locale.",
                fontSize = 12.sp,
                color = Theme.InkSoft,
                textAlign = TextAlign.Center,
            )
        }
    }
}
