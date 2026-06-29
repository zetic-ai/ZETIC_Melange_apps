package ai.zetic.skinclassifier.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.HelpOutline
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.MedicalServices
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.skinclassifier.model.Classification
import ai.zetic.skinclassifier.state.AnalysisState
import ai.zetic.skinclassifier.state.DiagnosisViewModel
import kotlin.math.roundToInt

@Composable
fun ResultsScreen(vm: DiagnosisViewModel) {
    Column(modifier = Modifier.fillMaxSize().padding(top = 10.dp)) {
        TopBar(vm)
        val classification = vm.classification
        val analysis = vm.analysis
        when {
            classification != null -> ResultContent(classification)
            analysis is AnalysisState.Failed -> EarlyFailure(vm, analysis.message)
            else -> Analyzing()
        }
    }
}

@Composable
private fun TopBar(vm: DiagnosisViewModel) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(38.dp)
                .clip(CircleShape)
                .background(Color.White.copy(alpha = 0.06f))
                .clickableNoRipple(true) { vm.reset() },
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Theme.Ink, modifier = Modifier.size(16.dp))
        }
        Spacer(Modifier.width(12.dp))
        vm.image?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = null,
                contentScale = ContentScale.Crop,
                modifier = Modifier.size(38.dp).clip(RoundedCornerShape(10.dp)),
            )
            Spacer(Modifier.width(12.dp))
        }
        Column(modifier = Modifier.weight(1f)) {
            Text("Analysis", color = Theme.Ink, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            Text("Fully on-device", color = Theme.InkFaint, fontSize = 11.sp)
        }
        Text(
            "New",
            color = Theme.Accent,
            fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.clickableNoRipple(true) { vm.reset() },
        )
    }
}

@Composable
private fun ResultContent(classification: Classification) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 18.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        VerdictCard(classification)
        ClassDistributionBars(classification.ranked)
        GuidanceCard(classification)
        DisclaimerBanner()
        Spacer(Modifier.size(8.dp))
    }
}

@Composable
private fun VerdictCard(c: Classification) {
    val cls = c.topClass
    Column(modifier = Modifier.glassCard(corner = 26.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Row(verticalAlignment = Alignment.Top) {
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SeverityBadge(cls)
                Text(cls.title, color = Theme.Ink, fontSize = 23.sp, fontWeight = FontWeight.Bold)
                Text(cls.blurb, color = Theme.InkSoft, fontSize = 13.5.sp)
            }
            Spacer(Modifier.width(12.dp))
            ConfidenceRing(value = c.confidence, tint = cls.tint, size = 116.dp)
        }

        if (c.isLowConfidence) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.AutoMirrored.Filled.HelpOutline, contentDescription = null, tint = Theme.Amber, modifier = Modifier.size(15.dp))
                Spacer(Modifier.width(6.dp))
                Text(
                    "Low confidence — treat this as a rough hint only.",
                    color = Theme.Amber,
                    fontSize = 12.5.sp,
                )
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Bolt, contentDescription = null, tint = Theme.InkFaint, modifier = Modifier.size(12.dp))
            Spacer(Modifier.width(5.dp))
            MonoCaption("Classified on-device in ${c.latencyMs.roundToInt()} ms")
        }
    }
}

@Composable
private fun GuidanceCard(c: Classification) {
    val cls = c.topClass
    Column(modifier = Modifier.glassCard(corner = 22.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
        GuidanceSection(Icons.Filled.Search, "What this result may suggest") {
            Text(cls.whatItMeans, color = Theme.InkSoft, fontSize = 14.sp)
        }
        GuidanceSection(Icons.Filled.FavoriteBorder, "General self-care") {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                cls.selfCare.forEach { bullet ->
                    Row(verticalAlignment = Alignment.Top) {
                        Box(
                            Modifier
                                .padding(top = 7.dp)
                                .size(5.dp)
                                .clip(CircleShape)
                                .background(Theme.Accent),
                        )
                        Spacer(Modifier.width(10.dp))
                        Text(bullet, color = Theme.InkSoft, fontSize = 14.sp)
                    }
                }
            }
        }
        GuidanceSection(Icons.Filled.MedicalServices, "When to seek medical care") {
            Text(cls.whenToSeeDoctor, color = Theme.InkSoft, fontSize = 14.sp)
        }
    }
}

@Composable
private fun GuidanceSection(icon: ImageVector, title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .size(width = 3.dp, height = 14.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(Theme.brandGradient),
            )
            Spacer(Modifier.width(8.dp))
            Icon(icon, contentDescription = null, tint = Theme.Accent2, modifier = Modifier.size(15.dp))
            Spacer(Modifier.width(6.dp))
            Text(title, color = Theme.Ink, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        }
        content()
    }
}

@Composable
private fun Analyzing() {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        CircularProgressIndicator(color = Theme.Accent, strokeWidth = 3.dp)
        Spacer(Modifier.size(20.dp))
        Text("Analyzing on-device", color = Theme.Ink, fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.size(6.dp))
        Text("Running the skin vision model…", color = Theme.InkSoft, fontSize = 13.sp)
    }
}

@Composable
private fun EarlyFailure(vm: DiagnosisViewModel, message: String) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Icon(Icons.Filled.Warning, contentDescription = null, tint = Theme.Coral, modifier = Modifier.size(40.dp))
        Spacer(Modifier.size(14.dp))
        Text("Analysis couldn't finish", color = Theme.Ink, fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.size(8.dp))
        Text(message, color = Theme.InkSoft, fontSize = 13.5.sp, textAlign = TextAlign.Center)
        Spacer(Modifier.size(20.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            SecondaryButton(title = "Back", icon = Icons.AutoMirrored.Filled.ArrowBack, onClick = { vm.reset() }, modifier = Modifier.weight(1f))
            PrimaryButton(title = "Retry", icon = Icons.Filled.Refresh, onClick = { vm.retryAnalyze() }, modifier = Modifier.weight(1f))
        }
    }
}
