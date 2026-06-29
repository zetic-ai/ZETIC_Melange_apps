package ai.zetic.skinclassifier.ui

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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CenterFocusWeak
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.MonitorHeart
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.skinclassifier.core.ModelStatus
import ai.zetic.skinclassifier.state.DiagnosisViewModel
import kotlin.math.roundToInt

@Composable
fun DownloadScreen(vm: DiagnosisViewModel) {
    val status = vm.classifierStatus
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 22.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        BrandMark()
        Spacer(Modifier.size(28.dp))
        Text(
            "Preparing on-device AI",
            color = Theme.Ink,
            fontSize = 21.sp,
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(Modifier.size(8.dp))
        Text(
            "The skin vision model is loading directly onto this device. Nothing is sent to the cloud.",
            color = Theme.InkSoft,
            fontSize = 13.5.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 8.dp),
        )
        Spacer(Modifier.size(28.dp))

        Column(modifier = Modifier.glassCard(corner = 26.dp)) {
            ModelLoadRow(status)
        }

        if (status is ModelStatus.Failed) {
            Spacer(Modifier.size(18.dp))
            Text(
                status.message,
                color = Theme.Coral,
                fontSize = 12.5.sp,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.size(6.dp))
            TextButton(onClick = { vm.retryLoad() }) {
                Text("Try again", color = Theme.Accent, fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
            }
        }

        Spacer(Modifier.size(28.dp))
        Text(
            "POWERED BY ZETIC MELANGE",
            color = Theme.InkFaint,
            fontSize = 11.sp,
            letterSpacing = 1.0.sp,
        )
    }
}

@Composable
private fun ModelLoadRow(status: ModelStatus) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        Box(
            modifier = Modifier
                .size(42.dp)
                .clip(CircleShape)
                .background(Color.White.copy(alpha = 0.06f)),
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Filled.CenterFocusWeak, contentDescription = null, tint = Theme.Accent, modifier = Modifier.size(20.dp))
        }
        Spacer(Modifier.width(14.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text("Skin Vision Model", color = Theme.Ink, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
            Text(statusText(status), color = Theme.InkSoft, fontSize = 12.sp)
        }
        Spacer(Modifier.width(12.dp))
        StatusIndicator(status)
    }
}

@Composable
private fun StatusIndicator(status: ModelStatus) {
    when (status) {
        is ModelStatus.Ready ->
            Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = Theme.Mint, modifier = Modifier.size(20.dp))
        is ModelStatus.Failed ->
            Icon(Icons.Filled.Warning, contentDescription = null, tint = Theme.Coral, modifier = Modifier.size(18.dp))
        is ModelStatus.Downloading ->
            CircularProgressIndicator(
                progress = { status.progress.coerceIn(0f, 1f) },
                modifier = Modifier.size(24.dp),
                color = Theme.Accent,
                trackColor = Color.White.copy(alpha = 0.10f),
                strokeWidth = 3.dp,
            )
        else ->
            CircularProgressIndicator(modifier = Modifier.size(22.dp), color = Theme.Accent, strokeWidth = 2.5.dp)
    }
}

private fun statusText(status: ModelStatus): String = when (status) {
    is ModelStatus.Downloading -> "Downloading ${(status.progress * 100).roundToInt()}%"
    is ModelStatus.Ready -> "Ready"
    is ModelStatus.Failed -> "Failed"
    else -> "Preparing…"
}

@Composable
private fun BrandMark() {
    Box(
        modifier = Modifier
            .size(84.dp)
            .clip(CircleShape)
            .background(Theme.brandGradient),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            Icons.Filled.MonitorHeart,
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier.size(38.dp),
        )
    }
}
