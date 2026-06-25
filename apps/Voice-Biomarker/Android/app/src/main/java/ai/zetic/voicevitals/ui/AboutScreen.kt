package ai.zetic.voicevitals.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.material.icons.filled.Air
import androidx.compose.material.icons.filled.Autorenew
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Shield
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

@Composable
fun AboutScreen() {
    val models = listOf(
        Triple(Icons.Filled.GraphicEq, "Emotion", "wav2vec2-large-xlsr · 7-class speech emotion"),
        Triple(Icons.Filled.Air, "Respiratory", "google/Sound Classification (YAMNET)"),
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Theme.Bg)
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        EditorialTitle("Voice Biomarkers", modifier = Modifier.padding(top = 8.dp, bottom = 2.dp))

        Column(Modifier.card()) {
            Text(
                "Every model runs fully on this phone through ZETIC Melange, accelerated on " +
                    "the device's NPU. Microphone audio is never uploaded — it works in Airplane Mode.",
                fontSize = 15.sp,
                color = Theme.InkSoft,
            )
        }

        Column(modifier = Modifier.card(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Two models, one pipeline", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = Theme.Ink)
            models.forEach { (icon, title, subtitle) ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconTile(icon = icon, size = 48)
                    Spacer(Modifier.width(14.dp))
                    Column {
                        Text(title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Theme.Ink)
                        Text(subtitle, fontSize = 12.sp, color = Theme.InkSoft)
                    }
                }
            }
        }

        Column(modifier = Modifier.card(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Shield, contentDescription = null, tint = Theme.TileInk, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text("Why on-device", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = Theme.TileInk)
            }
            Bullet("HIPAA-friendly: sensitive audio never leaves the device")
            Bullet("Real-time: NPU inference in milliseconds")
            Bullet("Offline: no connectivity required")
            Bullet("No cloud-GPU cost per inference")
        }

        Column(modifier = Modifier.card(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Autorenew, contentDescription = null, tint = Theme.Ink, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text("Deploy your own model", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = Theme.Ink)
            }
            Text("Upload your model to mlange.zetic.ai, then change one line:", fontSize = 12.sp, color = Theme.InkSoft)
            Text(
                "ZeticMLangeModel(context, personalKey, \"your-org/your-model\")",
                fontFamily = FontFamily.Monospace,
                fontSize = 11.sp,
                color = Theme.Ink,
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(Theme.CardAlt)
                    .padding(12.dp),
            )
        }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
private fun Bullet(text: String) {
    Row(verticalAlignment = Alignment.Top) {
        Icon(
            Icons.Filled.CheckCircle,
            contentDescription = null,
            tint = Theme.TileInk,
            modifier = Modifier.size(16.dp).padding(top = 2.dp),
        )
        Spacer(Modifier.width(8.dp))
        Text(text, fontSize = 15.sp, color = Theme.Ink)
    }
}
