package ai.zetic.demo.posemotion.ui

import ai.zetic.demo.posemotion.state.DemoViewModel
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.VideocamOff
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun MissingClipScreen(viewModel: DemoViewModel) {
    Column(
        modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Default.VideocamOff, contentDescription = null,
            tint = Theme.TextSecondary, modifier = Modifier.size(44.dp),
        )
        Text(
            "No demo clip found",
            color = Theme.TextPrimary, fontSize = 21.sp, fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(top = 12.dp, bottom = 14.dp),
        )
        Text(
            "1. Add GolfSwing.mp4 to app/src/main/assets/ and rebuild\n\n" +
                "2. Or push it without rebuilding:\nadb push GolfSwing.mp4 " +
                "/sdcard/Android/data/ai.zetic.demo.posemotion/files/\n\n" +
                "Side-on view, single athlete, visible ball, 720–1080p works best",
            color = Theme.TextSecondary, fontSize = 13.sp, textAlign = TextAlign.Start,
        )
        Button(
            onClick = { viewModel.recheckClip() },
            colors = ButtonDefaults.buttonColors(containerColor = Theme.Accent, contentColor = Color.Black),
            modifier = Modifier.padding(top = 20.dp),
        ) {
            Text("Check again", fontWeight = FontWeight.SemiBold)
        }
    }
}
