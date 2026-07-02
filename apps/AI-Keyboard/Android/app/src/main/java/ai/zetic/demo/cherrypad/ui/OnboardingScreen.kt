package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.ui.theme.CherryColors
import ai.zetic.demo.cherrypad.ui.theme.CherryDims
import android.content.Intent
import android.provider.Settings
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties

private data class Step(val title: String, val detail: String)

private val steps = listOf(
    Step("Enable the keyboard", "Settings ▸ System ▸ Languages & input ▸ On-screen keyboard ▸ Manage keyboards ▸ turn on CherryPad."),
    Step("Switch to it", "Tap the keyboard-switch icon (or 🌐) and pick CherryPad."),
    Step("Use it anywhere", "Select text (or just type), then tap Rewrite / Reply / Translate / Grammar. It runs right on the keyboard."),
)

@Composable
fun OnboardingScreen(onDone: () -> Unit) {
    val context = LocalContext.current
    Dialog(onDismissRequest = onDone, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Column(
            Modifier
                .fillMaxSize()
                .background(CherryColors.Background)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp),
        ) {
            // Hero
            Column(Modifier.padding(top = 24.dp, bottom = 28.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("🍒", fontSize = 44.sp)
                Text("CherryPad", fontSize = 30.sp, fontWeight = FontWeight.Black, color = CherryColors.TextPrimary)
                Text(
                    "On-device AI for rewriting, replying, translating, and fixing grammar — all private to your phone.",
                    fontSize = 16.sp,
                    color = CherryColors.TextSecondary,
                )
            }

            // Steps
            Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
                steps.forEachIndexed { i, step ->
                    Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                        Box(
                            Modifier.size(28.dp).clip(CircleShape).background(CherryColors.Cherry),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text("${i + 1}", color = CherryColors.OnCherry, fontSize = 15.sp, fontWeight = FontWeight.Bold)
                        }
                        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            Text(step.title, fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = CherryColors.TextPrimary)
                            Text(step.detail, fontSize = 14.sp, color = CherryColors.TextSecondary)
                        }
                    }
                }
            }

            Spacer(Modifier.size(28.dp))

            // Open Settings
            Text(
                "Open Settings",
                color = CherryColors.OnCherry,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(CherryDims.chipRadius))
                    .background(CherryColors.Cherry)
                    .clickable {
                        context.startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
                    }
                    .padding(vertical = 15.dp),
            )

            Text(
                "Start using CherryPad",
                color = CherryColors.Cherry,
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onDone() }
                    .padding(vertical = 12.dp),
            )

            Spacer(Modifier.size(8.dp))
        }
    }
}
