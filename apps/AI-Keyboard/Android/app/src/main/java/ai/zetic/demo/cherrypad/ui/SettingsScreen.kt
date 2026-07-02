package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.ui.theme.CherryColors
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties

@Composable
fun SettingsScreen(
    onShowOnboarding: () -> Unit,
    onDismiss: () -> Unit,
) {
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Column(
            Modifier
                .fillMaxSize()
                .background(CherryColors.Background)
                .verticalScroll(rememberScrollState()),
        ) {
            // Top bar
            Row(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Settings", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = CherryColors.TextPrimary)
                Spacer(Modifier.weight(1f))
                Text("Done", color = CherryColors.Cherry, fontSize = 16.sp, fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.clickable { onDismiss() })
            }

            // Section 1
            Card {
                Text(
                    "How to enable the keyboard",
                    color = CherryColors.Cherry,
                    fontSize = 16.sp,
                    modifier = Modifier.fillMaxWidth().clickable { onShowOnboarding() }.padding(16.dp),
                )
            }

            Text(
                "ABOUT",
                color = CherryColors.TextSecondary,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(start = 24.dp, top = 16.dp, bottom = 6.dp),
            )
            Card {
                Column {
                    LabeledRow("Model", "LFM2.5 350M")
                    LabeledRow("Runs", "100% on-device")
                }
            }
            Text(
                "A small on-device model powers Rewrite, Reply, Translate, and Grammar — right on the keyboard, no network needed after the first download.",
                color = CherryColors.TextSecondary,
                fontSize = 13.sp,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp),
            )
        }
    }
}

@Composable
private fun Card(content: @Composable () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(CherryColors.Surface),
    ) { content() }
}

@Composable
private fun LabeledRow(label: String, value: String) {
    Row(
        Modifier.fillMaxWidth().padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, color = CherryColors.TextPrimary, fontSize = 15.sp)
        Spacer(Modifier.weight(1f))
        Text(value, color = CherryColors.TextSecondary, fontSize = 15.sp)
    }
}
