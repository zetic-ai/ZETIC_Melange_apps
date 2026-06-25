package ai.zetic.demo.cameravitals.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.zetic.demo.cameravitals.state.MeasurementReport

/** Summary bottom sheet shown after a guided 30-second measurement. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReportSheet(report: MeasurementReport, onDismiss: () -> Unit) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Theme.background
    ) {
        Column(
            Modifier.fillMaxWidth().padding(horizontal = 24.dp).padding(bottom = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Measurement complete", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Theme.textPrimary)
                Text(
                    Theme.qualityLabel(report.avgQuality),
                    fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = Theme.quality(report.avgQuality)
                )
            }

            Row(verticalAlignment = Alignment.Bottom) {
                Icon(Icons.Filled.Favorite, null, tint = Theme.accent, modifier = Modifier.padding(bottom = 14.dp).size(28.dp))
                Spacer(Modifier.size(10.dp))
                Text("${report.avgBPM}", fontSize = 72.sp, fontWeight = FontWeight.Bold, color = Theme.textPrimary)
                Spacer(Modifier.size(8.dp))
                Text("BPM", fontSize = 18.sp, fontWeight = FontWeight.SemiBold, color = Theme.textSecondary, modifier = Modifier.padding(bottom = 14.dp))
            }

            Card(Modifier.fillMaxWidth().height(88.dp)) {
                WaveformChart(
                    samples = FloatArray(report.series.size) { report.series[it].toFloat() },
                    modifier = Modifier.fillMaxWidth().height(60.dp).padding(14.dp)
                )
            }

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Stat("Min", report.minBPM, Modifier.weight(1f))
                Stat("Average", report.avgBPM, Modifier.weight(1f))
                Stat("Max", report.maxBPM, Modifier.weight(1f))
            }

            Text("Not a medical device. For demonstration only.", fontSize = 12.sp, color = Theme.textSecondary)

            Box(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Theme.accent)
                    .clickable { onDismiss() }
                    .padding(vertical = 16.dp),
                contentAlignment = Alignment.Center
            ) {
                Text("Done", fontSize = 17.sp, fontWeight = FontWeight.Bold, color = Color.White)
            }
        }
    }
}

@Composable
private fun Stat(label: String, value: Int, modifier: Modifier = Modifier) {
    Column(
        modifier
            .clip(RoundedCornerShape(14.dp))
            .background(Theme.accentSoft)
            .padding(vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("$value", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Theme.textPrimary)
        Text(label, fontSize = 12.sp, fontWeight = FontWeight.Medium, color = Theme.textSecondary)
    }
}
