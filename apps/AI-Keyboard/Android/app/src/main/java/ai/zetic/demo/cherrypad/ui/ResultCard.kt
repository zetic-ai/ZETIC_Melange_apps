package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.AppModel
import ai.zetic.demo.cherrypad.ui.theme.CherryColors
import ai.zetic.demo.cherrypad.ui.theme.CherryDims
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** The streaming result card (mirrors iOS `ResultCard`). */
@Composable
fun ResultCard(model: AppModel, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .shadow(12.dp, RoundedCornerShape(CherryDims.cardRadius), clip = false)
            .clip(RoundedCornerShape(CherryDims.cardRadius))
            .background(CherryColors.Surface)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        // Header
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Icon(model.task.icon, contentDescription = null, tint = CherryColors.TextSecondary, modifier = Modifier.size(16.dp))
            Text("${model.task.title} result", color = CherryColors.TextSecondary, fontSize = 13.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.weight(1f))
            if (model.isGenerating) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp, color = CherryColors.Cherry)
            }
        }

        // Body
        val error = model.errorMessage
        when {
            error != null -> Text(error, color = CherryColors.CherryDark, fontSize = 15.sp)
            model.resultText.isEmpty() && model.isGenerating ->
                Text("Thinking…", color = CherryColors.TextSecondary, fontSize = 15.sp)
            else -> Text(model.resultText, color = CherryColors.TextPrimary, fontSize = 16.sp)
        }

        // Actions — only once a real result is present.
        if (!model.isGenerating && error == null && model.resultText.isNotEmpty()) {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(CherryDims.chipRadius))
                        .background(CherryColors.SurfaceMuted)
                        .clickable { model.retake() }
                        .padding(vertical = 11.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                ) {
                    Icon(Icons.Filled.Refresh, contentDescription = null, tint = CherryColors.TextPrimary, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.size(6.dp))
                    Text("Retake", color = CherryColors.TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                }
                Row(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(CherryDims.chipRadius))
                        .background(CherryColors.Cherry)
                        .clickable { model.apply() }
                        .padding(vertical = 11.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                ) {
                    Icon(
                        if (model.didApply) Icons.Filled.Check else Icons.Filled.ContentCopy,
                        contentDescription = null,
                        tint = CherryColors.OnCherry,
                        modifier = Modifier.size(16.dp),
                    )
                    Spacer(Modifier.size(6.dp))
                    Text(if (model.didApply) "Copied" else "Apply", color = CherryColors.OnCherry, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                }
            }
        }

        // Copied hint.
        if (model.didApply) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(CherryColors.CherrySoft)
                    .padding(10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = CherryColors.CherryDark, modifier = Modifier.size(16.dp))
                Text(
                    "Copied! Paste it wherever you like — or use the CherryPad keyboard's Insert result.",
                    color = CherryColors.CherryDark,
                    fontSize = 13.sp,
                )
            }
        }
    }
}
