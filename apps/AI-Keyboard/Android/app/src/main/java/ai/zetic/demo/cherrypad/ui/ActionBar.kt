package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.model.KeyboardTask
import ai.zetic.demo.cherrypad.ui.theme.CherryColors
import ai.zetic.demo.cherrypad.ui.theme.CherryDims
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** The four action pills. Selected = white-on-cherry; unselected = textSecondary-on-muted. */
@Composable
fun ActionBar(
    selected: KeyboardTask,
    onSelect: (KeyboardTask) -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        for (t in KeyboardTask.entries) {
            val isSelected = t == selected
            Column(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(CherryDims.chipRadius))
                    .background(if (isSelected) CherryColors.Cherry else CherryColors.SurfaceMuted)
                    .clickable { onSelect(t) }
                    .padding(vertical = 10.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                Icon(
                    imageVector = t.icon,
                    contentDescription = t.title,
                    tint = if (isSelected) CherryColors.OnCherry else CherryColors.TextSecondary,
                    modifier = Modifier.size(18.dp),
                )
                Text(
                    text = t.title,
                    color = if (isSelected) CherryColors.OnCherry else CherryColors.TextSecondary,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}
