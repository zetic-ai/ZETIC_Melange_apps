package ai.zetic.demo.cherrypad.ui

import ai.zetic.demo.cherrypad.AppModel
import ai.zetic.demo.cherrypad.model.KeyboardTask
import ai.zetic.demo.cherrypad.model.Stance
import ai.zetic.demo.cherrypad.model.Tone
import ai.zetic.demo.cherrypad.ui.theme.CherryColors
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.border
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Language
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Task-dependent options: rewrite → tones, reply → stances, translate → a single language
 * chip that opens the picker, grammar → nothing.
 */
@Composable
fun ChipRow(
    model: AppModel,
    onOpenLanguagePicker: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .horizontalScroll(rememberScrollState())
            .padding(horizontal = 2.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        when (model.task) {
            KeyboardTask.REWRITE -> Tone.entries.forEach { tone ->
                Chip(title = tone.label, selected = tone == model.tone) { model.selectTone(tone) }
            }
            KeyboardTask.REPLY -> Stance.entries.forEach { stance ->
                Chip(title = stance.label, selected = stance == model.stance) { model.selectStance(stance) }
            }
            KeyboardTask.TRANSLATE -> Chip(
                title = model.targetLanguage.englishName,
                selected = true,
                icon = Icons.Filled.Language,
                onClick = onOpenLanguagePicker,
            )
            KeyboardTask.GRAMMAR -> Unit
        }
    }
}

@Composable
private fun Chip(
    title: String,
    selected: Boolean,
    icon: ImageVector? = null,
    onClick: () -> Unit,
) {
    val fg = if (selected) CherryColors.CherryDark else CherryColors.TextSecondary
    val bg = if (selected) CherryColors.CherrySoft else CherryColors.SurfaceMuted
    val border = if (selected) BorderStroke(1.dp, CherryColors.Cherry.copy(alpha = 0.5f))
    else BorderStroke(1.dp, Color.Transparent)
    Row(
        modifier = Modifier
            .clip(CircleShape)
            .background(bg)
            .border(border, CircleShape)
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        if (icon != null) {
            Icon(imageVector = icon, contentDescription = null, tint = fg, modifier = Modifier.size(14.dp))
        }
        Text(text = title, color = fg, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
    }
}
