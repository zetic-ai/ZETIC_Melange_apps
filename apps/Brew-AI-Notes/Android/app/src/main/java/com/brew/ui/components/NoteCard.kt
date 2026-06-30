package com.brew.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.brew.data.NoteEntity
import com.brew.data.NoteStatus
import com.brew.ui.theme.BrewColors
import com.brew.ui.theme.cardBackground
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())

@OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
fun NoteCard(
    note: NoteEntity,
    onClick: () -> Unit,
    onLongPress: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .cardBackground(BrewColors.card, 20)
            .combinedClickable(onClick = onClick, onLongClick = onLongPress)
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        NoteIconTile(icon = iconFor(note.noteStatus))
        Spacer(Modifier.size(14.dp))
        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(
                note.displayTitle,
                fontSize = 18.sp,
                fontWeight = FontWeight.SemiBold,
                color = BrewColors.ink,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                statusOrTime(note),
                fontSize = 14.sp,
                color = BrewColors.inkSecondary,
            )
        }
        Spacer(Modifier.size(10.dp))
        Icon(Icons.Filled.Lock, contentDescription = null, tint = BrewColors.inkTertiary, modifier = Modifier.size(14.dp))
    }
}

@Composable
fun NoteIconTile(icon: ImageVector, size: Int = 56) {
    Box(
        modifier = Modifier
            .size(size.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(BrewColors.iconTile),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = null, tint = BrewColors.iconTileInk, modifier = Modifier.size(22.dp))
    }
}

private fun iconFor(status: NoteStatus): ImageVector = when (status) {
    NoteStatus.ENHANCED -> Icons.Filled.Description
    NoteStatus.ENHANCING -> Icons.Filled.AutoAwesome
    NoteStatus.TRANSCRIPTION_FAILED -> Icons.Filled.Warning
    else -> Icons.Filled.GraphicEq
}

private fun statusOrTime(note: NoteEntity): String = when (note.noteStatus) {
    NoteStatus.TRANSCRIBING -> "Transcribing…"
    NoteStatus.ENHANCING -> "Generating note…"
    NoteStatus.TRANSCRIPTION_FAILED -> "Transcription failed"
    else -> timeFormat.format(Date(note.createdAt))
}
