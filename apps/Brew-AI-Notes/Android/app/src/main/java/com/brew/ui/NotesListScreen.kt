package com.brew.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.clickable
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.brew.data.NoteEntity
import com.brew.engine.ModelPhase
import com.brew.ui.components.CircleIconButton
import com.brew.ui.components.ModelStatusChip
import com.brew.ui.components.NoteCard
import com.brew.ui.components.RecordingMiniBar
import com.brew.ui.theme.BrewColors
import com.brew.ui.theme.Serif
import com.brew.vm.NotesListViewModel
import com.brew.vm.RecordingState
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

@Composable
fun NotesListScreen(
    recording: RecordingState,
    onOpenNote: (String) -> Unit,
    onStartRecording: () -> Unit,
    onReopenRecording: () -> Unit,
    onOpenSettings: () -> Unit,
    aiReady: Boolean,
    vm: NotesListViewModel = viewModel(),
) {
    val notes by vm.notes.collectAsStateWithLifecycle()
    val phase by vm.llmPhase.collectAsStateWithLifecycle()
    val query by vm.query.collectAsStateWithLifecycle()
    var searchOpen by remember { mutableStateOf(false) }
    var pendingDelete by remember { mutableStateOf<NoteEntity?>(null) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BrewColors.canvas),
    ) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(
                start = 20.dp, end = 20.dp, top = 16.dp, bottom = 160.dp,
            ),
        ) {
            item {
                Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        ModelStatusChip(phase, onRetry = vm::retryLlm)
                        Spacer(Modifier.weight(1f))
                        CircleIconButton(Icons.Filled.Search, onClick = { searchOpen = !searchOpen })
                        Spacer(Modifier.size(4.dp))
                        ProfileButton(onClick = onOpenSettings)
                    }
                    Text(
                        buildTitle(),
                        fontFamily = Serif,
                        fontSize = 37.sp,
                        color = BrewColors.ink,
                    )
                    if (searchOpen) {
                        SearchField(query) { vm.query.value = it }
                    }
                    Spacer(Modifier.height(6.dp))
                }
            }

            if (notes.isEmpty()) {
                item {
                    Text(
                        if (query.isBlank()) "No notes yet." else "No notes match your search.",
                        fontSize = 16.sp,
                        color = BrewColors.inkSecondary,
                        modifier = Modifier.padding(top = 40.dp),
                    )
                }
            } else {
                val groups = groupByDay(notes)
                groups.forEach { (label, items) ->
                    item {
                        Text(
                            label,
                            fontSize = 16.sp,
                            color = BrewColors.inkSecondary,
                            modifier = Modifier.padding(top = 18.dp, bottom = 12.dp),
                        )
                    }
                    items(items.size) { i ->
                        val note = items[i]
                        NoteCard(
                            note = note,
                            onClick = { onOpenNote(note.id) },
                            onLongPress = { pendingDelete = note },
                            modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                        )
                    }
                }
            }
        }

        // Bottom controls.
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 16.dp),
        ) {
            if (recording.isRecording) {
                RecordingMiniBar(recording, onClick = onReopenRecording)
            } else {
                FloatingActionButtonBrew(
                    enabled = aiReady,
                    onClick = onStartRecording,
                    modifier = Modifier.align(Alignment.BottomEnd),
                )
            }
        }
    }

    pendingDelete?.let { note ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("Delete this note?") },
            text = { Text("The note, transcript, chat, and audio recording will be permanently deleted.") },
            confirmButton = {
                TextButton(onClick = {
                    vm.delete(note)
                    pendingDelete = null
                }) { Text("Delete", color = BrewColors.warning) }
            },
            dismissButton = {
                TextButton(onClick = { pendingDelete = null }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun ProfileButton(onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(46.dp)
            .clip(CircleShape)
            .background(BrewColors.profile)
            .clickable { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        Text("B", fontFamily = Serif, fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = BrewColors.ink)
    }
}

@Composable
private fun FloatingActionButtonBrew(enabled: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .size(64.dp)
            .clip(CircleShape)
            .background(if (enabled) BrewColors.accent else BrewColors.accent.copy(alpha = 0.45f))
            .clickable(enabled = enabled) { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        Icon(Icons.Filled.Add, contentDescription = "New recording", tint = androidx.compose.ui.graphics.Color.White, modifier = Modifier.size(26.dp))
    }
}

@Composable
private fun SearchField(query: String, onChange: (String) -> Unit) {
    androidx.compose.material3.OutlinedTextField(
        value = query,
        onValueChange = onChange,
        placeholder = { Text("Search notes") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

private fun buildTitle() = androidx.compose.ui.text.buildAnnotatedString {
    append("Your ")
    pushStyle(androidx.compose.ui.text.SpanStyle(fontStyle = FontStyle.Italic))
    append("Private")
    pop()
    append(" Notes")
}

private val dayHeaderFormat = SimpleDateFormat("EEE d MMM", Locale.getDefault())

private fun groupByDay(notes: List<NoteEntity>): List<Pair<String, List<NoteEntity>>> {
    val result = LinkedHashMap<String, MutableList<NoteEntity>>()
    for (note in notes) {
        val label = dayLabel(note.createdAt)
        result.getOrPut(label) { mutableListOf() }.add(note)
    }
    return result.map { it.key to it.value }
}

private fun dayLabel(epochMillis: Long): String {
    val cal = Calendar.getInstance().apply { timeInMillis = epochMillis }
    val today = Calendar.getInstance()
    val yesterday = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -1) }
    fun sameDay(a: Calendar, b: Calendar) =
        a.get(Calendar.YEAR) == b.get(Calendar.YEAR) &&
            a.get(Calendar.DAY_OF_YEAR) == b.get(Calendar.DAY_OF_YEAR)
    return when {
        sameDay(cal, today) -> "Today"
        sameDay(cal, yesterday) -> "Yesterday"
        else -> dayHeaderFormat.format(Date(epochMillis))
    }
}
