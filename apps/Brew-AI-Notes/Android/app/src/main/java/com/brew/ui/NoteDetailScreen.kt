package com.brew.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CalendarToday
import androidx.compose.material.icons.filled.ChatBubble
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.brew.data.NoteEntity
import com.brew.data.NoteStatus
import com.brew.engine.ModelPhase
import com.brew.ui.components.MarkdownText
import com.brew.ui.theme.BrewColors
import com.brew.ui.theme.Serif
import com.brew.vm.NoteDetailViewModel
import com.brew.vm.NoteScopedViewModelFactory
import java.text.DateFormat
import java.util.Date

@Composable
fun NoteDetailScreen(
    noteId: String,
    onBack: () -> Unit,
    onOpenChat: () -> Unit,
) {
    val app = androidx.compose.ui.platform.LocalContext.current.applicationContext as android.app.Application
    val vm: NoteDetailViewModel = viewModel(
        factory = NoteScopedViewModelFactory(app, noteId),
        key = "detail-$noteId",
    )
    val note by vm.note.collectAsStateWithLifecycle()
    val enhance by vm.enhance.collectAsStateWithLifecycle()
    val phase by vm.llmPhase.collectAsStateWithLifecycle()
    var tab by remember { mutableStateOf(0) } // 0 = Note, 1 = Transcript

    LaunchedEffect(note?.status, note?.transcript) {
        note?.let { vm.onNoteChanged(it) }
    }

    Box(modifier = Modifier.fillMaxSize().background(BrewColors.canvas)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 22.dp),
        ) {
            Spacer(Modifier.height(12.dp))
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(BrewColors.card)
                    .clickable { onBack() },
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = BrewColors.ink, modifier = Modifier.size(17.dp))
            }

            val current = note
            if (current == null) {
                Spacer(Modifier.height(40.dp))
                CircularProgressIndicator(color = BrewColors.iconTileInk)
            } else {
                Spacer(Modifier.height(14.dp))
                Text(current.displayTitle, fontFamily = Serif, fontSize = 31.sp, color = BrewColors.ink)
                Spacer(Modifier.height(10.dp))
                DatePill(current.createdAt)
                Spacer(Modifier.height(18.dp))
                SegmentedTabs(tab) { tab = it }
                Spacer(Modifier.height(16.dp))
                GatedContent(current, tab, enhance, phase, vm)
                Spacer(Modifier.height(120.dp))
            }
        }

        // Floating chat pill.
        if (note != null) {
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 16.dp)
                    .clip(RoundedCornerShape(28.dp))
                    .background(BrewColors.accent)
                    .clickable { onOpenChat() }
                    .padding(horizontal = 28.dp, vertical = 16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Filled.ChatBubble, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(10.dp))
                Text("Chat with note", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
            }
        }
    }
}

@Composable
private fun DatePill(createdAt: Long) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(BrewColors.cardElevated)
            .padding(horizontal = 14.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Filled.CalendarToday, contentDescription = null, tint = BrewColors.inkSecondary, modifier = Modifier.size(14.dp))
        Spacer(Modifier.width(8.dp))
        Text(
            DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(createdAt)),
            fontSize = 15.sp,
            color = BrewColors.inkSecondary,
        )
    }
}

@Composable
private fun SegmentedTabs(selected: Int, onSelect: (Int) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(BrewColors.card)
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        listOf("Note", "Transcript").forEachIndexed { i, label ->
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(9.dp))
                    .background(if (selected == i) BrewColors.cardElevated else Color.Transparent)
                    .clickable { onSelect(i) }
                    .padding(vertical = 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    label,
                    fontSize = 15.sp,
                    fontWeight = if (selected == i) FontWeight.SemiBold else FontWeight.Normal,
                    color = if (selected == i) BrewColors.ink else BrewColors.inkSecondary,
                )
            }
        }
    }
}

@Composable
private fun GatedContent(
    note: NoteEntity,
    tab: Int,
    enhance: com.brew.vm.EnhanceState,
    phase: ModelPhase,
    vm: NoteDetailViewModel,
) {
    when (note.noteStatus) {
        NoteStatus.TRANSCRIBING -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = BrewColors.inkSecondary)
            Spacer(Modifier.width(10.dp))
            Text("Transcribing recording…", fontSize = 16.sp, color = BrewColors.inkSecondary)
        }

        NoteStatus.TRANSCRIPTION_FAILED -> Column {
            Text("Transcription failed.", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = BrewColors.ink)
            note.transcriptionErrorMessage?.let {
                Spacer(Modifier.height(4.dp))
                Text(it, fontSize = 14.sp, color = BrewColors.inkSecondary)
            }
            Spacer(Modifier.height(4.dp))
            Text("Your recording is saved, so nothing is lost.", fontSize = 14.sp, color = BrewColors.inkSecondary)
            Spacer(Modifier.height(12.dp))
            ActionButton("Retry transcription") { vm.retryTranscription() }
        }

        else -> if (tab == 0) NoteTab(note, enhance, phase, vm) else TranscriptTab(note, vm)
    }
}

@Composable
private fun NoteTab(note: NoteEntity, enhance: com.brew.vm.EnhanceState, phase: ModelPhase, vm: NoteDetailViewModel) {
    when {
        enhance.generating -> Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = BrewColors.inkSecondary)
                Spacer(Modifier.width(10.dp))
                Text(phaseLabel(phase), fontSize = 16.sp, color = BrewColors.inkSecondary)
            }
            if (enhance.streamingText.isNotBlank()) {
                Spacer(Modifier.height(12.dp))
                MarkdownText(enhance.streamingText)
            }
        }

        enhance.error != null -> Column {
            Text("Couldn't generate the note.", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = BrewColors.ink)
            Spacer(Modifier.height(4.dp))
            Text(enhance.error!!, fontSize = 14.sp, color = BrewColors.inkSecondary)
            Spacer(Modifier.height(12.dp))
            ActionButton("Try again") { vm.generate() }
        }

        !note.enhancedNote.isNullOrEmpty() -> MarkdownText(note.enhancedNote!!)

        else -> Column {
            Text("No AI note yet.", fontSize = 17.sp, color = BrewColors.inkSecondary)
            Spacer(Modifier.height(12.dp))
            ActionButton("Generate note", prominent = true) { vm.generate() }
        }
    }
}

@Composable
private fun TranscriptTab(note: NoteEntity, vm: NoteDetailViewModel) {
    if (note.transcript.isBlank()) {
        Column {
            Text("No speech was detected in this recording.", fontSize = 17.sp, color = BrewColors.inkTertiary)
            Spacer(Modifier.height(12.dp))
            ActionButton("Retry transcription") { vm.retryTranscription() }
        }
    } else {
        Text(note.transcript, fontSize = 17.sp, color = BrewColors.ink)
    }
}

private fun phaseLabel(phase: ModelPhase): String = when (phase) {
    is ModelPhase.Downloading -> "Downloading AI model… ${(phase.progress * 100).toInt()}%"
    is ModelPhase.Preparing, is ModelPhase.Idle -> "Preparing AI…"
    is ModelPhase.Failed -> "AI unavailable"
    is ModelPhase.Ready -> "Generating note…"
}

@Composable
private fun ActionButton(label: String, prominent: Boolean = false, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(22.dp))
            .background(if (prominent) BrewColors.accent else BrewColors.card)
            .clickable { onClick() }
            .padding(horizontal = 20.dp, vertical = 12.dp),
    ) {
        Text(label, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = if (prominent) Color.White else BrewColors.ink)
    }
}
