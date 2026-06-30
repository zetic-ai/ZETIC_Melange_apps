package com.brew.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.brew.data.ChatMessageEntity
import com.brew.data.ChatRole
import com.brew.ui.components.MarkdownText
import com.brew.ui.components.ModelStatusChip
import com.brew.ui.components.TypingIndicator
import com.brew.ui.theme.BrewColors
import com.brew.vm.ChatViewModel
import com.brew.vm.NoteScopedViewModelFactory

@Composable
fun ChatSheet(noteId: String) {
    val app = androidx.compose.ui.platform.LocalContext.current.applicationContext as android.app.Application
    val vm: ChatViewModel = viewModel(
        factory = NoteScopedViewModelFactory(app, noteId),
        key = "chat-$noteId",
    )
    val messages by vm.messages.collectAsStateWithLifecycle()
    val ui by vm.ui.collectAsStateWithLifecycle()
    val phase by vm.llmPhase.collectAsStateWithLifecycle()
    var input by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    LaunchedEffect(messages.size, ui.streamingReply) {
        val count = messages.size + if (ui.responding) 1 else 0
        if (count > 0) listState.animateScrollToItem(count - 1)
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(BrewColors.canvas)
            .padding(horizontal = 4.dp),
    ) {
        Text(
            "Chat with note",
            fontSize = 16.sp,
            color = BrewColors.ink,
            modifier = Modifier.padding(16.dp).align(Alignment.CenterHorizontally),
        )
        LazyColumn(
            state = listState,
            modifier = Modifier.fillMaxWidth().heightIn(min = 200.dp, max = 460.dp).weight(1f, fill = false),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            if (messages.isEmpty() && !ui.responding) {
                item {
                    Text(
                        "Ask anything about this meeting — decisions, action items, who said what.",
                        fontSize = 16.sp,
                        color = BrewColors.inkSecondary,
                        modifier = Modifier.padding(top = 24.dp),
                    )
                }
            }
            items(messages.size) { i -> MessageBubble(messages[i]) }
            if (ui.responding) {
                item { StreamingBubble(ui.streamingReply) }
            }
            ui.error?.let { err ->
                item { Text(err, fontSize = 14.sp, color = BrewColors.warning) }
            }
        }

        ModelStatusChip(phase, modifier = Modifier.padding(horizontal = 16.dp))

        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                placeholder = { Text("Ask about this note…") },
                modifier = Modifier.weight(1f),
                shape = RoundedCornerShape(24.dp),
                maxLines = 4,
            )
            val canSend = input.isNotBlank() && !ui.responding
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(if (canSend) BrewColors.accent else BrewColors.inkTertiary)
                    .clickable(enabled = canSend) {
                        vm.send(input)
                        input = ""
                    },
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Filled.ArrowUpward, contentDescription = "Send", tint = Color.White, modifier = Modifier.size(18.dp))
            }
        }
    }
}

@Composable
private fun MessageBubble(message: ChatMessageEntity) {
    val isUser = message.chatRole == ChatRole.USER
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start) {
        if (isUser) Spacer(Modifier.width(40.dp))
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(18.dp))
                .background(if (isUser) BrewColors.accent else BrewColors.card)
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            if (isUser) {
                Text(message.content, fontSize = 16.sp, color = Color.White)
            } else {
                MarkdownText(message.content)
            }
        }
        if (!isUser) Spacer(Modifier.width(40.dp))
    }
}

@Composable
private fun StreamingBubble(text: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(18.dp))
                .background(BrewColors.card)
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            if (text.isBlank()) TypingIndicator() else MarkdownText(text)
        }
        Spacer(Modifier.width(40.dp))
    }
}
