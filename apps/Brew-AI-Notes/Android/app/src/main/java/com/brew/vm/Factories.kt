package com.brew.vm

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider

/** Factory for ViewModels that take a noteId alongside the Application. */
class NoteScopedViewModelFactory(
    private val app: Application,
    private val noteId: String,
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = when {
        modelClass.isAssignableFrom(NoteDetailViewModel::class.java) ->
            NoteDetailViewModel(app, noteId) as T
        modelClass.isAssignableFrom(ChatViewModel::class.java) ->
            ChatViewModel(app, noteId) as T
        else -> throw IllegalArgumentException("Unknown ViewModel ${modelClass.name}")
    }
}
