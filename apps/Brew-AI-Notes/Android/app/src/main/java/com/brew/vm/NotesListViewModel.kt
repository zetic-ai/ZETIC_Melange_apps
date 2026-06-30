package com.brew.vm

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.brew.BrewApplication
import com.brew.data.NoteEntity
import com.brew.engine.ModelPhase
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class NotesListViewModel(app: Application) : AndroidViewModel(app) {
    private val appCtx = app as BrewApplication
    private val repo = appCtx.repository
    private val coordinator = appCtx.coordinator

    val query = MutableStateFlow("")

    val notes: StateFlow<List<NoteEntity>> =
        combine(repo.observeNotes(), query) { all, q ->
            if (q.isBlank()) all
            else all.filter { note ->
                note.displayTitle.contains(q, true) ||
                    note.transcript.contains(q, true) ||
                    (note.enhancedNote?.contains(q, true) == true)
            }
        }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    val llmPhase: StateFlow<ModelPhase> = coordinator.llmPhase

    fun delete(note: NoteEntity) {
        viewModelScope.launch { repo.deleteNote(note) }
    }

    fun retryLlm() {
        coordinator.retryLlm()
        viewModelScope.launch { coordinator.preloadLlm() }
    }
}
