package com.brew.vm

import android.app.Application
import android.text.format.Formatter
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.brew.BrewApplication
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class StorageStats(val notesCount: Int = 0, val recordingsLabel: String = "—")

class SettingsViewModel(app: Application) : AndroidViewModel(app) {
    private val appCtx = app as BrewApplication
    private val repo = appCtx.repository
    private val coordinator = appCtx.coordinator

    val llmPhase = coordinator.llmPhase
    val versionLabel = "1.0 (1)"

    private val _stats = MutableStateFlow(StorageStats())
    val stats: StateFlow<StorageStats> = _stats

    init {
        viewModelScope.launch {
            val (count, bytes) = withContext(Dispatchers.IO) {
                repo.getAllNotes().size to repo.recordingsBytes()
            }
            _stats.value = StorageStats(
                notesCount = count,
                recordingsLabel = Formatter.formatShortFileSize(appCtx, bytes),
            )
        }
    }

    fun retryLlm() {
        coordinator.retryLlm()
        viewModelScope.launch { coordinator.preloadLlm() }
    }
}
