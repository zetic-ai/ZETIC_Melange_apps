package com.brew.vm

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.brew.BrewApplication
import com.brew.audio.WavAudioRecorder
import com.brew.data.NoteEntity
import com.brew.data.NoteStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

data class RecordingState(
    val isRecording: Boolean = false,
    val isPaused: Boolean = false,
    val elapsedSeconds: Int = 0,
    val level: Float = 0f,
    val healthy: Boolean = true,
)

/**
 * Owns the live recording session. Held once at the app shell so the mini-bar
 * on the list and the recording sheet share one source of truth.
 */
class RecordingViewModel(app: Application) : AndroidViewModel(app) {
    private val appCtx = app as BrewApplication
    private val repo = appCtx.repository
    private val worker = appCtx.transcriptionWorker

    private val _state = MutableStateFlow(RecordingState())
    val state: StateFlow<RecordingState> = _state.asStateFlow()

    private var recorder: WavAudioRecorder? = null
    private var audioFileName: String? = null
    private var lastFileCheckSamples = 0L

    fun start() {
        if (_state.value.isRecording) return
        val name = repo.newAudioFileName()
        audioFileName = name
        lastFileCheckSamples = 0L
        val file = repo.audioFile(name)
        val rec = WavAudioRecorder(
            outputFile = file,
            onLevel = { lvl -> _state.value = _state.value.copy(level = lvl) },
            onElapsed = { secs ->
                _state.value = _state.value.copy(elapsedSeconds = secs)
            },
        )
        recorder = rec
        rec.start()
        _state.value = RecordingState(isRecording = true)
    }

    fun pause() {
        recorder?.pause()
        _state.value = _state.value.copy(isPaused = true, level = 0f)
    }

    fun resume() {
        recorder?.resume()
        _state.value = _state.value.copy(isPaused = false)
    }

    /** Stops, persists a Note(transcribing), enqueues transcription. Returns its id. */
    fun stopAndSave(onSaved: (String) -> Unit) {
        val rec = recorder ?: return
        val name = audioFileName
        viewModelScope.launch {
            val duration = withContext(Dispatchers.IO) { rec.stop() }
            recorder = null
            _state.value = RecordingState()
            if (name == null) return@launch
            val id = UUID.randomUUID().toString()
            val note = NoteEntity(
                id = id,
                title = "",
                createdAt = System.currentTimeMillis(),
                durationSeconds = duration,
                audioFileName = name,
                status = NoteStatus.TRANSCRIBING.name,
            )
            repo.upsert(note)
            worker.enqueue(id)
            onSaved(id)
        }
    }

    fun cancel() {
        val rec = recorder ?: return
        val name = audioFileName
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                rec.stop()
                name?.let { repo.audioFile(it).delete() }
            }
            recorder = null
            audioFileName = null
            _state.value = RecordingState()
        }
    }
}
