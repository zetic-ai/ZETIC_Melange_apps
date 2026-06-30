package com.brew.asr

import com.brew.audio.WavIo
import com.brew.data.NoteStatus
import com.brew.data.NotesRepository
import com.brew.engine.ModelCoordinator
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.Collections

/**
 * Process-scoped background transcription (mirrors iOS `TranscriptionWorker`).
 * Reads a note's WAV, runs Whisper, and writes the transcript back. Resilient:
 * a failure keeps the audio for retry; empty transcript = "no speech", not an
 * error. Crash recovery resumes interrupted work on launch.
 */
class TranscriptionWorker(
    private val repo: NotesRepository,
    private val coordinator: ModelCoordinator,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val queueLock = Mutex()
    private val active = Collections.synchronizedSet(HashSet<String>())

    fun isActive(noteId: String): Boolean = active.contains(noteId)

    /** Enqueue a note for transcription (no-op if already running). */
    fun enqueue(noteId: String) {
        if (!active.add(noteId)) return
        scope.launch {
            try {
                transcribeNote(noteId)
            } finally {
                active.remove(noteId)
            }
        }
    }

    private suspend fun transcribeNote(noteId: String) {
        val note = repo.getNote(noteId) ?: return
        val fileName = note.audioFileName
        if (fileName == null) {
            repo.updateStatus(noteId, NoteStatus.TRANSCRIPTION_FAILED, "Recording file is missing.")
            return
        }
        val file = repo.audioFile(fileName)
        if (!file.exists() || file.length() <= 44L) {
            repo.updateStatus(noteId, NoteStatus.TRANSCRIPTION_FAILED, "Recording file is missing.")
            return
        }

        repo.updateStatus(noteId, NoteStatus.TRANSCRIBING)
        try {
            // Serialize the heavy read+inference so two notes never contend.
            val transcript = queueLock.withLock {
                withContext(Dispatchers.IO) {
                    val audio = WavIo.readMono16k(file)
                    coordinator.transcribe(audio)
                }
            }
            repo.setTranscript(noteId, transcript.trim(), NoteStatus.TRANSCRIBED)
        } catch (t: Throwable) {
            repo.updateStatus(
                noteId,
                NoteStatus.TRANSCRIPTION_FAILED,
                t.message ?: "Transcription failed.",
            )
        }
    }

    /**
     * On launch: notes stuck in TRANSCRIBING are retried; notes left ENHANCING
     * with no enhanced note revert to TRANSCRIBED so the detail screen can
     * regenerate. (Audio for failed notes is kept for manual retry.)
     */
    fun recoverInterruptedWork() {
        scope.launch {
            repo.getNotesWithStatus(NoteStatus.ENHANCING).forEach { note ->
                if (note.enhancedNote.isNullOrEmpty()) {
                    repo.updateStatus(note.id, NoteStatus.TRANSCRIBED)
                }
            }
            repo.getNotesWithStatus(NoteStatus.TRANSCRIBING).forEach { note ->
                enqueue(note.id)
            }
        }
    }
}
