package com.brew.data

import android.content.Context
import kotlinx.coroutines.flow.Flow
import java.io.File
import java.util.UUID

/**
 * Single source of truth over Room + the on-disk recordings directory.
 * Audio files live in `filesDir/recordings/rec-<uuid>.wav`; only the file name
 * is persisted on the note.
 */
class NotesRepository(context: Context) {
    private val appContext = context.applicationContext
    private val db = BrewDatabase.get(appContext)
    private val noteDao = db.noteDao()
    private val chatDao = db.chatDao()

    val recordingsDir: File =
        File(appContext.filesDir, "recordings").apply { mkdirs() }

    fun observeNotes(): Flow<List<NoteEntity>> = noteDao.observeNotes()
    fun observeNote(id: String): Flow<NoteEntity?> = noteDao.observeNote(id)
    fun observeMessages(noteId: String): Flow<List<ChatMessageEntity>> =
        chatDao.observeMessages(noteId)

    suspend fun getNote(id: String): NoteEntity? = noteDao.getNote(id)
    suspend fun getAllNotes(): List<NoteEntity> = noteDao.getAllNotes()
    suspend fun getMessages(noteId: String): List<ChatMessageEntity> =
        chatDao.getMessages(noteId)

    suspend fun getNotesWithStatus(vararg statuses: NoteStatus): List<NoteEntity> =
        noteDao.getNotesWithStatus(statuses.map { it.name })

    suspend fun upsert(note: NoteEntity) = noteDao.upsert(note)

    suspend fun updateStatus(
        id: String,
        status: NoteStatus,
        errorMessage: String? = null,
    ) {
        val note = noteDao.getNote(id) ?: return
        noteDao.upsert(note.copy(status = status.name, transcriptionErrorMessage = errorMessage))
    }

    suspend fun setTranscript(id: String, transcript: String, status: NoteStatus) {
        val note = noteDao.getNote(id) ?: return
        noteDao.upsert(
            note.copy(
                transcript = transcript,
                status = status.name,
                transcriptionErrorMessage = null,
            )
        )
    }

    suspend fun setEnhancedNote(id: String, enhanced: String, title: String) {
        val note = noteDao.getNote(id) ?: return
        noteDao.upsert(
            note.copy(
                enhancedNote = enhanced,
                title = title,
                status = NoteStatus.ENHANCED.name,
            )
        )
    }

    suspend fun addMessage(noteId: String, role: ChatRole, content: String) {
        chatDao.insert(
            ChatMessageEntity(
                id = UUID.randomUUID().toString(),
                role = role.name,
                content = content,
                createdAt = System.currentTimeMillis(),
                noteId = noteId,
            )
        )
    }

    suspend fun deleteNote(note: NoteEntity) {
        note.audioFileName?.let { File(recordingsDir, it).delete() }
        noteDao.delete(note.id)
    }

    fun audioFile(name: String): File = File(recordingsDir, name)

    fun newAudioFileName(): String = "rec-${UUID.randomUUID()}.wav"

    /** Total bytes of all recordings on disk (for settings storage stats). */
    fun recordingsBytes(): Long =
        recordingsDir.listFiles()?.sumOf { it.length() } ?: 0L
}
