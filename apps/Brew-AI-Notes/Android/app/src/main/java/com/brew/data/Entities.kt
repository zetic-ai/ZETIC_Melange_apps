package com.brew.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * A meeting note. Mirrors the iOS SwiftData `Note` model. Status is stored as
 * its enum name; the audio lives on disk (only the file name is persisted).
 */
@Entity(tableName = "notes")
data class NoteEntity(
    @PrimaryKey val id: String,
    val title: String = "",
    val createdAt: Long,                       // epoch millis
    val durationSeconds: Int = 0,
    val audioFileName: String? = null,         // file name in the recordings dir
    val transcript: String = "",
    val enhancedNote: String? = null,          // AI markdown note; null until generated
    val status: String = NoteStatus.TRANSCRIBING.name,
    val transcriptionErrorMessage: String? = null,
    val languageRaw: String? = "en-US",
) {
    val noteStatus: NoteStatus get() = NoteStatus.fromRaw(status)

    /** Trimmed title, or "New note" when empty (iOS `displayTitle`). */
    val displayTitle: String
        get() = title.trim().ifEmpty { "New note" }
}

/**
 * One chat message attached to a note. Cascade-deletes with its note.
 */
@Entity(
    tableName = "chat_messages",
    foreignKeys = [
        ForeignKey(
            entity = NoteEntity::class,
            parentColumns = ["id"],
            childColumns = ["noteId"],
            onDelete = ForeignKey.CASCADE,
        )
    ],
    indices = [Index("noteId")],
)
data class ChatMessageEntity(
    @PrimaryKey val id: String,
    val role: String,
    val content: String,
    val createdAt: Long,
    val noteId: String,
) {
    val chatRole: ChatRole get() = ChatRole.fromRaw(role)
}
