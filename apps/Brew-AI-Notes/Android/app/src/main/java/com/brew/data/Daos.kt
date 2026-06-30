package com.brew.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

@Dao
interface NoteDao {
    @Query("SELECT * FROM notes ORDER BY createdAt DESC")
    fun observeNotes(): Flow<List<NoteEntity>>

    @Query("SELECT * FROM notes WHERE id = :id")
    fun observeNote(id: String): Flow<NoteEntity?>

    @Query("SELECT * FROM notes WHERE id = :id")
    suspend fun getNote(id: String): NoteEntity?

    @Query("SELECT * FROM notes")
    suspend fun getAllNotes(): List<NoteEntity>

    @Query("SELECT * FROM notes WHERE status IN (:statuses)")
    suspend fun getNotesWithStatus(statuses: List<String>): List<NoteEntity>

    @Upsert
    suspend fun upsert(note: NoteEntity)

    @Query("DELETE FROM notes WHERE id = :id")
    suspend fun delete(id: String)
}

@Dao
interface ChatDao {
    @Query("SELECT * FROM chat_messages WHERE noteId = :noteId ORDER BY createdAt ASC")
    fun observeMessages(noteId: String): Flow<List<ChatMessageEntity>>

    @Query("SELECT * FROM chat_messages WHERE noteId = :noteId ORDER BY createdAt ASC")
    suspend fun getMessages(noteId: String): List<ChatMessageEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(message: ChatMessageEntity)
}
