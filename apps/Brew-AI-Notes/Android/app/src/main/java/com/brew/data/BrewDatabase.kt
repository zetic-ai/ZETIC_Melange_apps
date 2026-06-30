package com.brew.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [NoteEntity::class, ChatMessageEntity::class],
    version = 1,
    exportSchema = false,
)
abstract class BrewDatabase : RoomDatabase() {
    abstract fun noteDao(): NoteDao
    abstract fun chatDao(): ChatDao

    companion object {
        @Volatile
        private var instance: BrewDatabase? = null

        fun get(context: Context): BrewDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    BrewDatabase::class.java,
                    "brew.db",
                ).build().also { instance = it }
            }
    }
}
