package com.brew.data

/** Lifecycle of a note (ported from iOS `NoteStatus`). */
enum class NoteStatus {
    RECORDING,
    TRANSCRIBING,
    TRANSCRIPTION_FAILED,
    TRANSCRIBED,
    ENHANCING,
    ENHANCED;

    companion object {
        fun fromRaw(raw: String): NoteStatus =
            entries.firstOrNull { it.name == raw } ?: TRANSCRIBING
    }
}

/** Chat author (ported from iOS `ChatRole`). */
enum class ChatRole {
    USER,
    ASSISTANT;

    companion object {
        fun fromRaw(raw: String): ChatRole =
            entries.firstOrNull { it.name == raw } ?: USER
    }
}
